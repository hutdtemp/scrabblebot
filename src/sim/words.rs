//! Struct-of-arrays word index tuned for the simulator and the evolved policy.
//!
//! The central operation is [`WordIndex::analyze`]: a single pass over the
//! dictionary that reports, for a rack, the best word available now, the best
//! word after receiving any one extra letter, and the best word after
//! receiving the auctioned letter plus any one more. That is exactly the
//! "score now / with this letter / with this letter and the next" lookahead,
//! computed for all 26 follow-up letters at once.

use std::sync::{Arc, OnceLock};

use crate::core::dictionary::Dictionary;

pub type Counts = [u8; 26];

/// Words of at least this length get the 3x multiplier.
pub const LONG_LEN: u8 = 7;

pub struct WordIndex {
    // Parallel arrays in reward-descending order (ties: longer, then lex).
    masks: Vec<u32>,
    lens: Vec<u8>,
    rewards: Vec<u16>,
    counts: Vec<Counts>,
    texts: Vec<String>,
    /// Indices in upstream simulated-bot order: length desc, then lex asc.
    by_len_lex: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// Best reward playable now (0 if none) and its word index.
    pub best0: u32,
    pub best0_id: Option<u32>,
    /// Best reward among words of length >= LONG_LEN playable now.
    pub best0_long: u32,
    /// Best reward of a word needing exactly one extra copy of letter x.
    pub best1: [u32; 26],
    pub best1_long: [u32; 26],
    /// Given the auction letter L: best word needing L plus one extra x.
    pub best2l: [u32; 26],
    pub best2l_long: [u32; 26],
    /// Count of long words one / two letters away (rack flexibility).
    pub flex1_long: u32,
    pub flex2_long: u32,
}

impl Analysis {
    /// Best playable reward after adding letter `x`.
    pub fn with1(&self, x: usize) -> u32 {
        self.best0.max(self.best1[x])
    }
    pub fn with1_long(&self, x: usize) -> u32 {
        self.best0_long.max(self.best1_long[x])
    }
    /// Best playable reward after adding L (the analysis letter) and `x`.
    pub fn with2(&self, l: usize, x: usize) -> u32 {
        self.best0
            .max(self.best1[l])
            .max(self.best1[x])
            .max(self.best2l[x])
    }
    pub fn with2_long(&self, l: usize, x: usize) -> u32 {
        self.best0_long
            .max(self.best1_long[l])
            .max(self.best1_long[x])
            .max(self.best2l_long[x])
    }
}

impl WordIndex {
    pub fn from_dictionary(dict: &Dictionary) -> Self {
        let n = dict.word_count();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| {
            let wa = dict.word(a);
            let wb = dict.word(b);
            wb.reward
                .cmp(&wa.reward)
                .then(wb.len.cmp(&wa.len))
                .then(wa.text.cmp(&wb.text))
        });
        let mut idx = WordIndex {
            masks: Vec::with_capacity(n),
            lens: Vec::with_capacity(n),
            rewards: Vec::with_capacity(n),
            counts: Vec::with_capacity(n),
            texts: Vec::with_capacity(n),
            by_len_lex: Vec::new(),
        };
        for &i in &order {
            let w = dict.word(i);
            idx.masks.push(w.mask);
            idx.lens.push(w.len);
            idx.rewards.push(w.reward.min(u16::MAX as u32) as u16);
            idx.counts.push(w.counts);
            idx.texts.push(w.text.clone());
        }
        let mut by_len_lex: Vec<u32> = (0..n as u32).collect();
        by_len_lex.sort_by(|&a, &b| {
            idx.lens[b as usize]
                .cmp(&idx.lens[a as usize])
                .then(idx.texts[a as usize].cmp(&idx.texts[b as usize]))
        });
        idx.by_len_lex = by_len_lex;
        idx
    }

    pub fn bundled() -> Arc<WordIndex> {
        static IDX: OnceLock<Arc<WordIndex>> = OnceLock::new();
        IDX.get_or_init(|| Arc::new(WordIndex::from_dictionary(Dictionary::bundled())))
            .clone()
    }

    pub fn len(&self) -> usize {
        self.masks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.masks.is_empty()
    }
    pub fn text(&self, id: u32) -> &str {
        &self.texts[id as usize]
    }
    pub fn word_len(&self, id: u32) -> u8 {
        self.lens[id as usize]
    }
    pub fn reward(&self, id: u32) -> u32 {
        self.rewards[id as usize] as u32
    }
    pub fn counts(&self, id: u32) -> &Counts {
        &self.counts[id as usize]
    }

    /// Look up a word by text (linear; only for tests and live glue).
    pub fn find(&self, text: &str) -> Option<u32> {
        self.texts.iter().position(|t| t == text).map(|i| i as u32)
    }

    #[inline]
    fn fits(&self, i: usize, rack: &Counts) -> bool {
        let c = &self.counts[i];
        (0..26).all(|k| c[k] <= rack[k])
    }

    /// Highest-reward word of length >= `min_len` playable from `rack`.
    pub fn best_word(&self, rack: &Counts, min_len: u8) -> Option<u32> {
        let total: u32 = rack.iter().map(|&n| n as u32).sum();
        if total < 2 {
            return None;
        }
        let rmask = mask_of(rack);
        for i in 0..self.masks.len() {
            if self.masks[i] & !rmask != 0 {
                continue;
            }
            let len = self.lens[i];
            if len < min_len || len as u32 > total {
                continue;
            }
            if self.fits(i, rack) {
                return Some(i as u32);
            }
        }
        None
    }

    /// The upstream simulated bots' choice: the longest playable word of at
    /// least `min_len` letters, lexicographically first among equals.
    pub fn sim_word(&self, rack: &Counts, min_len: u8) -> Option<u32> {
        let total: u32 = rack.iter().map(|&n| n as u32).sum();
        if total < min_len as u32 {
            return None;
        }
        let rmask = mask_of(rack);
        for &i in &self.by_len_lex {
            let i = i as usize;
            let len = self.lens[i];
            if len < min_len {
                break;
            }
            if len as u32 > total || self.masks[i] & !rmask != 0 {
                continue;
            }
            if self.fits(i, rack) {
                return Some(i as u32);
            }
        }
        None
    }

    /// Greedy liquidation by reward. Returns (total reward, words).
    pub fn liquidate(&self, rack: &Counts) -> (u32, Vec<u32>) {
        let mut r = *rack;
        let mut words = Vec::new();
        let mut total = 0;
        while let Some(id) = self.best_word(&r, 2) {
            remove_word(&mut r, self.counts(id));
            total += self.reward(id);
            words.push(id);
        }
        (total, words)
    }

    /// One-pass lookahead analysis. `letter` is the auctioned letter, if any.
    pub fn analyze(&self, rack: &Counts, letter: Option<usize>) -> Analysis {
        let mut a = Analysis::default();
        let total: u32 = rack.iter().map(|&n| n as u32).sum();
        let rmask = mask_of(rack);
        let mut have0 = false;
        let mut have0_long = false;
        // Bit sets of letters whose best1/best1_long/best2l slots are filled.
        let (mut f1, mut f1l, mut f2, mut f2l) = (0u32, 0u32, 0u32, 0u32);
        for i in 0..self.masks.len() {
            let wmask = self.masks[i];
            let miss = wmask & !rmask;
            if miss.count_ones() > 2 {
                continue;
            }
            let len = self.lens[i];
            if len as u32 > total + 2 {
                continue;
            }
            // Deficit computation over the word's letters only.
            let c = &self.counts[i];
            let mut d = 0u32;
            let mut m = [0usize; 2];
            let mut bits = wmask;
            while bits != 0 {
                let k = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let need = c[k];
                let have = rack[k];
                if need > have {
                    let extra = (need - have) as u32;
                    if d + extra > 2 {
                        d = 3;
                        break;
                    }
                    for _ in 0..extra {
                        m[d as usize] = k;
                        d += 1;
                    }
                }
            }
            let reward = self.rewards[i] as u32;
            let long = len >= LONG_LEN;
            match d {
                0 => {
                    if !have0 {
                        have0 = true;
                        a.best0 = reward;
                        a.best0_id = Some(i as u32);
                    }
                    if long && !have0_long {
                        have0_long = true;
                        a.best0_long = reward;
                    }
                }
                1 => {
                    let x = m[0];
                    if f1 & (1 << x) == 0 {
                        f1 |= 1 << x;
                        a.best1[x] = reward;
                    }
                    if long {
                        a.flex1_long += 1;
                        if f1l & (1 << x) == 0 {
                            f1l |= 1 << x;
                            a.best1_long[x] = reward;
                        }
                    }
                }
                2 => {
                    if long {
                        a.flex2_long += 1;
                    }
                    if let Some(l) = letter {
                        let other = if m[0] == l {
                            Some(m[1])
                        } else if m[1] == l {
                            Some(m[0])
                        } else {
                            None
                        };
                        if let Some(x) = other {
                            if f2 & (1 << x) == 0 {
                                f2 |= 1 << x;
                                a.best2l[x] = reward;
                            }
                            if long && f2l & (1 << x) == 0 {
                                f2l |= 1 << x;
                                a.best2l_long[x] = reward;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        a
    }
}

#[inline]
pub fn mask_of(rack: &Counts) -> u32 {
    let mut m = 0u32;
    for (k, &n) in rack.iter().enumerate() {
        if n > 0 {
            m |= 1 << k;
        }
    }
    m
}

#[inline]
pub fn remove_word(rack: &mut Counts, word: &Counts) {
    for k in 0..26 {
        rack[k] -= word[k];
    }
}

#[inline]
pub fn fits(rack: &Counts, word: &Counts) -> bool {
    (0..26).all(|k| word[k] <= rack[k])
}

pub fn counts_from(letters: &str) -> Counts {
    let mut c = [0u8; 26];
    for b in letters.bytes() {
        c[(b.to_ascii_uppercase() - b'A') as usize] += 1;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx() -> WordIndex {
        WordIndex::from_dictionary(&Dictionary::from_wordlist(
            "QI\nQUIZ\nCAT\nCATS\nSCAT\nACT\nCOAT\nCOATS\nCOASTED\n",
        ))
    }

    #[test]
    fn analyze_matches_brute_force() {
        let w = idx();
        let rack = counts_from("CAT");
        let a = w.analyze(&rack, Some(18)); // S
        assert_eq!(a.best0, w.reward(w.find("CAT").unwrap()));
        // With S: CATS/SCAT (base 6, 4 letters -> 9).
        assert_eq!(a.with1(18), 9);
        // With S then O: COATS = 7 base * 2 = 14.
        assert_eq!(a.with2(18, 14), 14);
        // Brute force check for every follow-up letter.
        for x in 0..26 {
            let mut r = rack;
            r[18] += 1;
            r[x] += 1;
            let brute = w.best_word(&r, 2).map(|i| w.reward(i)).unwrap_or(0);
            assert_eq!(a.with2(18, x), brute, "letter {x}");
            let mut r1 = rack;
            r1[x] += 1;
            let brute1 = w.best_word(&r1, 2).map(|i| w.reward(i)).unwrap_or(0);
            assert_eq!(a.with1(x), brute1, "letter {x}");
        }
    }

    #[test]
    fn analyze_matches_brute_force_on_bundled_dictionary() {
        let w = WordIndex::bundled();
        let mut rng = crate::sim::rng::Rng::new(7);
        for _ in 0..30 {
            let mut rack = [0u8; 26];
            let n = 3 + rng.below(12) as usize;
            for _ in 0..n {
                rack[rng.below(26) as usize] += 1;
            }
            let l = rng.below(26) as usize;
            let a = w.analyze(&rack, Some(l));
            let b0 = w.best_word(&rack, 2).map(|i| w.reward(i)).unwrap_or(0);
            assert_eq!(a.best0, b0);
            let b0l = w
                .best_word(&rack, LONG_LEN)
                .map(|i| w.reward(i))
                .unwrap_or(0);
            assert_eq!(a.best0_long, b0l);
            for x in [0usize, 4, 18, 25, l] {
                let mut r = rack;
                r[l] += 1;
                r[x] += 1;
                let brute = w.best_word(&r, 2).map(|i| w.reward(i)).unwrap_or(0);
                assert_eq!(a.with2(l, x), brute);
                let brute_l = w.best_word(&r, LONG_LEN).map(|i| w.reward(i)).unwrap_or(0);
                assert_eq!(a.with2_long(l, x), brute_l);
            }
        }
    }

    #[test]
    fn sim_word_prefers_longest_then_lex() {
        let w = idx();
        let rack = counts_from("CATSO");
        assert_eq!(w.text(w.sim_word(&rack, 3).unwrap()), "COATS");
        let rack = counts_from("CATS");
        assert_eq!(w.text(w.sim_word(&rack, 3).unwrap()), "CATS");
        assert_eq!(w.sim_word(&counts_from("QI"), 3), None);
    }
}
