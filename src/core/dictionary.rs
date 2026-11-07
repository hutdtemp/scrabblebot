//! Exact dictionary word solver over the bundled ENABLE list.
//!
//! Two exact strategies are used for "best playable word":
//! - For small racks, enumerate every sub-multiset of the rack and look it up
//!   in a signature map. This is fast because a 10-letter rack has at most
//!   2^10 sub-multisets.
//! - For large racks, scan words in descending reward order and return the
//!   first one whose letters are all present. The presence-mask test rejects
//!   most words in a single bitwise operation.
//!
//! Both return the same answer: the playable word with the highest exact
//! module reward, tie-broken by length (longer first), then lexicographically.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::letters::score_word;
use super::rack::Rack;

/// Sub-multiset enumeration is used when the rack has at most this many
/// sub-multisets. Above it, the descending-reward scan is cheaper.
const SUBSET_ENUMERATION_LIMIT: u64 = 200_000;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordEntry {
    pub text: String,
    pub len: u8,
    pub base: u32,
    pub reward: u32,
    pub mask: u32,
    #[serde(skip)]
    pub counts: [u8; 26],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordChoice {
    pub word: String,
    pub len: u8,
    pub base: u32,
    pub reward: u32,
}

impl From<&WordEntry> for WordChoice {
    fn from(word: &WordEntry) -> Self {
        Self {
            word: word.text.clone(),
            len: word.len,
            base: word.base,
            reward: word.reward,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Partition {
    pub reward: u32,
    pub words: Vec<WordChoice>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Target {
    pub word: String,
    pub len: u8,
    pub base: u32,
    pub reward: u32,
    pub missing: Rack,
}

pub struct Dictionary {
    words: Vec<WordEntry>,
    /// Word indices sorted by reward desc, length desc, text asc.
    by_reward: Vec<u32>,
    /// Word indices grouped by length, each group sorted by reward desc, text asc.
    by_len: Vec<Vec<u32>>,
    /// Canonical letter-count signature -> best word index.
    signatures: HashMap<[u8; 26], u32>,
    max_word_len: usize,
}

impl Dictionary {
    /// Parse a newline-separated wordlist. Invalid lines are ignored.
    pub fn from_wordlist(text: &str) -> Self {
        let mut words = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            let Some(word) = super::letters::normalize_word(line) else {
                continue;
            };
            let Some(score) = score_word(&word) else {
                continue;
            };
            let mut counts = [0u8; 26];
            let mut mask = 0u32;
            for b in word.bytes() {
                let idx = (b - b'A') as usize;
                counts[idx] = counts[idx].saturating_add(1);
                mask |= 1 << idx;
            }
            words.push(WordEntry {
                text: word,
                len: counts.iter().sum::<u8>(),
                base: score.base,
                reward: score.reward,
                mask,
                counts,
            });
        }

        // The upstream list is unique and lex-sorted, but deduplicate
        // defensively so a hand-written test list cannot double-count.
        words.sort_by(|a, b| a.text.cmp(&b.text));
        words.dedup_by(|a, b| a.text == b.text);

        let mut by_reward: Vec<u32> = (0..words.len() as u32).collect();
        by_reward.sort_by(|&a, &b| {
            let wa = &words[a as usize];
            let wb = &words[b as usize];
            wb.reward
                .cmp(&wa.reward)
                .then(wb.len.cmp(&wa.len))
                .then(wa.text.cmp(&wb.text))
        });

        let max_word_len = words.iter().map(|w| w.len as usize).max().unwrap_or(0);
        let mut by_len: Vec<Vec<u32>> = vec![Vec::new(); max_word_len + 1];
        for (i, word) in words.iter().enumerate() {
            by_len[word.len as usize].push(i as u32);
        }
        for group in by_len.iter_mut() {
            group.sort_by(|&a, &b| {
                let wa = &words[a as usize];
                let wb = &words[b as usize];
                wb.reward.cmp(&wa.reward).then(wa.text.cmp(&wb.text))
            });
        }

        let mut signatures: HashMap<[u8; 26], u32> = HashMap::with_capacity(words.len());
        for (i, word) in words.iter().enumerate() {
            let i = i as u32;
            match signatures.get(&word.counts) {
                Some(&existing) if !is_better(word, &words[existing as usize]) => {}
                _ => {
                    signatures.insert(word.counts, i);
                }
            }
        }

        Self {
            words,
            by_reward,
            by_len,
            signatures,
            max_word_len,
        }
    }

    /// The bundled upstream ENABLE wordlist, parsed once per process.
    pub fn bundled() -> &'static Dictionary {
        static BUNDLED: OnceLock<std::sync::Arc<Dictionary>> = OnceLock::new();
        BUNDLED
            .get_or_init(|| {
                std::sync::Arc::new(Self::from_wordlist(include_str!(
                    "../../data/enable_wordlist.txt"
                )))
            })
            .as_ref()
    }

    /// Shared handle to the bundled dictionary, for the search engine.
    pub fn bundled_arc() -> std::sync::Arc<Dictionary> {
        static BUNDLED: OnceLock<std::sync::Arc<Dictionary>> = OnceLock::new();
        BUNDLED
            .get_or_init(|| {
                std::sync::Arc::new(Self::from_wordlist(include_str!(
                    "../../data/enable_wordlist.txt"
                )))
            })
            .clone()
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    pub fn max_word_len(&self) -> usize {
        self.max_word_len
    }

    pub fn word(&self, index: u32) -> &WordEntry {
        &self.words[index as usize]
    }

    /// The highest-reward playable word, or `None`.
    pub fn best_word(&self, rack: &Rack) -> Option<&WordEntry> {
        if rack.total() < 2 {
            return None;
        }
        if rack.submultiset_count_capped(SUBSET_ENUMERATION_LIMIT) <= SUBSET_ENUMERATION_LIMIT {
            self.best_word_by_submultisets(rack)
        } else {
            self.best_word_by_scan(rack)
        }
    }

    fn best_word_by_submultisets(&self, rack: &Rack) -> Option<&WordEntry> {
        let mut current = [0u8; 26];
        let mut best: Option<u32> = None;
        self.enumerate_submultisets(rack, 0, 0, &mut current, &mut best);
        best.map(|i| &self.words[i as usize])
    }

    fn enumerate_submultisets(
        &self,
        rack: &Rack,
        index: usize,
        len: usize,
        current: &mut [u8; 26],
        best: &mut Option<u32>,
    ) {
        if index == 26 {
            if len < 2 {
                return;
            }
            if let Some(&word_index) = self.signatures.get(current) {
                let word = &self.words[word_index as usize];
                match best {
                    Some(existing) if !is_better(word, &self.words[*existing as usize]) => {}
                    _ => *best = Some(word_index),
                }
            }
            return;
        }

        let max = rack.counts[index] as usize;
        for n in 0..=max {
            if len + n > self.max_word_len {
                break;
            }
            current[index] = n as u8;
            self.enumerate_submultisets(rack, index + 1, len + n, current, best);
        }
        current[index] = 0;
    }

    fn best_word_by_scan(&self, rack: &Rack) -> Option<&WordEntry> {
        let rack_mask = rack.mask();
        let total = rack.total();
        for &index in &self.by_reward {
            let word = &self.words[index as usize];
            if word.len as u32 > total {
                continue;
            }
            if word.mask & !rack_mask != 0 {
                continue;
            }
            if self.word_fits(word, rack) {
                return Some(word);
            }
        }
        None
    }

    fn word_fits(&self, word: &WordEntry, rack: &Rack) -> bool {
        for i in 0..26 {
            if word.counts[i] > rack.counts[i] {
                return false;
            }
        }
        true
    }

    /// Greedy liquidation: repeatedly play the highest-reward playable word.
    ///
    /// This is deliberately greedy, not an exact set-packing optimum. It is
    /// the value function used by the search and the playback policy; the
    /// README states this limitation honestly.
    pub fn greedy_partition(&self, rack: &Rack) -> Partition {
        let mut remaining = *rack;
        let mut words = Vec::new();
        let mut reward = 0u32;
        while let Some(word) = self.best_word(&remaining) {
            let choice = WordChoice::from(word);
            let text = choice.word.clone();
            if !remaining.remove_word(&text) {
                debug_assert!(false, "best_word returned an unplayable word");
                break;
            }
            reward += choice.reward;
            words.push(choice);
        }
        Partition { reward, words }
    }

    /// The best word that is not playable yet but is within `max_missing`
    /// letters of the rack, provided it beats the best playable word.
    ///
    /// Used by the target-aware playback policy. The scan is exact and
    /// descending by reward, so the first qualifying word is the best target.
    pub fn best_target(&self, rack: &Rack, max_missing: u8) -> Option<Target> {
        let best_now = self.best_word(rack).map(|w| w.reward).unwrap_or(0);
        let total = rack.total();
        for &index in &self.by_reward {
            let word = &self.words[index as usize];
            if word.reward <= best_now {
                break;
            }
            if word.len as u32 > total + max_missing as u32 {
                continue;
            }
            // A quick mask test: a word with a missing letter is still a
            // candidate, so we cannot reject on `word.mask & !rack_mask`.
            let mut missing = 0u32;
            let mut missing_rack = Rack::new();
            let mut fits = true;
            for i in 0..26 {
                let need = word.counts[i];
                let have = rack.counts[i];
                if need > have {
                    missing += (need - have) as u32;
                    if missing > max_missing as u32 {
                        fits = false;
                        break;
                    }
                    missing_rack.add(i, need - have);
                }
            }
            if fits && missing > 0 {
                return Some(Target {
                    word: word.text.clone(),
                    len: word.len,
                    base: word.base,
                    reward: word.reward,
                    missing: missing_rack,
                });
            }
        }
        None
    }

    /// All playable words, descending by reward. Intended for tests and
    /// diagnostics; the live path uses `best_word`.
    pub fn playable_words(&self, rack: &Rack) -> Vec<&WordEntry> {
        let rack_mask = rack.mask();
        let total = rack.total();
        self.by_reward
            .iter()
            .filter_map(|&index| {
                let word = &self.words[index as usize];
                if word.len as u32 <= total
                    && word.mask & !rack_mask == 0
                    && self.word_fits(word, rack)
                {
                    Some(word)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Whether a word is in the dictionary (case-insensitive).
    pub fn contains(&self, word: &str) -> bool {
        let Some(normalized) = super::letters::normalize_word(word) else {
            return false;
        };
        let mut counts = [0u8; 26];
        for b in normalized.bytes() {
            counts[(b - b'A') as usize] += 1;
        }
        self.signatures.contains_key(&counts)
    }

    /// Words grouped by length, for diagnostics.
    pub fn words_of_len(&self, len: usize) -> usize {
        self.by_len.get(len).map(|v| v.len()).unwrap_or(0)
    }
}

fn is_better(candidate: &WordEntry, current: &WordEntry) -> bool {
    candidate.reward > current.reward
        || (candidate.reward == current.reward
            && (candidate.len > current.len
                || (candidate.len == current.len && candidate.text < current.text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_dict() -> Dictionary {
        Dictionary::from_wordlist(
            "AA\nAAL\nAALII\nAB\nABLY\nQUIZ\nQUIZZES\nJAZZ\nJAZZY\nCAT\nDOG\n",
        )
    }

    #[test]
    fn repeated_letters_are_required() {
        let dict = small_dict();
        let rack = Rack::from_letters("A").unwrap();
        assert!(dict.best_word(&rack).is_none());
        let rack = Rack::from_letters("AA").unwrap();
        assert_eq!(dict.best_word(&rack).unwrap().text, "AA");
        let rack = Rack::from_letters("AAB").unwrap();
        // AA is playable; AB is also playable but AA has the same base and
        // shorter length, so the longer AB wins the tie-break.
        assert_eq!(dict.best_word(&rack).unwrap().text, "AB");
    }

    #[test]
    fn best_word_maximizes_reward_not_just_length() {
        let dict = small_dict();
        // QUIZ (4 letters, 33) beats CAT (3 letters, 5) even though both fit.
        let rack = Rack::from_letters("QUIZCAT").unwrap();
        assert_eq!(dict.best_word(&rack).unwrap().text, "QUIZ");
        // QUIZZES (7 letters, 102) beats QUIZ when the letters exist.
        let rack = Rack::from_letters("QUIZZES").unwrap();
        assert_eq!(dict.best_word(&rack).unwrap().text, "QUIZZES");
    }

    #[test]
    fn greedy_partition_uses_multiple_words() {
        let dict = small_dict();
        let rack = Rack::from_letters("QUIZCAT").unwrap();
        let partition = dict.greedy_partition(&rack);
        assert_eq!(partition.words[0].word, "QUIZ");
        assert_eq!(partition.reward, 33 + 5);
        assert_eq!(partition.words.len(), 2);
    }

    #[test]
    fn target_requires_missing_letters_and_beats_current_best() {
        let dict = small_dict();
        let rack = Rack::from_letters("QUI").unwrap();
        let target = dict.best_target(&rack, 2).unwrap();
        assert_eq!(target.word, "QUIZ");
        assert_eq!(target.missing.to_compact_string(), "Z");
        assert_eq!(target.reward, 33);

        let rack = Rack::from_letters("QUIZ").unwrap();
        assert!(dict.best_target(&rack, 2).is_none());
    }

    #[test]
    fn bundled_dictionary_has_expected_size_and_words() {
        let dict = Dictionary::bundled();
        assert!(dict.word_count() > 170_000, "got {}", dict.word_count());
        assert!(dict.contains("AA"));
        assert!(dict.contains("quiz"));
        assert!(dict.contains("ZYZZYVAS"));
        assert!(!dict.contains("QQQQ"));
        assert!(dict.max_word_len() >= 15);
    }

    #[test]
    fn bundled_best_word_is_playable() {
        let dict = Dictionary::bundled();
        let rack = Rack::from_letters("QUIZZES").unwrap();
        let word = dict.best_word(&rack).unwrap();
        assert!(rack.contains_word(&word.text), "{}", word.text);
        assert!(word.reward >= 102);
    }

    #[test]
    fn scan_and_subset_paths_agree() {
        let dict = Dictionary::bundled();
        // A rack with many distinct letters forces the scan path, while still
        // being small enough to enumerate exhaustively for the comparison.
        let rack = Rack::from_letters("ABCDEFGHIJKL").unwrap();
        let scan = dict.best_word_by_scan(&rack).map(|w| w.text.clone());
        let subset = dict
            .best_word_by_submultisets(&rack)
            .map(|w| w.text.clone());
        assert_eq!(scan, subset);
    }
}
