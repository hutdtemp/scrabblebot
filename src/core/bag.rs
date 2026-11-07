//! A model of the shared bag.
//!
//! The module never exposes the bag composition. It does expose:
//! - our own rack (`my_rack`),
//! - every auction letter, including letters nobody won (`auction`),
//! - the current number of tiles left (`match.bag_total`).
//!
//! Word plays do not change the bag: a played tile moves from a rack to the
//! discard pile, while the bag only shrinks when an auction draws a letter.
//! The only genuinely hidden removal is the opponents' starting deal (and any
//! tiles they still hold), whose *count* is known but whose letters are not.
//!
//! We therefore start from the full 98-tile bag, subtract our initial rack and
//! every auction letter we have seen, then reconcile the total to the
//! authoritative `bag_total` by subtracting the hidden opponent deal
//! proportionally. That is an estimate, not an oracle; it is documented in the
//! README.

use serde::{Deserialize, Serialize};

use super::letters::{BAG_SIZE, DEFAULT_BAG};
use super::rack::Rack;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BagModel {
    pub counts: [u32; 26],
    pub total: u32,
}

impl Default for BagModel {
    fn default() -> Self {
        Self::full()
    }
}

impl BagModel {
    pub fn full() -> Self {
        let counts = DEFAULT_BAG.map(|n| n as u32);
        Self {
            counts,
            total: BAG_SIZE,
        }
    }

    pub fn from_counts(counts: [u32; 26]) -> Self {
        let total = counts.iter().sum();
        Self { counts, total }
    }

    pub fn total(&self) -> u32 {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    pub fn count(&self, index: usize) -> u32 {
        self.counts[index]
    }

    /// Remove a rack's letters. Saturates at zero (a malformed snapshot must
    /// not panic the bot).
    pub fn remove_rack(&mut self, rack: &Rack) {
        for (i, &n) in rack.counts.iter().enumerate() {
            let n = n as u32;
            let removed = self.counts[i].min(n);
            self.counts[i] -= removed;
            self.total = self.total.saturating_sub(removed);
        }
    }

    /// Remove `n` copies of one letter. Returns false if the model did not
    /// contain that many.
    pub fn remove_letter(&mut self, index: usize, n: u32) -> bool {
        if self.counts[index] < n {
            return false;
        }
        self.counts[index] -= n;
        self.total -= n;
        true
    }

    pub fn add_letter(&mut self, index: usize, n: u32) {
        self.counts[index] += n;
        self.total += n;
    }

    /// Probability of drawing `index` next, given this model.
    pub fn draw_probability(&self, index: usize) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.counts[index] as f64 / self.total as f64
        }
    }

    /// Exact probability of drawing a particular multiset of `draws` from this
    /// bag, where `draws[i]` is the number of copies of letter `i`.
    ///
    /// This is the multivariate hypergeometric probability
    /// `prod C(c_i, d_i) / C(total, sum d)`.
    pub fn multiset_probability(&self, draws: &[u32; 26]) -> f64 {
        let drawn: u32 = draws.iter().sum();
        if drawn > self.total {
            return 0.0;
        }
        let mut log_p = 0.0f64;
        for (i, &d) in draws.iter().enumerate() {
            let c = self.counts[i];
            if d > c {
                return 0.0;
            }
            log_p += log_choose(c, d);
        }
        log_p -= log_choose(self.total, drawn);
        log_p.exp()
    }

    /// Reconcile the model's total to `target` by removing or adding letters
    /// proportionally to the current counts.
    ///
    /// The hidden opponent starting deal is the usual reason for a positive
    /// excess. Apportionment uses largest remainders, so the result is
    /// deterministic and the counts always sum exactly to `target`.
    pub fn reconcile_to_total(&mut self, target: u32) {
        if self.total == target {
            return;
        }
        if self.total > target {
            let excess = self.total - target;
            self.remove_proportionally(excess);
        } else {
            let deficit = target - self.total;
            self.add_proportionally(deficit);
        }
        debug_assert_eq!(self.total, target);
    }

    fn remove_proportionally(&mut self, amount: u32) {
        if amount == 0 {
            return;
        }
        let total = self.total;
        if total == 0 {
            return;
        }
        let mut shares = [0u32; 26];
        let mut remainders = [(0u64, 0usize); 26];
        let mut assigned = 0u32;
        for i in 0..26 {
            let numerator = self.counts[i] as u64 * amount as u64;
            shares[i] = (numerator / total as u64) as u32;
            remainders[i] = (numerator % total as u64, i);
            assigned += shares[i];
        }
        let mut leftover = amount.saturating_sub(assigned);
        remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, i) in remainders.iter() {
            if leftover == 0 {
                break;
            }
            if shares[i] < self.counts[i] {
                shares[i] += 1;
                leftover -= 1;
            }
        }
        // If rounding still left mass (possible only with zero-count slots),
        // take it from the largest counts deterministically.
        let mut i = 0usize;
        while leftover > 0 && i < 26 {
            let available = self.counts[i].saturating_sub(shares[i]);
            let take = available.min(leftover);
            shares[i] += take;
            leftover -= take;
            i += 1;
        }
        for (count, share) in self.counts.iter_mut().zip(shares) {
            *count -= share;
        }
        self.total -= amount;
    }

    fn add_proportionally(&mut self, amount: u32) {
        if amount == 0 {
            return;
        }
        let total = self.total;
        let weights: [u32; 26] = if total == 0 {
            DEFAULT_BAG.map(|n| n as u32)
        } else {
            self.counts
        };
        let weight_total: u32 = weights.iter().sum();
        if weight_total == 0 {
            return;
        }
        let mut shares = [0u32; 26];
        let mut remainders = [(0u64, 0usize); 26];
        let mut assigned = 0u32;
        for i in 0..26 {
            let numerator = weights[i] as u64 * amount as u64;
            shares[i] = (numerator / weight_total as u64) as u32;
            remainders[i] = (numerator % weight_total as u64, i);
            assigned += shares[i];
        }
        let mut leftover = amount.saturating_sub(assigned);
        remainders.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, i) in remainders.iter() {
            if leftover == 0 {
                break;
            }
            shares[i] += 1;
            leftover -= 1;
        }
        for (count, share) in self.counts.iter_mut().zip(shares) {
            *count += share;
        }
        self.total += amount;
    }

    /// Build the match bag estimate described in the module docs.
    pub fn estimate_for_match(
        my_initial_rack: &Rack,
        observed_auction_letters: impl IntoIterator<Item = u8>,
        current_bag_total: u32,
    ) -> Self {
        let mut bag = Self::full();
        bag.remove_rack(my_initial_rack);
        for letter in observed_auction_letters {
            if let Some(idx) = super::letters::letter_index(letter) {
                bag.remove_letter(idx, 1);
            }
        }
        bag.reconcile_to_total(current_bag_total);
        bag
    }
}

/// `ln C(n, k)`, with `C(n, k) = 0` when `k > n`.
fn log_choose(n: u32, k: u32) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let k = k.min(n - k);
    let mut acc = 0.0f64;
    for i in 0..k {
        acc += ((n - i) as f64).ln() - ((i + 1) as f64).ln();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rack::Rack;

    #[test]
    fn full_bag_has_98_tiles() {
        let bag = BagModel::full();
        assert_eq!(bag.total(), 98);
        assert_eq!(bag.count(4), 12); // E
        assert!((bag.draw_probability(4) - 12.0 / 98.0).abs() < 1e-12);
    }

    #[test]
    fn removing_rack_and_auctions_then_reconciling_hits_public_total() {
        let rack = Rack::from_letters("AEIOU").unwrap();
        let auctions = b"EEAAS".iter().copied();
        // 98 - 5 own tiles - 5 auctions - 25 hidden opponents = 63.
        let bag = BagModel::estimate_for_match(&rack, auctions, 63);
        assert_eq!(bag.total(), 63);
        assert_eq!(bag.counts.iter().sum::<u32>(), 63);
        // E should lose its own tile, the two auction E's, and a share of the
        // hidden deal, so it must be below 12.
        assert!(bag.count(4) < 12);
    }

    #[test]
    fn reconcile_never_goes_negative() {
        let mut bag = BagModel::from_counts([1; 26]);
        bag.reconcile_to_total(0);
        assert_eq!(bag.total(), 0);
        assert!(bag.counts.iter().all(|&n| n == 0));
    }

    #[test]
    fn reconcile_adds_when_target_is_larger() {
        let mut bag = BagModel::from_counts([0; 26]);
        bag.reconcile_to_total(26);
        assert_eq!(bag.total(), 26);
        assert_eq!(bag.counts.iter().sum::<u32>(), 26);
    }

    #[test]
    fn multiset_probability_is_hypergeometric() {
        // Bag: 2 A, 1 B. Probability of drawing A then B in two draws is 2/3,
        // and the multiset {A, B} has that same probability.
        let mut bag = BagModel::from_counts([0; 26]);
        bag.add_letter(0, 2);
        bag.add_letter(1, 1);
        let mut draws = [0u32; 26];
        draws[0] = 1;
        draws[1] = 1;
        assert!((bag.multiset_probability(&draws) - 2.0 / 3.0).abs() < 1e-12);

        // {A, A} has probability 1/3.
        let mut aa = [0u32; 26];
        aa[0] = 2;
        assert!((bag.multiset_probability(&aa) - 1.0 / 3.0).abs() < 1e-12);

        // Probabilities over all multisets of size 2 sum to 1.
        let mut sum = 0.0;
        for a in 0..=2 {
            for b in 0..=1 {
                if a + b == 2 {
                    let mut d = [0u32; 26];
                    d[0] = a;
                    d[1] = b;
                    sum += bag.multiset_probability(&d);
                }
            }
        }
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn impossible_multiset_has_zero_probability() {
        let bag = BagModel::full();
        let mut draws = [0u32; 26];
        draws[9] = 2; // only one J
        assert_eq!(bag.multiset_probability(&draws), 0.0);
    }
}
