//! Rack counts and small helpers shared by the solver and live client.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::letters::{ALPHABET, letter_index, normalize_word};

/// A multiset of letters, indexed A..Z.
///
/// Counts are `u8` because a single match can never put more than 98 tiles in
/// one rack; arithmetic saturates rather than wrapping if a malformed server
/// snapshot ever exceeds that.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rack {
    pub counts: [u8; 26],
}

impl Rack {
    pub const fn new() -> Self {
        Self { counts: [0; 26] }
    }

    pub fn from_counts(counts: [u8; 26]) -> Self {
        Self { counts }
    }

    /// Parse a string of A-Z letters (case-insensitive).
    pub fn from_letters(letters: &str) -> Result<Self, RackParseError> {
        let mut rack = Self::new();
        for b in letters.bytes() {
            let idx = letter_index(b).ok_or(RackParseError(b))?;
            rack.add(idx, 1);
        }
        Ok(rack)
    }

    /// Add `n` copies of the letter at `index`, saturating at 255.
    pub fn add(&mut self, index: usize, n: u8) {
        self.counts[index] = self.counts[index].saturating_add(n);
    }

    /// Remove `n` copies. Returns false and leaves the rack unchanged if the
    /// rack does not have that many.
    pub fn remove(&mut self, index: usize, n: u8) -> bool {
        if self.counts[index] < n {
            return false;
        }
        self.counts[index] -= n;
        true
    }

    pub fn count(&self, index: usize) -> u8 {
        self.counts[index]
    }

    pub fn total(&self) -> u32 {
        self.counts.iter().map(|&n| n as u32).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&n| n == 0)
    }

    /// Bitmask of letters present at least once.
    pub fn mask(&self) -> u32 {
        let mut mask = 0u32;
        for (i, &n) in self.counts.iter().enumerate() {
            if n > 0 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Whether `word` can be spelled from this rack, counting repeats.
    pub fn contains_word(&self, word: &str) -> bool {
        let Some(normalized) = normalize_word(word) else {
            return false;
        };
        let mut need = [0u8; 26];
        for b in normalized.bytes() {
            let idx = (b - b'A') as usize;
            need[idx] = need[idx].saturating_add(1);
            if need[idx] > self.counts[idx] {
                return false;
            }
        }
        true
    }

    /// Add every letter of `word` to the rack. Returns false if the word is
    /// invalid (but leaves any letters already added by earlier iterations).
    pub fn add_word(&mut self, word: &str) -> bool {
        let Some(normalized) = normalize_word(word) else {
            return false;
        };
        for b in normalized.bytes() {
            self.add((b - b'A') as usize, 1);
        }
        true
    }

    /// Remove every letter of `word`. Returns false and leaves the rack
    /// unchanged if the word is invalid or not spellable.
    pub fn remove_word(&mut self, word: &str) -> bool {
        if !self.contains_word(word) {
            return false;
        }
        let normalized = normalize_word(word).expect("contains_word validated the word");
        for b in normalized.bytes() {
            self.counts[(b - b'A') as usize] -= 1;
        }
        true
    }

    /// Compact A..Z string, useful for logs and plan displays.
    pub fn to_compact_string(&self) -> String {
        let mut out = String::new();
        for (i, &n) in self.counts.iter().enumerate() {
            for _ in 0..n {
                out.push(ALPHABET[i] as char);
            }
        }
        out
    }

    /// Number of sub-multisets of this rack, capped at `cap`.
    ///
    /// Used to choose between exact subset enumeration and a linear scan of
    /// the dictionary.
    pub fn submultiset_count_capped(&self, cap: u64) -> u64 {
        let mut product = 1u64;
        for &n in &self.counts {
            if n == 0 {
                continue;
            }
            product = product.saturating_mul(n as u64 + 1);
            if product > cap {
                return cap.saturating_add(1);
            }
        }
        product
    }
}

impl fmt::Debug for Rack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rack({})", self.to_compact_string())
    }
}

impl fmt::Display for Rack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_compact_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RackParseError(pub u8);

impl fmt::Display for RackParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "not an A-Z letter: 0x{:02x}", self.0)
    }
}

impl std::error::Error for RackParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_letters_are_counted() {
        let rack = Rack::from_letters("AAB").unwrap();
        assert_eq!(rack.count(0), 2);
        assert_eq!(rack.count(1), 1);
        assert_eq!(rack.total(), 3);
        assert!(rack.contains_word("AA"));
        assert!(rack.contains_word("AB"));
        assert!(!rack.contains_word("AAA"));
        assert!(!rack.contains_word("ABC"));
    }

    #[test]
    fn remove_word_respects_repeats() {
        let mut rack = Rack::from_letters("AAB").unwrap();
        assert!(rack.remove_word("AA"));
        assert_eq!(rack.to_compact_string(), "B");
        assert!(!rack.remove_word("AA"));
        assert_eq!(rack.to_compact_string(), "B");
    }

    #[test]
    fn add_word_and_mask() {
        let mut rack = Rack::new();
        assert!(rack.add_word("quiz"));
        assert_eq!(rack.to_compact_string(), "IQUZ");
        assert_eq!(rack.mask(), (1 << 8) | (1 << 16) | (1 << 20) | (1 << 25));
    }

    #[test]
    fn parse_rejects_non_letters() {
        assert!(Rack::from_letters("AB1").is_err());
        assert!(Rack::from_letters("AB CD").is_err());
    }

    #[test]
    fn submultiset_count_capped_stops_early() {
        let rack = Rack::from_letters("AABBCC").unwrap();
        assert_eq!(rack.submultiset_count_capped(1000), 27);
        let big = Rack::from_letters("ABCDEFGHIJKLMNOPQRSTUVWXYZ").unwrap();
        assert_eq!(big.submultiset_count_capped(1000), 1001);
    }
}
