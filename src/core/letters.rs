//! Letter values, the standard 98-tile bag, and exact word scoring.
//!
//! The rules mirror `spacetimedb/src/letters.ts` in the upstream module:
//! standard Scrabble letter values, no blanks, and integer reward
//! `base * numerator / denominator` with truncating division.

use serde::{Deserialize, Serialize};

pub const ALPHABET: [u8; 26] = *b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// Standard Scrabble letter values, indexed A..Z.
pub const LETTER_VALUES: [u8; 26] = [
    1, 3, 3, 2, 1, 4, 2, 4, 1, 8, 5, 1, 3, 1, 1, 3, 10, 1, 1, 1, 1, 4, 4, 8, 4, 10,
];

/// Standard 98-tile Scrabble distribution, indexed A..Z. No blanks.
pub const DEFAULT_BAG: [u8; 26] = [
    9, 2, 2, 4, 12, 2, 3, 2, 9, 1, 1, 4, 2, 6, 8, 2, 1, 6, 4, 6, 4, 2, 2, 1, 2, 1,
];

pub const BAG_SIZE: u32 = 98;

#[inline]
pub fn letter_index(byte: u8) -> Option<usize> {
    match byte.to_ascii_uppercase() {
        b'A'..=b'Z' => Some((byte.to_ascii_uppercase() - b'A') as usize),
        _ => None,
    }
}

#[inline]
pub fn letter_char(index: usize) -> char {
    ALPHABET[index] as char
}

#[inline]
pub fn letter_value(index: usize) -> u32 {
    LETTER_VALUES[index] as u32
}

/// Length-based reward multiplier as `(numerator, denominator)`.
///
/// `reward = base * numerator / denominator` with integer truncation, exactly
/// like the module's bigint arithmetic.
pub fn length_multiplier(len: usize) -> (u32, u32) {
    match len {
        0..=3 => (1, 1),
        4 => (3, 2),
        5 => (2, 1),
        6 => (5, 2),
        _ => (3, 1),
    }
}

/// Exact score for an uppercase A-Z word of length >= 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WordScore {
    pub base: u32,
    pub multiplier_num: u32,
    pub multiplier_den: u32,
    pub reward: u32,
    pub bonus: u32,
}

/// Score a word exactly as the module does.
///
/// Returns `None` for words shorter than two letters or containing anything
/// outside ASCII A-Z. Lowercase ASCII is accepted and normalized, matching the
/// module's `asciiUppercase` before validation.
pub fn score_word(word: &str) -> Option<WordScore> {
    let bytes = word.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let mut base = 0u32;
    for &b in bytes {
        let idx = letter_index(b)?;
        base += letter_value(idx);
    }
    let (num, den) = length_multiplier(bytes.len());
    let reward = base * num / den;
    Some(WordScore {
        base,
        multiplier_num: num,
        multiplier_den: den,
        reward,
        bonus: reward - base,
    })
}

/// Canonical uppercase form used on the wire and in the dictionary.
pub fn normalize_word(word: &str) -> Option<String> {
    let bytes = word.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len());
    for &b in bytes {
        let idx = letter_index(b)?;
        out.push(ALPHABET[idx]);
    }
    Some(String::from_utf8(out).expect("A-Z bytes are valid UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bag_is_98_tiles() {
        let total: u32 = DEFAULT_BAG.iter().map(|&n| n as u32).sum();
        assert_eq!(total, BAG_SIZE);
        assert_eq!(DEFAULT_BAG.len(), 26);
    }

    #[test]
    fn letter_values_match_standard_scrabble() {
        let expected = [
            ('A', 1),
            ('B', 3),
            ('C', 3),
            ('D', 2),
            ('E', 1),
            ('F', 4),
            ('G', 2),
            ('H', 4),
            ('I', 1),
            ('J', 8),
            ('K', 5),
            ('L', 1),
            ('M', 3),
            ('N', 1),
            ('O', 1),
            ('P', 3),
            ('Q', 10),
            ('R', 1),
            ('S', 1),
            ('T', 1),
            ('U', 1),
            ('V', 4),
            ('W', 4),
            ('X', 8),
            ('Y', 4),
            ('Z', 10),
        ];
        for (ch, value) in expected {
            assert_eq!(letter_value(letter_index(ch as u8).unwrap()), value);
        }
    }

    #[test]
    fn three_letter_word_has_no_multiplier() {
        assert_eq!(
            score_word("CAT"),
            Some(WordScore {
                base: 5,
                multiplier_num: 1,
                multiplier_den: 1,
                reward: 5,
                bonus: 0
            })
        );
    }

    #[test]
    fn four_letter_word_truncates_half_points() {
        // JAZZ: 8 + 1 + 10 + 10 = 29; 29 * 3 / 2 = 43.5 -> 43.
        let score = score_word("JAZZ").unwrap();
        assert_eq!(score.base, 29);
        assert_eq!(score.reward, 43);
        assert_eq!(score.bonus, 14);
    }

    #[test]
    fn six_letter_word_truncates_half_points() {
        // JAZZES: 8 + 1 + 10 + 10 + 1 + 1 = 31; 31 * 5 / 2 = 77.5 -> 77.
        let score = score_word("JAZZES").unwrap();
        assert_eq!(score.base, 31);
        assert_eq!(score.reward, 77);
        assert_eq!(score.bonus, 46);
    }

    #[test]
    fn seven_letter_word_gets_triple() {
        // QUIZZES: 10 + 1 + 1 + 10 + 10 + 1 + 1 = 34; 34 * 3 = 102.
        let score = score_word("QUIZZES").unwrap();
        assert_eq!(score.base, 34);
        assert_eq!(score.reward, 102);
        assert_eq!(score.bonus, 68);
    }

    #[test]
    fn lowercase_is_accepted_but_normalized() {
        assert_eq!(score_word("quiz"), score_word("QUIZ"));
        assert_eq!(normalize_word("quiz").as_deref(), Some("QUIZ"));
    }

    #[test]
    fn invalid_words_are_rejected() {
        assert_eq!(score_word("A"), None);
        assert_eq!(score_word(""), None);
        assert_eq!(score_word("AB1"), None);
        assert_eq!(score_word("A B"), None);
        assert_eq!(score_word("ÄB"), None);
    }
}
