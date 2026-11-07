//! Pure helpers for match bookkeeping and authoritative-state reconciliation.

use super::letters::letter_index;
use super::rack::Rack;

/// Reconstruct the rack a bot was dealt at match start.
///
/// Every tile in the current rack came from either the initial deal or an
/// auction this bot won. Every tile this bot played left the rack. Therefore
///
/// ```text
/// initial = current + played - won
/// ```
///
/// holds even when a won tile was later played: adding it back as "played"
/// and subtracting it as "won" cancels. This is used to rebuild the bag model
/// after a reconnect, when the initial rack was not observed live.
pub fn reconstruct_initial_rack<'a>(
    current: &Rack,
    played_words: impl IntoIterator<Item = &'a str>,
    won_letters: impl IntoIterator<Item = u8>,
) -> Rack {
    let mut initial = *current;
    for word in played_words {
        initial.add_word(word);
    }
    for letter in won_letters {
        if let Some(idx) = letter_index(letter)
            && !initial.remove(idx, 1)
        {
            // A malformed/partial history should not panic; the bag model
            // will reconcile to the public bag total anyway.
            debug_assert!(false, "won letter missing from reconstructed rack");
        }
    }
    initial
}

/// 1-based competition placement: one plus the number of participants with a
/// strictly higher score. Ties share a placement.
pub fn placement(scores: impl IntoIterator<Item = i64>, my_score: i64) -> usize {
    1 + scores.into_iter().filter(|&score| score > my_score).count()
}

/// Convert a module timestamp (microseconds since the Unix epoch) to
/// milliseconds since the Unix epoch, saturating at zero.
pub fn micros_to_millis(micros: i64) -> u64 {
    if micros <= 0 {
        0
    } else {
        (micros as u64) / 1_000
    }
}

/// Elapsed milliseconds between two module timestamps.
pub fn elapsed_millis(start_micros: i64, end_micros: i64) -> u64 {
    let diff = end_micros.saturating_sub(start_micros);
    if diff <= 0 { 0 } else { (diff as u64) / 1_000 }
}

/// Parse the module's one-letter auction string.
pub fn auction_letter_index(letter: &str) -> Option<usize> {
    let bytes = letter.as_bytes();
    if bytes.len() != 1 {
        return None;
    }
    letter_index(bytes[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_rack_reconstruction_handles_played_auction_tiles() {
        // Initial: A B C D E. Won Q, played "QUIZ" using Q, U (won), I, Z
        // (won). Current rack is A B C D E minus I plus nothing.
        let current = Rack::from_letters("ABCDE").unwrap();
        let initial = reconstruct_initial_rack(&current, ["QUIZ"], *b"QUZ");
        // QUIZ adds Q,U,I,Z; subtracting Q,U,Z leaves I, so initial has I.
        assert_eq!(initial.to_compact_string(), "ABCDEI");
    }

    #[test]
    fn initial_rack_reconstruction_without_events_is_current() {
        let current = Rack::from_letters("AAB").unwrap();
        let initial = reconstruct_initial_rack(&current, std::iter::empty(), std::iter::empty());
        assert_eq!(initial, current);
    }

    #[test]
    fn placement_counts_strictly_higher_scores() {
        assert_eq!(placement([10, 20, 30], 20), 2);
        assert_eq!(placement([10, 20, 20], 20), 1);
        assert_eq!(placement([10, 20, 20], 10), 3);
        assert_eq!(placement([0], 0), 1);
    }

    #[test]
    fn timestamp_conversions() {
        assert_eq!(micros_to_millis(1_700_000_000_123_456), 1_700_000_000_123);
        assert_eq!(elapsed_millis(1_000_000, 1_250_000), 250);
        assert_eq!(elapsed_millis(1_250_000, 1_000_000), 0);
    }

    #[test]
    fn auction_letter_parsing() {
        assert_eq!(auction_letter_index("Q"), Some(16));
        assert_eq!(auction_letter_index("q"), Some(16));
        assert_eq!(auction_letter_index("QQ"), None);
        assert_eq!(auction_letter_index(""), None);
    }
}
