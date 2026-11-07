use std::time::{Duration, Instant};

use scrabblebot_bot::core::bag::BagModel;
use scrabblebot_bot::core::dictionary::Dictionary;
use scrabblebot_bot::core::letters::letter_index;
use scrabblebot_bot::core::rack::Rack;
use scrabblebot_bot::core::search::{BeamScore, SearchConfig, SearchEngine};

#[test]
fn bundled_search_completes_a_depth_with_a_realistic_rack() {
    let engine = SearchEngine::new(
        Dictionary::bundled_arc(),
        SearchConfig {
            beam_width: 128,
            max_depth: 6,
            exact_depth: 2,
            max_time_ms: 500,
            max_states: 4096,
            beam_score: BeamScore::ExpectedValue,
        },
    );
    let rack = Rack::from_letters("AEIRSTU").unwrap();
    let initial = Rack::from_letters("AEIOU").unwrap();
    let bag = BagModel::estimate_for_match(&initial, b"EEAAS".iter().copied(), 63);
    let deadline = Instant::now() + Duration::from_millis(500);
    let result = engine.search(&rack, letter_index(b'Q').unwrap(), &bag, deadline);
    assert!(
        result.depth_completed >= 1,
        "search fell back entirely: {result:?}"
    );
    assert!(
        result.elapsed_ms <= 3000,
        "search took {} ms",
        result.elapsed_ms
    );
    assert!(result.value_without >= 0.0);
    assert!(result.value_with >= 0.0);
}
