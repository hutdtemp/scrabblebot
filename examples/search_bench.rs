use std::time::{Duration, Instant};

use scrabblebot_bot::core::bag::BagModel;
use scrabblebot_bot::core::dictionary::Dictionary;
use scrabblebot_bot::core::letters::letter_index;
use scrabblebot_bot::core::rack::Rack;
use scrabblebot_bot::core::search::{BeamScore, SearchConfig, SearchEngine};

fn main() {
    let engine = SearchEngine::new(
        Dictionary::bundled_arc(),
        SearchConfig {
            beam_width: 128,
            max_depth: 12,
            exact_depth: 2,
            max_time_ms: 800,
            max_states: 4096,
            beam_score: BeamScore::ExpectedValue,
        },
    );
    let initial = Rack::from_letters("AEIOU").unwrap();
    let bag = BagModel::estimate_for_match(&initial, b"EEAAS".iter().copied(), 63);
    for rack_text in ["AEIRSTU", "QUIZ", "AEEIRST", "ABCDEFGH", "AAEILNRS"] {
        let rack = Rack::from_letters(rack_text).unwrap();
        let deadline = Instant::now() + Duration::from_millis(800);
        let result = engine.search(&rack, letter_index(b'Q').unwrap(), &bag, deadline);
        println!(
            "rack={rack_text} depth={} elapsed_ms={} states={} value_without={:.1} value_with={:.1} marginal={:.1} fallback={}",
            result.depth_completed,
            result.elapsed_ms,
            result.states_evaluated,
            result.value_without,
            result.value_with,
            result.marginal,
            result.fallback_used
        );
    }
}
