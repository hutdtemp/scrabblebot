//! Adaptive-depth, probability-weighted beam search over hypothetical future
//! bag draws.
//!
//! At an auction we know the auctioned letter `L` is already out of the bag,
//! whether or not we win it. The search compares two racks against the *same*
//! future bag:
//!
//! - `rack` (we do not win `L`)
//! - `rack + L` (we win `L`)
//!
//! For a horizon of `d` future draws it enumerates or beams over multisets of
//! drawn letters with their exact multivariate-hypergeometric probabilities,
//! evaluates each resulting rack with greedy liquidation, and returns the
//! probability-weighted expected reward. The marginal value is
//! `V(with L) - V(without L)`.
//!
//! Important properties:
//! - Futures are weighted by probability, not maximized. A rare lucky draw
//!   cannot masquerade as expected value.
//! - Equivalent rack states are merged by summing their probability mass.
//! - Depths are attempted in increasing order until the deadline; an
//!   incomplete depth is discarded and the last completed depth is kept.
//! - If no depth completes, the result is the safe immediate-liquidation
//!   fallback with `fallback_used = true`.
//!
//! The value function is greedy liquidation (repeatedly play the best word),
//! not an exact set-packing optimum. That is a deliberate baseline limitation.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::bag::BagModel;
use super::dictionary::{Dictionary, Partition, WordChoice};
use super::rack::Rack;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeamScore {
    /// `probability * (value_without + value_with)`.
    ExpectedValue,
    /// Keep the most likely futures.
    Probability,
    /// Keep the highest-value futures (diagnostic; can be optimistic).
    Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum number of draw-multiset states kept after each depth.
    pub beam_width: usize,
    /// Maximum future draws to consider.
    pub max_depth: u8,
    /// Depths up to this value are evaluated exhaustively (no beam pruning).
    pub exact_depth: u8,
    /// Hard cap on search wall time, in addition to the per-auction deadline.
    pub max_time_ms: u64,
    /// Safety cap on states kept in a single layer.
    pub max_states: usize,
    /// How to rank states when pruning.
    pub beam_score: BeamScore,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 128,
            max_depth: 12,
            exact_depth: 2,
            max_time_ms: 800,
            max_states: 4096,
            beam_score: BeamScore::ExpectedValue,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub candidate: u8,
    pub value_without: f64,
    pub value_with: f64,
    pub marginal: f64,
    pub depth_completed: u8,
    pub states_evaluated: u64,
    pub elapsed_ms: u64,
    pub timed_out: bool,
    pub fallback_used: bool,
    pub retained_probability: f64,
    pub best_now_without: Option<WordChoice>,
    pub best_now_with: Option<WordChoice>,
}

impl SearchResult {
    pub fn fallback(
        dictionary: &Dictionary,
        rack: &Rack,
        rack_with: &Rack,
        candidate: u8,
        elapsed_ms: u64,
    ) -> Self {
        let best_without = dictionary.best_word(rack).map(WordChoice::from);
        let best_with = dictionary.best_word(rack_with).map(WordChoice::from);
        let value_without = dictionary.greedy_partition(rack).reward as f64;
        let value_with = dictionary.greedy_partition(rack_with).reward as f64;
        Self {
            candidate,
            value_without,
            value_with,
            marginal: value_with - value_without,
            depth_completed: 0,
            states_evaluated: 0,
            elapsed_ms,
            timed_out: false,
            fallback_used: true,
            retained_probability: 0.0,
            best_now_without: best_without,
            best_now_with: best_with,
        }
    }
}

pub struct SearchEngine {
    dictionary: Arc<Dictionary>,
    config: SearchConfig,
}

impl SearchEngine {
    pub fn new(dictionary: Arc<Dictionary>, config: SearchConfig) -> Self {
        Self { dictionary, config }
    }

    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }

    pub fn config(&self) -> &SearchConfig {
        &self.config
    }

    /// Run the search until `deadline` (also capped by `config.max_time_ms`).
    pub fn search(
        &self,
        rack: &Rack,
        candidate: usize,
        bag: &BagModel,
        deadline: Instant,
    ) -> SearchResult {
        let start = Instant::now();
        if candidate >= 26 {
            return SearchResult::fallback(&self.dictionary, rack, rack, 0, 0);
        }
        let candidate = candidate as u8;
        let mut rack_with = *rack;
        rack_with.add(candidate as usize, 1);

        let config_deadline = start + Duration::from_millis(self.config.max_time_ms.max(1));
        let effective_deadline = deadline.min(config_deadline);

        let mut evaluator = RackEvaluator::new(&self.dictionary);
        let base_without = evaluator.partition_value(rack);
        let base_with = evaluator.partition_value(&rack_with);

        let best_without = self.dictionary.best_word(rack).map(WordChoice::from);
        let best_with = self.dictionary.best_word(&rack_with).map(WordChoice::from);

        let max_depth = self
            .config
            .max_depth
            .min(bag.total.min(u8::MAX as u32) as u8);
        let mut result = SearchResult {
            candidate,
            value_without: base_without,
            value_with: base_with,
            marginal: base_with - base_without,
            depth_completed: 0,
            states_evaluated: evaluator.evaluations(),
            elapsed_ms: 0,
            timed_out: false,
            fallback_used: true,
            retained_probability: 0.0,
            best_now_without: best_without,
            best_now_with: best_with,
        };

        if max_depth == 0 {
            result.elapsed_ms = start.elapsed().as_millis() as u64;
            return result;
        }

        for depth in 1..=max_depth {
            if Instant::now() >= effective_deadline {
                result.timed_out = true;
                break;
            }
            let evals_before = evaluator.evaluations();
            match self.run_depth(
                &mut evaluator,
                rack,
                &rack_with,
                bag,
                depth,
                effective_deadline,
            ) {
                DepthOutcome::Complete {
                    value_without,
                    value_with,
                    retained_probability,
                } => {
                    // The bot can always liquidate now, so the value of a rack
                    // is at least its immediate greedy partition value.
                    result.value_without = value_without.max(base_without);
                    result.value_with = value_with.max(base_with);
                    result.marginal = result.value_with - result.value_without;
                    result.depth_completed = depth;
                    result.retained_probability = retained_probability;
                    result.fallback_used = false;
                    result.states_evaluated += evaluator.evaluations() - evals_before;
                }
                DepthOutcome::Timeout => {
                    result.timed_out = true;
                    break;
                }
            }
        }

        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn run_depth(
        &self,
        evaluator: &mut RackEvaluator<'_>,
        rack_without: &Rack,
        rack_with: &Rack,
        bag: &BagModel,
        depth: u8,
        deadline: Instant,
    ) -> DepthOutcome {
        let beam_width = if depth <= self.config.exact_depth {
            usize::MAX
        } else {
            self.config
                .beam_width
                .max(1)
                .min(self.config.max_states.max(1))
        };

        let mut states: Vec<DrawState> = vec![DrawState {
            drawn: [0u8; 26],
            prob: 1.0,
            value_without: evaluator.partition_value(rack_without),
            value_with: evaluator.partition_value(rack_with),
        }];

        for step in 0..depth {
            let remaining = bag.total.saturating_sub(step as u32);
            if remaining == 0 {
                break;
            }
            let mut next: HashMap<[u8; 26], f64> = HashMap::new();
            let mut expansions = 0u64;
            for state in &states {
                for i in 0..26 {
                    let available = bag.counts[i].saturating_sub(state.drawn[i] as u32);
                    if available == 0 {
                        continue;
                    }
                    let p = available as f64 / remaining as f64;
                    let mut drawn = state.drawn;
                    drawn[i] += 1;
                    *next.entry(drawn).or_insert(0.0) += state.prob * p;
                    expansions += 1;
                    if expansions.is_multiple_of(512) && Instant::now() >= deadline {
                        return DepthOutcome::Timeout;
                    }
                }
            }
            if Instant::now() >= deadline {
                return DepthOutcome::Timeout;
            }

            let mut next_states: Vec<DrawState> = Vec::with_capacity(next.len());
            for (index, (drawn, prob)) in next.into_iter().enumerate() {
                if index % 64 == 0 && Instant::now() >= deadline {
                    return DepthOutcome::Timeout;
                }
                let with_draw_without = rack_plus_drawn(rack_without, &drawn);
                let with_draw_with = rack_plus_drawn(rack_with, &drawn);
                next_states.push(DrawState {
                    drawn,
                    prob,
                    value_without: evaluator.partition_value(&with_draw_without),
                    value_with: evaluator.partition_value(&with_draw_with),
                });
            }

            if next_states.len() > beam_width {
                next_states.sort_by(|a, b| {
                    self.state_score(b)
                        .total_cmp(&self.state_score(a))
                        .then_with(|| b.prob.total_cmp(&a.prob))
                        .then_with(|| a.drawn.cmp(&b.drawn))
                });
                next_states.truncate(beam_width);
            }
            if next_states.is_empty() {
                // The bag ran out before the requested depth.
                break;
            }
            states = next_states;
        }

        let retained_probability: f64 = states.iter().map(|s| s.prob).sum();
        let expected_without: f64 = states.iter().map(|s| s.prob * s.value_without).sum::<f64>();
        let expected_with: f64 = states.iter().map(|s| s.prob * s.value_with).sum::<f64>();

        // Pruned probability mass is assigned the safe immediate-liquidation
        // fallback, never the best pruned future.
        let missing_probability = (1.0 - retained_probability).max(0.0);
        let fallback_without = evaluator.partition_value(rack_without);
        let fallback_with = evaluator.partition_value(rack_with);

        DepthOutcome::Complete {
            value_without: expected_without + missing_probability * fallback_without,
            value_with: expected_with + missing_probability * fallback_with,
            retained_probability,
        }
    }

    fn state_score(&self, state: &DrawState) -> f64 {
        match self.config.beam_score {
            BeamScore::Probability => state.prob,
            BeamScore::Value => state.value_without + state.value_with,
            BeamScore::ExpectedValue => state.prob * (state.value_without + state.value_with + 1.0),
        }
    }
}

#[derive(Clone, Debug)]
struct DrawState {
    drawn: [u8; 26],
    prob: f64,
    value_without: f64,
    value_with: f64,
}

enum DepthOutcome {
    Complete {
        value_without: f64,
        value_with: f64,
        retained_probability: f64,
    },
    Timeout,
}

struct RackEvaluator<'a> {
    dictionary: &'a Dictionary,
    cache: HashMap<Rack, Partition>,
    evaluations: u64,
}

impl<'a> RackEvaluator<'a> {
    fn new(dictionary: &'a Dictionary) -> Self {
        Self {
            dictionary,
            cache: HashMap::new(),
            evaluations: 0,
        }
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn partition_value(&mut self, rack: &Rack) -> f64 {
        self.partition(rack).reward as f64
    }

    fn partition(&mut self, rack: &Rack) -> &Partition {
        if !self.cache.contains_key(rack) {
            if self.cache.len() >= 200_000 {
                self.cache.clear();
            }
            let partition = self.dictionary.greedy_partition(rack);
            self.evaluations += 1;
            self.cache.insert(*rack, partition);
        }
        self.cache.get(rack).expect("inserted above")
    }
}

fn rack_plus_drawn(base: &Rack, drawn: &[u8; 26]) -> Rack {
    let mut rack = *base;
    for (i, &count) in drawn.iter().enumerate() {
        if count > 0 {
            rack.add(i, count);
        }
    }
    rack
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::bag::BagModel;
    use crate::core::dictionary::Dictionary;
    use crate::core::letters::letter_index;
    use crate::core::rack::Rack;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn engine(words: &str, config: SearchConfig) -> SearchEngine {
        SearchEngine::new(Arc::new(Dictionary::from_wordlist(words)), config)
    }

    fn bag_from(letters: &str) -> BagModel {
        let mut bag = BagModel::from_counts([0; 26]);
        for b in letters.bytes() {
            let idx = letter_index(b).unwrap();
            bag.add_letter(idx, 1);
        }
        bag
    }

    fn generous() -> Instant {
        Instant::now() + Duration::from_secs(2)
    }

    #[test]
    fn candidate_completes_a_word_and_has_positive_marginal() {
        let engine = engine(
            "QUIZ\n",
            SearchConfig {
                max_depth: 1,
                ..SearchConfig::default()
            },
        );
        let rack = Rack::from_letters("QUI").unwrap();
        let bag = bag_from("AAAA");
        let result = engine.search(&rack, letter_index(b'Z').unwrap(), &bag, generous());
        assert!(result.depth_completed >= 1);
        assert_eq!(result.best_now_with.as_ref().unwrap().word, "QUIZ");
        assert!(result.value_with >= 33.0);
        assert!(result.marginal > 0.0);
    }

    #[test]
    fn rare_future_is_probability_weighted_not_maximized() {
        // X + Z + Q makes XZQ (3 letters, base 8+10+10 = 28).
        // The bag has Z, Q, A. With X in the rack, only the ZQ draw (1/3)
        // completes the word. A luckiest-future search would report 28;
        // the correct expectation is 28/3.
        let engine = engine(
            "XZQ\n",
            SearchConfig {
                max_depth: 2,
                exact_depth: 2,
                ..SearchConfig::default()
            },
        );
        let rack = Rack::new();
        let bag = bag_from("ZQA");
        let result = engine.search(&rack, letter_index(b'X').unwrap(), &bag, generous());
        assert_eq!(result.depth_completed, 2);
        assert!(
            (result.value_with - 28.0 / 3.0).abs() < 1e-9,
            "{}",
            result.value_with
        );
        assert!(result.value_with < 28.0);
        assert_eq!(result.value_without, 0.0);
    }

    #[test]
    fn equivalent_rack_states_are_merged() {
        // Bag {A, B}; X + A + B makes XAB (base 12, len 3 -> 12). The only
        // size-2 multiset is {A, B}, with probability 1. If draw order were
        // double-counted, the value would be 24 instead of 12.
        let engine = engine(
            "XAB\n",
            SearchConfig {
                max_depth: 2,
                exact_depth: 2,
                ..SearchConfig::default()
            },
        );
        let rack = Rack::new();
        let bag = bag_from("AB");
        let result = engine.search(&rack, letter_index(b'X').unwrap(), &bag, generous());
        assert_eq!(result.depth_completed, 2);
        assert!(
            (result.value_with - 12.0).abs() < 1e-9,
            "{}",
            result.value_with
        );
        assert!((result.retained_probability - 1.0).abs() < 1e-9);
    }

    #[test]
    fn repeated_letters_use_bag_counts() {
        // Bag {A, A}; X + A + A makes XAA (base 10, len 3 -> 10).
        let engine = engine(
            "XAA\n",
            SearchConfig {
                max_depth: 2,
                exact_depth: 2,
                ..SearchConfig::default()
            },
        );
        let rack = Rack::new();
        let bag = bag_from("AA");
        let result = engine.search(&rack, letter_index(b'X').unwrap(), &bag, generous());
        assert_eq!(result.depth_completed, 2);
        assert!(
            (result.value_with - 10.0).abs() < 1e-9,
            "{}",
            result.value_with
        );
    }

    #[test]
    fn short_deadline_returns_safe_fallback() {
        let engine = engine("QUIZ\n", SearchConfig::default());
        let rack = Rack::from_letters("QUI").unwrap();
        let bag = bag_from("AAAA");
        let deadline = Instant::now() - Duration::from_millis(1);
        let result = engine.search(&rack, letter_index(b'Z').unwrap(), &bag, deadline);
        assert_eq!(result.depth_completed, 0);
        assert!(result.fallback_used);
        assert_eq!(result.value_with, 33.0);
        assert_eq!(result.value_without, 0.0);
    }

    #[test]
    fn generous_deadline_reaches_configured_depth() {
        let engine = engine(
            "QUIZ\n",
            SearchConfig {
                max_depth: 3,
                exact_depth: 1,
                beam_width: 8,
                ..SearchConfig::default()
            },
        );
        let rack = Rack::from_letters("QUI").unwrap();
        let bag = bag_from("AAAAAA");
        let result = engine.search(&rack, letter_index(b'Z').unwrap(), &bag, generous());
        assert!(
            result.depth_completed >= 2,
            "depth {}",
            result.depth_completed
        );
        assert!(!result.fallback_used);
    }

    #[test]
    fn search_is_deterministic() {
        let engine = engine(
            "QUIZ\nQUIZZES\nJAZZ\n",
            SearchConfig {
                max_depth: 3,
                exact_depth: 1,
                beam_width: 16,
                ..SearchConfig::default()
            },
        );
        let rack = Rack::from_letters("QUIZ").unwrap();
        let bag = bag_from("AAAAZZZ");
        let a = engine.search(&rack, letter_index(b'E').unwrap(), &bag, generous());
        let b = engine.search(&rack, letter_index(b'E').unwrap(), &bag, generous());
        assert_eq!(a.depth_completed, b.depth_completed);
        assert!((a.value_with - b.value_with).abs() < 1e-12);
        assert!((a.value_without - b.value_without).abs() < 1e-12);
    }

    #[test]
    fn empty_bag_is_immediate_fallback() {
        let engine = engine("QUIZ\n", SearchConfig::default());
        let rack = Rack::from_letters("QUI").unwrap();
        let bag = BagModel::from_counts([0; 26]);
        let result = engine.search(&rack, letter_index(b'Z').unwrap(), &bag, generous());
        assert_eq!(result.depth_completed, 0);
        assert_eq!(result.value_with, 33.0);
    }
}
