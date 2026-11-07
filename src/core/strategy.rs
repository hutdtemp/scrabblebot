//! Bidding and playback policy.
//!
//! The default search strategy bids a shaded fraction of the search-derived
//! marginal value. Vickrey auctions are close to truth-telling, so the default
//! fraction is 0.9; first-price auctions need more shading, so the default is
//! 0.7. The bid is always capped by the authoritative balance.
//!
//! Playback is deliberately greedy. The default target-aware policy compares
//! the best currently playable word with the best word within two missing
//! letters. It holds only when the future target is worth at least
//! `1 / target_ratio` times the current word, and it force-plays when the bag
//! is nearly empty or after `target_hold_limit` auctions without a play. This
//! is a heuristic, not a proof of optimality; the README states it plainly.

use serde::{Deserialize, Serialize};

use super::dictionary::{Dictionary, Target, WordChoice};
use super::letters::letter_value;
use super::rack::Rack;
use super::search::SearchResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuctionKind {
    Vickrey,
    FirstPrice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategyKind {
    /// No lookahead: face value plus a small premium, and the best playable
    /// word immediately.
    Greedy,
    /// Beam-search valuation and target-aware playback.
    Search,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaybackPolicy {
    /// Play the best playable word immediately.
    Greedy,
    /// Hold a playable word only for a clearly better near-future target.
    TargetAware,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub vickrey_bid_fraction: f64,
    pub first_price_bid_fraction: f64,
    pub min_marginal_for_bid: f64,
    pub max_bid_fraction: f64,
    pub playback: PlaybackPolicy,
    pub target_max_missing: u8,
    pub target_ratio: f64,
    pub target_hold_limit: u32,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            vickrey_bid_fraction: 0.9,
            first_price_bid_fraction: 0.7,
            min_marginal_for_bid: 0.5,
            max_bid_fraction: 1.0,
            playback: PlaybackPolicy::TargetAware,
            target_max_missing: 2,
            target_ratio: 0.5,
            target_hold_limit: 2,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BidDecision {
    pub amount: i64,
    pub marginal_value: f64,
    pub fallback_used: bool,
    pub reason: BidReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BidReason {
    SearchMarginal,
    GreedyFallback,
    NoPositiveValue,
    BalanceCapped,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayDecision {
    pub word: Option<WordChoice>,
    pub target: Option<Target>,
    pub reason: PlayReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlayReason {
    GreedyBestNow,
    TargetBeatsThreshold,
    ForcedPlay,
    HoldForTarget,
    NoPlayableWord,
    NoTarget,
}

/// Simple no-lookahead bid: face value, plus two if we do not hold the letter.
pub fn greedy_bid(letter: usize, rack: &Rack, balance: i64) -> i64 {
    if balance <= 0 {
        return 0;
    }
    let face = letter_value(letter) as i64;
    let premium = if rack.count(letter) == 0 { 2 } else { 0 };
    (face + premium).clamp(0, balance)
}

/// Turn a search marginal value into a bid.
pub fn search_bid(
    result: &SearchResult,
    rack: &Rack,
    balance: i64,
    auction: AuctionKind,
    config: &StrategyConfig,
) -> BidDecision {
    if result.fallback_used {
        let amount = greedy_bid(result.candidate as usize, rack, balance);
        return BidDecision {
            amount,
            marginal_value: result.marginal,
            fallback_used: true,
            reason: if amount > 0 {
                BidReason::GreedyFallback
            } else {
                BidReason::NoPositiveValue
            },
        };
    }
    bid_from_marginal(result.marginal, balance, auction, config)
}

pub fn bid_from_marginal(
    marginal: f64,
    balance: i64,
    auction: AuctionKind,
    config: &StrategyConfig,
) -> BidDecision {
    if balance <= 0 {
        return BidDecision {
            amount: 0,
            marginal_value: marginal,
            fallback_used: false,
            reason: BidReason::NoPositiveValue,
        };
    }
    let marginal = if marginal.is_finite() {
        marginal.max(0.0)
    } else {
        0.0
    };
    if marginal < config.min_marginal_for_bid {
        return BidDecision {
            amount: 0,
            marginal_value: marginal,
            fallback_used: false,
            reason: BidReason::NoPositiveValue,
        };
    }
    let fraction = match auction {
        AuctionKind::Vickrey => config.vickrey_bid_fraction,
        AuctionKind::FirstPrice => config.first_price_bid_fraction,
    }
    .clamp(0.0, 1.0);
    let raw = (marginal * fraction).round() as i64;
    let raw = raw.max(1);
    let max_bid = ((balance as f64) * config.max_bid_fraction.clamp(0.0, 1.0)).floor() as i64;
    let max_bid = max_bid.clamp(0, balance);
    let amount = raw.min(max_bid);
    BidDecision {
        amount,
        marginal_value: marginal,
        fallback_used: false,
        reason: if amount == max_bid && raw > max_bid {
            BidReason::BalanceCapped
        } else {
            BidReason::SearchMarginal
        },
    }
}

/// Choose a word to play, or deliberately hold.
pub fn choose_play(
    rack: &Rack,
    dictionary: &Dictionary,
    config: &StrategyConfig,
    hold_auctions: u32,
    bag_total: u32,
) -> PlayDecision {
    let best_now = dictionary.best_word(rack).map(WordChoice::from);

    if config.playback == PlaybackPolicy::Greedy {
        return match best_now {
            Some(word) => PlayDecision {
                word: Some(word),
                target: None,
                reason: PlayReason::GreedyBestNow,
            },
            None => PlayDecision {
                word: None,
                target: None,
                reason: PlayReason::NoPlayableWord,
            },
        };
    }

    let target = dictionary.best_target(rack, config.target_max_missing);
    match (best_now, target) {
        (Some(best), Some(target)) => {
            let force = bag_total <= 2 || hold_auctions >= config.target_hold_limit;
            let ratio_ok =
                best.reward as f64 >= target.reward as f64 * config.target_ratio.clamp(0.0, 1.0);
            if force {
                PlayDecision {
                    word: Some(best),
                    target: Some(target),
                    reason: PlayReason::ForcedPlay,
                }
            } else if ratio_ok {
                PlayDecision {
                    word: Some(best),
                    target: Some(target),
                    reason: PlayReason::TargetBeatsThreshold,
                }
            } else {
                PlayDecision {
                    word: None,
                    target: Some(target),
                    reason: PlayReason::HoldForTarget,
                }
            }
        }
        (Some(best), None) => PlayDecision {
            word: Some(best),
            target: None,
            reason: PlayReason::NoTarget,
        },
        (None, Some(target)) => PlayDecision {
            word: None,
            target: Some(target),
            reason: PlayReason::HoldForTarget,
        },
        (None, None) => PlayDecision {
            word: None,
            target: None,
            reason: PlayReason::NoPlayableWord,
        },
    }
}

/// Convenience for the no-lookahead strategy.
pub fn greedy_word(rack: &Rack, dictionary: &Dictionary) -> Option<WordChoice> {
    dictionary.best_word(rack).map(WordChoice::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::dictionary::Dictionary;
    use crate::core::rack::Rack;
    use crate::core::search::SearchResult;

    fn result(marginal: f64, fallback: bool) -> SearchResult {
        SearchResult {
            candidate: 25,
            value_without: 10.0,
            value_with: 10.0 + marginal,
            marginal,
            depth_completed: if fallback { 0 } else { 3 },
            states_evaluated: 10,
            elapsed_ms: 5,
            timed_out: false,
            fallback_used: fallback,
            retained_probability: 1.0,
            best_now_without: None,
            best_now_with: None,
        }
    }

    #[test]
    fn greedy_bid_pays_face_value_plus_premium_when_missing() {
        let rack = Rack::from_letters("AE").unwrap();
        assert_eq!(greedy_bid(25, &rack, 100), 12); // Z=10 + 2
        assert_eq!(greedy_bid(0, &rack, 100), 1); // A=1, already held
        assert_eq!(greedy_bid(25, &rack, 5), 5); // capped by balance
        assert_eq!(greedy_bid(25, &rack, 0), 0);
    }

    #[test]
    fn search_bid_shades_vickrey_and_first_price() {
        let config = StrategyConfig::default();
        let vickrey = bid_from_marginal(10.0, 100, AuctionKind::Vickrey, &config);
        assert_eq!(vickrey.amount, 9);
        let first = bid_from_marginal(10.0, 100, AuctionKind::FirstPrice, &config);
        assert_eq!(first.amount, 7);
    }

    #[test]
    fn search_bid_ignores_tiny_marginals_and_caps_balance() {
        let config = StrategyConfig::default();
        assert_eq!(
            bid_from_marginal(0.4, 100, AuctionKind::Vickrey, &config).amount,
            0
        );
        assert_eq!(
            bid_from_marginal(1000.0, 7, AuctionKind::Vickrey, &config).amount,
            7
        );
        assert_eq!(
            bid_from_marginal(10.0, 0, AuctionKind::Vickrey, &config).amount,
            0
        );
    }

    #[test]
    fn fallback_uses_greedy_bid() {
        let config = StrategyConfig::default();
        let rack = Rack::from_letters("AE").unwrap();
        let decision = search_bid(
            &result(0.0, true),
            &rack,
            100,
            AuctionKind::Vickrey,
            &config,
        );
        assert_eq!(decision.amount, 12);
        assert!(decision.fallback_used);
    }

    fn dict() -> Dictionary {
        Dictionary::from_wordlist("QI\nQUIZ\n")
    }

    #[test]
    fn target_aware_holds_a_small_word_for_a_big_target() {
        let rack = Rack::from_letters("QUI").unwrap();
        let decision = choose_play(&rack, &dict(), &StrategyConfig::default(), 0, 20);
        assert!(decision.word.is_none());
        assert_eq!(decision.reason, PlayReason::HoldForTarget);
        assert_eq!(decision.target.unwrap().word, "QUIZ");
    }

    #[test]
    fn target_aware_forces_play_when_bag_is_nearly_empty() {
        let rack = Rack::from_letters("QUI").unwrap();
        let decision = choose_play(&rack, &dict(), &StrategyConfig::default(), 0, 2);
        assert_eq!(decision.word.unwrap().word, "QI");
        assert_eq!(decision.reason, PlayReason::ForcedPlay);
    }

    #[test]
    fn target_aware_forces_play_after_hold_limit() {
        let rack = Rack::from_letters("QUI").unwrap();
        let decision = choose_play(&rack, &dict(), &StrategyConfig::default(), 2, 20);
        assert_eq!(decision.word.unwrap().word, "QI");
        assert_eq!(decision.reason, PlayReason::ForcedPlay);
    }

    #[test]
    fn target_aware_plays_target_when_it_is_playable() {
        let rack = Rack::from_letters("QUIZ").unwrap();
        let decision = choose_play(&rack, &dict(), &StrategyConfig::default(), 0, 20);
        assert_eq!(decision.word.unwrap().word, "QUIZ");
        assert_eq!(decision.reason, PlayReason::NoTarget);
    }

    #[test]
    fn greedy_playback_always_plays_best_now() {
        let rack = Rack::from_letters("QUI").unwrap();
        let config = StrategyConfig {
            playback: PlaybackPolicy::Greedy,
            ..StrategyConfig::default()
        };
        let decision = choose_play(&rack, &dict(), &config, 0, 20);
        assert_eq!(decision.word.unwrap().word, "QI");
        assert_eq!(decision.reason, PlayReason::GreedyBestNow);
    }
}
