//! Fixed opponents used to keep evolution grounded outside its own lineage.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;

use super::engine::{Action, Agent, Obs, face};
use super::rng::Rng;
use super::words::{Counts, LONG_LEN, WordIndex, remove_word};

const VOWELS: [usize; 5] = [0, 4, 8, 14, 20];

/// Upstream simulated strategies (`decideBid` / `simulateWordPlay`).
#[derive(Clone, Copy, Debug)]
pub enum SimKind {
    Cheapskate,
    ValueBidder,
    Aggressive,
}

pub struct SimBot(pub SimKind);

impl Agent for SimBot {
    fn name(&self) -> String {
        format!("sim-{:?}", self.0)
    }
    fn act(&mut self, obs: &Obs, _w: &WordIndex, rng: &mut Rng) -> Action {
        let value = face(obs.letter);
        let base = match self.0 {
            SimKind::Cheapskate => value - 1,
            SimKind::ValueBidder => value,
            SimKind::Aggressive => value + if VOWELS.contains(&obs.letter) { 4 } else { 2 },
        };
        let jitter = rng.range_i64(-2, 2);
        let bid = (base + jitter).max(1).min(obs.balance).max(0);
        Action {
            plays: Vec::new(),
            bid,
        }
    }
    fn on_win(&mut self, rack: &Counts, w: &WordIndex) -> Option<u32> {
        w.sim_word(rack, 3)
    }
}

/// The repository's original `--strategy greedy`: face (+2 if not held), and
/// play the best word immediately.
pub struct OldGreedy;

impl Agent for OldGreedy {
    fn name(&self) -> String {
        "old-greedy".into()
    }
    fn act(&mut self, obs: &Obs, w: &WordIndex, _rng: &mut Rng) -> Action {
        let mut rack = obs.rack;
        let mut plays = Vec::new();
        let mut balance = obs.balance;
        while let Some(id) = w.best_word(&rack, 2) {
            remove_word(&mut rack, w.counts(id));
            balance += w.reward(id) as i64;
            plays.push(id);
        }
        let premium = if rack[obs.letter] == 0 { 2 } else { 0 };
        Action {
            plays,
            bid: (face(obs.letter) + premium).min(balance),
        }
    }
}

/// Hoarding long-word player with a configurable bid rule. Used for the
/// "spender" archetype and as the play policy of the empirical opponent.
fn hoard_plays(obs: &Obs, w: &WordIndex, min_len: u8, pressure: u32) -> (Vec<u32>, Counts, i64) {
    let mut rack = obs.rack;
    let mut balance = obs.balance;
    let mut plays = Vec::new();
    if obs.bag_total == 0 {
        let (_, ws) = w.liquidate(&rack);
        for &id in &ws {
            balance += w.reward(id) as i64;
        }
        return (ws, [0; 26], balance);
    }
    loop {
        let total: u32 = rack.iter().map(|&c| c as u32).sum();
        let need = if total >= pressure { 5 } else { min_len };
        match w.best_word(&rack, need) {
            Some(id) => {
                remove_word(&mut rack, w.counts(id));
                balance += w.reward(id) as i64;
                plays.push(id);
            }
            None => break,
        }
    }
    (plays, rack, balance)
}

pub struct Spender;

impl Agent for Spender {
    fn name(&self) -> String {
        "spender".into()
    }
    fn act(&mut self, obs: &Obs, w: &WordIndex, _rng: &mut Rng) -> Action {
        let (plays, _, balance) = hoard_plays(obs, w, 5, 14);
        let bid = if obs.bag_total == 0 {
            0
        } else {
            (4 * face(obs.letter) + 4).min(balance)
        };
        Action { plays, bid }
    }
}

/// Bid quantiles of strong Maincloud bots, per letter and game phase.
#[derive(Clone, Debug, Deserialize)]
pub struct OpponentModel {
    pub winning_bid_quantiles: HashMap<String, Vec<f64>>,
}

impl OpponentModel {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Sample from a piecewise-linear quantile curve (q10..q90, tails
    /// extended linearly).
    fn sample(&self, letter: usize, phase: usize, rng: &mut Rng) -> f64 {
        let key = format!("{}{}", (b'A' + letter as u8) as char, phase);
        let Some(q) = self.winning_bid_quantiles.get(&key) else {
            return (3 * face(letter) + 3) as f64;
        };
        let ps = [0.1, 0.25, 0.5, 0.75, 0.9];
        let u = rng.f64();
        if u <= ps[0] {
            return (q[0] - (ps[0] - u) / 0.15 * (q[1] - q[0])).max(1.0);
        }
        for i in 0..4 {
            if u <= ps[i + 1] {
                let t = (u - ps[i]) / (ps[i + 1] - ps[i]);
                return q[i] + t * (q[i + 1] - q[i]);
            }
        }
        q[4] + (u - 0.9) / 0.15 * (q[4] - q[3])
    }
}

/// Opponent built from Maincloud data: bids drawn from strong bots' winning
/// bid distribution (scaled down because winning bids are the upper tail of
/// what they bid), and long-word hoarding play.
pub struct Empirical {
    pub model: Arc<OpponentModel>,
    pub scale: f64,
}

impl Agent for Empirical {
    fn name(&self) -> String {
        format!("empirical-{:.2}", self.scale)
    }
    fn act(&mut self, obs: &Obs, w: &WordIndex, rng: &mut Rng) -> Action {
        let (plays, _, balance) = hoard_plays(obs, w, LONG_LEN, 16);
        if obs.bag_total == 0 {
            return Action { plays, bid: 0 };
        }
        let round = obs.history.len();
        let phase = (round * 3 / 68).min(2);
        let bid = (self.model.sample(obs.letter, phase, rng) * self.scale).round() as i64;
        Action {
            plays,
            bid: bid.clamp(0, balance),
        }
    }
}
