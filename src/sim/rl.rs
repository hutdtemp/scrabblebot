//! Reinforcement-learning policy: shared inputs, action space, and a small
//! Rust inference runtime for trained weights (used in the simulator league,
//! the evaluation panel and the live bot).
//!
//! For an apples-to-apples comparison with the evolved policy, the RL policy
//! sees exactly the GA's inputs: the 63 bid features plus the 27 play
//! features of the best available word, all computed before any play.
//!
//! Actions:
//! - play: 0 = hold; k in 1..=7 = repeatedly play the best-reward word of at
//!   least `k + 1` letters (thresholds 2..=8).
//! - bid: an index into [`BID_BINS`], capped by the post-play balance.
//!
//! In the final auction the rack is always flushed and the bid is 0.

use serde::{Deserialize, Serialize};

use super::engine::{Action, Agent, Obs};
use super::policy::{
    Context, Decision, Genome, N_FEATURES, N_PLAY, Progress, bid_features, decide, lookahead,
    play_features,
};
use super::rng::Rng;
use super::words::{WordIndex, remove_word};

pub const N_OBS: usize = N_FEATURES + N_PLAY;
pub const N_PLAY_ACTIONS: usize = 8;
pub const BID_BINS: [i64; 28] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50,
    60, 75,
];
pub const N_BID_ACTIONS: usize = BID_BINS.len();

/// Observation vector for the RL policy (GA inputs, pre-play).
pub fn rl_obs(obs: &Obs, words: &WordIndex) -> [f32; N_OBS] {
    let mut out = [0f32; N_OBS];
    let l = obs.letter;
    let prog = Progress::from_obs(obs);
    let ctx = Context::from_obs(obs);
    let a = words.analyze(&obs.rack, Some(l));
    let la = lookahead(&a, l, &ctx.probs);
    let f = bid_features(obs, &obs.rack, obs.balance, &a, &la, &ctx, &prog);
    out[..N_FEATURES].copy_from_slice(&f);
    if let Some(id) = a.best0_id {
        let pf = play_features(
            words,
            id,
            &obs.rack,
            obs,
            &a,
            obs.balance,
            &prog,
            &la,
            ctx.recent_paid,
        );
        out[N_FEATURES..].copy_from_slice(&pf);
    }
    out
}

/// Turn discrete actions into concrete plays and a bid.
pub fn decode(obs: &Obs, words: &WordIndex, play: usize, bid: usize) -> Decision {
    let mut d = Decision::default();
    let mut rack = obs.rack;
    if obs.bag_total == 0 {
        d.plays = words.liquidate(&rack).1;
        return d;
    }
    let mut balance = obs.balance;
    if play > 0 {
        let min_len = (play + 1) as u8;
        for _ in 0..8 {
            let Some(id) = words.best_word(&rack, min_len) else {
                break;
            };
            remove_word(&mut rack, words.counts(id));
            balance += words.reward(id) as i64;
            d.plays.push(id);
        }
    }
    d.bid = BID_BINS[bid.min(N_BID_ACTIONS - 1)].min(balance);
    d
}

/// Map a teacher (evolved) decision onto the RL action space, for imitation.
pub fn teacher_actions(genome: &Genome, obs: &Obs, words: &WordIndex) -> (usize, usize) {
    let d = decide(genome, obs, words);
    let play = d
        .plays
        .iter()
        .map(|&id| words.word_len(id) as usize)
        .min()
        .map(|len| len.clamp(2, 8) - 1)
        .unwrap_or(0);
    let bid = BID_BINS
        .iter()
        .enumerate()
        .min_by_key(|(_, b)| (*b - d.bid).abs())
        .map(|(i, _)| i)
        .unwrap_or(0);
    (play, bid)
}

/// Bid head: a discretized Gaussian over [`BID_BINS`] with
/// `mu = 10 * z0 + 10` and `sigma = exp(clamp(z1)) + 0.5`. Greedy action =
/// the bin nearest `mu`. Keeps bins ordered (12 is close to 13) and needs
/// only two outputs.
pub fn bid_mu(z0: f32) -> f32 {
    10.0 * z0 + 10.0
}

pub fn nearest_bin(mu: f32) -> usize {
    BID_BINS
        .iter()
        .enumerate()
        .min_by(|a, b| {
            (*a.1 as f32 - mu)
                .abs()
                .total_cmp(&(*b.1 as f32 - mu).abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Trained actor weights exported from PyTorch (one tanh hidden layer).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RlModel {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub wp: Vec<Vec<f32>>,
    pub bp: Vec<f32>,
    pub wb: Vec<Vec<f32>>,
    pub bb: Vec<f32>,
}

impl RlModel {
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let m: RlModel = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        anyhow::ensure!(
            m.w1.iter().all(|r| r.len() == N_OBS)
                && m.wp.len() == N_PLAY_ACTIONS
                && m.wb.len() == 2,
            "RL model {} does not match the current inputs/actions",
            path.display()
        );
        Ok(m)
    }

    pub fn params(&self) -> usize {
        let h = self.b1.len();
        h * N_OBS + h + (N_PLAY_ACTIONS + 2) * (h + 1)
    }

    /// Greedy (argmax) actions.
    pub fn act(&self, x: &[f32; N_OBS]) -> (usize, usize) {
        let h: Vec<f32> = self
            .w1
            .iter()
            .zip(&self.b1)
            .map(|(row, b)| (row.iter().zip(x).map(|(w, v)| w * v).sum::<f32>() + b).tanh())
            .collect();
        let out = |w: &Vec<Vec<f32>>, b: &Vec<f32>| -> Vec<f32> {
            w.iter()
                .zip(b)
                .map(|(row, b)| row.iter().zip(&h).map(|(w, v)| w * v).sum::<f32>() + b)
                .collect()
        };
        let play = out(&self.wp, &self.bp)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let z = out(&self.wb, &self.bb);
        (play, nearest_bin(bid_mu(z[0])))
    }

    pub fn decide(&self, obs: &Obs, words: &WordIndex) -> Decision {
        if obs.bag_total == 0 {
            return decode(obs, words, 0, 0);
        }
        let (p, b) = self.act(&rl_obs(obs, words));
        decode(obs, words, p, b)
    }
}

impl super::policy::Policy for RlModel {
    fn decide(&self, obs: &Obs, words: &WordIndex) -> Decision {
        RlModel::decide(self, obs, words)
    }
}

pub struct RlAgent {
    pub model: std::sync::Arc<RlModel>,
}

impl Agent for RlAgent {
    fn name(&self) -> String {
        "rl".into()
    }
    fn act(&mut self, obs: &Obs, words: &WordIndex, _rng: &mut Rng) -> Action {
        let d = self.model.decide(obs, words);
        Action {
            plays: d.plays,
            bid: d.bid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::engine::{AuctionKind, AuctionRecord};
    use crate::sim::words::counts_from;

    #[test]
    fn decode_respects_threshold_and_balance() {
        let words = WordIndex::bundled();
        let hist = vec![
            AuctionRecord {
                letter: 4,
                winner: None,
                paid: 0,
                top_bid: 0
            };
            10
        ];
        let obs = Obs {
            me: 0,
            rack: counts_from("RETAINSCAT"),
            initial_rack: counts_from("RETAI"),
            balance: 3,
            score: 0,
            letter: 0,
            bag_total: 30,
            balances: &[3, 50],
            scores: &[0, 10],
            history: &hist,
            kind: AuctionKind::Vickrey,
        };
        let hold = decode(&obs, &words, 0, 27);
        assert!(hold.plays.is_empty());
        assert_eq!(hold.bid, 3); // capped by balance
        let long = decode(&obs, &words, 6, 27); // threshold 7
        assert!(long.plays.iter().all(|&id| words.word_len(id) >= 7));
        assert!(!long.plays.is_empty());
        assert!(long.bid > 3); // balance grew from the play
        let x = rl_obs(&obs, &words);
        assert!(x.iter().all(|v| v.is_finite()));
    }
}
