//! The evolvable policy, shared by the simulator and the live client.
//!
//! Each decision point (an auction's open):
//!
//! 1. **Play.** Repeatedly consider the best-reward playable word. A linear
//!    play head scores it from its length, reward, rack size, game phase and
//!    balance; the word is played if the score is positive. During the final
//!    auction every playable word is flushed, since the match ends when that
//!    auction closes.
//! 2. **Bid.** A hand-shaped base bid (face value plus lookahead marginals)
//!    is corrected by a small tanh MLP over ~54 features, then capped by the
//!    balance. The final auction's letter can never be played, so it gets 0.
//!
//! The base terms make a fresh genome behave sensibly; evolution adjusts the
//! base weights, the play head, and the MLP correction together.

use serde::{Deserialize, Serialize};

use super::engine::{Action, Obs, face};
use super::rng::Rng;
use super::words::{Analysis, Counts, WordIndex, remove_word};
use crate::core::letters::DEFAULT_BAG;

pub const N_FEATURES: usize = 63;
pub const HIDDEN: usize = 16;
pub const N_BASE: usize = 6;
pub const N_PLAY: usize = 27;
pub const MLP_CORRECTION: f32 = 20.0;

// Genome layout offsets.
const BASE: usize = 0;
const PLAY: usize = BASE + N_BASE; // N_PLAY weights + bias
const W1: usize = PLAY + N_PLAY + 1;
const B1: usize = W1 + HIDDEN * N_FEATURES;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + HIDDEN;
pub const GENOME_LEN: usize = B2 + 1;

const VOWELS: u32 = (1 << 0) | (1 << 4) | (1 << 8) | (1 << 14) | (1 << 20);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Genome {
    pub params: Vec<f32>,
    /// Per-genome mutation step multiplier (self-adaptive).
    #[serde(default = "one")]
    pub sigma: f32,
}

fn one() -> f32 {
    1.0
}

impl Genome {
    /// A hand-written starting point: bid ~2.5x face + 3 plus lookahead
    /// marginals, and play only 7+ letter words before the final auction.
    pub fn prior() -> Self {
        let mut p = vec![0.0f32; GENOME_LEN];
        // base: face, marg, marg_long, e_marg, e_marg_long, bias
        p[BASE..BASE + N_BASE].copy_from_slice(&[2.5, 0.4, 0.4, 0.3, 0.3, 3.0]);
        // play features: reward, len{2-3,4,5,6,7,8+}, rack, left, bal, leftover, flex, bias
        let play = [
            0.5, -6.0, -6.0, -6.0, -3.0, 4.0, 5.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, // v1 inputs
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // v2 match-progress inputs
            0.0, 0.0, 0.0, 0.0, 0.0, // v3 standing and risk inputs
            0.0, 0.0, 0.0, // v3 liquidity inputs
        ];
        p[PLAY..PLAY + N_PLAY].copy_from_slice(&play);
        p[PLAY + N_PLAY] = 0.0;
        Self {
            params: p,
            sigma: 1.0,
        }
    }

    /// Mutation scale per gene: base and play genes in their natural units,
    /// MLP weights small.
    pub fn gene_scale(i: usize) -> f32 {
        if i < PLAY {
            [0.4, 0.1, 0.1, 0.1, 0.1, 1.0][i - BASE]
        } else if i < W1 {
            0.5
        } else {
            0.15
        }
    }

    pub fn len_ok(&self) -> bool {
        self.params.len() == GENOME_LEN
    }

    /// Upgrade an older layout by inserting zero weights for inputs added
    /// since, so an upgraded genome behaves exactly like the original.
    /// Layouts: v1 = 13 play inputs / 54 bid features; v2 = 19 / 57.
    pub fn upgrade(mut self) -> Self {
        const LAYOUTS: [(usize, usize); 2] = [(13, 54), (19, 57)];
        if self.len_ok() {
            return self;
        }
        for (np, nf) in LAYOUTS {
            let old_len = N_BASE + np + 1 + HIDDEN * nf + HIDDEN + HIDDEN + 1;
            if self.params.len() != old_len {
                continue;
            }
            let o = &self.params;
            let mut p = Vec::with_capacity(GENOME_LEN);
            p.extend_from_slice(&o[..N_BASE]);
            p.extend_from_slice(&o[N_BASE..N_BASE + np]);
            p.extend(std::iter::repeat_n(0.0, N_PLAY - np));
            p.push(o[N_BASE + np]); // play bias
            let w1 = N_BASE + np + 1;
            for h in 0..HIDDEN {
                p.extend_from_slice(&o[w1 + h * nf..w1 + (h + 1) * nf]);
                p.extend(std::iter::repeat_n(0.0, N_FEATURES - nf));
            }
            p.extend_from_slice(&o[w1 + HIDDEN * nf..]); // b1, w2, b2
            self.params = p;
            break;
        }
        self
    }

    /// Load from JSON, upgrading older layouts.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let g: Genome = serde_json::from_str(&text)?;
        let g = g.upgrade();
        anyhow::ensure!(
            g.len_ok(),
            "genome {} has an unknown layout",
            path.display()
        );
        Ok(g)
    }
}

/// Match-progress signals shared by the bid and play heads.
pub struct Progress {
    /// Fraction of the match's auctions already closed, 0..1.
    pub frac: f64,
    /// Letters we can expect to still win at our win rate so far.
    pub exp_wins_left: f64,
}

impl Progress {
    pub fn from_obs(obs: &Obs) -> Self {
        let done = obs.history.len() as f64;
        let total = done + obs.auctions_left() as f64;
        let wins = obs
            .history
            .iter()
            .filter(|r| r.winner == Some(obs.me as u8))
            .count() as f64;
        // Smoothed toward one win in five auctions early in the match.
        let rate = (wins + 1.0) / (done + 5.0);
        Self {
            frac: if total > 0.0 { done / total } else { 1.0 },
            exp_wins_left: rate * obs.auctions_left() as f64,
        }
    }
}

/// Public-information bag estimate: full bag minus our starting rack minus
/// every auctioned letter, scaled to the authoritative remaining total.
/// Returns draw probabilities for the next letter.
pub fn bag_probs(initial: &Counts, seen: &[u32; 26]) -> [f64; 26] {
    let mut est = [0f64; 26];
    let mut sum = 0.0;
    for k in 0..26 {
        let v = DEFAULT_BAG[k] as f64 - initial[k] as f64 - seen[k] as f64;
        est[k] = v.max(0.0);
        sum += est[k];
    }
    if sum <= 0.0 {
        return [1.0 / 26.0; 26];
    }
    for v in est.iter_mut() {
        *v /= sum;
    }
    est
}

pub struct Context {
    pub probs: [f64; 26],
    pub recent_paid: f64,
    pub letter_paid: f64,
}

impl Context {
    pub fn from_obs(obs: &Obs) -> Self {
        let mut seen = [0u32; 26];
        for r in obs.history {
            seen[r.letter as usize] += 1;
        }
        seen[obs.letter] += 1;
        let mut recent = 0.0;
        let mut n = 0.0;
        for r in obs
            .history
            .iter()
            .rev()
            .filter(|r| r.winner.is_some())
            .take(10)
        {
            recent += r.paid as f64;
            n += 1.0;
        }
        let mut lp = 0.0;
        let mut ln = 0.0;
        for r in obs
            .history
            .iter()
            .filter(|r| r.winner.is_some() && r.letter as usize == obs.letter)
        {
            lp += r.paid as f64;
            ln += 1.0;
        }
        Self {
            probs: bag_probs(&obs.initial_rack, &seen),
            recent_paid: if n > 0.0 { recent / n } else { 6.0 },
            letter_paid: if ln > 0.0 {
                lp / ln
            } else {
                3.0 * face(obs.letter) as f64 + 3.0
            },
        }
    }
}

pub struct Lookahead {
    pub now: f64,
    pub with_l: f64,
    pub now_long: f64,
    pub with_l_long: f64,
    pub e1: f64,
    pub e2: f64,
    pub e1_long: f64,
    pub e2_long: f64,
    /// Standard deviations of the same quantities over the next draw: how
    /// much is riding on luck, as opposed to the expected value.
    pub sd1: f64,
    pub sd2: f64,
    pub sd1_long: f64,
    pub sd2_long: f64,
}

pub fn lookahead(a: &Analysis, l: usize, probs: &[f64; 26]) -> Lookahead {
    let (mut e1, mut e2, mut e1l, mut e2l) = (0.0, 0.0, 0.0, 0.0);
    let (mut q1, mut q2, mut q1l, mut q2l) = (0.0, 0.0, 0.0, 0.0);
    for (x, &p) in probs.iter().enumerate() {
        if p == 0.0 {
            continue;
        }
        let (v1, v2) = (a.with1(x) as f64, a.with2(l, x) as f64);
        let (v1l, v2l) = (a.with1_long(x) as f64, a.with2_long(l, x) as f64);
        e1 += p * v1;
        e2 += p * v2;
        e1l += p * v1l;
        e2l += p * v2l;
        q1 += p * v1 * v1;
        q2 += p * v2 * v2;
        q1l += p * v1l * v1l;
        q2l += p * v2l * v2l;
    }
    let sd = |q: f64, e: f64| (q - e * e).max(0.0).sqrt();
    Lookahead {
        now: a.best0 as f64,
        with_l: a.with1(l) as f64,
        now_long: a.best0_long as f64,
        with_l_long: a.with1_long(l) as f64,
        e1,
        e2,
        e1_long: e1l,
        e2_long: e2l,
        sd1: sd(q1, e1),
        sd2: sd(q2, e2),
        sd1_long: sd(q1l, e1l),
        sd2_long: sd(q2l, e2l),
    }
}

fn rack_total(r: &Counts) -> u32 {
    r.iter().map(|&c| c as u32).sum()
}

pub fn bid_features(
    obs: &Obs,
    rack: &Counts,
    balance: i64,
    a: &Analysis,
    la: &Lookahead,
    ctx: &Context,
    prog: &Progress,
) -> [f32; N_FEATURES] {
    let mut f = [0f32; N_FEATURES];
    let l = obs.letter;
    f[l] = 1.0;
    let total = rack_total(rack) as f64;
    let vowels: u32 = (0..26)
        .filter(|k| VOWELS & (1 << k) != 0)
        .map(|k| rack[k] as u32)
        .sum();
    let left = obs.auctions_left() as f64;
    let max_other_score = (0..obs.n_players())
        .filter(|&j| j != obs.me)
        .map(|j| obs.scores[j])
        .max()
        .unwrap_or(0) as f64;
    let max_other_bal = (0..obs.n_players())
        .filter(|&j| j != obs.me)
        .map(|j| obs.balances[j])
        .max()
        .unwrap_or(0) as f64;
    let score = obs.scores[obs.me] as f64;
    let vals = [
        face(l) as f64 / 10.0,
        if VOWELS & (1 << l) != 0 { 1.0 } else { 0.0 },
        rack[l] as f64 / 2.0,
        total / 20.0,
        vowels as f64 / total.max(1.0),
        la.now / 50.0,
        la.with_l / 50.0,
        la.now_long / 60.0,
        la.with_l_long / 60.0,
        (la.with_l - la.now) / 20.0,
        (la.with_l_long - la.now_long) / 30.0,
        la.e1 / 50.0,
        la.e2 / 50.0,
        (la.e2 - la.e1) / 20.0,
        la.e1_long / 60.0,
        la.e2_long / 60.0,
        (la.e2_long - la.e1_long) / 30.0,
        (1.0 + a.flex1_long as f64).ln() / 5.0,
        (1.0 + a.flex2_long as f64).ln() / 8.0,
        balance as f64 / 100.0,
        left / 68.0,
        balance as f64 / left / 10.0,
        score / 150.0,
        (score - max_other_score) / 100.0,
        max_other_bal / 100.0,
        ctx.recent_paid / 15.0,
        ctx.letter_paid / 20.0,
        1.0,
        // v2: match progress
        prog.frac,
        prog.exp_wins_left / 10.0,
        if a.best0_long == 0 && a.flex1_long == 0 {
            prog.frac
        } else {
            0.0
        },
        // v3: risk (spread of the lookahead outcomes)
        la.sd1 / 30.0,
        la.sd2 / 30.0,
        la.sd1_long / 30.0,
        la.sd2_long / 30.0,
        (la.sd2_long - la.sd1_long) / 20.0,
        // v3: liquidity (how many market-priced letters the balance buys)
        (balance as f64 / ctx.recent_paid.max(1.0)).min(5.0) / 5.0,
    ];
    f[26..26 + vals.len()].copy_from_slice(&vals.map(|v| v as f32));
    f
}

fn mlp(g: &[f32], f: &[f32; N_FEATURES]) -> f32 {
    let mut z = g[B2];
    for h in 0..HIDDEN {
        let row = &g[W1 + h * N_FEATURES..W1 + (h + 1) * N_FEATURES];
        let mut s = g[B1 + h];
        for k in 0..N_FEATURES {
            s += row[k] * f[k];
        }
        z += g[W2 + h] * s.tanh();
    }
    z
}

/// Inputs for the decision to play word `id` now.
#[allow(clippy::too_many_arguments)]
pub fn play_features(
    words: &WordIndex,
    id: u32,
    rack: &Counts,
    obs: &Obs,
    a: &Analysis,
    balance: i64,
    prog: &Progress,
    la: &Lookahead,
    price: f64,
) -> [f32; N_PLAY] {
    let len = words.word_len(id);
    let bucket = match len {
        0..=3 => 0,
        4 => 1,
        5 => 2,
        6 => 3,
        7 => 4,
        _ => 5,
    };
    let total = rack_total(rack) as f32;
    let mut pf = [0f32; N_PLAY];
    pf[0] = words.reward(id) as f32 / 50.0;
    pf[1 + bucket] = 1.0;
    pf[7] = total / 20.0;
    pf[8] = obs.auctions_left() as f32 / 68.0;
    pf[9] = balance as f32 / 100.0;
    pf[10] = (total - len as f32) / 20.0;
    pf[11] = (1.0 + a.flex1_long as f32).ln() / 5.0;
    pf[12] = if a.best0_long > 0 { 1.0 } else { 0.0 };
    // v2: progress, and progress x length so late matches can accept
    // shorter words while early ones still hold for 7+.
    let t = prog.frac as f32;
    pf[13] = t;
    pf[14] = if len == 4 { t } else { 0.0 };
    pf[15] = if len == 5 { t } else { 0.0 };
    pf[16] = if len == 6 { t } else { 0.0 };
    pf[17] = t * t;
    pf[18] = prog.exp_wins_left as f32 / 10.0;
    // v3: standing and risk. Margin vs the best opponent, the spread of what
    // holding could turn into, and that spread split by behind / ahead so a
    // linear head can be risk-seeking when behind and risk-averse when ahead.
    let me = obs.scores[obs.me];
    let best_other = (0..obs.n_players())
        .filter(|&j| j != obs.me)
        .map(|j| obs.scores[j])
        .max()
        .unwrap_or(0);
    let margin = (me - best_other) as f32 / 100.0;
    let risk = la.sd1_long.max(la.sd1) as f32 / 30.0;
    pf[19] = margin;
    pf[20] = la.sd1 as f32 / 30.0;
    pf[21] = la.sd1_long as f32 / 30.0;
    pf[22] = risk * (-margin).max(0.0);
    pf[23] = risk * margin.max(0.0);
    // v3: liquidity. Holding a word while unable to afford the market price
    // forfeits every auction until it is played; playing now buys letters.
    let starved = (1.0 - balance as f64 / price.max(1.0)).clamp(0.0, 1.0) as f32;
    pf[24] = starved;
    pf[25] = starved * words.reward(id) as f32 / 50.0;
    pf[26] = starved * obs.auctions_left() as f32 / 68.0;
    pf
}

/// Linear play head: positive means play.
fn play_score(g: &[f32], pf: &[f32; N_PLAY]) -> f32 {
    let w = &g[PLAY..PLAY + N_PLAY];
    let mut s = g[PLAY + N_PLAY];
    for k in 0..N_PLAY {
        s += w[k] * pf[k];
    }
    s
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Decision {
    pub plays: Vec<u32>,
    pub bid: i64,
    pub base_bid: f64,
    pub correction: f64,
}

/// The full policy decision for one auction open.
pub fn decide(genome: &Genome, obs: &Obs, words: &WordIndex) -> Decision {
    let g = &genome.params;
    let mut rack = obs.rack;
    let mut balance = obs.balance;
    let l = obs.letter;
    let mut d = Decision::default();

    if obs.bag_total == 0 {
        // Final auction: flush everything; the letter is worthless.
        let (_, ws) = words.liquidate(&rack);
        d.plays = ws;
        return d;
    }

    let prog = Progress::from_obs(obs);
    let ctx = Context::from_obs(obs);
    let mut a = words.analyze(&rack, Some(l));
    let mut la = lookahead(&a, l, &ctx.probs);
    for _ in 0..8 {
        let Some(id) = a.best0_id else { break };
        let pf = play_features(
            words,
            id,
            &rack,
            obs,
            &a,
            balance,
            &prog,
            &la,
            ctx.recent_paid,
        );
        if play_score(g, &pf) <= 0.0 {
            break;
        }
        remove_word(&mut rack, words.counts(id));
        balance += words.reward(id) as i64;
        d.plays.push(id);
        a = words.analyze(&rack, Some(l));
        la = lookahead(&a, l, &ctx.probs);
    }
    let base = g[BASE] as f64 * face(l) as f64
        + g[BASE + 1] as f64 * (la.with_l - la.now).max(0.0)
        + g[BASE + 2] as f64 * (la.with_l_long - la.now_long).max(0.0)
        + g[BASE + 3] as f64 * (la.e2 - la.e1).max(0.0)
        + g[BASE + 4] as f64 * (la.e2_long - la.e1_long).max(0.0)
        + g[BASE + 5] as f64;
    let f = bid_features(obs, &rack, balance, &a, &la, &ctx, &prog);
    let corr = MLP_CORRECTION as f64 * (mlp(g, &f) as f64).tanh();
    let bid = (base + corr).round().max(0.0) as i64;
    d.bid = bid.min(balance);
    d.base_bid = base;
    d.correction = corr;
    d
}

/// Mutate in place (Gaussian, self-adaptive sigma).
pub fn mutate(g: &mut Genome, rng: &mut Rng, rate: f64) {
    g.sigma = (g.sigma * (0.2 * rng.normal()).exp() as f32).clamp(0.05, 4.0);
    for i in 0..g.params.len() {
        if rng.f64() < rate {
            g.params[i] += g.sigma * Genome::gene_scale(i) * rng.normal() as f32;
        }
    }
}

/// Uniform crossover in blocks: the base genes, the play head, and each
/// hidden unit (its input row, bias, and output weight) move together.
pub fn crossover(a: &Genome, b: &Genome, rng: &mut Rng) -> Genome {
    let mut c = a.clone();
    let pick = |rng: &mut Rng| rng.f64() < 0.5;
    if pick(rng) {
        c.params[BASE..PLAY].copy_from_slice(&b.params[BASE..PLAY]);
    }
    if pick(rng) {
        c.params[PLAY..W1].copy_from_slice(&b.params[PLAY..W1]);
    }
    for h in 0..HIDDEN {
        if pick(rng) {
            let r = W1 + h * N_FEATURES..W1 + (h + 1) * N_FEATURES;
            c.params[r.clone()].copy_from_slice(&b.params[r]);
            c.params[B1 + h] = b.params[B1 + h];
            c.params[W2 + h] = b.params[W2 + h];
        }
    }
    if pick(rng) {
        c.params[B2] = b.params[B2];
    }
    c.sigma = 0.5 * (a.sigma + b.sigma);
    c
}

/// Anything that can make the per-auction decision (evolved genome or a
/// trained RL actor). Used by the live client.
pub trait Policy: Send + Sync {
    fn decide(&self, obs: &Obs, words: &WordIndex) -> Decision;
}

impl std::fmt::Debug for dyn Policy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Policy")
    }
}

impl Policy for Genome {
    fn decide(&self, obs: &Obs, words: &WordIndex) -> Decision {
        decide(self, obs, words)
    }
}

/// Wrapper: in the last `window` auctions, bid at least balance / auctions
/// left (money left at the end of a match is worth nothing). The final
/// auction still bids 0 because its letter can never be played.
pub struct SpendDown {
    pub inner: std::sync::Arc<dyn Policy>,
    pub window: u32,
}

impl Policy for SpendDown {
    fn decide(&self, obs: &Obs, words: &WordIndex) -> Decision {
        let mut d = self.inner.decide(obs, words);
        if obs.bag_total > 0 && obs.bag_total < self.window {
            let reward: i64 = d.plays.iter().map(|&id| words.reward(id) as i64).sum();
            let balance = obs.balance + reward;
            // Exclude the final (worthless) auction from the divisor.
            let floor = balance / obs.bag_total as i64;
            d.bid = d.bid.max(floor).min(balance);
        }
        d
    }
}

/// Load a policy file: `*.rl.json` is an RL actor, anything else a genome.
/// A `spend<K>:` prefix wraps it in [`SpendDown`] with window K.
pub fn load_policy(path: &std::path::Path) -> anyhow::Result<std::sync::Arc<dyn Policy>> {
    let s = path.to_string_lossy();
    if let Some(rest) = s.strip_prefix("spend")
        && let Some((k, p)) = rest.split_once(':')
    {
        return Ok(std::sync::Arc::new(SpendDown {
            inner: load_policy(std::path::Path::new(p))?,
            window: k.parse()?,
        }));
    }
    if path.to_string_lossy().ends_with(".rl.json") {
        Ok(std::sync::Arc::new(super::rl::RlModel::load(path)?))
    } else {
        Ok(std::sync::Arc::new(Genome::load(path)?))
    }
}

/// Simulator agent wrapper.
pub struct Evolved {
    pub genome: std::sync::Arc<Genome>,
    pub label: String,
}

impl super::engine::Agent for Evolved {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn act(&mut self, obs: &Obs, words: &WordIndex, _rng: &mut Rng) -> Action {
        let d = decide(&self.genome, obs, words);
        Action {
            plays: d.plays,
            bid: d.bid,
        }
    }
}

/// Experiment: same bidding as the genome, but never play a word until the
/// final auction, then liquidate everything at once ("hold to the end").
pub struct HoldToEnd {
    pub genome: std::sync::Arc<Genome>,
}

impl super::engine::Agent for HoldToEnd {
    fn name(&self) -> String {
        "hold-to-end".into()
    }
    fn act(&mut self, obs: &Obs, words: &WordIndex, _rng: &mut Rng) -> Action {
        let d = decide(&self.genome, obs, words);
        if obs.bag_total == 0 {
            return Action {
                plays: d.plays,
                bid: 0,
            };
        }
        Action {
            plays: Vec::new(),
            bid: d.bid.min(obs.balance),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upgraded_v1_genome_decides_identically() {
        use crate::sim::engine::{AuctionKind, AuctionRecord, Obs};
        use crate::sim::words::counts_from;
        // Build a v1-layout genome with random weights, upgrade it, and check
        // the new zero inputs leave every decision unchanged.
        let mut rng = Rng::new(3);
        let old_len = N_BASE + 13 + 1 + HIDDEN * 54 + HIDDEN + HIDDEN + 1;
        let mut v1 = Genome::prior().params[..N_BASE + 13].to_vec();
        v1.push(0.3);
        while v1.len() < old_len {
            v1.push((rng.normal() * 0.2) as f32);
        }
        let g = Genome {
            params: v1.clone(),
            sigma: 1.0,
        }
        .upgrade();
        assert!(g.len_ok());
        // Same MLP output on features whose new inputs are zero.
        let mut f = [0f32; N_FEATURES];
        for (k, v) in f.iter_mut().enumerate().take(54) {
            *v = ((k * 7 % 11) as f32 - 5.0) / 5.0;
        }
        let w1 = N_BASE + 13 + 1;
        let mut z = v1[w1 + HIDDEN * 54 + 2 * HIDDEN];
        for h in 0..HIDDEN {
            let mut s = v1[w1 + HIDDEN * 54 + h];
            for k in 0..54 {
                s += v1[w1 + h * 54 + k] * f[k];
            }
            z += v1[w1 + HIDDEN * 54 + HIDDEN + h] * s.tanh();
        }
        assert!((mlp(&g.params, &f) - z).abs() < 1e-5);
        // And a full decision runs on the upgraded genome.
        let words = WordIndex::bundled();
        let hist = vec![
            AuctionRecord {
                letter: 4,
                winner: Some(0),
                paid: 5,
                top_bid: 0
            };
            30
        ];
        let obs = Obs {
            me: 0,
            rack: counts_from("RETAINSX"),
            initial_rack: counts_from("RETAI"),
            balance: 50,
            score: 10,
            letter: 18,
            bag_total: 20,
            balances: &[50, 60],
            scores: &[10, 20],
            history: &hist,
            kind: AuctionKind::Vickrey,
        };
        let _ = decide(&g, &obs, &words);
    }

    #[test]
    fn layout_is_consistent() {
        let g = Genome::prior();
        assert_eq!(g.params.len(), GENOME_LEN);
        assert_eq!(super::super::words::LONG_LEN, 7);
    }
}
