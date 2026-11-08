//! Step-wise match engine for reinforcement learning.
//!
//! Identical rules and RNG consumption to [`super::engine::play_match`], but
//! one seat (the learner) is driven from outside: the match runs until the
//! learner's turn, exposes its observation, and resumes with its action.
//! `tests::matches_play_match_exactly` checks equivalence.

use super::engine::{
    AUCTION_RESERVE, Action, Agent, AuctionKind, AuctionRecord, MatchConfig, MatchResult, Obs,
    PlayerResult, STARTING_BALANCE,
};
use super::rng::Rng;
use super::words::{Counts, WordIndex, fits, remove_word};
use crate::core::letters::DEFAULT_BAG;

pub struct StepMatch {
    pub learner: usize,
    agents: Vec<Option<Box<dyn Agent>>>,
    kind: AuctionKind,
    rng: Rng,
    agent_rng: Rng,
    bag: [u32; 26],
    bag_total: u32,
    pub racks: Vec<Counts>,
    pub initial: Vec<Counts>,
    pub balances: Vec<i64>,
    pub scores: Vec<i64>,
    pub history: Vec<AuctionRecord>,
    stats: Vec<(u32, u32, i64)>,
    order: Vec<usize>,
    pos: usize,
    bids: Vec<(usize, i64)>,
    letter: Option<usize>,
    /// Letters each seat has won and played (public), for opponent features.
    pub won: Vec<u32>,
    pub played: Vec<u32>,
}

impl StepMatch {
    /// `agents[learner]` is ignored (pass any placeholder or None).
    pub fn new(
        mut agents: Vec<Option<Box<dyn Agent>>>,
        learner: usize,
        cfg: &MatchConfig,
        seed: u64,
    ) -> Self {
        let n = agents.len();
        agents[learner] = None;
        let mut bag = [0u32; 26];
        for (k, &c) in DEFAULT_BAG.iter().enumerate() {
            bag[k] = c as u32;
        }
        let mut m = Self {
            learner,
            agents,
            kind: cfg.kind,
            rng: Rng::new(seed),
            agent_rng: Rng::new(seed ^ 0xA5A5_5A5A_DEAD_BEEF),
            bag_total: bag.iter().sum(),
            bag,
            racks: vec![[0u8; 26]; n],
            initial: Vec::new(),
            balances: vec![STARTING_BALANCE; n],
            scores: vec![0; n],
            history: Vec::with_capacity(70),
            stats: vec![(0, 0, 0); n],
            order: (0..n).collect(),
            pos: 0,
            bids: Vec::with_capacity(n),
            letter: None,
            won: vec![0; n],
            played: vec![0; n],
        };
        for i in 0..n {
            for _ in 0..cfg.starting_rack {
                if let Some(k) = m.draw() {
                    m.racks[i][k] += 1;
                }
            }
        }
        m.initial = m.racks.clone();
        m.letter = m.draw();
        if m.letter.is_some() {
            m.open_auction();
        }
        m
    }

    fn draw(&mut self) -> Option<usize> {
        if self.bag_total == 0 {
            return None;
        }
        let mut idx = self.rng.below(self.bag_total as u64) as u32;
        for k in 0..26 {
            if idx < self.bag[k] {
                self.bag[k] -= 1;
                self.bag_total -= 1;
                return Some(k);
            }
            idx -= self.bag[k];
        }
        unreachable!()
    }

    fn open_auction(&mut self) {
        self.bids.clear();
        self.rng.shuffle(&mut self.order);
        self.pos = 0;
    }

    pub fn done(&self) -> bool {
        self.letter.is_none()
    }

    pub fn n(&self) -> usize {
        self.racks.len()
    }

    /// Observation for seat `i` at the current auction.
    pub fn obs(&self, i: usize) -> Obs<'_> {
        Obs {
            me: i,
            rack: self.racks[i],
            initial_rack: self.initial[i],
            balance: self.balances[i],
            score: self.scores[i],
            letter: self.letter.unwrap_or(0),
            bag_total: self.bag_total,
            balances: &self.balances,
            scores: &self.scores,
            history: &self.history,
            kind: self.kind,
        }
    }

    fn apply(&mut self, i: usize, action: Action, words: &WordIndex) {
        for id in action.plays {
            let wc = words.counts(id);
            if fits(&self.racks[i], wc) {
                remove_word(&mut self.racks[i], wc);
                let r = words.reward(id) as i64;
                self.balances[i] += r;
                self.scores[i] += r;
                self.stats[i].0 += 1;
                self.played[i] += words.word_len(id) as u32;
            }
        }
        if action.bid > 0 && action.bid <= self.balances[i] {
            self.bids.push((i, action.bid));
        }
    }

    /// Run non-learner seats until the learner must act or the match ends.
    pub fn advance(&mut self, words: &WordIndex) {
        while let Some(l) = self.letter {
            while self.pos < self.order.len() {
                let i = self.order[self.pos];
                if i == self.learner {
                    return;
                }
                // Move the agent and its RNG out so the observation can borrow
                // the match state immutably.
                let mut agent = self.agents[i]
                    .take()
                    .expect("non-learner seat has an agent");
                let mut rng = std::mem::replace(&mut self.agent_rng, Rng::new(0));
                let action = agent.act(&self.obs(i), words, &mut rng);
                self.agent_rng = rng;
                self.agents[i] = Some(agent);
                self.apply(i, action, words);
                self.pos += 1;
            }
            self.close(l, words);
        }
    }

    /// Apply the learner's action, then run until its next turn.
    pub fn step_learner(&mut self, action: Action, words: &WordIndex) {
        debug_assert_eq!(self.order.get(self.pos), Some(&self.learner));
        let i = self.learner;
        self.apply(i, action, words);
        self.pos += 1;
        self.advance(words);
    }

    fn close(&mut self, l: usize, words: &WordIndex) {
        self.rng.shuffle(&mut self.bids);
        self.bids.sort_by_key(|b| std::cmp::Reverse(b.1));
        let mut rec = AuctionRecord {
            letter: l as u8,
            winner: None,
            paid: 0,
            top_bid: 0,
        };
        if let Some(&(w, top)) = self.bids.first() {
            let paid = match self.kind {
                AuctionKind::FirstPrice => top,
                AuctionKind::Vickrey => self
                    .bids
                    .get(1)
                    .map(|b| b.1)
                    .unwrap_or(AUCTION_RESERVE.min(top)),
            };
            rec.winner = Some(w as u8);
            rec.top_bid = top;
            rec.paid = paid;
            if self.balances[w] >= paid {
                self.balances[w] -= paid;
                self.racks[w][l] += 1;
                self.stats[w].1 += 1;
                self.stats[w].2 += paid;
                self.won[w] += 1;
                let sim_play = self.agents[w]
                    .as_mut()
                    .and_then(|a| a.on_win(&self.racks[w], words));
                if let Some(id) = sim_play {
                    let wc = words.counts(id);
                    if fits(&self.racks[w], wc) {
                        remove_word(&mut self.racks[w], wc);
                        let r = words.reward(id) as i64;
                        self.balances[w] += r;
                        self.scores[w] += r;
                        self.stats[w].0 += 1;
                        self.played[w] += words.word_len(id) as u32;
                    }
                }
            }
        }
        self.history.push(rec);
        self.letter = self.draw();
        if self.letter.is_some() {
            self.open_auction();
        }
    }

    pub fn result(&self) -> MatchResult {
        MatchResult {
            players: (0..self.n())
                .map(|i| PlayerResult {
                    score: self.scores[i],
                    balance: self.balances[i],
                    words: self.stats[i].0,
                    letters_won: self.stats[i].1,
                    spent: self.stats[i].2,
                    rack_left: self.racks[i].iter().map(|&c| c as u32).sum(),
                })
                .collect(),
            auctions: self.history.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::agents::{SimBot, SimKind};
    use crate::sim::engine::play_match;
    use crate::sim::policy::{Evolved, Genome};
    use std::sync::Arc;

    fn table(learner_genome: &Arc<Genome>) -> Vec<Box<dyn Agent>> {
        vec![
            Box::new(SimBot(SimKind::Aggressive)),
            Box::new(Evolved {
                genome: learner_genome.clone(),
                label: "x".into(),
            }),
            Box::new(SimBot(SimKind::ValueBidder)),
            Box::new(Evolved {
                genome: Arc::new(Genome::prior()),
                label: "p".into(),
            }),
            Box::new(SimBot(SimKind::Cheapskate)),
            Box::new(SimBot(SimKind::Aggressive)),
        ]
    }

    #[test]
    fn matches_play_match_exactly() {
        let words = WordIndex::bundled();
        let g = Arc::new(Genome::prior());
        for seed in 0..6u64 {
            let mut agents = table(&g);
            let reference = play_match(&mut agents, &words, &MatchConfig::default(), seed);

            let agents: Vec<Option<Box<dyn Agent>>> = table(&g).into_iter().map(Some).collect();
            let mut m = StepMatch::new(agents, 1, &MatchConfig::default(), seed);
            let mut driver = Evolved {
                genome: g.clone(),
                label: "x".into(),
            };
            let mut dummy = Rng::new(0);
            m.advance(&words);
            while !m.done() {
                let action = {
                    let obs = m.obs(1);
                    driver.act(&obs, &words, &mut dummy)
                };
                m.step_learner(action, &words);
            }
            let got = m.result();
            for (a, b) in reference.players.iter().zip(&got.players) {
                assert_eq!(
                    (a.score, a.balance, a.letters_won),
                    (b.score, b.balance, b.letters_won),
                    "seed {seed}"
                );
            }
        }
    }
}
