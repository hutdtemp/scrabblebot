//! Headless match engine mirroring the pinned upstream module
//! (`spacetimedb/src/game.ts`):
//!
//! - 98-tile bag, no blanks; each participant is dealt `starting_rack` tiles.
//! - One letter is drawn and auctioned per round. Simulated bots bid when the
//!   auction opens. Bids above the bidder's balance are rejected.
//! - Close: shuffle, stable sort descending; Vickrey winner pays the
//!   runner-up's bid (or `min(reserve, top)` if alone). The letter is awarded
//!   only if the winner can still pay. An unbid letter is lost.
//! - A simulated winner immediately plays its longest word of >= 3 letters.
//! - The next letter is drawn; if the bag is empty the match ends at once, so
//!   the final auction's letter can never be played.
//!
//! Real bots may play words at any time. Here every agent gets one decision
//! point per auction, at open, where it may play any number of words and then
//! bid with its updated balance.

use super::rng::Rng;
use super::words::{Counts, WordIndex, fits, remove_word};
use crate::core::letters::{DEFAULT_BAG, letter_value};

pub const STARTING_BALANCE: i64 = 100;
pub const STARTING_RACK: usize = 5;
pub const AUCTION_RESERVE: i64 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuctionKind {
    Vickrey,
    FirstPrice,
}

#[derive(Clone, Copy, Debug)]
pub struct AuctionRecord {
    pub letter: u8,
    pub winner: Option<u8>,
    pub paid: i64,
    /// Highest bid. Hidden from bots during a match; kept for analysis.
    pub top_bid: i64,
}

/// Everything a bot can legitimately observe at an auction's open.
pub struct Obs<'a> {
    pub me: usize,
    pub rack: Counts,
    pub initial_rack: Counts,
    pub balance: i64,
    pub score: i64,
    pub letter: usize,
    /// Tiles left in the bag *after* drawing the current letter. Zero means
    /// this is the final auction.
    pub bag_total: u32,
    pub balances: &'a [i64],
    pub scores: &'a [i64],
    pub history: &'a [AuctionRecord],
    pub kind: AuctionKind,
}

impl Obs<'_> {
    /// Auctions left including the current one.
    pub fn auctions_left(&self) -> u32 {
        self.bag_total + 1
    }
    pub fn n_players(&self) -> usize {
        self.balances.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Action {
    pub plays: Vec<u32>,
    pub bid: i64,
}

pub trait Agent: Send {
    fn name(&self) -> String;
    /// Decide words to play (applied first, in order) and a bid.
    fn act(&mut self, obs: &Obs, words: &WordIndex, rng: &mut Rng) -> Action;
    /// Called after this agent wins a letter. Upstream simulated bots play a
    /// word here; real bots return None and act at the next open.
    fn on_win(&mut self, _rack: &Counts, _words: &WordIndex) -> Option<u32> {
        None
    }
}

#[derive(Clone, Debug)]
pub struct PlayerResult {
    pub score: i64,
    pub balance: i64,
    pub words: u32,
    pub letters_won: u32,
    pub spent: i64,
    pub rack_left: u32,
}

#[derive(Clone, Debug)]
pub struct MatchResult {
    pub players: Vec<PlayerResult>,
    pub auctions: Vec<AuctionRecord>,
}

impl MatchResult {
    /// Fraction of opponents this seat beat (ties count half).
    pub fn pairwise(&self, seat: usize) -> f64 {
        let me = self.players[seat].score;
        let n = self.players.len();
        let mut s = 0.0;
        for (j, p) in self.players.iter().enumerate() {
            if j == seat {
                continue;
            }
            if me > p.score {
                s += 1.0;
            } else if me == p.score {
                s += 0.5;
            }
        }
        s / (n - 1) as f64
    }
}

pub struct MatchConfig {
    pub kind: AuctionKind,
    pub starting_rack: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            kind: AuctionKind::Vickrey,
            starting_rack: STARTING_RACK,
        }
    }
}

struct Bag {
    counts: [u32; 26],
    total: u32,
}

impl Bag {
    fn full() -> Self {
        let mut counts = [0u32; 26];
        for (k, &n) in DEFAULT_BAG.iter().enumerate() {
            counts[k] = n as u32;
        }
        Self {
            counts,
            total: counts.iter().sum(),
        }
    }
    fn draw(&mut self, rng: &mut Rng) -> Option<usize> {
        if self.total == 0 {
            return None;
        }
        let mut idx = rng.below(self.total as u64) as u32;
        for k in 0..26 {
            if idx < self.counts[k] {
                self.counts[k] -= 1;
                self.total -= 1;
                return Some(k);
            }
            idx -= self.counts[k];
        }
        unreachable!()
    }
}

pub fn play_match(
    agents: &mut [Box<dyn Agent>],
    words: &WordIndex,
    cfg: &MatchConfig,
    seed: u64,
) -> MatchResult {
    let n = agents.len();
    let mut rng = Rng::new(seed);
    // Separate stream for agent-internal randomness so a seat's choices do
    // not perturb the bag sequence (common random numbers across variants).
    let mut agent_rng = Rng::new(seed ^ 0xA5A5_5A5A_DEAD_BEEF);
    let mut bag = Bag::full();
    let mut racks = vec![[0u8; 26]; n];
    let mut balances = vec![STARTING_BALANCE; n];
    let mut scores = vec![0i64; n];
    let mut stats = vec![(0u32, 0u32, 0i64); n]; // words, letters won, spent
    for rack in racks.iter_mut() {
        for _ in 0..cfg.starting_rack {
            if let Some(k) = bag.draw(&mut rng) {
                rack[k] += 1;
            }
        }
    }
    let initial = racks.clone();
    let mut history: Vec<AuctionRecord> = Vec::with_capacity(70);
    let mut order: Vec<usize> = (0..n).collect();

    let mut letter = bag.draw(&mut rng);
    while let Some(l) = letter {
        // Open: every agent may play words, then bid.
        let mut bids: Vec<(usize, i64)> = Vec::with_capacity(n);
        rng.shuffle(&mut order);
        for &i in &order {
            let action = {
                let obs = Obs {
                    me: i,
                    rack: racks[i],
                    initial_rack: initial[i],
                    balance: balances[i],
                    score: scores[i],
                    letter: l,
                    bag_total: bag.total,
                    balances: &balances,
                    scores: &scores,
                    history: &history,
                    kind: cfg.kind,
                };
                agents[i].act(&obs, words, &mut agent_rng)
            };
            for id in action.plays {
                let wc = words.counts(id);
                if fits(&racks[i], wc) {
                    remove_word(&mut racks[i], wc);
                    let r = words.reward(id) as i64;
                    balances[i] += r;
                    scores[i] += r;
                    stats[i].0 += 1;
                }
            }
            if action.bid > 0 && action.bid <= balances[i] {
                bids.push((i, action.bid));
            }
        }
        // Close.
        rng.shuffle(&mut bids);
        bids.sort_by_key(|b| std::cmp::Reverse(b.1)); // stable
        let mut rec = AuctionRecord {
            letter: l as u8,
            winner: None,
            paid: 0,
            top_bid: 0,
        };
        if let Some(&(w, top)) = bids.first() {
            let paid = match cfg.kind {
                AuctionKind::FirstPrice => top,
                AuctionKind::Vickrey => {
                    bids.get(1).map(|b| b.1).unwrap_or(AUCTION_RESERVE.min(top))
                }
            };
            rec.winner = Some(w as u8);
            rec.top_bid = top;
            rec.paid = paid;
            if balances[w] >= paid {
                balances[w] -= paid;
                racks[w][l] += 1;
                stats[w].1 += 1;
                stats[w].2 += paid;
                if let Some(id) = agents[w].on_win(&racks[w], words) {
                    let wc = words.counts(id);
                    if fits(&racks[w], wc) {
                        remove_word(&mut racks[w], wc);
                        let r = words.reward(id) as i64;
                        balances[w] += r;
                        scores[w] += r;
                        stats[w].0 += 1;
                    }
                }
            }
        }
        history.push(rec);
        letter = bag.draw(&mut rng);
    }

    MatchResult {
        players: (0..n)
            .map(|i| PlayerResult {
                score: scores[i],
                balance: balances[i],
                words: stats[i].0,
                letters_won: stats[i].1,
                spent: stats[i].2,
                rack_left: racks[i].iter().map(|&c| c as u32).sum(),
            })
            .collect(),
        auctions: history,
    }
}

/// Face value of a letter index.
pub fn face(letter: usize) -> i64 {
    letter_value(letter) as i64
}
