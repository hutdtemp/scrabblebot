//! Genetic algorithm with mixed rooms.
//!
//! Every training match seats some current candidates, one or more past
//! champions (hall of fame), and fixed outside opponents, so the population
//! cannot drift into only beating its own lineage. Fitness is relative
//! (fraction of opponents out-scored), which removes any reward for tacit
//! collusion among clones. A frozen test panel, never used for selection,
//! tracks real progress.

use std::fs;
use std::io::Write;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::agents::{Empirical, OldGreedy, OpponentModel, SimBot, SimKind, Spender};
use super::engine::{Agent, MatchConfig, play_match};
use super::policy::{Evolved, Genome, crossover, mutate};
use super::rng::Rng;
use super::words::WordIndex;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Opp {
    Sim(String),
    OldGreedy,
    Spender,
    Empirical(f64),
    /// A fixed genome loaded from disk (e.g. the model currently deployed).
    Frozen(String),
    /// A genome's bidding with no word plays until the final auction.
    HoldToEnd(String),
    /// A trained RL actor (JSON weights).
    Rl(String),
    /// Any policy spec accepted by `policy::load_policy` (e.g. `spend10:x.rl.json`).
    Spec(String),
    /// A policy (genome or `.rl.json`) whose bids are multiplied, to mimic a
    /// specific real bot's spending level.
    Scaled(String, f64),
}

fn frozen_genome(path: &str) -> Arc<Genome> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<Genome>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut c = cache.lock().unwrap();
    c.entry(path.to_string())
        .or_insert_with(|| {
            Arc::new(
                Genome::load(std::path::Path::new(path))
                    .unwrap_or_else(|e| panic!("frozen genome {path}: {e}")),
            )
        })
        .clone()
}

pub fn rl_model(path: &str) -> Arc<super::rl::RlModel> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<String, Arc<super::rl::RlModel>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut c = cache.lock().unwrap();
    c.entry(path.to_string())
        .or_insert_with(|| {
            Arc::new(
                super::rl::RlModel::load(std::path::Path::new(path))
                    .unwrap_or_else(|e| panic!("rl model {path}: {e}")),
            )
        })
        .clone()
}

/// Competent policy with bids scaled to a real bot's spending level.
pub struct ScaledAgent {
    pub policy: Arc<dyn super::policy::Policy>,
    pub mult: f64,
}

impl Agent for ScaledAgent {
    fn name(&self) -> String {
        format!("scaled-{:.2}", self.mult)
    }
    fn act(
        &mut self,
        obs: &super::engine::Obs,
        words: &WordIndex,
        _rng: &mut Rng,
    ) -> super::engine::Action {
        let d = self.policy.decide(obs, words);
        let bid = ((d.bid as f64) * self.mult).round() as i64;
        super::engine::Action {
            plays: d.plays,
            bid: bid.clamp(0, obs.balance),
        }
    }
}

/// The deployed model, used as a competent fixed opponent in realistic rooms.
pub const FROZEN_V1: &str = "genomes/panel_v2.json";
/// Empirical-opponent bid scale that matches real full-room clearing prices.
pub const EMPIRICAL_FULL: f64 = 0.85;

impl Opp {
    pub fn build(&self, model: &Option<Arc<OpponentModel>>) -> Box<dyn Agent> {
        match self {
            Opp::Sim(k) => Box::new(SimBot(match k.as_str() {
                "cheap" => SimKind::Cheapskate,
                "value" => SimKind::ValueBidder,
                _ => SimKind::Aggressive,
            })),
            Opp::OldGreedy => Box::new(OldGreedy),
            Opp::Spender => Box::new(Spender),
            Opp::Spec(spec) => Box::new(ScaledAgent {
                policy: super::policy::load_policy(std::path::Path::new(spec))
                    .unwrap_or_else(|e| panic!("policy {spec}: {e}")),
                mult: 1.0,
            }),
            Opp::Scaled(path, mult) => Box::new(ScaledAgent {
                policy: if path.ends_with(".rl.json") {
                    rl_model(path) as Arc<dyn super::policy::Policy>
                } else {
                    frozen_genome(path) as Arc<dyn super::policy::Policy>
                },
                mult: *mult,
            }),
            Opp::Rl(path) => Box::new(super::rl::RlAgent {
                model: rl_model(path),
            }),
            Opp::HoldToEnd(path) => Box::new(super::policy::HoldToEnd {
                genome: frozen_genome(path),
            }),
            Opp::Frozen(path) => Box::new(Evolved {
                genome: frozen_genome(path),
                label: "frozen".into(),
            }),
            Opp::Empirical(s) => match model {
                Some(m) => Box::new(Empirical {
                    model: m.clone(),
                    scale: *s,
                }),
                None => Box::new(Spender),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub pop: usize,
    pub games_per: usize,
    pub seats_pop: usize,
    pub seats_hof: usize,
    pub seats_ext: usize,
    pub elite: usize,
    pub tournament: usize,
    pub rate_head: f64,
    pub rate_mlp: f64,
    pub crossover: f64,
    pub panel_every: usize,
    pub panel_games: usize,
    pub external: Vec<Opp>,
    pub seed: u64,
    /// "" (legacy fixed seats) or "realmix" (Maincloud room-mix sampling).
    #[serde(default)]
    pub profile: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            pop: 64,
            games_per: 24,
            seats_pop: 3,
            seats_hof: 1,
            seats_ext: 2,
            elite: 4,
            tournament: 3,
            rate_head: 0.3,
            rate_mlp: 0.03,
            crossover: 0.5,
            panel_every: 5,
            panel_games: 300,
            external: vec![
                Opp::Empirical(0.75),
                Opp::Empirical(0.9),
                Opp::Empirical(1.0),
                Opp::Sim("aggr".into()),
                Opp::Sim("value".into()),
                Opp::Sim("cheap".into()),
                Opp::OldGreedy,
                Opp::Spender,
            ],
            seed: 1,
            profile: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub generation: usize,
    pub pop: Vec<Genome>,
    pub hof: Vec<Genome>,
    pub config: TrainConfig,
}

/// A seat in a room: either a genome or a fixed opponent.
#[derive(Clone)]
pub enum Seat {
    Genome(Arc<Genome>, String),
    Fixed(Opp),
}

pub struct SeatResult {
    pub score: i64,
    pub place: usize,
    pub pairwise: f64,
}

pub fn run_room(
    seats: &[Seat],
    words: &WordIndex,
    model: &Option<Arc<OpponentModel>>,
    seed: u64,
) -> Vec<SeatResult> {
    let mut agents: Vec<Box<dyn Agent>> = seats
        .iter()
        .map(|s| match s {
            Seat::Genome(g, label) => Box::new(Evolved {
                genome: g.clone(),
                label: label.clone(),
            }) as Box<dyn Agent>,
            Seat::Fixed(o) => o.build(model),
        })
        .collect();
    let res = play_match(&mut agents, words, &MatchConfig::default(), seed);
    (0..seats.len())
        .map(|i| SeatResult {
            score: res.players[i].score,
            place: 1 + res
                .players
                .iter()
                .filter(|p| p.score > res.players[i].score)
                .count(),
            pairwise: res.pairwise(i),
        })
        .collect()
}

fn mix(a: u64, b: u64) -> u64 {
    let mut r = Rng::new(a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b);
    r.next_u64()
}

pub struct GenStats {
    pub best_fit: f64,
    pub mean_fit: f64,
    pub best_score: f64,
    pub mean_score: f64,
    pub champion: usize,
}

/// Evaluate the population for one generation.
pub fn evaluate(
    state: &State,
    words: &WordIndex,
    model: &Option<Arc<OpponentModel>>,
) -> (Vec<f64>, Vec<f64>) {
    let cfg = &state.config;
    let mut rng = Rng::new(mix(cfg.seed, state.generation as u64));
    let mut slots: Vec<usize> = (0..state.pop.len())
        .flat_map(|i| std::iter::repeat_n(i, cfg.games_per))
        .collect();
    rng.shuffle(&mut slots);
    let pop: Vec<Arc<Genome>> = state.pop.iter().cloned().map(Arc::new).collect();
    let hof: Vec<Arc<Genome>> = state.hof.iter().cloned().map(Arc::new).collect();
    let prior = Arc::new(Genome::prior());
    let rooms: Vec<(Vec<usize>, Vec<Seat>, u64)> = slots
        .chunks(cfg.seats_pop)
        .enumerate()
        .map(|(gi, chunk)| {
            let mut seats: Vec<Seat> = chunk
                .iter()
                .map(|&i| Seat::Genome(pop[i].clone(), format!("p{i}")))
                .collect();
            if cfg.profile == "realmix" {
                // Mirror the recent Maincloud mix: ~50% full real rooms,
                // ~40% three real bots + sims, ~10% two real bots + sims.
                let pick_hof = |rng: &mut Rng| -> Seat {
                    if hof.is_empty() {
                        Seat::Fixed(Opp::Frozen(FROZEN_V1.into()))
                    } else {
                        let n = hof.len();
                        let idx = if rng.f64() < 0.5 {
                            n - 1 - rng.below(n.min(10) as u64) as usize
                        } else {
                            rng.below(n as u64) as usize
                        };
                        Seat::Genome(hof[idx].clone(), "hof".into())
                    }
                };
                let sim = |rng: &mut Rng| -> Seat {
                    let k = ["aggr", "value", "cheap"][rng.below(3) as usize];
                    Seat::Fixed(Opp::Sim(k.into()))
                };
                let u = rng.f64();
                if u < 0.5 {
                    seats.push(pick_hof(&mut rng));
                    seats.push(if rng.f64() < 0.5 {
                        Seat::Fixed(Opp::Frozen(FROZEN_V1.into()))
                    } else {
                        pick_hof(&mut rng)
                    });
                    let s = EMPIRICAL_FULL * (0.9 + 0.2 * rng.f64());
                    seats.push(Seat::Fixed(Opp::Empirical(s)));
                    seats.push(Seat::Fixed(Opp::Empirical(s)));
                } else if u < 0.9 {
                    seats.push(pick_hof(&mut rng));
                    for _ in 0..3 {
                        seats.push(sim(&mut rng));
                    }
                } else {
                    for _ in 0..4 {
                        seats.push(sim(&mut rng));
                    }
                }
                return (
                    chunk.to_vec(),
                    seats,
                    mix(
                        cfg.seed ^ 0x5EED,
                        ((state.generation as u64) << 20) | gi as u64,
                    ),
                );
            }
            for _ in 0..cfg.seats_hof {
                let g = if hof.is_empty() {
                    prior.clone()
                } else {
                    // Favor recent champions slightly but keep all eras alive.
                    let n = hof.len();
                    let u = rng.f64();
                    let idx = if u < 0.5 {
                        n - 1 - rng.below(n.min(10) as u64) as usize
                    } else {
                        rng.below(n as u64) as usize
                    };
                    hof[idx].clone()
                };
                seats.push(Seat::Genome(g, "hof".into()));
            }
            for _ in 0..cfg.seats_ext {
                let o = cfg.external[rng.below(cfg.external.len() as u64) as usize].clone();
                seats.push(Seat::Fixed(o));
            }
            (
                chunk.to_vec(),
                seats,
                mix(
                    cfg.seed ^ 0x5EED,
                    (state.generation as u64) << 20 | gi as u64,
                ),
            )
        })
        .collect();

    let results: Vec<(Vec<usize>, Vec<SeatResult>)> = rooms
        .par_iter()
        .map(|(members, seats, seed)| (members.clone(), run_room(seats, words, model, *seed)))
        .collect();

    let n = state.pop.len();
    let mut fit = vec![0.0; n];
    let mut score = vec![0.0; n];
    let mut count = vec![0.0f64; n];
    for (members, res) in results {
        for (k, &i) in members.iter().enumerate() {
            fit[i] += res[k].pairwise;
            score[i] += res[k].score as f64;
            count[i] += 1.0;
        }
    }
    for i in 0..n {
        fit[i] /= count[i].max(1.0);
        score[i] /= count[i].max(1.0);
    }
    (fit, score)
}

pub fn next_generation(state: &mut State, fit: &[f64], score: &[f64]) -> GenStats {
    let cfg = state.config.clone();
    let n = state.pop.len();
    let mut order: Vec<usize> = (0..n).collect();
    let key = |i: usize| fit[i] + 0.0005 * score[i];
    order.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap());
    let champion = order[0];
    let stats = GenStats {
        best_fit: fit[champion],
        mean_fit: fit.iter().sum::<f64>() / n as f64,
        best_score: score[champion],
        mean_score: score.iter().sum::<f64>() / n as f64,
        champion,
    };
    state.hof.push(state.pop[champion].clone());

    let mut rng = Rng::new(mix(cfg.seed ^ 0xB4EED, state.generation as u64));
    let mut next: Vec<Genome> = order[..cfg.elite.min(n)]
        .iter()
        .map(|&i| state.pop[i].clone())
        .collect();
    let tour = |rng: &mut Rng| -> usize {
        let mut best = rng.below(n as u64) as usize;
        for _ in 1..cfg.tournament {
            let c = rng.below(n as u64) as usize;
            if key(c) > key(best) {
                best = c;
            }
        }
        best
    };
    while next.len() < n {
        let a = tour(&mut rng);
        let mut child = if rng.f64() < cfg.crossover {
            let b = tour(&mut rng);
            crossover(&state.pop[a], &state.pop[b], &mut rng)
        } else {
            state.pop[a].clone()
        };
        mutate_split(&mut child, &mut rng, cfg.rate_head, cfg.rate_mlp);
        next.push(child);
    }
    state.pop = next;
    state.generation += 1;
    stats
}

fn mutate_split(g: &mut Genome, rng: &mut Rng, rate_head: f64, rate_mlp: f64) {
    // Head genes (base bid + play head) mutate often; MLP weights sparsely.
    let head = super::policy::N_BASE + super::policy::N_PLAY + 1;
    let mut head_g = Genome {
        params: g.params[..head].to_vec(),
        sigma: g.sigma,
    };
    mutate(&mut head_g, rng, rate_head);
    g.params[..head].copy_from_slice(&head_g.params);
    let sigma = head_g.sigma;
    for i in head..g.params.len() {
        if rng.f64() < rate_mlp {
            g.params[i] += sigma * Genome::gene_scale(i) * rng.normal() as f32;
        }
    }
    g.sigma = sigma;
}

/// Frozen evaluation panel: fixed rooms and fixed seeds, never used for
/// selection.
pub fn panel_rooms() -> Vec<(&'static str, Vec<Opp>)> {
    use Opp::*;
    let f = || Frozen(FROZEN_V1.into());
    let e = || Empirical(EMPIRICAL_FULL);
    vec![
        // Six real bots, as in ~half of recent Maincloud matches.
        ("full", vec![f(), f(), f(), e(), e()]),
        // Three real bots plus three built-in sims (~40%).
        (
            "three-real",
            vec![
                f(),
                f(),
                Sim("aggr".into()),
                Sim("value".into()),
                Sim("cheap".into()),
            ],
        ),
        // The original generation-0 panel room, kept for continuity.
        (
            "legacy",
            vec![
                Empirical(0.85),
                Empirical(0.85),
                Empirical(1.0),
                Sim("aggr".into()),
                Sim("value".into()),
            ],
        ),
        (
            "sims",
            vec![
                Sim("aggr".into()),
                Sim("aggr".into()),
                Sim("value".into()),
                Sim("value".into()),
                Sim("cheap".into()),
            ],
        ),
    ]
}

/// Single number used to pick models: mean place in the two realistic rooms
/// weighted by how often they occur (lower is better), returned negated so
/// that larger is better.
pub fn headline(ps: &[PanelScore]) -> f64 {
    let get = |name: &str| ps.iter().find(|p| p.room == name).map(|p| p.mean_place);
    match (get("full"), get("three-real")) {
        (Some(a), Some(b)) => -(0.55 * a + 0.45 * b),
        _ => ps.first().map(|p| -p.mean_place).unwrap_or(0.0),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct PanelScore {
    pub room: String,
    pub mean_score: f64,
    pub mean_place: f64,
    pub win_rate: f64,
}

pub fn run_panel(
    me: Seat,
    games: usize,
    seed_base: u64,
    words: &WordIndex,
    model: &Option<Arc<OpponentModel>>,
) -> Vec<PanelScore> {
    panel_rooms()
        .into_iter()
        .map(|(name, opps)| {
            let res: Vec<SeatResult> = (0..games)
                .into_par_iter()
                .map(|g| {
                    let mut seats = vec![me.clone()];
                    seats.extend(opps.iter().cloned().map(Seat::Fixed));
                    // Rotate our seat so seat order can't bias the panel.
                    let rot = g % seats.len();
                    seats.rotate_left(rot);
                    let my = (seats.len() - rot) % seats.len();
                    let r = run_room(&seats, words, model, mix(0xFA4E1 ^ seed_base, g as u64));
                    r.into_iter().nth(my).unwrap()
                })
                .collect();
            let n = res.len() as f64;
            PanelScore {
                room: name.into(),
                mean_score: res.iter().map(|r| r.score as f64).sum::<f64>() / n,
                mean_place: res.iter().map(|r| r.place as f64).sum::<f64>() / n,
                win_rate: res.iter().filter(|r| r.place == 1).count() as f64 / n,
            }
        })
        .collect()
}

pub fn init_state(config: TrainConfig, init: Option<Genome>) -> State {
    let mut rng = Rng::new(config.seed);
    let prior = init.unwrap_or_else(Genome::prior);
    let pop = (0..config.pop)
        .map(|i| {
            let mut g = prior.clone();
            if i > 0 {
                mutate_split(&mut g, &mut rng, 0.5, 0.2);
            }
            g
        })
        .collect();
    State {
        generation: 0,
        pop,
        hof: Vec::new(),
        config,
    }
}

pub fn save_state(dir: &std::path::Path, state: &State) -> anyhow::Result<()> {
    let tmp = dir.join("state.json.tmp");
    fs::write(&tmp, serde_json::to_vec(state)?)?;
    fs::rename(tmp, dir.join("state.json"))?;
    Ok(())
}

pub fn append_line(path: &std::path::Path, line: &str) -> anyhow::Result<()> {
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}
