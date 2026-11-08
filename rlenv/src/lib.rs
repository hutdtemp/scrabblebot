//! Python bindings: a batch of simulated matches with one learner seat each.
//!
//! Observations, actions and rewards cross the boundary as flat little-endian
//! byte buffers (numpy `frombuffer`), so no numpy crate is needed.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

use scrabblebot_bot::sim::agents::OpponentModel;
use scrabblebot_bot::sim::engine::{Action, Agent, MatchConfig};
use scrabblebot_bot::sim::ga::{EMPIRICAL_FULL, Opp};
use scrabblebot_bot::sim::policy::Genome;
use scrabblebot_bot::sim::rl::{N_BID_ACTIONS, N_OBS, N_PLAY_ACTIONS, decode, rl_obs, teacher_actions};
use scrabblebot_bot::sim::rng::Rng;
use scrabblebot_bot::sim::stepper::StepMatch;
use scrabblebot_bot::sim::words::WordIndex;

/// Reward shaping weight on the change in (our score - best opponent score).
const SHAPING: f64 = 0.005;

struct Slot {
    m: StepMatch,
    prev_margin: f64,
    prev_score: f64,
}

#[pyclass(unsendable)]
struct VecEnv {
    slots: Vec<Slot>,
    words: Arc<WordIndex>,
    model: Option<Arc<OpponentModel>>,
    teacher: Option<Arc<Genome>>,
    pool: Vec<Opp>,
    rng: Rng,
    episodes: u64,
    /// Share of matches that are full real-bot rooms (vs rooms padded with
    /// built-in sims). Recent Maincloud matches are ~100% full rooms.
    full_frac: f64,
    /// "score": reward = our points scored this step / 100 (sums to our
    /// final score / 100). "placement": end-of-match pairwise placement plus
    /// a small margin-vs-leader shaping term (what v3 was trained on).
    reward_mode: String,
}

fn margin(m: &StepMatch) -> f64 {
    let me = m.scores[m.learner];
    let best = (0..m.n())
        .filter(|&j| j != m.learner)
        .map(|j| m.scores[j])
        .max()
        .unwrap_or(0);
    (me - best) as f64
}

fn pairwise(m: &StepMatch) -> f64 {
    let me = m.scores[m.learner];
    let mut s = 0.0;
    for j in 0..m.n() {
        if j == m.learner {
            continue;
        }
        let o = m.scores[j];
        s += if me > o {
            1.0
        } else if me == o {
            0.5
        } else {
            0.0
        };
    }
    s / (m.n() - 1) as f64
}

impl VecEnv {
    fn new_match(&mut self) -> Slot {
        let seed = self.rng.next_u64();
        let mut r = Rng::new(seed ^ 0x51DE);
        let pick = |r: &mut Rng, pool: &[Opp]| pool[r.below(pool.len() as u64) as usize].clone();
        let sim = |r: &mut Rng| Opp::Sim(["aggr", "value", "cheap"][r.below(3) as usize].into());
        let u = r.f64();
        let mut opps: Vec<Opp> = Vec::new();
        if u < self.full_frac {
            // Full real room: 3-4 competent opponents, the rest empirical.
            let k = if r.f64() < 0.5 { 3 } else { 4 };
            for _ in 0..k {
                opps.push(pick(&mut r, &self.pool));
            }
            let s = EMPIRICAL_FULL * (0.9 + 0.2 * r.f64());
            while opps.len() < 5 {
                opps.push(Opp::Empirical(s));
            }
        } else if u < self.full_frac + (1.0 - self.full_frac) * 0.8 {
            for _ in 0..2 {
                opps.push(pick(&mut r, &self.pool));
            }
            for _ in 0..3 {
                opps.push(sim(&mut r));
            }
        } else {
            opps.push(pick(&mut r, &self.pool));
            for _ in 0..4 {
                opps.push(sim(&mut r));
            }
        }
        let learner = r.below(6) as usize;
        let mut agents: Vec<Option<Box<dyn Agent>>> = Vec::with_capacity(6);
        let mut it = opps.into_iter();
        for i in 0..6 {
            agents.push(if i == learner {
                None
            } else {
                Some(it.next().unwrap().build(&self.model))
            });
        }
        let mut m = StepMatch::new(agents, learner, &MatchConfig::default(), seed);
        advance(&mut m, &self.words);
        self.episodes += 1;
        Slot {
            m,
            prev_margin: 0.0,
            prev_score: 0.0,
        }
    }

    fn pack_obs(&mut self, py: Python<'_>, with_teacher: bool) -> (Py<PyBytes>, Py<PyBytes>) {
        let words = &self.words;
        let teacher = if with_teacher { &self.teacher } else { &None };
        let rows: Vec<([f32; N_OBS], (u8, u8))> = self
            .slots
            .par_iter_mut()
            .map(|s| {
                let obs = s.m.obs(s.m.learner);
                let x = rl_obs(&obs, words);
                let t = teacher
                    .as_ref()
                    .map(|g| {
                        let (p, b) = teacher_actions(g, &obs, words);
                        (p as u8, b as u8)
                    })
                    .unwrap_or((0, 0));
                (x, t)
            })
            .collect();
        let mut ob = Vec::with_capacity(rows.len() * N_OBS * 4);
        let mut tb = Vec::with_capacity(rows.len() * 2);
        for (x, (p, b)) in rows {
            for v in x {
                ob.extend_from_slice(&v.to_le_bytes());
            }
            tb.push(p);
            tb.push(b);
        }
        (PyBytes::new(py, &ob).into(), PyBytes::new(py, &tb).into())
    }
}

/// Run until the learner must act; the final auction (flush, bid 0) is
/// played automatically since it has no decision in it.
fn advance(m: &mut StepMatch, words: &WordIndex) {
    m.advance(words);
    while !m.done() {
        let obs = m.obs(m.learner);
        if obs.bag_total != 0 {
            return;
        }
        let d = decode(&obs, words, 0, 0);
        m.step_learner(Action { plays: d.plays, bid: 0 }, words);
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed, pool, model_path, teacher_path=None, full_frac=0.5, reward_mode="placement".to_string()))]
    fn py_new(
        n: usize,
        seed: u64,
        pool: Vec<String>,
        model_path: String,
        teacher_path: Option<String>,
        full_frac: f64,
        reward_mode: String,
    ) -> PyResult<Self> {
        if reward_mode != "score" && reward_mode != "placement" {
            return Err(pyo3::exceptions::PyValueError::new_err("reward_mode must be score|placement"));
        }
        let words = WordIndex::bundled();
        let model = OpponentModel::load(&model_path).ok().map(Arc::new);
        let teacher = match teacher_path {
            Some(p) => Some(Arc::new(
                Genome::load(std::path::Path::new(&p))
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            )),
            None => None,
        };
        let mut env = VecEnv {
            slots: Vec::new(),
            words,
            model,
            teacher,
            pool: Vec::new(),
            rng: Rng::new(seed),
            episodes: 0,
            full_frac,
            reward_mode,
        };
        env.set_pool(pool)?;
        for _ in 0..n {
            let s = env.new_match();
            env.slots.push(s);
        }
        Ok(env)
    }

    /// Competent opponents: genome JSON paths, or RL actors ending `.rl.json`.
    fn set_pool(&mut self, pool: Vec<String>) -> PyResult<()> {
        if pool.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("empty pool"));
        }
        self.pool = pool
            .into_iter()
            .map(|p| match p.split_once('@') {
                Some((path, mult)) => Opp::Scaled(path.to_string(), mult.parse().unwrap_or(1.0)),
                None if p.ends_with(".rl.json") => Opp::Rl(p),
                None => Opp::Frozen(p),
            })
            .collect();
        Ok(())
    }

    #[getter]
    fn n_obs(&self) -> usize {
        N_OBS
    }
    #[getter]
    fn n_play(&self) -> usize {
        N_PLAY_ACTIONS
    }
    #[getter]
    fn n_bid(&self) -> usize {
        N_BID_ACTIONS
    }
    #[getter]
    fn episodes(&self) -> u64 {
        self.episodes
    }

    /// (obs f32[n*N_OBS], teacher u8[n*2]) for the current learner turns.
    /// Teacher actions are computed only when requested (imitation phase).
    #[pyo3(signature = (with_teacher=false))]
    fn observe(&mut self, py: Python<'_>, with_teacher: bool) -> (Py<PyBytes>, Py<PyBytes>) {
        self.pack_obs(py, with_teacher)
    }

    /// Apply actions; returns (reward f32[n], done u8[n], final pairwise
    /// placement f32[n] or -1, final score f32[n]). Done slots restart.
    fn step(
        &mut self,
        py: Python<'_>,
        play: Vec<u8>,
        bid: Vec<u8>,
    ) -> PyResult<(Py<PyBytes>, Py<PyBytes>, Py<PyBytes>, Py<PyBytes>)> {
        let n = self.slots.len();
        if play.len() != n || bid.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err("action length"));
        }
        let words = self.words.clone();
        let score_mode = self.reward_mode == "score";
        let out: Vec<(f32, bool, f32, f32)> = self
            .slots
            .par_iter_mut()
            .enumerate()
            .map(|(i, s)| {
                let d = {
                    let obs = s.m.obs(s.m.learner);
                    decode(&obs, &words, play[i] as usize, bid[i] as usize)
                };
                s.m.step_learner(Action { plays: d.plays, bid: d.bid }, &words);
                advance(&mut s.m, &words);
                let mut r;
                if score_mode {
                    let sc = s.m.scores[s.m.learner] as f64;
                    r = (sc - s.prev_score) / 100.0;
                    s.prev_score = sc;
                } else {
                    let mg = margin(&s.m);
                    r = SHAPING * (mg - s.prev_margin);
                    s.prev_margin = mg;
                }
                if s.m.done() {
                    let p = pairwise(&s.m);
                    if !score_mode {
                        r += 2.0 * p - 1.0;
                    }
                    (r as f32, true, p as f32, s.m.scores[s.m.learner] as f32)
                } else {
                    (r as f32, false, -1.0, 0.0)
                }
            })
            .collect();
        for (i, o) in out.iter().enumerate() {
            if o.1 {
                self.slots[i] = self.new_match();
            }
        }
        let mut rb = Vec::with_capacity(n * 4);
        let mut db = Vec::with_capacity(n);
        let mut pb = Vec::with_capacity(n * 4);
        let mut sb = Vec::with_capacity(n * 4);
        for (r, d, p, sc) in out {
            rb.extend_from_slice(&r.to_le_bytes());
            db.push(d as u8);
            pb.extend_from_slice(&p.to_le_bytes());
            sb.extend_from_slice(&sc.to_le_bytes());
        }
        Ok((
            PyBytes::new(py, &rb).into(),
            PyBytes::new(py, &db).into(),
            PyBytes::new(py, &pb).into(),
            PyBytes::new(py, &sb).into(),
        ))
    }
}

#[pymodule]
fn scrabble_rlenv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    Ok(())
}
