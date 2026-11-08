//! Offline simulator, evaluation panel and genetic-algorithm trainer.
//!
//! ```sh
//! cargo run --release --bin train -- bench
//! cargo run --release --bin train -- panel --genome runs/x/best.json
//! cargo run --release --bin train -- panel --opp old-greedy
//! cargo run --release --bin train -- train --out runs/x --generations 300
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use clap::{Parser, Subcommand};

use scrabblebot_bot::sim::agents::OpponentModel;
use scrabblebot_bot::sim::ga::{
    Opp, Seat, State, TrainConfig, append_line, evaluate, headline, init_state, next_generation,
    run_panel, save_state,
};
use scrabblebot_bot::sim::policy::Genome;
use scrabblebot_bot::sim::words::WordIndex;

#[derive(Parser)]
struct Cli {
    /// Empirical opponent model from tools/analyze_maincloud.py.
    #[arg(long, default_value = "data/maincloud/opponent_model.json")]
    model: String,
    #[arg(long)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Time simulated matches.
    Bench {
        #[arg(long, default_value_t = 200)]
        games: usize,
    },
    /// Score a genome or fixed opponent on the frozen panel.
    Panel {
        #[arg(long)]
        genome: Option<PathBuf>,
        /// prior | old-greedy | spender | empirical | sim-aggr
        #[arg(long)]
        opp: Option<String>,
        #[arg(long, default_value_t = 600)]
        games: usize,
    },
    /// Print per-seat stats for a room, e.g. `--seats empirical,empirical,prior,sim-aggr`.
    Room {
        #[arg(long, value_delimiter = ',')]
        seats: Vec<String>,
        #[arg(long, default_value_t = 300)]
        games: usize,
        #[arg(long)]
        scale: Option<f64>,
    },
    /// Re-score candidate genomes on fresh seeds (held out from training's
    /// panel) and copy the best to --out.
    Select {
        #[arg(required = true)]
        genomes: Vec<PathBuf>,
        #[arg(long, default_value_t = 1500)]
        games: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Measure cash-starved holding: auctions where the genome cannot afford
    /// the market price yet holds a playable word.
    Starve {
        #[arg(long)]
        genome: PathBuf,
        #[arg(long, default_value_t = 400)]
        games: usize,
    },
    /// Run (or resume) the genetic algorithm.
    Train {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 500)]
        generations: usize,
        #[arg(long, default_value_t = 64)]
        pop: usize,
        #[arg(long, default_value_t = 24)]
        games_per: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Room sampling profile: "realmix" mirrors Maincloud's room mix.
        #[arg(long, default_value = "")]
        profile: String,
        /// Warm-start the population around this genome.
        #[arg(long)]
        init: Option<PathBuf>,
    },
}

fn seat_for(genome: &Option<PathBuf>, opp: &Option<String>) -> anyhow::Result<Seat> {
    if let Some(p) = genome {
        let g = Genome::load(p)?;
        return Ok(Seat::Genome(Arc::new(g), "genome".into()));
    }
    if let Some(path) = opp.as_deref().and_then(|o| o.strip_prefix("rl:")) {
        return Ok(Seat::Fixed(Opp::Rl(path.to_string())));
    }
    Ok(match opp.as_deref().unwrap_or("prior") {
        "prior" => Seat::Genome(Arc::new(Genome::prior()), "prior".into()),
        "old-greedy" => Seat::Fixed(Opp::OldGreedy),
        "spender" => Seat::Fixed(Opp::Spender),
        "empirical" => Seat::Fixed(Opp::Empirical(1.0)),
        "sim-aggr" => Seat::Fixed(Opp::Sim("aggr".into())),
        "sim-value" => Seat::Fixed(Opp::Sim("value".into())),
        "sim-cheap" => Seat::Fixed(Opp::Sim("cheap".into())),
        other => anyhow::bail!("unknown opp {other}"),
    })
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(t) = cli.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()?;
    }
    let words = WordIndex::bundled();
    let model = match OpponentModel::load(&cli.model) {
        Ok(m) => Some(Arc::new(m)),
        Err(e) => {
            eprintln!("warning: no opponent model ({e}); empirical seats fall back to spender");
            None
        }
    };

    match cli.cmd {
        Cmd::Bench { games } => {
            let t = Instant::now();
            let s = run_panel(
                Seat::Genome(Arc::new(Genome::prior()), "prior".into()),
                games / 3,
                0,
                &words,
                &model,
            );
            let dt = t.elapsed().as_secs_f64();
            println!(
                "{} games in {:.2}s = {:.1} games/s",
                games / 3 * 3,
                dt,
                (games / 3 * 3) as f64 / dt
            );
            for p in s {
                println!(
                    "{:<12} score {:>6.1} place {:.2} win {:.2}",
                    p.room, p.mean_score, p.mean_place, p.win_rate
                );
            }
        }
        Cmd::Panel { genome, opp, games } => {
            let seat = seat_for(&genome, &opp)?;
            for p in run_panel(seat, games, 0, &words, &model) {
                println!(
                    "{:<12} score {:>6.1} place {:.2} win {:.2}",
                    p.room, p.mean_score, p.mean_place, p.win_rate
                );
            }
        }
        Cmd::Select {
            genomes,
            games,
            seed,
            out,
        } => {
            let mut best: Option<(f64, PathBuf)> = None;
            for path in &genomes {
                let seat = if path.to_string_lossy().ends_with(".rl.json") {
                    Seat::Fixed(Opp::Rl(path.to_string_lossy().into_owned()))
                } else {
                    seat_for(&Some(path.clone()), &None)?
                };
                let ps = run_panel(seat, games, seed.wrapping_mul(0x1000_0001), &words, &model);
                // Headline: Maincloud-like room score, with place as tie context.
                let head = headline(&ps);
                println!(
                    "{:<40} {}",
                    path.display(),
                    ps.iter()
                        .map(|p| format!("{} {:.1}/{:.2}", p.room, p.mean_score, p.mean_place))
                        .collect::<Vec<_>>()
                        .join("  ")
                );
                if best.as_ref().is_none_or(|(b, _)| head > *b) {
                    best = Some((head, path.clone()));
                }
            }
            if let (Some((score, path)), Some(out)) = (best, out) {
                std::fs::copy(&path, &out)?;
                println!(
                    "selected {} ({score:.1}) -> {}",
                    path.display(),
                    out.display()
                );
            }
        }
        Cmd::Starve { genome, games } => {
            use rayon::prelude::*;
            use scrabblebot_bot::sim::engine::{Action, Agent, MatchConfig, Obs, play_match};
            use scrabblebot_bot::sim::policy::{Context, decide};
            use scrabblebot_bot::sim::rng::Rng;
            use scrabblebot_bot::sim::words::WordIndex as WI;
            struct Probe {
                g: Arc<Genome>,
                stats: [f64; 5], // auctions, starved, starved_holding, held_reward, flushed_reward
            }
            impl Agent for Probe {
                fn name(&self) -> String {
                    "probe".into()
                }
                fn act(&mut self, obs: &Obs, w: &WI, _r: &mut Rng) -> Action {
                    let d = decide(&self.g, obs, w);
                    let price = Context::from_obs(obs).recent_paid;
                    self.stats[0] += 1.0;
                    if obs.bag_total == 0 {
                        self.stats[4] += d.plays.iter().map(|&i| w.reward(i) as f64).sum::<f64>();
                    } else if (obs.balance as f64) < 0.5 * price {
                        self.stats[1] += 1.0;
                        if d.plays.is_empty()
                            && let Some(id) = w.best_word(&obs.rack, 2)
                            && w.reward(id) >= 10
                        {
                            self.stats[2] += 1.0;
                            self.stats[3] += w.reward(id) as f64;
                        }
                    }
                    Action {
                        plays: d.plays,
                        bid: d.bid,
                    }
                }
            }
            let g = Arc::new(Genome::load(&genome)?);
            let v2 = scrabblebot_bot::sim::ga::FROZEN_V1;
            let totals: Vec<[f64; 6]> = (0..games)
                .into_par_iter()
                .map(|i| {
                    let mut probe = Probe {
                        g: g.clone(),
                        stats: [0.0; 5],
                    };
                    let p: *mut Probe = &mut probe;
                    let mut agents: Vec<Box<dyn Agent>> = vec![Box::new(ProbeRef(p))];
                    for _ in 0..3 {
                        agents.push(Opp::Frozen(v2.into()).build(&model));
                    }
                    for _ in 0..2 {
                        agents.push(Opp::Empirical(0.85).build(&model));
                    }
                    let r = play_match(
                        &mut agents,
                        &words,
                        &MatchConfig::default(),
                        5000 + i as u64,
                    );
                    drop(agents);
                    let s = probe.stats;
                    [s[0], s[1], s[2], s[3], s[4], r.players[0].score as f64]
                })
                .collect();
            struct ProbeRef(*mut Probe);
            unsafe impl Send for ProbeRef {}
            impl Agent for ProbeRef {
                fn name(&self) -> String {
                    "probe".into()
                }
                fn act(&mut self, obs: &Obs, w: &WI, r: &mut Rng) -> Action {
                    unsafe { (*self.0).act(obs, w, r) }
                }
            }
            let n = games as f64;
            let sum = |k: usize| totals.iter().map(|t| t[k]).sum::<f64>();
            let matches_with_long_starve = totals.iter().filter(|t| t[2] >= 5.0).count();
            println!(
                "per match: score {:.1}, starved auctions {:.1}, starved while holding a word >=10 pts {:.1}",
                sum(5) / n,
                sum(1) / n,
                sum(2) / n
            );
            println!(
                "matches with >=5 starved-holding auctions: {:.0}%",
                100.0 * matches_with_long_starve as f64 / n
            );
            println!("final-auction flush reward per match: {:.1}", sum(4) / n);
        }
        Cmd::Room {
            seats,
            games,
            scale,
        } => {
            use rayon::prelude::*;
            use scrabblebot_bot::sim::engine::{MatchConfig, play_match};
            let specs: Vec<Seat> = seats
                .iter()
                .map(|s| match (s.as_str(), scale) {
                    ("empirical", Some(x)) => Ok(Seat::Fixed(Opp::Empirical(x))),
                    (s, _) if s.starts_with("hold:") => {
                        Ok(Seat::Fixed(Opp::HoldToEnd(s[5..].to_string())))
                    }
                    (s, _) if s.starts_with("rl:") => Ok(Seat::Fixed(Opp::Rl(s[3..].to_string()))),
                    (s, _) if s.starts_with("spec:") => {
                        Ok(Seat::Fixed(Opp::Spec(s[5..].to_string())))
                    }
                    (s, _) if s.contains('@') => {
                        let (p, m) = s.split_once('@').unwrap();
                        Ok(Seat::Fixed(Opp::Scaled(
                            p.to_string(),
                            m.parse().unwrap_or(1.0),
                        )))
                    }
                    (s, _) if s.ends_with(".json") => seat_for(&Some(PathBuf::from(s)), &None),
                    (s, _) => seat_for(&None, &Some(s.to_string())),
                })
                .collect::<anyhow::Result<_>>()?;
            let n = specs.len();
            let rows: Vec<Vec<[f64; 6]>> = (0..games)
                .into_par_iter()
                .map(|g| {
                    let mut agents: Vec<Box<dyn scrabblebot_bot::sim::engine::Agent>> = specs
                        .iter()
                        .map(|s| match s {
                            Seat::Genome(gn, l) => Box::new(scrabblebot_bot::sim::policy::Evolved {
                                genome: gn.clone(),
                                label: l.clone(),
                            })
                                as Box<dyn scrabblebot_bot::sim::engine::Agent>,
                            Seat::Fixed(o) => o.build(&model),
                        })
                        .collect();
                    let r = play_match(
                        &mut agents,
                        &words,
                        &MatchConfig::default(),
                        1000 + g as u64,
                    );
                    (0..n)
                        .map(|i| {
                            let p = &r.players[i];
                            let place = 1 + r.players.iter().filter(|q| q.score > p.score).count();
                            [
                                p.score as f64,
                                place as f64,
                                p.letters_won as f64,
                                p.spent as f64 / (p.letters_won.max(1)) as f64,
                                p.balance as f64,
                                p.words as f64,
                            ]
                        })
                        .collect()
                })
                .collect();
            println!(
                "{:<14}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}",
                "seat", "score", "place", "won", "paid/l", "endbal", "words"
            );
            for i in 0..n {
                let mut m = [0.0; 6];
                for r in &rows {
                    for k in 0..6 {
                        m[k] += r[i][k] / games as f64;
                    }
                }
                println!(
                    "{:<14}{:>8.1}{:>8.2}{:>8.1}{:>8.1}{:>8.1}{:>8.1}",
                    seats[i], m[0], m[1], m[2], m[3], m[4], m[5]
                );
            }
        }
        Cmd::Train {
            out,
            generations,
            pop,
            games_per,
            seed,
            profile,
            init,
        } => {
            std::fs::create_dir_all(out.join("champions"))?;
            let state_path = out.join("state.json");
            let mut state: State = if state_path.exists() {
                serde_json::from_str(&std::fs::read_to_string(&state_path)?)
                    .context("state.json")?
            } else {
                let init = match &init {
                    Some(p) => Some(Genome::load(p)?),
                    None => None,
                };
                init_state(
                    TrainConfig {
                        pop,
                        games_per,
                        seed,
                        seats_pop: if profile == "realmix" { 2 } else { 3 },
                        profile,
                        ..TrainConfig::default()
                    },
                    init,
                )
            };
            let log = out.join("log.csv");
            let panel_log = out.join("panel.csv");
            if !log.exists() {
                append_line(
                    &log,
                    "gen,best_fit,mean_fit,best_score,mean_score,sigma,secs",
                )?;
                append_line(&panel_log, "gen,room,mean_score,mean_place,win_rate")?;
            }
            let mut best_panel = f64::NEG_INFINITY;
            if let Ok(s) = std::fs::read_to_string(out.join("best_panel.txt")) {
                best_panel = s.trim().parse().unwrap_or(best_panel);
            }
            let end = state.generation + generations;
            while state.generation < end {
                let t = Instant::now();
                let (fit, score) = evaluate(&state, &words, &model);
                let g_idx = state.generation;
                let sigma =
                    state.pop.iter().map(|g| g.sigma as f64).sum::<f64>() / state.pop.len() as f64;
                let st = next_generation(&mut state, &fit, &score);
                let champ = state.hof.last().unwrap().clone();
                std::fs::write(
                    out.join("champions").join(format!("gen_{g_idx:04}.json")),
                    serde_json::to_vec(&champ)?,
                )?;
                append_line(
                    &log,
                    &format!(
                        "{g_idx},{:.4},{:.4},{:.1},{:.1},{:.3},{:.1}",
                        st.best_fit,
                        st.mean_fit,
                        st.best_score,
                        st.mean_score,
                        sigma,
                        t.elapsed().as_secs_f64()
                    ),
                )?;
                println!(
                    "gen {g_idx}: best fit {:.3} (score {:.1}) mean fit {:.3} (score {:.1}) sigma {:.2} [{:.1}s]",
                    st.best_fit,
                    st.best_score,
                    st.mean_fit,
                    st.mean_score,
                    sigma,
                    t.elapsed().as_secs_f64()
                );
                if g_idx.is_multiple_of(state.config.panel_every) {
                    let ps = run_panel(
                        Seat::Genome(Arc::new(champ.clone()), "champ".into()),
                        state.config.panel_games,
                        0,
                        &words,
                        &model,
                    );
                    for p in &ps {
                        append_line(
                            &panel_log,
                            &format!(
                                "{g_idx},{},{:.2},{:.3},{:.3}",
                                p.room, p.mean_score, p.mean_place, p.win_rate
                            ),
                        )?;
                    }
                    let headline = headline(&ps);
                    println!(
                        "  panel: {}",
                        ps.iter()
                            .map(|p| format!("{} {:.1}/{:.2}", p.room, p.mean_score, p.mean_place))
                            .collect::<Vec<_>>()
                            .join("  ")
                    );
                    if headline > best_panel {
                        best_panel = headline;
                        std::fs::write(out.join("best.json"), serde_json::to_vec_pretty(&champ)?)?;
                        std::fs::write(out.join("best_panel.txt"), format!("{best_panel}"))?;
                        println!("  new best panel: weighted mean place {:.3}", -best_panel);
                    }
                }
                save_state(&out, &state)?;
            }
        }
    }
    Ok(())
}
