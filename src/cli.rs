//! Command-line interface.

use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};

use crate::core::search::{BeamScore, SearchConfig};
use crate::core::strategy::{PlaybackPolicy, StrategyConfig, StrategyKind};
use crate::live::{LiveConfig, TelemetrySink};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StrategyArg {
    /// No lookahead: face-value bids and immediate best-word playback.
    Greedy,
    /// Adaptive-depth beam-search valuation and target-aware playback.
    Search,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PlaybackArg {
    /// Always play the best playable word immediately.
    Greedy,
    /// Hold a playable word only for a clearly better near-future target.
    Target,
}

#[derive(Parser, Debug)]
#[command(
    name = "scrabblebot-bot",
    version,
    about = "Rust ScrabbleBot auction bot (local-only by default)",
    long_about = "Rust bot for Clockwork Labs' ScrabbleBot.\n\n\
        Defaults point at a local SpacetimeDB server: http://127.0.0.1:3010 and \
        database scrabblebot-local. Non-loopback URIs are refused unless \
        --allow-remote is passed. Tokens are read from --token-env or \
        --token-file and are never logged.\n\n\
        The default strategy values each auctioned letter with an adaptive-depth, \
        probability-weighted beam search over future bag draws, bids a shaded \
        fraction of the marginal value, and plays words greedily. The greedy \
        strategy is a no-lookahead baseline for paired comparison.",
    after_help = "Examples:\n  \
        # One bounded local match with the coordinator's token file\n  \
        scrabblebot-bot --token-file /path/to/beam-local.token --matches 1 \\\n    \
          --telemetry telemetry.jsonl\n\n  \
        # No-lookahead baseline\n  \
        scrabblebot-bot --strategy greedy --matches 1 \\\n    \
          --token-file /path/to/beam-local.token --telemetry greedy.jsonl\n\n  \
        # Fresh local-only team/bot bootstrap (creates a team, mints and redeems a nonce)\n  \
        scrabblebot-bot --bootstrap-team my-local-team --bootstrap-bot my-local-bot \\\n    \
          --matches 1 --telemetry bootstrap.jsonl\n\n\
        Telemetry is JSONL: run_start, auction, word, match_summary, reducer_error, run_summary."
)]
struct Cli {
    /// SpacetimeDB host URI (default is the local Docker server).
    #[arg(long, default_value = "http://127.0.0.1:3010")]
    uri: String,

    /// SpacetimeDB database name.
    #[arg(long, default_value = "scrabblebot-local")]
    database: String,

    /// Bot name used in logs and as the default token-file stem.
    #[arg(long, default_value = "bot")]
    bot_name: String,

    /// File containing a SpacetimeDB token. Defaults to .token-<bot-name>.
    #[arg(long)]
    token_file: Option<PathBuf>,

    /// Do not read or save a token file (an anonymous identity will not survive reconnect).
    #[arg(long)]
    no_token_file: bool,

    /// Environment variable containing a SpacetimeDB token (fallback if the token file is absent).
    #[arg(long, default_value = "SPACETIMEDB_TOKEN")]
    token_env: String,

    /// Strategy: search (default) or greedy.
    #[arg(long, value_enum, default_value_t = StrategyArg::Search)]
    strategy: StrategyArg,

    /// Playback policy. Defaults to target for search and greedy for the
    /// no-lookahead baseline.
    #[arg(long, value_enum)]
    playback: Option<PlaybackArg>,

    /// Stop after this many completed matches; 0 means unlimited.
    #[arg(long, default_value_t = 1)]
    matches: usize,

    /// JSONL telemetry path, "-" for stdout, or "none".
    #[arg(long, default_value = "telemetry.jsonl")]
    telemetry: String,

    /// Maximum search wall time per auction, in milliseconds. The default
    /// targets about 0.8 s of the 1 s auction window.
    #[arg(long, default_value_t = 800)]
    search_max_time_ms: u64,

    /// Beam width for depths above --exact-depth.
    #[arg(long, default_value_t = 128)]
    beam_width: usize,

    /// Safety cap on future bag draws considered. The deadline is the primary
    /// limit; the default lets the search use the full ~0.8 s budget.
    #[arg(long, default_value_t = 12)]
    max_depth: u8,

    /// Depths up to this value are evaluated exhaustively.
    #[arg(long, default_value_t = 2)]
    exact_depth: u8,

    /// Keep this much time before auction close for the reducer submission and
    /// acknowledgment. Local reducer round-trip is ~2 ms; 150 ms is a safe
    /// margin after an ~800 ms search.
    #[arg(long, default_value_t = 150)]
    submit_margin_ms: u64,

    /// Stop if no match progress happens for this many seconds; 0 disables.
    #[arg(long, default_value_t = 300)]
    match_timeout_secs: u64,

    /// Redeem a credential nonce on connect (local provisioning path).
    #[arg(long)]
    claim_nonce: Option<String>,

    /// Local-only bootstrap: create this team if the identity has no credential.
    #[arg(long)]
    bootstrap_team: Option<String>,

    /// Local-only bootstrap: create this bot with --bootstrap-team.
    #[arg(long)]
    bootstrap_bot: Option<String>,

    /// Allow a non-loopback URI. Off by default; this bot never targets Maincloud implicitly.
    #[arg(long)]
    allow_remote: bool,

    /// Suppress human-readable progress logs (telemetry is unaffected).
    #[arg(long)]
    quiet: bool,

    /// Print the effective configuration (without token values) and exit.
    #[arg(long)]
    print_config: bool,
}

pub fn main() -> Result<()> {
    let cli = Cli::parse();
    let print_config = cli.print_config;
    let config = cli.into_config()?;
    if print_config {
        print_config_summary(&config);
        return Ok(());
    }
    crate::live::run(config)
}

impl Cli {
    fn into_config(self) -> Result<LiveConfig> {
        if self.bootstrap_team.is_some() != self.bootstrap_bot.is_some() {
            bail!("--bootstrap-team and --bootstrap-bot must be provided together");
        }
        if self.bootstrap_team.is_some() && self.claim_nonce.is_some() {
            bail!("use either --claim-nonce or --bootstrap-team/--bootstrap-bot, not both");
        }
        if self.bot_name.trim().is_empty() {
            bail!("--bot-name must not be empty");
        }
        if self.beam_width == 0 {
            bail!("--beam-width must be at least 1");
        }
        if self.max_depth == 0 {
            bail!("--max-depth must be at least 1");
        }
        if self.exact_depth > self.max_depth {
            bail!("--exact-depth must not exceed --max-depth");
        }
        if self.search_max_time_ms == 0 {
            bail!("--search-max-time-ms must be at least 1");
        }

        let token_file = if self.no_token_file {
            None
        } else {
            Some(
                self.token_file
                    .unwrap_or_else(|| PathBuf::from(format!(".token-{}", self.bot_name))),
            )
        };
        let telemetry = match self.telemetry.as_str() {
            "-" => TelemetrySink::Stdout,
            "none" => TelemetrySink::Disabled,
            path => TelemetrySink::File(PathBuf::from(path)),
        };
        let strategy = match self.strategy {
            StrategyArg::Greedy => StrategyKind::Greedy,
            StrategyArg::Search => StrategyKind::Search,
        };
        let playback = match self.playback {
            Some(PlaybackArg::Greedy) => PlaybackPolicy::Greedy,
            Some(PlaybackArg::Target) => PlaybackPolicy::TargetAware,
            None => match strategy {
                StrategyKind::Greedy => PlaybackPolicy::Greedy,
                StrategyKind::Search => PlaybackPolicy::TargetAware,
            },
        };
        let search = SearchConfig {
            beam_width: self.beam_width,
            max_depth: self.max_depth,
            exact_depth: self.exact_depth,
            max_time_ms: self.search_max_time_ms,
            max_states: 4096,
            beam_score: BeamScore::ExpectedValue,
        };
        let strategy_config = StrategyConfig {
            playback,
            ..StrategyConfig::default()
        };

        Ok(LiveConfig {
            uri: self.uri,
            database: self.database,
            token_file,
            token_env: self.token_env,
            bot_name: self.bot_name,
            strategy,
            matches: self.matches,
            telemetry,
            search,
            strategy_config,
            submit_margin_ms: self.submit_margin_ms,
            match_timeout_secs: self.match_timeout_secs,
            claim_nonce: self.claim_nonce,
            bootstrap_team: self.bootstrap_team,
            bootstrap_bot: self.bootstrap_bot,
            allow_remote: self.allow_remote,
            quiet: self.quiet,
        })
    }
}

fn print_config_summary(config: &LiveConfig) {
    let token_source = if std::env::var(&config.token_env)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
    {
        "env"
    } else if config.token_file.as_ref().is_some_and(|path| path.exists()) {
        "file"
    } else {
        "none"
    };
    println!("uri={}", config.uri);
    println!("database={}", config.database);
    println!("bot_name={}", config.bot_name);
    println!("strategy={}", config.strategy_name());
    println!("playback={}", config.playback_name());
    println!("matches={}", config.matches);
    println!(
        "token_file={}",
        config
            .token_file
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!("token_env={}", config.token_env);
    println!("token_source={token_source}");
    println!(
        "telemetry={}",
        match &config.telemetry {
            TelemetrySink::File(path) => path.display().to_string(),
            TelemetrySink::Stdout => "-".to_string(),
            TelemetrySink::Disabled => "none".to_string(),
        }
    );
    println!(
        "search=beam_width:{} max_depth:{} exact_depth:{} max_time_ms:{}",
        config.search.beam_width,
        config.search.max_depth,
        config.search.exact_depth,
        config.search.max_time_ms
    );
    println!("submit_margin_ms={}", config.submit_margin_ms);
    println!("match_timeout_secs={}", config.match_timeout_secs);
    println!("claim_nonce={}", config.claim_nonce.is_some());
    println!(
        "bootstrap_team={}",
        config.bootstrap_team.as_deref().unwrap_or("none")
    );
    println!(
        "bootstrap_bot={}",
        config.bootstrap_bot.as_deref().unwrap_or("none")
    );
    println!("allow_remote={}", config.allow_remote);
}
