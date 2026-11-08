//! SpacetimeDB live client.
//!
//! The client owns a single connection at a time, processes SDK messages on a
//! short polling loop, and runs searches on background threads so a search
//! never stalls auction/rack event processing. All reducer inputs are checked
//! against the authoritative client cache immediately before submission.

mod state;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use spacetimedb_sdk::table::{WithDelete, WithInsert, WithUpdate};
use spacetimedb_sdk::{DbContext, Identity};
use url::Url;

use crate::core::dictionary::Dictionary;
use crate::core::search::{SearchConfig, SearchEngine};
use crate::core::strategy::{PlaybackPolicy, StrategyConfig, StrategyKind};
use crate::core::telemetry::TelemetryWriter;
use crate::module_bindings::*;

pub use state::BotRuntime;

/// Base subscription: only small tables. Per-match rows (participants,
/// auctions, results, word plays) are subscribed per match by the runtime
/// (`BotRuntime::manage_subscriptions`). Maincloud holds tens of thousands of
/// finished matches; loading them, or every live match, makes the client fall
/// behind the server's message stream.
fn subscribe_scoped(conn: &DbConnection, shared: &Arc<Shared>) {
    let applied = shared.clone();
    let err = shared.clone();
    conn.subscription_builder()
        .on_applied(move |_ctx| {
            applied.subscription_applied.store(true, Ordering::SeqCst);
            applied.mark_dirty();
        })
        .on_error(move |_ctx, e| {
            err.record_error(None, None, None, "subscription", format!("{e:?}"));
        })
        .subscribe(vec![
            "SELECT * FROM match_state".to_string(),
            "SELECT * FROM lobby".to_string(),
            "SELECT * FROM bot".to_string(),
            "SELECT * FROM bot_credential".to_string(),
            "SELECT * FROM team_member".to_string(),
            "SELECT * FROM my_rack".to_string(),
            "SELECT * FROM my_nonces".to_string(),
        ]);
}

#[derive(Clone, Debug)]
pub enum TelemetrySink {
    File(PathBuf),
    Stdout,
    Disabled,
}

#[derive(Clone, Debug)]
pub struct LiveConfig {
    pub uri: String,
    pub database: String,
    pub token_file: Option<PathBuf>,
    pub token_env: String,
    pub bot_name: String,
    pub strategy: StrategyKind,
    pub genome: Option<Arc<dyn crate::sim::policy::Policy>>,
    pub matches: usize,
    pub telemetry: TelemetrySink,
    pub search: SearchConfig,
    pub strategy_config: StrategyConfig,
    pub submit_margin_ms: u64,
    pub match_timeout_secs: u64,
    pub claim_nonce: Option<String>,
    pub bootstrap_team: Option<String>,
    pub bootstrap_bot: Option<String>,
    pub allow_remote: bool,
    pub quiet: bool,
}

impl LiveConfig {
    pub fn strategy_name(&self) -> &'static str {
        match self.strategy {
            StrategyKind::Greedy => "greedy",
            StrategyKind::Search => "search",
            StrategyKind::Evolved => "evolved",
        }
    }

    pub fn playback_name(&self) -> &'static str {
        match self.strategy_config.playback {
            PlaybackPolicy::Greedy => "greedy",
            PlaybackPolicy::TargetAware => "target",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ReducerErrorRecord {
    pub match_id: Option<u64>,
    pub auction_id: Option<u64>,
    pub word: Option<String>,
    pub action: String,
    pub error: String,
    pub at_ms: u64,
}

pub struct Shared {
    pub dirty: AtomicBool,
    pub connected: AtomicBool,
    pub disconnected: AtomicBool,
    pub subscription_applied: AtomicBool,
    pub identity: Mutex<Option<Identity>>,
    pub errors: Mutex<Vec<ReducerErrorRecord>>,
    pub bid_acks: Mutex<Vec<(u64, i64, u64)>>,
    pub connect_error: Mutex<Option<String>>,
    pub disconnect_reason: Mutex<Option<String>>,
}

impl Shared {
    pub fn new() -> Self {
        Self {
            dirty: AtomicBool::new(false),
            connected: AtomicBool::new(false),
            disconnected: AtomicBool::new(false),
            subscription_applied: AtomicBool::new(false),
            identity: Mutex::new(None),
            errors: Mutex::new(Vec::new()),
            bid_acks: Mutex::new(Vec::new()),
            connect_error: Mutex::new(None),
            disconnect_reason: Mutex::new(None),
        }
    }

    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::SeqCst);
    }

    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::SeqCst)
    }

    pub fn record_error(
        &self,
        match_id: Option<u64>,
        auction_id: Option<u64>,
        word: Option<String>,
        action: impl Into<String>,
        error: impl Into<String>,
    ) {
        let mut errors = self.errors.lock().expect("error mutex poisoned");
        if errors.len() >= 10_000 {
            errors.drain(0..1_000);
        }
        errors.push(ReducerErrorRecord {
            match_id,
            auction_id,
            word,
            action: action.into(),
            error: error.into(),
            at_ms: crate::core::telemetry::now_millis(),
        });
    }

    pub fn drain_errors(&self) -> Vec<ReducerErrorRecord> {
        std::mem::take(&mut *self.errors.lock().expect("error mutex poisoned"))
    }

    pub fn record_bid_ack(&self, auction_id: u64, amount: i64) {
        let mut acks = self.bid_acks.lock().expect("bid ack mutex poisoned");
        if acks.len() >= 10_000 {
            acks.drain(0..1_000);
        }
        acks.push((auction_id, amount, crate::core::telemetry::now_millis()));
    }

    pub fn drain_bid_acks(&self) -> Vec<(u64, i64, u64)> {
        std::mem::take(&mut *self.bid_acks.lock().expect("bid ack mutex poisoned"))
    }

    pub fn set_connect_error(&self, error: String) {
        *self
            .connect_error
            .lock()
            .expect("connect error mutex poisoned") = Some(error);
    }

    pub fn take_connect_error(&self) -> Option<String> {
        self.connect_error
            .lock()
            .expect("connect error mutex poisoned")
            .take()
    }

    pub fn set_disconnect_reason(&self, reason: String) {
        *self
            .disconnect_reason
            .lock()
            .expect("disconnect mutex poisoned") = Some(reason);
    }

    pub fn take_disconnect_reason(&self) -> Option<String> {
        self.disconnect_reason
            .lock()
            .expect("disconnect mutex poisoned")
            .take()
    }
}

impl Default for Shared {
    fn default() -> Self {
        Self::new()
    }
}

pub fn run(config: LiveConfig) -> Result<()> {
    validate_uri(&config.uri, config.allow_remote)?;
    let dictionary = Dictionary::bundled_arc();
    let engine = Arc::new(SearchEngine::new(dictionary, config.search.clone()));
    let telemetry = match &config.telemetry {
        TelemetrySink::File(path) => TelemetryWriter::file(path)
            .with_context(|| format!("opening telemetry file {}", path.display()))?,
        TelemetrySink::Stdout => TelemetryWriter::stdout(),
        TelemetrySink::Disabled => TelemetryWriter::disabled(),
    };
    let mut runtime = BotRuntime::new(config.clone(), engine, telemetry);
    let mut last_connect_error: Option<String> = None;
    let mut last_matches_run = runtime.matches_run();
    let mut last_match_progress = Instant::now();

    loop {
        if runtime.matches_run() != last_matches_run {
            last_matches_run = runtime.matches_run();
            last_match_progress = Instant::now();
        }
        if runtime.has_running_match() {
            last_match_progress = Instant::now();
        }
        if config.matches > 0 && runtime.matches_run() >= config.matches {
            break;
        }
        if config.match_timeout_secs > 0
            && last_match_progress.elapsed() > Duration::from_secs(config.match_timeout_secs)
        {
            bail!(
                "no match progress within {}s (matches run: {}); last connection error: {}",
                config.match_timeout_secs,
                runtime.matches_run(),
                last_connect_error.as_deref().unwrap_or("none")
            );
        }

        let shared = Arc::new(Shared::new());
        match connect(&config, shared.clone()) {
            Ok(conn) => {
                last_connect_error = None;
                let stop = run_connection(&mut runtime, &config, &conn, &shared);
                if stop {
                    break;
                }
                if config.matches > 0 && runtime.matches_run() >= config.matches {
                    break;
                }
                if let Some(reason) = shared.take_disconnect_reason()
                    && !config.quiet
                {
                    eprintln!("[{}] disconnected: {reason}", config.bot_name);
                }
            }
            Err(err) => {
                let message = format!("{err:#}");
                if last_connect_error.as_deref() != Some(message.as_str()) {
                    eprintln!(
                        "[{}] connect to {} / {} failed: {message}",
                        config.bot_name, config.uri, config.database
                    );
                    last_connect_error = Some(message);
                }
            }
        }

        if config.matches > 0 && runtime.matches_run() >= config.matches {
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }

    runtime.write_run_summary();
    if config.matches > 0 && runtime.matches_run() < config.matches {
        bail!(
            "stopped after {} of {} requested matches",
            runtime.matches_run(),
            config.matches
        );
    }
    Ok(())
}

/// Returns `true` when the whole run should stop, `false` when reconnecting is
/// reasonable.
fn run_connection(
    runtime: &mut BotRuntime,
    config: &LiveConfig,
    conn: &DbConnection,
    shared: &Arc<Shared>,
) -> bool {
    let mut last_tick = Instant::now();
    loop {
        if shared.disconnected.load(Ordering::SeqCst) || !conn.is_active() {
            break;
        }
        // Drain everything queued before ticking, so a burst of server
        // messages can never leave the decision loop acting on stale state.
        let mut processed = 0usize;
        let mut failed = false;
        loop {
            match conn.advance_one_message() {
                Ok(true) => {
                    processed += 1;
                    if processed >= 10_000 {
                        break;
                    }
                }
                Ok(false) => break,
                Err(err) => {
                    if !matches!(err, spacetimedb_sdk::Error::Disconnected) && !config.quiet {
                        eprintln!("[{}] connection error: {err:?}", config.bot_name);
                    }
                    failed = true;
                    break;
                }
            }
        }
        if failed {
            break;
        }
        if processed == 0 {
            thread::sleep(Duration::from_millis(2));
        }

        if last_tick.elapsed() >= Duration::from_millis(1) {
            last_tick = Instant::now();
            if let Err(err) = runtime.tick(conn, shared) {
                eprintln!("[{}] runtime stopped: {err:#}", config.bot_name);
                return true;
            }
        }

        if config.matches > 0 && runtime.matches_run() >= config.matches {
            let _ = conn.disconnect();
            break;
        }

        if config.match_timeout_secs > 0
            && runtime.idle_for() > Duration::from_secs(config.match_timeout_secs)
        {
            eprintln!(
                "[{}] no match progress for {}s; stopping",
                config.bot_name, config.match_timeout_secs
            );
            let _ = conn.disconnect();
            return true;
        }
    }
    false
}

fn connect(config: &LiveConfig, shared: Arc<Shared>) -> Result<DbConnection> {
    let token = load_token(config);
    let token_was_provided = token.is_some();
    let mut builder = DbConnection::builder()
        .with_uri(config.uri.clone())
        .with_database_name(config.database.clone());
    if let Some(token) = token {
        builder = builder.with_token(Some(token));
    }

    let config_for_connect = config.clone();
    let shared_for_connect = shared.clone();
    builder = builder
        .on_connect(move |conn, identity, token| {
            *shared_for_connect
                .identity
                .lock()
                .expect("identity mutex poisoned") = Some(identity);
            shared_for_connect.connected.store(true, Ordering::SeqCst);

            if !token_was_provided {
                if let Some(path) = &config_for_connect.token_file {
                    if let Err(err) = save_token_secure(path, token) {
                        eprintln!(
                            "[{}] warning: could not save issued token to {}: {err}",
                            config_for_connect.bot_name,
                            path.display()
                        );
                    }
                } else {
                    eprintln!(
                        "[{}] warning: no token file configured; a reconnect will use a new identity",
                        config_for_connect.bot_name
                    );
                }
            }

            register_table_callbacks(conn, &shared_for_connect);
            subscribe_scoped(conn, &shared_for_connect);

            if !config_for_connect.quiet {
                eprintln!(
                    "[{}] connected as {}",
                    config_for_connect.bot_name,
                    identity.to_hex()
                );
            }
        })
        .on_connect_error({
            let shared = shared.clone();
            move |_ctx, err| {
                shared.set_connect_error(format!("{err:?}"));
                shared.disconnected.store(true, Ordering::SeqCst);
            }
        })
        .on_disconnect({
            let shared = shared.clone();
            move |_ctx, err| {
                shared.disconnected.store(true, Ordering::SeqCst);
                if let Some(err) = err {
                    shared.set_disconnect_reason(format!("{err:?}"));
                }
            }
        });

    builder.build().context("building SpacetimeDB connection")
}

fn register_table_callbacks(conn: &DbConnection, shared: &Arc<Shared>) {
    macro_rules! watch {
        ($table:expr) => {{
            let s = shared.clone();
            WithInsert::on_insert(&$table, move |_, _| s.mark_dirty());
            let s = shared.clone();
            WithDelete::on_delete(&$table, move |_, _| s.mark_dirty());
            let s = shared.clone();
            WithUpdate::on_update(&$table, move |_, _, _| s.mark_dirty());
        }};
    }

    watch!(conn.db.match_state());
    watch!(conn.db.match_participant());
    watch!(conn.db.my_rack());
    watch!(conn.db.auction());
    watch!(conn.db.auction_result());
    watch!(conn.db.word_play());
    watch!(conn.db.bot());
    watch!(conn.db.bot_credential());
    watch!(conn.db.lobby());
    watch!(conn.db.lobby_member());
    watch!(conn.db.my_nonces());
    watch!(conn.db.team_member());
}

fn validate_uri(uri: &str, allow_remote: bool) -> Result<()> {
    let parsed = Url::parse(uri).with_context(|| format!("invalid SpacetimeDB URI {uri:?}"))?;
    let host = parsed.host_str().unwrap_or("");
    let local = matches!(host, "127.0.0.1" | "localhost" | "::1");
    if !local && !allow_remote {
        bail!(
            "refusing non-loopback URI {uri:?} (host {host:?}); this bot is local-only by default. \
             Pass --allow-remote only if you explicitly intend to connect elsewhere."
        );
    }
    if !local {
        eprintln!("warning: --allow-remote is set; connecting to non-loopback host {host:?}");
    }
    Ok(())
}

fn load_token(config: &LiveConfig) -> Option<String> {
    // Prefer the explicit local token file over a possibly unrelated global
    // environment token. The env var remains the fallback for callers that
    // deliberately do not use a file.
    if let Some(path) = &config.token_file
        && let Ok(value) = std::fs::read_to_string(path)
    {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Some(value);
        }
    }
    if let Ok(value) = std::env::var(&config.token_env) {
        let value = value.trim().to_string();
        if !value.is_empty() {
            return Some(value);
        }
    }
    None
}

fn save_token_secure(path: &Path, token: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(path)?;
        file.write_all(token.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    #[cfg(not(unix))]
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        file.write_all(token.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
    }
    Ok(())
}

pub(crate) fn flatten_reducer_result<E: std::fmt::Debug>(
    result: Result<Result<(), String>, E>,
) -> Result<(), String> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(message)) => Err(message),
        Err(err) => Err(format!("{err:?}")),
    }
}
