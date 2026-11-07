//! Authoritative-state reconciliation, bidding/playback driving, and telemetry.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use spacetimedb_sdk::Identity;
use spacetimedb_sdk::table::TableLike;

use crate::core::bag::BagModel;
use crate::core::game::{elapsed_millis, micros_to_millis, placement, reconstruct_initial_rack};
use crate::core::letters::letter_index;
use crate::core::rack::Rack;
use crate::core::search::{SearchEngine, SearchResult};
use crate::core::strategy::{
    AuctionKind, PlayReason, StrategyKind, choose_play, greedy_bid, search_bid,
};
use crate::core::telemetry::{
    AuctionTelemetry, MatchParticipantTelemetry, MatchSummary, MatchWordTelemetry,
    ReducerErrorTelemetry, RunStart, RunSummary, SearchDepthTelemetry, SearchTelemetry,
    TelemetryRecord, TelemetryWriter, WinningBidTelemetry, WordTelemetry, now_millis,
};
use crate::module_bindings::*;

use super::{LiveConfig, ReducerErrorRecord, Shared, flatten_reducer_result};

pub struct BotRuntime {
    config: LiveConfig,
    engine: Arc<SearchEngine>,
    telemetry: TelemetryWriter,
    matches: HashMap<u64, LiveMatch>,
    seen_running: HashSet<u64>,
    summarized: HashSet<u64>,
    matches_run: usize,
    total_reducer_errors: u64,
    search_tx: Sender<SearchJobResult>,
    search_rx: Receiver<SearchJobResult>,
    started_at: Instant,
    last_progress: Instant,
    run_start_written: bool,
    bot_id: Option<u64>,
    bot_name: String,
    bootstrap: BootstrapState,
    bootstrap_error: Option<String>,
    claim_attempted: bool,
    no_credential_since: Option<Instant>,
    was_connected: bool,
    last_join_lobby: Option<Instant>,
}

impl BotRuntime {
    pub fn new(config: LiveConfig, engine: Arc<SearchEngine>, telemetry: TelemetryWriter) -> Self {
        let (search_tx, search_rx) = mpsc::channel();
        let bot_name = config.bot_name.clone();
        Self {
            config,
            engine,
            telemetry,
            matches: HashMap::new(),
            seen_running: HashSet::new(),
            summarized: HashSet::new(),
            matches_run: 0,
            total_reducer_errors: 0,
            search_tx,
            search_rx,
            started_at: Instant::now(),
            last_progress: Instant::now(),
            run_start_written: false,
            bot_id: None,
            bot_name,
            bootstrap: BootstrapState::Idle,
            bootstrap_error: None,
            claim_attempted: false,
            no_credential_since: None,
            was_connected: false,
            last_join_lobby: None,
        }
    }

    pub fn matches_run(&self) -> usize {
        self.matches_run
    }

    pub fn idle_for(&self) -> Duration {
        self.last_progress.elapsed()
    }

    pub fn has_running_match(&self) -> bool {
        self.matches
            .values()
            .any(|live| live.snapshot.match_row.status == MatchStatus::Running)
    }

    pub fn tick(&mut self, conn: &DbConnection, shared: &Arc<Shared>) -> Result<()> {
        let errors = shared.drain_errors();
        self.handle_errors(errors);
        for (auction_id, amount, ack_ms) in shared.drain_bid_acks() {
            for live in self.matches.values_mut() {
                if let Some(trace) = live.traces.get_mut(&auction_id) {
                    trace.bid_ack_at_ms = Some(ack_ms);
                    trace.bid_ack_amount = Some(amount);
                }
            }
        }

        if shared.connected.load(std::sync::atomic::Ordering::SeqCst) && !self.was_connected {
            self.was_connected = true;
            self.no_credential_since = None;
            self.bootstrap = BootstrapState::Idle;
            self.bootstrap_error = None;
            self.claim_attempted = false;
            self.matches.clear();
            self.last_progress = Instant::now();
            while self.search_rx.try_recv().is_ok() {}
        }
        if !shared.connected.load(std::sync::atomic::Ordering::SeqCst) {
            self.was_connected = false;
        }

        if shared
            .disconnected
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }
        if !shared
            .subscription_applied
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(());
        }
        let identity = match *shared.identity.lock().expect("identity mutex poisoned") {
            Some(identity) => identity,
            None => return Ok(()),
        };

        self.reconcile(conn, shared, identity)?;
        self.poll_search_results(conn, shared);
        self.drive(conn, shared)?;
        self.maybe_join_lobby(conn, shared);
        Ok(())
    }

    fn reconcile(
        &mut self,
        conn: &DbConnection,
        shared: &Arc<Shared>,
        identity: Identity,
    ) -> Result<()> {
        let my_bot_id = conn
            .db
            .bot_credential()
            .iter()
            .find(|cred| cred.identity == identity)
            .map(|cred| cred.bot_id);

        let Some(my_bot_id) = my_bot_id else {
            self.handle_no_credential(conn, shared, &identity)?;
            return Ok(());
        };

        self.no_credential_since = None;
        if self.bot_id != Some(my_bot_id) {
            self.bot_id = Some(my_bot_id);
            if let Some(bot) = conn.db.bot().id().find(&my_bot_id) {
                self.bot_name = bot.name;
            }
            self.write_run_start();
            if !self.config.quiet {
                eprintln!(
                    "[{}] acting as bot '{}' (id {})",
                    self.config.bot_name, self.bot_name, my_bot_id
                );
            }
        }

        let dirty = shared.take_dirty();
        if !dirty && !self.matches.is_empty() {
            return Ok(());
        }

        let snapshots = build_all_snapshots(conn, my_bot_id)?;
        let mut running_ids = HashSet::new();
        let mut progress = false;
        for snapshot in snapshots {
            let match_id = snapshot.match_row.id;
            match snapshot.match_row.status {
                MatchStatus::Running => {
                    running_ids.insert(match_id);
                    self.seen_running.insert(match_id);
                    let live = self
                        .matches
                        .entry(match_id)
                        .or_insert_with(|| LiveMatch::new(snapshot.clone()));
                    if live.update_snapshot(snapshot) {
                        progress = true;
                    }
                }
                MatchStatus::Ended => {
                    // Only count matches this process actually observed running.
                    // Historical ended matches in the client cache must not
                    // consume the bounded --matches budget on a fresh start.
                    if self.seen_running.contains(&match_id) && !self.summarized.contains(&match_id)
                    {
                        // The terminal auction and Ended status arrive in the
                        // same transaction, so no running-match drive will flush it.
                        if let Some(live) = self.matches.get_mut(&match_id) {
                            record_auction_results(
                                &mut self.telemetry,
                                conn,
                                &snapshot,
                                live,
                                my_bot_id,
                            );
                        }
                        let live = self.matches.get(&match_id);
                        write_match_summary(
                            &mut self.telemetry,
                            &self.config,
                            conn,
                            &snapshot,
                            live,
                            my_bot_id,
                            &self.bot_name,
                        );
                        self.summarized.insert(match_id);
                        self.seen_running.remove(&match_id);
                        self.matches.remove(&match_id);
                        self.matches_run += 1;
                        progress = true;
                        if !self.config.quiet {
                            eprintln!(
                                "[{}] match {} ended: score {} placement {}",
                                self.config.bot_name,
                                match_id,
                                snapshot.my_participant.score,
                                placement(
                                    snapshot.participants.iter().map(|p| p.score),
                                    snapshot.my_participant.score
                                )
                            );
                        }
                    }
                }
                MatchStatus::Lobby => {}
            }
        }
        self.matches.retain(|id, _| running_ids.contains(id));
        if progress {
            self.last_progress = Instant::now();
        }
        Ok(())
    }

    fn handle_no_credential(
        &mut self,
        conn: &DbConnection,
        shared: &Arc<Shared>,
        identity: &Identity,
    ) -> Result<()> {
        let since = *self.no_credential_since.get_or_insert_with(Instant::now);

        if let Some(code) = &self.config.claim_nonce {
            if !self.claim_attempted {
                self.claim_attempted = true;
                let shared_for_callback = shared.clone();
                let result = conn
                    .reducers
                    .claim_credential_then(code.clone(), move |_ctx, res| {
                        if let Err(message) = flatten_reducer_result(res) {
                            shared_for_callback.record_error(
                                None,
                                None,
                                None,
                                "claim_credential",
                                message,
                            );
                        }
                    });
                if let Err(err) = result {
                    shared.record_error(
                        None,
                        None,
                        None,
                        "claim_credential_send",
                        format!("{err:?}"),
                    );
                }
                if !self.config.quiet {
                    eprintln!(
                        "[{}] claiming bot credential with nonce",
                        self.config.bot_name
                    );
                }
            }
            if since.elapsed() > Duration::from_secs(8) {
                bail!("claim_credential did not produce a BotCredential; bad or expired nonce?");
            }
            return Ok(());
        }

        if self.config.bootstrap_team.is_some() && self.config.bootstrap_bot.is_some() {
            self.bootstrap_step(conn, shared, identity)?;
            return Ok(());
        }

        if since.elapsed() > Duration::from_secs(3) {
            bail!(
                "no BotCredential for this identity. Provide a local token with --token-file or \
                 --token-env, use --claim-nonce CODE, or use --bootstrap-team NAME --bootstrap-bot NAME \
                 on a local-only server."
            );
        }
        Ok(())
    }

    fn bootstrap_step(
        &mut self,
        conn: &DbConnection,
        shared: &Arc<Shared>,
        identity: &Identity,
    ) -> Result<()> {
        if let Some(error) = &self.bootstrap_error {
            bail!("local bootstrap failed: {error}");
        }
        let team = self
            .config
            .bootstrap_team
            .clone()
            .expect("checked by caller");
        let bot = self
            .config
            .bootstrap_bot
            .clone()
            .expect("checked by caller");

        match self.bootstrap {
            BootstrapState::Idle => {
                let shared_for_callback = shared.clone();
                let result =
                    conn.reducers
                        .create_team_then(team.clone(), bot.clone(), move |_ctx, res| {
                            if let Err(message) = flatten_reducer_result(res) {
                                shared_for_callback.record_error(
                                    None,
                                    None,
                                    None,
                                    "create_team",
                                    message,
                                );
                            }
                        });
                if let Err(err) = result {
                    shared.record_error(None, None, None, "create_team_send", format!("{err:?}"));
                }
                self.bootstrap = BootstrapState::CreateTeamSent(Instant::now());
                if !self.config.quiet {
                    eprintln!(
                        "[{}] local bootstrap: creating team '{}' bot '{}'",
                        self.config.bot_name, team, bot
                    );
                }
            }
            BootstrapState::CreateTeamSent(at) => {
                if at.elapsed() > Duration::from_secs(8) {
                    bail!("create_team did not create a team membership");
                }
                if conn
                    .db
                    .team_member()
                    .iter()
                    .any(|member| member.user == *identity)
                {
                    let shared_for_callback = shared.clone();
                    let result = conn.reducers.mint_credential_nonce_then(move |_ctx, res| {
                        if let Err(message) = flatten_reducer_result(res) {
                            shared_for_callback.record_error(
                                None,
                                None,
                                None,
                                "mint_nonce",
                                message,
                            );
                        }
                    });
                    if let Err(err) = result {
                        shared.record_error(
                            None,
                            None,
                            None,
                            "mint_nonce_send",
                            format!("{err:?}"),
                        );
                    }
                    self.bootstrap = BootstrapState::MintNonceSent(Instant::now());
                }
            }
            BootstrapState::MintNonceSent(at) => {
                if at.elapsed() > Duration::from_secs(8) {
                    bail!("mint_credential_nonce did not produce a nonce");
                }
                if let Some(nonce) = conn.db.my_nonces().iter().next() {
                    let shared_for_callback = shared.clone();
                    let result = conn.reducers.claim_credential_then(
                        nonce.code.clone(),
                        move |_ctx, res| {
                            if let Err(message) = flatten_reducer_result(res) {
                                shared_for_callback.record_error(
                                    None,
                                    None,
                                    None,
                                    "claim_credential",
                                    message,
                                );
                            }
                        },
                    );
                    if let Err(err) = result {
                        shared.record_error(
                            None,
                            None,
                            None,
                            "claim_credential_send",
                            format!("{err:?}"),
                        );
                    }
                    self.bootstrap = BootstrapState::ClaimSent(Instant::now());
                }
            }
            BootstrapState::ClaimSent(at) => {
                if at.elapsed() > Duration::from_secs(8) {
                    bail!("claim_credential did not produce a BotCredential");
                }
                // The next reconcile resolves the credential from the cache.
                self.bootstrap = BootstrapState::Idle;
            }
        }
        Ok(())
    }

    fn handle_errors(&mut self, errors: Vec<ReducerErrorRecord>) {
        for error in errors {
            self.total_reducer_errors += 1;
            self.telemetry
                .write(TelemetryRecord::ReducerError(ReducerErrorTelemetry {
                    match_id: error.match_id,
                    auction_id: error.auction_id,
                    word: error.word.clone(),
                    action: error.action.clone(),
                    error: error.error.clone(),
                    at_ms: error.at_ms,
                }));

            if let Some(match_id) = error.match_id
                && let Some(live) = self.matches.get_mut(&match_id)
            {
                live.reducer_errors += 1;
                if error.action == "word"
                    && let Some(pending) = &live.pending_word
                    && error.word.as_deref() == Some(pending.word.as_str())
                {
                    live.pending_word = None;
                }
                if error.action == "bid"
                    && let Some(auction_id) = error.auction_id
                    && let Some(trace) = live.traces.get_mut(&auction_id)
                {
                    trace.reducer_error = Some(error.error.clone());
                }
            }

            if matches!(
                error.action.as_str(),
                "create_team" | "mint_nonce" | "claim_credential"
            ) {
                self.bootstrap_error = Some(error.error);
            }
        }
    }

    fn poll_search_results(&mut self, conn: &DbConnection, shared: &Arc<Shared>) {
        while let Ok(job) = self.search_rx.try_recv() {
            self.handle_search_result(conn, shared, job);
        }
    }

    fn handle_search_result(
        &mut self,
        conn: &DbConnection,
        shared: &Arc<Shared>,
        job: SearchJobResult,
    ) {
        let Some(live) = self.matches.get_mut(&job.match_id) else {
            return;
        };
        let Some(trace) = live.traces.get_mut(&job.auction_id) else {
            return;
        };
        trace.search = Some(job.result.clone());

        if job.rack_version != live.rack_version {
            trace.stale = true;
            return;
        }
        let Some(auction) = live
            .snapshot
            .auctions
            .iter()
            .find(|auction| auction.id == job.auction_id)
        else {
            trace.stale = true;
            return;
        };
        if auction.status != AuctionStatus::Open
            || live.snapshot.match_row.current_auction_id != Some(job.auction_id)
        {
            trace.late_search = true;
            return;
        }
        let remaining = remaining_auction_ms(auction);
        if remaining <= self.config.submit_margin_ms as i64 + 5 {
            trace.late_search = true;
            return;
        }

        let kind = auction_kind(&live.snapshot.match_row);
        let decision = search_bid(
            &job.result,
            &live.snapshot.rack,
            live.snapshot.my_participant.balance,
            kind,
            &self.config.strategy_config,
        );
        trace.search_bid = Some(decision.amount);
        // The greedy fallback is a floor, not a throwaway. Greedy liquidation
        // can report zero marginal value for a letter that is not yet part of
        // a complete word; retracting the fallback to zero in that case makes
        // the bot needlessly passive. The search can still raise the bid.
        let (target_amount, floor_applied) = apply_bid_floor(decision.amount, trace.fallback_bid);
        trace.floor_applied = floor_applied;
        if target_amount != trace.final_bid
            && submit_bid(conn, shared, job.match_id, job.auction_id, target_amount)
        {
            trace.final_bid = target_amount;
            trace.submitted = true;
            trace.submitted_at_ms = Some(now_millis());
            if !self.config.quiet {
                eprintln!(
                    "[{}] match {} auction {} '{}' final bid {} (search {}, marginal {:.1}, depth {})",
                    self.config.bot_name,
                    job.match_id,
                    job.auction_id,
                    trace.letter_str,
                    target_amount,
                    decision.amount,
                    decision.marginal_value,
                    job.result.depth_completed
                );
            }
        }
    }

    fn drive(&mut self, conn: &DbConnection, shared: &Arc<Shared>) -> Result<()> {
        let Some(my_bot_id) = self.bot_id else {
            return Ok(());
        };
        let ids: Vec<u64> = self.matches.keys().copied().collect();
        let Self {
            matches,
            config,
            engine,
            telemetry,
            search_tx,
            ..
        } = self;
        for id in ids {
            let Some(live) = matches.get_mut(&id) else {
                continue;
            };
            drive_match(
                conn, shared, config, engine, telemetry, search_tx, live, my_bot_id,
            );
        }
        Ok(())
    }

    fn maybe_join_lobby(&mut self, conn: &DbConnection, shared: &Arc<Shared>) {
        let Some(my_bot_id) = self.bot_id else {
            return;
        };
        if self
            .matches
            .values()
            .any(|live| live.snapshot.match_row.status == MatchStatus::Running)
        {
            return;
        }
        if self.config.matches > 0 && self.matches_run >= self.config.matches {
            return;
        }
        if let Some(last) = self.last_join_lobby
            && last.elapsed() < Duration::from_secs(2)
        {
            return;
        }
        let Some(lobby) = conn
            .db
            .lobby()
            .iter()
            .find(|lobby| lobby.status == LobbyStatus::Open)
        else {
            return;
        };
        if conn
            .db
            .lobby_member()
            .iter()
            .any(|member| member.lobby_id == lobby.id && member.bot_id == my_bot_id)
        {
            return;
        }
        let shared_for_callback = shared.clone();
        let result = conn.reducers.join_lobby_then(move |_ctx, res| {
            if let Err(message) = flatten_reducer_result(res)
                && !is_benign_join_error(&message)
            {
                shared_for_callback.record_error(None, None, None, "join_lobby", message);
            }
        });
        if let Err(err) = result {
            shared.record_error(None, None, None, "join_lobby_send", format!("{err:?}"));
        }
        self.last_join_lobby = Some(Instant::now());
        if !self.config.quiet {
            eprintln!("[{}] joining lobby {}", self.config.bot_name, lobby.id);
        }
    }

    fn write_run_start(&mut self) {
        if self.run_start_written {
            return;
        }
        self.run_start_written = true;
        self.telemetry.write(TelemetryRecord::RunStart(RunStart {
            bot_name: self.bot_name.clone(),
            bot_id: self.bot_id,
            strategy: self.config.strategy_name().to_string(),
            playback: self.config.playback_name().to_string(),
            uri: self.config.uri.clone(),
            database: self.config.database.clone(),
            matches_target: self.config.matches,
            beam_width: self.config.search.beam_width,
            max_depth: self.config.search.max_depth,
            exact_depth: self.config.search.exact_depth,
            search_max_time_ms: self.config.search.max_time_ms,
            started_at_ms: now_millis(),
        }));
    }

    pub fn write_run_summary(&mut self) {
        let summary = RunSummary {
            matches_played: self.matches_run,
            matches_target: self.config.matches,
            strategy: self.config.strategy_name().to_string(),
            elapsed_ms: self.started_at.elapsed().as_millis() as u64,
            reducer_errors: self.total_reducer_errors,
            telemetry_path: self.telemetry.path().map(|path| path.display().to_string()),
        };
        self.telemetry
            .write(TelemetryRecord::RunSummary(summary.clone()));
        if !self.config.quiet {
            eprintln!(
                "[{}] run summary: {} match(es), {} reducer error(s), {} ms",
                self.config.bot_name,
                summary.matches_played,
                summary.reducer_errors,
                summary.elapsed_ms
            );
        }
    }
}

#[derive(Clone, Debug)]
struct MatchSnapshot {
    match_row: Match,
    my_participant: MatchParticipant,
    participants: Vec<MatchParticipant>,
    rack: Rack,
    bag: BagModel,
    auctions: Vec<Auction>,
    results: Vec<AuctionResult>,
    my_word_plays: Vec<WordPlay>,
}

fn build_all_snapshots(conn: &DbConnection, my_bot_id: u64) -> Result<Vec<MatchSnapshot>> {
    let mut snapshots = Vec::new();
    for participant in conn.db.match_participant().iter() {
        if participant.bot_id != my_bot_id {
            continue;
        }
        let Some(match_row) = conn.db.match_state().id().find(&participant.match_id) else {
            continue;
        };
        if match_row.status == MatchStatus::Lobby {
            continue;
        }
        if let Some(snapshot) = build_snapshot(conn, &match_row, my_bot_id)? {
            snapshots.push(snapshot);
        }
    }
    Ok(snapshots)
}

fn build_snapshot(
    conn: &DbConnection,
    match_row: &Match,
    my_bot_id: u64,
) -> Result<Option<MatchSnapshot>> {
    let match_id = match_row.id;
    let participants: Vec<MatchParticipant> = conn
        .db
        .match_participant()
        .iter()
        .filter(|participant| participant.match_id == match_id)
        .collect();
    let Some(my_participant) = participants
        .iter()
        .find(|participant| participant.bot_id == my_bot_id)
        .cloned()
    else {
        return Ok(None);
    };

    let mut auctions: Vec<Auction> = conn
        .db
        .auction()
        .iter()
        .filter(|auction| auction.match_id == match_id)
        .collect();
    auctions.sort_by_key(|auction| auction.id);

    let mut results: Vec<AuctionResult> = conn
        .db
        .auction_result()
        .iter()
        .filter(|result| result.match_id == match_id)
        .collect();
    results.sort_by_key(|result| result.auction_id);

    let mut word_plays: Vec<WordPlay> = conn
        .db
        .word_play()
        .iter()
        .filter(|play| play.match_id == match_id)
        .collect();
    word_plays.sort_by_key(|play| play.id);
    let my_word_plays: Vec<WordPlay> = word_plays
        .iter()
        .filter(|play| play.bot_id == my_bot_id)
        .cloned()
        .collect();

    let mut rack = Rack::new();
    for holding in conn.db.my_rack().iter() {
        if holding.match_id != match_id || holding.bot_id != my_bot_id {
            continue;
        }
        if let Some(byte) = holding.letter.as_bytes().first()
            && let Some(index) = letter_index(*byte)
        {
            rack.add(index, holding.count.min(255) as u8);
        }
    }

    let won_letters: Vec<u8> = results
        .iter()
        .filter(|result| result.winner_bot_id == Some(my_bot_id))
        .filter_map(|result| result.letter.as_bytes().first().copied())
        .collect();
    let played_words: Vec<&str> = my_word_plays
        .iter()
        .map(|play| play.word.as_str())
        .collect();
    let initial_rack = reconstruct_initial_rack(&rack, played_words, won_letters);
    let auction_letters: Vec<u8> = auctions
        .iter()
        .filter_map(|auction| auction.letter.as_bytes().first().copied())
        .collect();
    let bag = BagModel::estimate_for_match(&initial_rack, auction_letters, match_row.bag_total);

    Ok(Some(MatchSnapshot {
        match_row: match_row.clone(),
        my_participant,
        participants,
        rack,
        bag,
        auctions,
        results,
        my_word_plays,
    }))
}

struct LiveMatch {
    snapshot: MatchSnapshot,
    first_seen_ms: u64,
    last_auction_id: Option<u64>,
    hold_auctions: u32,
    rack_version: u64,
    last_rack: Rack,
    pending_word: Option<PendingWord>,
    last_word_attempt: Option<Instant>,
    last_played_word: Option<String>,
    recorded_word_ids: HashSet<u64>,
    traces: BTreeMap<u64, AuctionTrace>,
    reducer_errors: u64,
    last_round: u32,
    last_word_count: usize,
    last_result_count: usize,
}

impl LiveMatch {
    fn new(snapshot: MatchSnapshot) -> Self {
        Self {
            last_auction_id: snapshot.match_row.current_auction_id,
            last_rack: snapshot.rack,
            last_round: snapshot.match_row.current_round,
            last_word_count: snapshot.my_word_plays.len(),
            last_result_count: snapshot.results.len(),
            snapshot,
            first_seen_ms: now_millis(),
            hold_auctions: 0,
            rack_version: 0,
            pending_word: None,
            last_word_attempt: None,
            last_played_word: None,
            recorded_word_ids: HashSet::new(),
            traces: BTreeMap::new(),
            reducer_errors: 0,
        }
    }

    fn update_snapshot(&mut self, snapshot: MatchSnapshot) -> bool {
        let progress = snapshot.match_row.current_round != self.last_round
            || snapshot.my_word_plays.len() != self.last_word_count
            || snapshot.results.len() != self.last_result_count;

        if snapshot.rack != self.last_rack {
            self.rack_version += 1;
            self.last_rack = snapshot.rack;
        }
        if snapshot.match_row.current_auction_id != self.last_auction_id {
            if self.last_auction_id.is_some() {
                self.hold_auctions = self.hold_auctions.saturating_add(1);
            }
            self.last_auction_id = snapshot.match_row.current_auction_id;
        }
        self.last_round = snapshot.match_row.current_round;
        self.last_word_count = snapshot.my_word_plays.len();
        self.last_result_count = snapshot.results.len();
        self.snapshot = snapshot;
        progress
    }
}

struct PendingWord {
    word: String,
    submitted_at: Instant,
    submitted_at_micros: i64,
    source: PlayReason,
}

struct AuctionTrace {
    auction_id: u64,
    round: u32,
    letter: u8,
    letter_str: String,
    kind: AuctionKind,
    balance_before: i64,
    fallback_bid: i64,
    search_bid: Option<i64>,
    final_bid: i64,
    submitted: bool,
    submitted_at_ms: Option<u64>,
    search: Option<SearchResult>,
    search_started: bool,
    floor_applied: bool,
    late_search: bool,
    stale: bool,
    reducer_error: Option<String>,
    result: Option<AuctionResult>,
    result_at_ms: Option<u64>,
    auction_open_at_ms: Option<u64>,
    search_started_at_ms: Option<u64>,
    search_budget_ms: Option<u64>,
    bid_ack_at_ms: Option<u64>,
    bid_ack_amount: Option<i64>,
    written: bool,
}

impl AuctionTrace {
    fn new(auction: &Auction, round: u32, kind: AuctionKind, balance_before: i64) -> Self {
        let letter = auction
            .letter
            .as_bytes()
            .first()
            .and_then(|byte| letter_index(*byte))
            .unwrap_or(0) as u8;
        Self {
            auction_id: auction.id,
            round,
            letter,
            letter_str: auction.letter.clone(),
            kind,
            balance_before,
            fallback_bid: 0,
            search_bid: None,
            final_bid: 0,
            submitted: false,
            submitted_at_ms: None,
            search: None,
            search_started: false,
            floor_applied: false,
            late_search: false,
            stale: false,
            reducer_error: None,
            result: None,
            result_at_ms: None,
            auction_open_at_ms: Some(micros_to_millis(
                auction.opens_at.to_micros_since_unix_epoch(),
            )),
            search_started_at_ms: None,
            search_budget_ms: None,
            bid_ack_at_ms: None,
            bid_ack_amount: None,
            written: false,
        }
    }

    fn from_result(result: &AuctionResult, snapshot: &MatchSnapshot) -> Self {
        let letter = result
            .letter
            .as_bytes()
            .first()
            .and_then(|byte| letter_index(*byte))
            .unwrap_or(0) as u8;
        Self {
            auction_id: result.auction_id,
            round: 0,
            letter,
            letter_str: result.letter.clone(),
            kind: auction_kind(&snapshot.match_row),
            balance_before: snapshot.my_participant.balance,
            fallback_bid: 0,
            search_bid: None,
            final_bid: 0,
            submitted: false,
            submitted_at_ms: None,
            search: None,
            search_started: true,
            floor_applied: false,
            late_search: true,
            stale: false,
            reducer_error: None,
            result: Some(result.clone()),
            result_at_ms: Some(micros_to_millis(
                result.closed_at.to_micros_since_unix_epoch(),
            )),
            auction_open_at_ms: None,
            search_started_at_ms: None,
            search_budget_ms: None,
            bid_ack_at_ms: None,
            bid_ack_amount: None,
            written: false,
        }
    }
}

struct SearchJobResult {
    match_id: u64,
    auction_id: u64,
    rack_version: u64,
    result: SearchResult,
}

#[allow(clippy::too_many_arguments)]
fn drive_match(
    conn: &DbConnection,
    shared: &Arc<Shared>,
    config: &LiveConfig,
    engine: &Arc<SearchEngine>,
    telemetry: &mut TelemetryWriter,
    search_tx: &Sender<SearchJobResult>,
    live: &mut LiveMatch,
    my_bot_id: u64,
) {
    let snapshot = live.snapshot.clone();
    record_new_word_plays(telemetry, live, &snapshot);
    drive_pending_word(live, &snapshot);
    drive_auctions(
        conn, shared, config, engine, telemetry, search_tx, live, &snapshot, my_bot_id,
    );
    drive_word_play(conn, shared, config, engine, live, &snapshot);
}

fn record_new_word_plays(
    telemetry: &mut TelemetryWriter,
    live: &mut LiveMatch,
    snapshot: &MatchSnapshot,
) {
    for play in &snapshot.my_word_plays {
        if !live.recorded_word_ids.insert(play.id) {
            continue;
        }
        let source = live
            .pending_word
            .as_ref()
            .filter(|pending| pending.word == play.word)
            .map(|pending| play_reason_name(pending.source))
            .unwrap_or("observed");
        telemetry.write(TelemetryRecord::Word(WordTelemetry {
            match_id: snapshot.match_row.id,
            word: play.word.clone(),
            base: play.base_score,
            bonus: play.bonus,
            reward: play.total_reward,
            balance_after: snapshot.my_participant.balance,
            score_after: snapshot.my_participant.score,
            source: source.to_string(),
            reducer_error: None,
            played_at_ms: micros_to_millis(play.played_at.to_micros_since_unix_epoch()),
        }));
        if live
            .pending_word
            .as_ref()
            .is_some_and(|pending| pending.word == play.word)
        {
            live.pending_word = None;
            live.hold_auctions = 0;
            live.last_played_word = Some(play.word.clone());
        }
    }
}

fn drive_pending_word(live: &mut LiveMatch, snapshot: &MatchSnapshot) {
    let Some(pending) = &live.pending_word else {
        return;
    };
    let confirmed = snapshot.my_word_plays.iter().any(|play| {
        play.word == pending.word
            && play.played_at.to_micros_since_unix_epoch() >= pending.submitted_at_micros
    });
    if confirmed
        || pending.submitted_at.elapsed() > Duration::from_millis(1500)
        || !snapshot.rack.contains_word(&pending.word)
    {
        live.pending_word = None;
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_auctions(
    conn: &DbConnection,
    shared: &Arc<Shared>,
    config: &LiveConfig,
    engine: &Arc<SearchEngine>,
    telemetry: &mut TelemetryWriter,
    search_tx: &Sender<SearchJobResult>,
    live: &mut LiveMatch,
    snapshot: &MatchSnapshot,
    my_bot_id: u64,
) {
    let current_auction = snapshot
        .auctions
        .iter()
        .find(|auction| {
            Some(auction.id) == snapshot.match_row.current_auction_id
                && auction.status == AuctionStatus::Open
        })
        .or_else(|| {
            snapshot
                .auctions
                .iter()
                .rev()
                .find(|auction| auction.status == AuctionStatus::Open)
        });

    if let Some(auction) = current_auction {
        let round = snapshot
            .auctions
            .iter()
            .position(|candidate| candidate.id == auction.id)
            .unwrap_or(0) as u32
            + 1;
        let kind = auction_kind(&snapshot.match_row);
        let trace = live.traces.entry(auction.id).or_insert_with(|| {
            AuctionTrace::new(auction, round, kind, snapshot.my_participant.balance)
        });

        if !trace.search_started {
            trace.search_started = true;
            let remaining = remaining_auction_ms(auction);
            let fallback = greedy_bid(
                trace.letter as usize,
                &snapshot.rack,
                snapshot.my_participant.balance,
            );
            trace.fallback_bid = fallback;
            trace.final_bid = fallback;
            if remaining > config.submit_margin_ms as i64 + 10
                && fallback > 0
                && submit_bid(conn, shared, snapshot.match_row.id, auction.id, fallback)
            {
                trace.submitted = true;
                trace.submitted_at_ms = Some(now_millis());
                if !config.quiet {
                    eprintln!(
                        "[{}] match {} auction {} '{}' fallback bid {}",
                        config.bot_name,
                        snapshot.match_row.id,
                        auction.id,
                        trace.letter_str,
                        fallback
                    );
                }
            }

            if config.strategy == StrategyKind::Search
                && remaining > config.submit_margin_ms as i64 + 30
            {
                let budget_ms = (remaining - config.submit_margin_ms as i64 - 20)
                    .min(config.search.max_time_ms as i64)
                    .max(10) as u64;
                trace.search_started_at_ms = Some(now_millis());
                trace.search_budget_ms = Some(budget_ms);
                let deadline = Instant::now() + Duration::from_millis(budget_ms);
                let engine = engine.clone();
                let tx = search_tx.clone();
                let rack = snapshot.rack;
                let bag = snapshot.bag.clone();
                let candidate = trace.letter as usize;
                let match_id = snapshot.match_row.id;
                let auction_id = auction.id;
                let rack_version = live.rack_version;
                std::thread::spawn(move || {
                    let result = engine.search(&rack, candidate, &bag, deadline);
                    let _ = tx.send(SearchJobResult {
                        match_id,
                        auction_id,
                        rack_version,
                        result,
                    });
                });
            }
        }
    }

    record_auction_results(telemetry, conn, snapshot, live, my_bot_id);
}

fn record_auction_results(
    telemetry: &mut TelemetryWriter,
    conn: &DbConnection,
    snapshot: &MatchSnapshot,
    live: &mut LiveMatch,
    my_bot_id: u64,
) {
    for result in &snapshot.results {
        let trace = live
            .traces
            .entry(result.auction_id)
            .or_insert_with(|| AuctionTrace::from_result(result, snapshot));
        if let Some(auction) = snapshot
            .auctions
            .iter()
            .find(|auction| auction.id == result.auction_id)
        {
            if trace.round == 0 {
                trace.round = snapshot
                    .auctions
                    .iter()
                    .position(|candidate| candidate.id == auction.id)
                    .unwrap_or(0) as u32
                    + 1;
            }
            if trace.auction_open_at_ms.is_none() {
                trace.auction_open_at_ms = Some(micros_to_millis(
                    auction.opens_at.to_micros_since_unix_epoch(),
                ));
            }
        }
        trace.result = Some(result.clone());
        trace.result_at_ms = Some(micros_to_millis(
            result.closed_at.to_micros_since_unix_epoch(),
        ));
        if !trace.written {
            write_auction_telemetry(telemetry, conn, snapshot, trace, my_bot_id);
            trace.written = true;
        }
    }
}

fn drive_word_play(
    conn: &DbConnection,
    shared: &Arc<Shared>,
    config: &LiveConfig,
    engine: &Arc<SearchEngine>,
    live: &mut LiveMatch,
    snapshot: &MatchSnapshot,
) {
    if live.pending_word.is_some() {
        return;
    }
    if let Some(last) = live.last_word_attempt
        && last.elapsed() < Duration::from_millis(500)
    {
        return;
    }
    let decision = choose_play(
        &snapshot.rack,
        engine.dictionary(),
        &config.strategy_config,
        live.hold_auctions,
        snapshot.match_row.bag_total,
    );
    let Some(word) = decision.word else {
        return;
    };
    if word.word.len() < 2
        || !snapshot.rack.contains_word(&word.word)
        || !engine.dictionary().contains(&word.word)
    {
        return;
    }

    let shared_for_callback = shared.clone();
    let match_id = snapshot.match_row.id;
    let submitted_word = word.word.clone();
    let result =
        conn.reducers
            .submit_word_then(match_id, submitted_word.clone(), move |_ctx, res| {
                if let Err(message) = flatten_reducer_result(res) {
                    shared_for_callback.record_error(
                        Some(match_id),
                        None,
                        Some(submitted_word.clone()),
                        "word",
                        message,
                    );
                }
            });
    match result {
        Ok(()) => {
            live.pending_word = Some(PendingWord {
                word: word.word.clone(),
                submitted_at: Instant::now(),
                submitted_at_micros: now_micros(),
                source: decision.reason,
            });
            live.last_word_attempt = Some(Instant::now());
            if !config.quiet {
                eprintln!(
                    "[{}] match {} playing '{}' (reward {}, reason {:?})",
                    config.bot_name, match_id, word.word, word.reward, decision.reason
                );
            }
        }
        Err(err) => {
            shared.record_error(
                Some(match_id),
                None,
                Some(word.word),
                "word_send",
                format!("{err:?}"),
            );
        }
    }
}

fn write_auction_telemetry(
    telemetry: &mut TelemetryWriter,
    conn: &DbConnection,
    snapshot: &MatchSnapshot,
    trace: &AuctionTrace,
    my_bot_id: u64,
) {
    let Some(result) = &trace.result else {
        return;
    };
    let winner_name = result
        .winner_bot_id
        .and_then(|bot_id| conn.db.bot().id().find(&bot_id).map(|bot| bot.name));
    let search = trace.search.as_ref().map(search_telemetry);
    telemetry.write(TelemetryRecord::Auction(Box::new(AuctionTelemetry {
        match_id: snapshot.match_row.id,
        auction_id: trace.auction_id,
        round: trace.round,
        letter: trace.letter_str.clone(),
        auction_kind: auction_kind_name(trace.kind).to_string(),
        balance_before: trace.balance_before,
        fallback_bid: trace.fallback_bid,
        search_bid: trace.search_bid,
        floor_applied: trace.floor_applied,
        final_bid: trace.final_bid,
        submitted: trace.submitted,
        search,
        winner_bot_id: result.winner_bot_id,
        winner_name,
        paid: result.paid,
        won: result.winner_bot_id == Some(my_bot_id),
        late_search: trace.late_search,
        stale: trace.stale,
        reducer_error: trace.reducer_error.clone(),
        auction_open_at_ms: trace.auction_open_at_ms,
        search_started_at_ms: trace.search_started_at_ms,
        search_start_delay_ms: trace
            .search_started_at_ms
            .zip(trace.auction_open_at_ms)
            .map(|(start, open)| start.saturating_sub(open)),
        search_budget_ms: trace.search_budget_ms,
        bid_submitted_at_ms: trace.submitted_at_ms,
        bid_ack_at_ms: trace.bid_ack_at_ms,
        bid_ack_amount: trace.bid_ack_amount,
        bid_ack_latency_ms: trace
            .bid_ack_at_ms
            .zip(trace.submitted_at_ms)
            .map(|(ack, submitted)| ack.saturating_sub(submitted)),
        time_left_after_ack_ms: trace
            .bid_ack_at_ms
            .zip(trace.result_at_ms)
            .map(|(ack, close)| close.saturating_sub(ack)),
        total_to_ack_ms: trace
            .bid_ack_at_ms
            .zip(trace.auction_open_at_ms)
            .map(|(ack, open)| ack.saturating_sub(open)),
        result_at_ms: trace.result_at_ms,
    })));
}

fn write_match_summary(
    telemetry: &mut TelemetryWriter,
    config: &LiveConfig,
    conn: &DbConnection,
    snapshot: &MatchSnapshot,
    live: Option<&LiveMatch>,
    my_bot_id: u64,
    bot_name: &str,
) {
    let my_participant = &snapshot.my_participant;
    let scores: Vec<i64> = snapshot.participants.iter().map(|p| p.score).collect();
    let participants = snapshot
        .participants
        .iter()
        .map(|participant| {
            let bot = conn.db.bot().id().find(&participant.bot_id);
            MatchParticipantTelemetry {
                bot_id: participant.bot_id,
                name: bot
                    .as_ref()
                    .map(|bot| bot.name.clone())
                    .unwrap_or_else(|| format!("bot-{}", participant.bot_id)),
                is_simulated: bot.as_ref().map(|bot| bot.is_simulated).unwrap_or(false),
                score: participant.score,
                balance: participant.balance,
                placement: placement(scores.iter().copied(), participant.score),
            }
        })
        .collect();

    let words = snapshot
        .my_word_plays
        .iter()
        .map(|play| MatchWordTelemetry {
            word: play.word.clone(),
            base: play.base_score,
            bonus: play.bonus,
            reward: play.total_reward,
            played_at_ms: micros_to_millis(play.played_at.to_micros_since_unix_epoch()),
        })
        .collect();

    let winning_bids = snapshot
        .results
        .iter()
        .map(|result| {
            let trace = live.and_then(|live| live.traces.get(&result.auction_id));
            let winner_name = result
                .winner_bot_id
                .and_then(|bot_id| conn.db.bot().id().find(&bot_id).map(|bot| bot.name));
            WinningBidTelemetry {
                auction_id: result.auction_id,
                round: trace.map(|trace| trace.round).unwrap_or(0),
                letter: result.letter.clone(),
                winner_bot_id: result.winner_bot_id,
                winner_name,
                paid: result.paid,
                our_bid: trace.map(|trace| trace.final_bid),
                won: result.winner_bot_id == Some(my_bot_id),
            }
        })
        .collect();

    let search_depths = live
        .map(|live| {
            live.traces
                .values()
                .filter_map(|trace| {
                    trace.search.as_ref().map(|search| SearchDepthTelemetry {
                        auction_id: trace.auction_id,
                        letter: trace.letter_str.clone(),
                        depth_completed: search.depth_completed,
                        elapsed_ms: search.elapsed_ms,
                        timed_out: search.timed_out,
                        fallback_used: search.fallback_used,
                        marginal: search.marginal,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let start_micros = snapshot
        .match_row
        .started_at
        .map(|time| time.to_micros_since_unix_epoch());
    let end_micros = snapshot
        .match_row
        .ended_at
        .map(|time| time.to_micros_since_unix_epoch());
    let started_at_ms = start_micros.map(micros_to_millis).unwrap_or_else(|| {
        live.map(|live| live.first_seen_ms)
            .unwrap_or_else(now_millis)
    });
    let ended_at_ms = end_micros.map(micros_to_millis).unwrap_or_else(now_millis);
    let elapsed_ms = match (start_micros, end_micros) {
        (Some(start), Some(end)) => elapsed_millis(start, end),
        _ => ended_at_ms.saturating_sub(started_at_ms),
    };

    telemetry.write(TelemetryRecord::MatchSummary(MatchSummary {
        match_id: snapshot.match_row.id,
        bot_id: my_bot_id,
        bot_name: bot_name.to_string(),
        strategy: config.strategy_name().to_string(),
        final_score: my_participant.score,
        placement: placement(scores.iter().copied(), my_participant.score),
        budget_start: 100,
        budget_end: my_participant.balance,
        participants,
        words,
        winning_bids,
        search_depths,
        reducer_errors: live.map(|live| live.reducer_errors).unwrap_or(0),
        elapsed_ms,
        started_at_ms,
        ended_at_ms,
    }));
}

fn search_telemetry(result: &SearchResult) -> SearchTelemetry {
    SearchTelemetry {
        depth_completed: result.depth_completed,
        elapsed_ms: result.elapsed_ms,
        states_evaluated: result.states_evaluated,
        value_without: result.value_without,
        value_with: result.value_with,
        marginal: result.marginal,
        timed_out: result.timed_out,
        fallback_used: result.fallback_used,
        retained_probability: result.retained_probability,
        best_now_without: result
            .best_now_without
            .as_ref()
            .map(|word| word.word.clone()),
        best_now_with: result.best_now_with.as_ref().map(|word| word.word.clone()),
    }
}

fn submit_bid(
    conn: &DbConnection,
    shared: &Arc<Shared>,
    match_id: u64,
    auction_id: u64,
    amount: i64,
) -> bool {
    let shared_for_callback = shared.clone();
    let result = conn
        .reducers
        .submit_bid_then(auction_id, amount, move |_ctx, res| {
            shared_for_callback.record_bid_ack(auction_id, amount);
            if let Err(message) = flatten_reducer_result(res) {
                shared_for_callback.record_error(
                    Some(match_id),
                    Some(auction_id),
                    None,
                    "bid",
                    message,
                );
            }
        });
    match result {
        Ok(()) => true,
        Err(err) => {
            shared.record_error(
                Some(match_id),
                Some(auction_id),
                None,
                "bid_send",
                format!("{err:?}"),
            );
            false
        }
    }
}

fn auction_kind(match_row: &Match) -> AuctionKind {
    match match_row.auction_type {
        AuctionType::Vickrey => AuctionKind::Vickrey,
        AuctionType::FirstPrice => AuctionKind::FirstPrice,
    }
}

fn apply_bid_floor(search_bid: i64, fallback_bid: i64) -> (i64, bool) {
    let final_bid = search_bid.max(fallback_bid);
    (final_bid, final_bid > search_bid)
}

fn is_benign_join_error(message: &str) -> bool {
    message.contains("already in a running match")
        || message.contains("already in this lobby")
        || message.contains("tournament")
}

fn auction_kind_name(kind: AuctionKind) -> &'static str {
    match kind {
        AuctionKind::Vickrey => "vickrey",
        AuctionKind::FirstPrice => "first_price",
    }
}

fn play_reason_name(reason: PlayReason) -> &'static str {
    match reason {
        PlayReason::GreedyBestNow => "greedy_best_now",
        PlayReason::TargetBeatsThreshold => "target_threshold",
        PlayReason::ForcedPlay => "forced",
        PlayReason::HoldForTarget => "hold",
        PlayReason::NoPlayableWord => "none",
        PlayReason::NoTarget => "no_target",
    }
}

fn remaining_auction_ms(auction: &Auction) -> i64 {
    let now = now_micros();
    let closes = auction.closes_at.to_micros_since_unix_epoch();
    (closes - now).max(0) / 1_000
}

fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_micros() as i64)
        .unwrap_or(0)
}

enum BootstrapState {
    Idle,
    CreateTeamSent(Instant),
    MintNonceSent(Instant),
    ClaimSent(Instant),
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacetimedb_sdk::Timestamp;

    #[test]
    fn bid_floor_keeps_the_greedy_fallback_when_search_is_lower() {
        assert_eq!(apply_bid_floor(0, 3), (3, true));
        assert_eq!(apply_bid_floor(2, 3), (3, true));
        assert_eq!(apply_bid_floor(8, 3), (8, false));
        assert_eq!(apply_bid_floor(0, 0), (0, false));
    }

    #[test]
    fn auction_trace_letter_is_an_index_not_an_ascii_byte() {
        let auction = Auction {
            id: 1,
            match_id: 1,
            letter: "Q".to_string(),
            opens_at: Timestamp::from_micros_since_unix_epoch(1_000_000),
            closes_at: Timestamp::from_micros_since_unix_epoch(2_000_000),
            status: AuctionStatus::Open,
        };
        let trace = AuctionTrace::new(&auction, 1, AuctionKind::Vickrey, 100);
        assert_eq!(trace.letter, 16);
        assert_eq!(trace.letter_str, "Q");
    }
}
