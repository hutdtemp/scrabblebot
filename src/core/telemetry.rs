//! Machine-readable JSONL telemetry.
//!
//! One JSON object per line. Every line carries `"schema": 1` and a `"type"`.
//! The live client writes an `auction` record when an auction result is known,
//! a `word` record for each word this bot plays, a `match_summary` when a match
//! ends, and a `run_summary` at exit.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const TELEMETRY_SCHEMA: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Envelope {
    pub schema: u32,
    #[serde(flatten)]
    pub record: TelemetryRecord,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TelemetryRecord {
    RunStart(RunStart),
    Auction(Box<AuctionTelemetry>),
    Word(WordTelemetry),
    MatchSummary(MatchSummary),
    RunSummary(RunSummary),
    ReducerError(ReducerErrorTelemetry),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchTelemetry {
    pub depth_completed: u8,
    pub elapsed_ms: u64,
    pub states_evaluated: u64,
    pub value_without: f64,
    pub value_with: f64,
    pub marginal: f64,
    pub timed_out: bool,
    pub fallback_used: bool,
    pub retained_probability: f64,
    pub best_now_without: Option<String>,
    pub best_now_with: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunStart {
    pub bot_name: String,
    pub bot_id: Option<u64>,
    pub strategy: String,
    pub playback: String,
    pub uri: String,
    pub database: String,
    pub matches_target: usize,
    pub beam_width: usize,
    pub max_depth: u8,
    pub exact_depth: u8,
    pub search_max_time_ms: u64,
    pub started_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuctionTelemetry {
    pub match_id: u64,
    pub auction_id: u64,
    pub round: u32,
    pub letter: String,
    pub auction_kind: String,
    pub balance_before: i64,
    pub fallback_bid: i64,
    pub search_bid: Option<i64>,
    pub floor_applied: bool,
    pub final_bid: i64,
    pub submitted: bool,
    pub search: Option<SearchTelemetry>,
    pub winner_bot_id: Option<u64>,
    pub winner_name: Option<String>,
    pub paid: i64,
    pub won: bool,
    pub late_search: bool,
    pub stale: bool,
    pub reducer_error: Option<String>,
    pub auction_open_at_ms: Option<u64>,
    pub search_started_at_ms: Option<u64>,
    pub search_start_delay_ms: Option<u64>,
    pub search_budget_ms: Option<u64>,
    pub bid_submitted_at_ms: Option<u64>,
    pub bid_ack_at_ms: Option<u64>,
    pub bid_ack_amount: Option<i64>,
    pub bid_ack_latency_ms: Option<u64>,
    pub time_left_after_ack_ms: Option<u64>,
    pub total_to_ack_ms: Option<u64>,
    pub result_at_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WordTelemetry {
    pub match_id: u64,
    pub word: String,
    pub base: i64,
    pub bonus: i64,
    pub reward: i64,
    pub balance_after: i64,
    pub score_after: i64,
    pub source: String,
    pub reducer_error: Option<String>,
    pub played_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchParticipantTelemetry {
    pub bot_id: u64,
    pub name: String,
    pub is_simulated: bool,
    pub score: i64,
    pub balance: i64,
    pub placement: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchWordTelemetry {
    pub word: String,
    pub base: i64,
    pub bonus: i64,
    pub reward: i64,
    pub played_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WinningBidTelemetry {
    pub auction_id: u64,
    pub round: u32,
    pub letter: String,
    pub winner_bot_id: Option<u64>,
    pub winner_name: Option<String>,
    pub paid: i64,
    pub our_bid: Option<i64>,
    pub won: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchDepthTelemetry {
    pub auction_id: u64,
    pub letter: String,
    pub depth_completed: u8,
    pub elapsed_ms: u64,
    pub timed_out: bool,
    pub fallback_used: bool,
    pub marginal: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MatchSummary {
    pub match_id: u64,
    pub bot_id: u64,
    pub bot_name: String,
    pub strategy: String,
    pub final_score: i64,
    pub placement: usize,
    pub budget_start: i64,
    pub budget_end: i64,
    pub participants: Vec<MatchParticipantTelemetry>,
    pub words: Vec<MatchWordTelemetry>,
    pub winning_bids: Vec<WinningBidTelemetry>,
    pub search_depths: Vec<SearchDepthTelemetry>,
    pub reducer_errors: u64,
    pub elapsed_ms: u64,
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunSummary {
    pub matches_played: usize,
    pub matches_target: usize,
    pub strategy: String,
    pub elapsed_ms: u64,
    pub reducer_errors: u64,
    pub telemetry_path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReducerErrorTelemetry {
    pub match_id: Option<u64>,
    pub auction_id: Option<u64>,
    pub word: Option<String>,
    pub action: String,
    pub error: String,
    pub at_ms: u64,
}

impl Envelope {
    pub fn new(record: TelemetryRecord) -> Self {
        Self {
            schema: TELEMETRY_SCHEMA,
            record,
        }
    }
}

pub struct TelemetryWriter {
    out: Box<dyn Write + Send>,
    path: Option<PathBuf>,
    write_errors: u64,
}

impl TelemetryWriter {
    pub fn file(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            out: Box::new(BufWriter::new(file)),
            path: Some(path.to_path_buf()),
            write_errors: 0,
        })
    }

    pub fn stdout() -> Self {
        Self {
            out: Box::new(BufWriter::new(std::io::stdout())),
            path: None,
            write_errors: 0,
        }
    }

    pub fn disabled() -> Self {
        Self {
            out: Box::new(std::io::sink()),
            path: None,
            write_errors: 0,
        }
    }

    /// Test/diagnostic constructor.
    pub fn from_writer(out: Box<dyn Write + Send>) -> Self {
        Self {
            out,
            path: None,
            write_errors: 0,
        }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn write_errors(&self) -> u64 {
        self.write_errors
    }

    pub fn write(&mut self, record: TelemetryRecord) {
        let envelope = Envelope::new(record);
        match serde_json::to_vec(&envelope) {
            Ok(mut bytes) => {
                bytes.push(b'\n');
                if let Err(err) = self.out.write_all(&bytes).and_then(|_| self.out.flush()) {
                    self.write_errors += 1;
                    eprintln!("telemetry write failed: {err}");
                }
            }
            Err(err) => {
                self.write_errors += 1;
                eprintln!("telemetry serialization failed: {err}");
            }
        }
    }
}

pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl Write for SharedBuffer {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn auction_record_is_tagged_jsonl() {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let mut writer = TelemetryWriter::from_writer(Box::new(SharedBuffer(buf.clone())));
        writer.write(TelemetryRecord::Auction(Box::new(AuctionTelemetry {
            match_id: 1,
            auction_id: 2,
            round: 3,
            letter: "E".into(),
            auction_kind: "vickrey".into(),
            balance_before: 100,
            fallback_bid: 3,
            search_bid: Some(7),
            floor_applied: false,
            final_bid: 7,
            submitted: true,
            search: Some(SearchTelemetry {
                depth_completed: 4,
                elapsed_ms: 80,
                states_evaluated: 1234,
                value_without: 10.0,
                value_with: 18.0,
                marginal: 8.0,
                timed_out: false,
                fallback_used: false,
                retained_probability: 0.98,
                best_now_without: Some("QI".into()),
                best_now_with: Some("QUIZ".into()),
            }),
            winner_bot_id: None,
            winner_name: None,
            paid: 0,
            won: false,
            late_search: false,
            stale: false,
            reducer_error: None,
            auction_open_at_ms: Some(90),
            search_started_at_ms: Some(95),
            search_start_delay_ms: Some(5),
            search_budget_ms: Some(800),
            bid_submitted_at_ms: Some(100),
            bid_ack_at_ms: Some(102),
            bid_ack_amount: Some(7),
            bid_ack_latency_ms: Some(2),
            time_left_after_ack_ms: Some(98),
            total_to_ack_ms: Some(12),
            result_at_ms: Some(200),
        })));
        let text = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(text.ends_with('\n'));
        let value: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!(value["schema"], 1);
        assert_eq!(value["type"], "auction");
        assert_eq!(value["letter"], "E");
        assert_eq!(value["search"]["depth_completed"], 4);
    }

    #[test]
    fn match_summary_round_trips() {
        let summary = MatchSummary {
            match_id: 9,
            bot_id: 4,
            bot_name: "test".into(),
            strategy: "search".into(),
            final_score: 250,
            placement: 2,
            budget_start: 100,
            budget_end: 40,
            participants: vec![MatchParticipantTelemetry {
                bot_id: 4,
                name: "test".into(),
                is_simulated: false,
                score: 250,
                balance: 40,
                placement: 2,
            }],
            words: vec![MatchWordTelemetry {
                word: "QUIZ".into(),
                base: 22,
                bonus: 11,
                reward: 33,
                played_at_ms: 123,
            }],
            winning_bids: vec![WinningBidTelemetry {
                auction_id: 1,
                round: 1,
                letter: "Q".into(),
                winner_bot_id: Some(4),
                winner_name: Some("test".into()),
                paid: 5,
                our_bid: Some(7),
                won: true,
            }],
            search_depths: vec![SearchDepthTelemetry {
                auction_id: 1,
                letter: "Q".into(),
                depth_completed: 3,
                elapsed_ms: 90,
                timed_out: false,
                fallback_used: false,
                marginal: 8.0,
            }],
            reducer_errors: 0,
            elapsed_ms: 60_000,
            started_at_ms: 1,
            ended_at_ms: 60_001,
        };
        let json =
            serde_json::to_string(&Envelope::new(TelemetryRecord::MatchSummary(summary))).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["type"], "match_summary");
        assert_eq!(value["final_score"], 250);
        assert_eq!(value["words"][0]["word"], "QUIZ");
    }
}
