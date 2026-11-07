//! Check that every authoritative auction in each ended match has a JSONL record.
//! Usage: cargo run --example verify_telemetry -- /path/to/run.jsonl

use std::collections::HashSet;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, ensure};
use scrabblebot_bot::core::telemetry::{Envelope, TelemetryRecord};

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("provide a telemetry JSONL path")?;
    let input = std::fs::File::open(&path).with_context(|| format!("opening {path}"))?;
    let mut auctions = HashSet::new();
    let mut summaries = Vec::new();
    for (index, line) in BufReader::new(input).lines().enumerate() {
        let envelope: Envelope =
            serde_json::from_str(&line?).with_context(|| format!("parsing line {}", index + 1))?;
        match envelope.record {
            TelemetryRecord::Auction(auction) => {
                auctions.insert((auction.match_id, auction.auction_id));
            }
            TelemetryRecord::MatchSummary(summary) => summaries.push(summary),
            _ => {}
        }
    }
    ensure!(!summaries.is_empty(), "no completed matches to verify");
    for summary in &summaries {
        let missing: Vec<_> = summary
            .winning_bids
            .iter()
            .filter(|bid| !auctions.contains(&(summary.match_id, bid.auction_id)))
            .map(|bid| bid.auction_id)
            .collect();
        ensure!(
            missing.is_empty(),
            "match {} is missing auction telemetry: {:?}",
            summary.match_id,
            missing
        );
        println!(
            "match {}: all {} auctions recorded",
            summary.match_id,
            summary.winning_bids.len()
        );
    }
    Ok(())
}
