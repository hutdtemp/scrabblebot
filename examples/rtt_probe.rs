//! Measure the local SpacetimeDB reducer round-trip time without changing game
//! state. It submits bids for a deliberately unknown auction id; the module
//! rejects them, but the request/response path is the same one used for real
//! bids. Run with:
//!
//! cargo run --release --example rtt_probe -- --token-file /path/to/token

use std::sync::mpsc;
use std::time::{Duration, Instant};

use clap::Parser;
use scrabblebot_bot::module_bindings::*;
use spacetimedb_sdk::DbContext;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "http://127.0.0.1:3010")]
    uri: String,
    #[arg(long, default_value = "scrabblebot-local")]
    database: String,
    #[arg(long)]
    token_file: std::path::PathBuf,
    #[arg(long, default_value_t = 10)]
    probes: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let token = std::fs::read_to_string(&args.token_file)?;
    let conn = DbConnection::builder()
        .with_uri(args.uri.clone())
        .with_database_name(args.database.clone())
        .with_token(Some(token.trim().to_string()))
        .on_connect(|_conn, identity, _token| {
            eprintln!("connected as {}", identity.to_hex());
        })
        .on_connect_error(|_ctx, err| eprintln!("connect error: {err:?}"))
        .build()?;

    let (tx, rx) = mpsc::channel();
    let probes = args.probes.max(1);
    conn.subscription_builder()
        .on_applied(move |ctx| {
            for i in 0..probes {
                let start = Instant::now();
                let tx = tx.clone();
                // Unknown auction id: the reducer rejects it, but the
                // request/response path is identical to a real bid.
                let fake_auction_id = 9_000_000 + i as u64;
                let _ = ctx
                    .reducers
                    .submit_bid_then(fake_auction_id, 0, move |_ctx, result| {
                        let elapsed = start.elapsed();
                        let _ = tx.send((elapsed, format!("{result:?}")));
                    });
            }
        })
        .subscribe_to_all_tables();

    let mut samples = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while samples.len() < probes && Instant::now() < deadline {
        match conn.advance_one_message() {
            Ok(true) => {}
            Ok(false) => std::thread::sleep(Duration::from_millis(1)),
            Err(err) => anyhow::bail!("connection error: {err:?}"),
        }
        while let Ok((elapsed, result)) = rx.try_recv() {
            samples.push(elapsed.as_micros() as f64 / 1000.0);
            eprintln!(
                "probe {}: {:.1} ms ({result})",
                samples.len(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
    if samples.is_empty() {
        anyhow::bail!("no reducer responses within 10s");
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let avg = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let max = *samples.last().unwrap();
    println!(
        "reducer_rtt_ms: n={} avg={avg:.1} p50={p50:.1} max={max:.1}",
        samples.len()
    );
    Ok(())
}
