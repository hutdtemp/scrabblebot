# Local viability check — 2026-11-07

## Setup and scope

- Original upstream game revision: `b5ccb36aa238ad16692f8d6af54da15ef9f11324`.
- Official server image: `clockworklabs/spacetime:v2.12.0`, digest
  `sha256:a47a9ac98ca69f5a82d727dfa0d9dfe34ede8c2335c8cdd30e95b9499f4e46f8`.
- Isolated local endpoint `http://127.0.0.1:3010`, database `scrabblebot-local`.
- Original upstream source and rules unchanged; default six players, five
  starting tiles, 100 balance, Vickrey auctions, 1,000 ms auction duration.
- Each game contained `beam-local` (search), `greedy-local` (baseline), and
  four of the upstream built-in simulated bots. Both strategies faced the
  same public auction sequence but different private starting racks.
- These are four unseeded shared games, not deterministic replays or a large
  statistical benchmark. Simulator lineups varied. No Maincloud play or
  deployment was performed.

## Scores verified against the server

| Local match | Search score | Place / 6 | Greedy score | Place / 6 |
|---|---:|---:|---:|---:|
| 21 | 42 | 2 | 21 | 4 |
| 22 | 44 | 2 | 11 | 4 |
| 23 | 49 | 2 | 14 | 4 |
| 25 | **112** | **1** | 59 | 2 |
| Mean | **61.75** | | **26.25** | |

The search bot beat this simple baseline in all four shared games and won the
last table. It did not beat the strongest simulated opponent in the first
three. Match 25 had a weaker simulated lineup, so its large score is not
attributable solely to search quality.

Best final-source game: 39 auction wins and 15 word plays, ending with balance
72. Words and rewards: XYLAN 30, GO 3, BY 7, LI 2, WAUR 10, IS 2, IT 2, REV 6,
EAU 3, AW 5, SEI 3, DE 3, DE 3, CARL 9, BRACH 24.

All eight final score/balance pairs matched authoritative `match_participant`
SQL. For each summary, word rewards summed to the reported score and
`100 + rewards - auction payments` equaled the final balance. Both strategies
reported **zero reducer errors**.

## Search and timing

Settings: beam width 128, exhaustive depth 2, maximum depth 12, search budget
800 ms, submission margin 150 ms. A valid face-value-based fallback is sent
immediately; the asynchronous search may raise it before the deadline.

In the first three search matches:

- 201 per-auction search records (see reporting limitation below).
- Completed depth 1–12; median depth 10.
- Search elapsed time: median 803 ms, p95 819 ms, max 828 ms. The configured
  budget is checked periodically, not a preemptive hard real-time limit.
- 32 search-raised bids; latest acknowledgment was 836 ms after auction open.
- These raised-bid acknowledgments had at least 165 ms until the observed
  server settlement, and none produced an expired-auction reducer error.
- 15 stale and 4 late search outputs were rejected; their early fallback bids
  remained valid. These are discarded refinements, not missing initial bids.
- Median retained probability mass was about 0.000866 (0.087%). The deep beam
  often falls back on its safe baseline for most future probability mass.
  Increasing depth alone is therefore not a reliable quality improvement.

The hybrid default includes target-aware word holding and a greedy bid floor.
The baseline uses immediate word play and face-value-plus-premium bids.
This comparison does **not** isolate the causal benefit of beam search from
holding policy, bid-floor behavior, random starting racks, or opponent mix.

## Reporting regression and final-source check

The first three games ran the handed-back policy. Independent verification
caught that each JSONL stream recorded 67 auctions although the authoritative
summary contained 68: the terminal auction and match-end status arrive in the
same transaction, bypassing the running-match logger. Scores and accounting
were unaffected.

A retained verifier reproduced the failure:

```text
Error: match 21 is missing auction telemetry: [1428]
```

The coordinator flushed auction results before summarizing an ended match,
cleaned up Clippy findings without changing the bidding/playback policy, then
rebuilt and replayed both strategies in match 25. Both runs exited successfully
and the verifier confirmed all 68 auctions for each bot:

```sh
cargo run --example verify_telemetry -- /path/to/run.jsonl
```

Thus the final-source live acceptance covers match 25; matches 21–23 are
additional policy evidence from before the reporting-only repair.

## Build and safety checks

- `cargo fmt --all -- --check`
- `cargo clippy --all-targets --locked -- -D warnings`
- `cargo test --locked`: 52 unit tests plus 1 bundled-dictionary integration test
- `cargo check --all-targets --locked`
- `cargo build --release --locked`
- Explicit non-loopback URI rejected before connection without `--allow-remote`.
- Incomplete local bootstrap configuration rejected.
- Dictionary SHA-256 matched the unmodified upstream wordlist byte-for-byte.
- All 141 generated binding files matched independent schema regeneration
  byte-for-byte after the project's Rust 2024 formatting; regeneration script
  also executed successfully.
- No credentials or local telemetry files are committed.

Forced network reconnect, first-price live play, and tournament registration
were not exercised by this acceptance run. The implementation has reconnect
logic and first-price shading, but those are not claimed as independently
live-verified here.

## Reproduce the match runs

With the original upstream local server running and two separate locally
provisioned bot credential files, run these concurrently so both enter the
same rolling lobby:

```sh
cargo build --release --locked

./target/release/scrabblebot-bot --token-file /path/to/search.token \
  --bot-name beam-local --matches 3 --telemetry /tmp/search.jsonl

./target/release/scrabblebot-bot --token-file /path/to/greedy.token \
  --bot-name greedy-local --strategy greedy --matches 3 \
  --telemetry /tmp/greedy.jsonl
```

Names used for logs do not select a server identity: each token must actually
be a credential for a distinct local bot. Do not reuse production credentials.
The RNG and simulator seating are not seeded, so exact scores will differ.
