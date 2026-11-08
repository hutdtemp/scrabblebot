# scrabblebot-bot

A Rust-only bot for [Clockwork Labs' ScrabbleBot](https://github.com/clockworklabs/scrabblebot).
It plays the upstream 1-second sealed-bid letter auctions, builds a private
rack, and plays dictionary words for reward.

> **Recommended: `--strategy evolved --genome genomes/v4.rl.json`** (the RL
> policy deployed for the tournament). `--genome` also accepts an evolved
> genome such as `genomes/ga7_latest.json`. Both are trained in an offline
> simulator against Maincloud-derived opponents; see [TRAINING.md](TRAINING.md). The rest of this README describes the
> original search baseline.

The implementation is deliberately a **reasonable baseline**, not an ML
project:

- exact ENABLE dictionary validity and exact upstream scoring;
- an adaptive-depth, probability-weighted beam search over hypothetical future
  bag draws for auction valuation;
- greedy search-derived bidding with a safe greedy fallback floor;
- greedily submitted words, with a target-aware option that can hold a small
  word for a clearly better near-future one;
- a no-lookahead `--strategy greedy` baseline for honest comparison;
- JSONL per-auction and per-match telemetry;
- local-only defaults, explicit token file/env, and bounded `--matches N`.

## Rules this bot assumes

These are the upstream module's actual rules, not a reimplementation:

- 98-tile standard Scrabble bag, no blanks.
- 6-player matches by default; each bot starts with 5 tiles and 100 balance.
- One letter is revealed per round and auctioned for 1 second.
- Vickrey by default: the winner pays the runner-up's bid (reserve 1).
  First-price is configurable per lobby.
- A word may be played at any time while the match is running, must be in the
  dictionary, and must be at least 2 letters.
- `reward = floor(base_score * length_multiplier)`; the reward is added to both
  balance and score. Multipliers: 1x for <=3 letters, 1.5x at 4, 2x at 5,
  2.5x at 6, 3x at 7+.
- The bag only shrinks when an auction draws a letter. A letter nobody wins is
  lost; word plays do not change the bag. This follows `closeAuction` in the
  pinned upstream source; its bot guide currently says unbid letters return,
  but that is not what this source revision implements.
- `my_rack` is the only rack visibility; opponents' racks and the bag
  composition are not exposed.

## Quick start (local only)

Defaults are:

```text
URI:      http://127.0.0.1:3010
Database: scrabblebot-local
```

Non-loopback URIs are refused unless `--allow-remote` is passed. This bot never
targets Maincloud implicitly and never publishes anything.

```sh
# Build and test
cargo build --release
cargo test

# One bounded local match with a coordinator-provided token file.
# Never commit the token file or paste its contents into logs.
cargo run --release -- \
  --token-file /path/to/beam-local.token \
  --bot-name beam-local \
  --matches 1 \
  --telemetry /tmp/telemetry.jsonl

# No-lookahead baseline
cargo run --release -- \
  --token-file /path/to/beam-local.token \
  --bot-name beam-local \
  --strategy greedy \
  --matches 1 \
  --telemetry /tmp/greedy.jsonl
```

`--matches 0` means unlimited; the default is 1. `--match-timeout-secs 0`
disables the no-progress timeout. `--telemetry -` writes JSONL to stdout;
`--telemetry none` disables it.

Useful diagnostics:

```sh
cargo run --quiet -- --help
cargo run --quiet -- --print-config --token-file /path/to/token
cargo run --release --example rtt_probe -- --token-file /path/to/token
cargo run --release --example search_bench
```

`rtt_probe` submits bids for deliberately unknown auction ids and measures the
same reducer request/response path without changing game state. `search_bench`
prints search depth, elapsed time, and values for a few realistic racks.

## Build, bindings, and reproducibility

- Rust: `rust-version = "1.97"` (the SpacetimeDB SDK 2.12.0 requires it).
- SpacetimeDB SDK: pinned to `=2.12.0`.
- Bindings: generated from the upstream module schema with the official
  SpacetimeDB CLI 2.12.0 Docker image, not the old host CLI.
- `Cargo.lock` is tracked because this is a binary.
- `src/module_bindings/` is tracked so a clean checkout builds without a
  running server or a codegen step. Regenerate only when the upstream schema
  changes.

Preferred regeneration route (coordinator's isolated container):

```sh
CONTAINER=scrabble-local-e99a53b4 ./scripts/generate-bindings.sh
```

The script also supports a standalone official image if the upstream module's
`dist/bundle.js` exists:

```sh
UPSTREAM=/path/to/clockworklabs/scrabblebot ./scripts/generate-bindings.sh
```

The container's CLI cannot run `rustfmt`; the generated files are unformatted
but compile. The bindings header records the CLI version.

## Token and credential handling

The token is read from the first available source:

1. `--token-file` (default `.token-<bot-name>`);
2. the environment variable named by `--token-env` (default
   `SPACETIMEDB_TOKEN`).

The token is never logged, printed by `--print-config`, or written to
telemetry. If the bot connects anonymously, the issued token is saved to the
token file with mode `0600` so a reconnect keeps the same identity.

Local provisioning options:

```sh
# Redeem a coordinator-minted nonce (fresh local identity)
cargo run --release -- --claim-nonce CODE --matches 1 ...

# Self-bootstrap a fresh local team and bot, then mint/redeem its own nonce
cargo run --release -- \
  --bootstrap-team my-local-team --bootstrap-bot my-local-bot \
  --matches 1 ...
```

Bootstrap is local-only and intentionally small; it is not a provisioning
system. If the identity already has a `BotCredential`, bootstrap is skipped.

## Design

### 1. Exact dictionary solver

`data/enable_wordlist.txt` is the upstream ENABLE list bundled into the binary
(172,820 words, SHA-256 in `THIRD_PARTY_NOTICES.md`).

`Dictionary::best_word(rack)` returns the playable word with the highest exact
module reward, tie-broken by longer length and then lexicographically. It uses
one of two exact strategies:

- for small racks, enumerate every sub-multiset and look it up in a
  letter-count signature map;
- for large racks, scan words in descending reward order and return the first
  one whose letters fit.

`greedy_partition(rack)` repeatedly plays the best word and removes its
letters. This is the value function used by the search and playback. It is
**greedy, not an exact set-packing optimum**; that limitation is deliberate.

### 2. Bag model

The bag composition is private. The bot reconstructs a model from public data:

```text
bag = full 98-tile bag
    - my initial rack
    - every auction letter observed (won or lost)
    - a proportional share of the hidden opponents' starting deal
```

The proportional subtraction is reconciled so the counts sum exactly to the
authoritative `match.bag_total`. Word plays do not change the bag. The hidden
opponent starting deal is an estimate, not an oracle.

After a reconnect, the initial rack is reconstructed from authoritative state:

```text
initial = current + words I played - auctions I won
```

This works even when a won letter was later played.

### 3. Adaptive-depth beam search

At an auction, the auctioned letter `L` is already out of the bag whether or not
the bot wins it. The search compares two racks against the **same** future bag:

```text
V_without = value(rack)
V_with    = value(rack + L)
marginal  = V_with - V_without
```

Future draws are hypothetical free additions to our rack. This estimates
letter-completion opportunities, **not** the expected score under actual
competition: future acquisition prices and the chance of losing those auctions
are deliberately outside this initial solver's model.

For a horizon of `d` future draws, it enumerates or beams over multisets of
drawn letters with exact multivariate-hypergeometric probabilities. Equivalent
rack states are merged by summing their probability mass. Each state is valued
by greedy liquidation, and the expected value is probability-weighted. A rare
lucky future cannot masquerade as expected value.

Pruning keeps states by `probability * (value_without + value_with)`. Pruned
probability mass is assigned the safe immediate-liquidation fallback, never the
best pruned future. Depths are attempted in increasing order:

- depths up to `--exact-depth` are exhaustive;
- deeper depths use `--beam-width` states;
- an incomplete depth is discarded;
- the last completed depth is kept;
- if no depth completes, the immediate-liquidation fallback is used.

`--max-depth` is a safety cap; the deadline is the primary limit.

### 4. Bidding

The default search strategy bids a shaded fraction of the marginal value:

- Vickrey: 0.9 of marginal (near truth-telling);
- first-price: 0.7 of marginal (shading for the winner's curse);
- minimum marginal to bid: 0.5;
- always capped by the authoritative balance.

A greedy fallback bid is submitted immediately when an auction opens:

```text
face value + 2 if the letter is not already held, otherwise face value
```

The fallback is a **floor**, not a throwaway. Greedy liquidation can report zero
marginal value for a letter that is not yet part of a complete word; retracting
the fallback to zero in that case makes the bot needlessly passive. The search
can raise the bid above the floor, but does not lower it below the floor. This
is a deliberate hybrid baseline and is visible in telemetry
(`fallback_bid`, `search_bid`, `floor_applied`, `final_bid`).

### 5. Playback

`--playback target` (default for search) compares:

- the best currently playable word;
- the best word that is at most the internal `target_max_missing`
  (default 2) letters away.

It holds only when the current word is worth less than the internal
`target_ratio` (default 0.5) times the future target's reward. It force-plays
when:

- `bag_total <= 2`, or
- the bot has held for the internal `target_hold_limit` (default 2) auctions.

Once the target becomes playable, it is the best current word (or a better one),
so it is submitted greedily. `--playback greedy` always plays the best current
word and is used automatically by `--strategy greedy`.

### 6. Live lifecycle and safety

- Subscribes to all public tables/views, including `my_rack`.
- Rebuilds match state from the authoritative client cache on every dirty
  table change, not from optimistic local mutations.
- Filters `my_rack` by `match_id` and `bot_id`; never reads another bot's rack.
- Rejects stale search results by auction id, rack version, auction status,
  and current-auction id before any reducer call.
- Submits a fallback bid early; search updates are only sent while the auction
  is still open and the submit margin remains.
- Waits for authoritative `word_play`/`my_rack` updates after a word
  submission; does not optimistically spend letters. A 1.5 s timeout clears a
  lost pending word.
- Records reducer errors and clears the matching pending word/bid state.
- Handles initial subscription population, rack insert/update/delete, match
  end, and rejoin.
- On reconnect, uses the saved token, rebuilds from the client cache, and does
  not count historical ended matches against `--matches`.
- Searches run on background threads; the SDK event loop is never blocked by a
  search.

## Timing: targeting ~0.8 s of the 1 s auction

The measured local reducer round-trip (10 probes against the local Docker
server, `rtt_probe`) was:

```text
n=10 avg=2.1 ms p50=2.2 ms max=2.2 ms
```

The live client computes the search budget as:

```text
budget = min(remaining_auction_ms - submit_margin_ms - 20, search_max_time_ms)
```

Defaults:

```text
search_max_time_ms = 800
submit_margin_ms   = 150
```

So the search typically ends around 0.8 s after the auction opens, and the bid
is acknowledged with roughly 150-200 ms left before the 1 s close. The raw
reducer round-trip is ~2 ms; the live ack latency including the main loop's
poll is 6-9 ms. The search itself checks its deadline during expansion and
state evaluation, discards an incomplete depth, and keeps the last completed
depth.

In the local smoke match, the two auctions where the full-budget search bid was
submitted recorded:

```text
search.elapsed_ms       = 801-804
search_budget_ms        = 800
bid_ack_latency_ms      = 6-9
total_to_ack_ms         = 822-829  (avg 825.5)
time_left_after_ack_ms  = 171-179  (avg 176)
```

The 6-9 ms live ack latency is the reducer RTT plus the main loop's 1-2 ms poll
and search-result delivery; the raw reducer RTT probe measured ~2 ms. The
measured 800 ms cap therefore lands at ~0.82-0.83 s from auction open to ack;
use `--search-max-time-ms 780` if you want the total closer to 0.80 s. Later
auctions completed their configured max depth early and therefore used less
than the full budget. `--max-depth` now defaults to 12 so the deadline, rather
than an early depth cap, is the usual binding constraint.

For a remote server, increase `--submit-margin-ms` and/or reduce
`--search-max-time-ms` based on measured RTT. The defaults are tuned for the
local loopback server.

## Telemetry

JSONL, one object per line. Every line has `"schema": 1` and a `"type"`:

- `run_start` — bot identity, strategy, search config, target match count;
- `auction` — letter, fallback/search/final bid, floor flag, search depth,
  elapsed, values, winner, paid, reducer error, and timing fields:
  `auction_open_at_ms`, `search_started_at_ms`, `search_start_delay_ms`,
  `search_budget_ms`, `bid_submitted_at_ms`, `bid_ack_at_ms`,
  `bid_ack_amount`, `bid_ack_latency_ms`, `time_left_after_ack_ms`,
  `total_to_ack_ms`, `result_at_ms`;
- `word` — word, base, bonus, reward, source, reducer error;
- `match_summary` — final score, placement, budget start/end, participants,
  words, winning bids, per-auction search depths, reducer errors, elapsed;
- `reducer_error` — action, match/auction/word, error text;
- `run_summary` — matches played, total elapsed, reducer errors, telemetry path.

Example timing extraction:

```sh
jq -r 'select(.type=="auction") |
  [.auction_id, .letter, .search.depth_completed, .search.elapsed_ms,
   .search_budget_ms, .bid_ack_latency_ms, .time_left_after_ack_ms,
   .total_to_ack_ms, .fallback_bid, .search_bid, .floor_applied,
   .final_bid, .won] | @tsv' telemetry.jsonl
```

Example match summary extraction:

```sh
jq -r 'select(.type=="match_summary") |
  [.match_id, .final_score, .placement, .budget_end, .elapsed_ms,
   .reducer_errors, (.words|length), (.winning_bids|length)] | @tsv' telemetry.jsonl
```

## Verification performed

- `cargo test`: 52 unit tests plus 1 bundled-dictionary integration test.
- `cargo check --all-targets`.
- `cargo run --release --example rtt_probe`: local reducer RTT measured as
  above.
- Two bounded local smoke matches against the coordinator's `beam-local`
  credential and unchanged simulated opponents:
  - match 9: final score 16, placement 3, 0 reducer errors;
  - match 13: final score 24, placement 3, 0 reducer errors, with the 800 ms
    search/timing path exercised.
- These are smoke checks, not seeded or paired benchmarks. The upstream RNG is
  not deterministic merely because the server is local; the parent should run
  several bounded matches for score/timing comparisons.

Bugs found and fixed during the smoke work:

- an auction letter was initially treated as an ASCII byte instead of a 0..25
  index, causing a panic; fixed and covered by
  `auction_trace_letter_is_an_index_not_an_ascii_byte`;
- a fresh process summarized a historical ended match from the client cache and
  consumed the `--matches` budget; fixed by only counting matches actually
  observed running (`seen_running`).

## Known limitations

- `greedy_partition` is greedy, not an exact optimal partition of the rack.
- The bag model estimates the hidden opponents' starting deal proportionally;
  it is not a Bayesian posterior.
- The search is a finite-horizon beam approximation, not a full-game expected
  value. Beam pruning can miss rare high-value futures. At deep horizons a
  narrow beam can retain very little probability mass, so the fallback
  dominates and greater depth is not necessarily a better estimate. Telemetry
  exposes `retained_probability` for this reason.
- First-price shading (0.7) and the Vickrey fraction (0.9) are heuristics.
- Target-aware playback can hold a small word and lose it if the target never
  arrives; the bag/depth force conditions bound this, but it is not optimal.
- The fallback bid floor can overpay relative to the search's marginal value;
  it is a deliberate safe baseline.
- No tournament registration; the bot only uses the rolling lobby.
- No ML/LSTM, no opponent modeling beyond public auctions/word plays.
- In-flight search traces are lost across a reconnect; per-auction telemetry
  already written before the disconnect remains.
- Auction deadlines use local wall-clock time against server timestamps; the
  defaults assume the bot and server clocks are roughly synchronized (the
  intended local loopback deployment).
- The local Docker server and token are coordinator-owned; this repository does
  not contain credentials.

## Local acceptance results

The coordinator independently tested the final policy in four shared,
unseeded six-player matches. Search scores were **42, 44, 49, 112** (mean
61.75); the no-lookahead baseline scored **21, 11, 14, 59** (mean 26.25).
The final-source match scored 112 and won its table. All eight bot/match
summaries were checked against authoritative server SQL and word/bid
accounting; no reducer errors occurred.

See [BENCHMARK.md](BENCHMARK.md) for the exact setup, reporting-fix boundary,
timing observations, and comparison limits. This is an initial viability
check, not a claim that beam search alone caused the improvement.

Validate complete final-auction reporting on a finished run with:

```sh
cargo run --example verify_telemetry -- /path/to/run.jsonl
```

## License and attribution

Our code is MIT (see `LICENSE`). Upstream ScrabbleBot is MIT (Copyright (c)
2026 Clockwork Labs); the ENABLE wordlist is public domain. See
`THIRD_PARTY_NOTICES.md` and `LICENSES/`.
