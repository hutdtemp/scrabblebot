# Simulator, evolved policy, and training

The original search bot scored 16–112 locally because it played short 1x
words immediately and finished matches with unspent balance. Maincloud data
shows the strong bots do the opposite: about 60% of their words are exactly
7 letters (3x). This subsystem trains a policy for that game offline.

## Pieces

| Path | What it is |
|---|---|
| `tools/maincloud_dump.py` | Read-only dump of Maincloud's public tables (anonymous identity, HTTP SQL). Resumable. Output in `data/maincloud/` (gitignored). |
| `tools/analyze_maincloud.py` | Per-bot summaries and `data/maincloud/opponent_model.json` (strong bots' winning-bid quantiles per letter and phase). |
| `src/sim/engine.rs` | Headless match engine mirroring the upstream `closeAuction`, built-in bot bidding and word play, and the end rule (the final auction's letter is never playable). |
| `src/sim/words.rs` | Word index. `analyze()` makes one pass that returns the best word now, after any one extra letter, and after the auction letter plus any one more. Tested against brute force on the full dictionary. |
| `src/sim/policy.rs` | Evolvable policy used both in simulation and live: a linear play head, plus a base bid with a 54→16→1 tanh MLP correction (~920 parameters). |
| `src/sim/agents.rs` | Outside opponents: the three upstream built-in bot types, the old greedy bot, a big spender, and an empirical opponent sampled from Maincloud bid data. |
| `src/sim/ga.rs`, `src/bin/train.rs` | Genetic algorithm, test panel, and CLI. |

## Training design

- **Rooms:** 3 current candidates, 1 old winner and 2 outside opponents.
  Old winners are drawn from every past generation, favoring recent ones.
  The outside opponents come from beyond the population's own lineage, which
  keeps self-play grounded.
- **Fitness:** the fraction of opponents out-scored, so clones gain nothing by
  colluding on low bids.
- **Selection:** elitism, tournament selection, block-wise crossover (whole
  hidden units move together), and Gaussian mutation with a self-adapting
  step size.
- **Frozen panel:** fixed rooms with fixed seeds, never used for selection.
  - `maincloud`: 3 empirical opponents and 2 built-in bots.
  - `sims`: 5 built-in bots.
  - `local-test`: the old greedy bot and 4 built-in bots.
- **Shipping:** `train select` re-scores candidates on fresh seeds to remove
  the selection bias of picking the best panel result.

The simulator is CPU-bound, at about 400 matches/s on 24 cores. Policy
inference is a ~920-parameter MLP run once per auction, so a GPU would not
speed anything up.

## Calibrated profile (`--profile realmix`)

After 17 real Maincloud matches, we turned letters into points as well as the
match winners did (5.5 vs 5.6 points per letter used). We lost on volume: we
won 1.3 fewer letters and finished with 8 more unspent. The training market
had cleared at about 10 per letter, while real full rooms clear at about 12.3.

`realmix` samples rooms in the recent Maincloud mix:

| Share | Room |
|---:|---|
| 50% | 6 real bots: 2 candidates, old winners or the deployed model, and 2 empirical opponents |
| 40% | 3 real bots plus 3 built-in sims |
| 10% | 2 real bots plus 4 built-in sims |

The deployed model (`genomes/panel_v1.json`) is also the frozen competent
opponent on the panel. A simulated full room of that model clears at about
13 per letter, close to reality. Our live score was 93 and the model scores
94 in a simulated full room. Models are chosen on weighted mean place:
0.55 x full room + 0.45 x three-real room.

## Commands

```sh
python3 -I tools/maincloud_dump.py            # refresh data/maincloud/
python3 -I tools/analyze_maincloud.py         # opponent_model.json + bot table

cargo build --release
./target/release/train bench
./target/release/train panel --opp old-greedy       # baseline on the panel
./target/release/train room --seats genomes/evolved.json,empirical,empirical,empirical,sim-aggr,sim-value
./target/release/train train --out runs/ga1 --generations 1500   # resumes from runs/ga1/state.json
./target/release/train select runs/ga1/champions/gen_1*.json --out genomes/evolved.json

# Live (local by default; Maincloud needs --allow-remote and a real token)
./target/release/scrabblebot-bot --strategy evolved --genome genomes/evolved.json \
  --token-file /path/to/token --matches 1
```

A run writes `log.csv` (fitness per generation), `panel.csv`,
`champions/gen_NNNN.json`, `best.json`, and a resumable `state.json`.

## Live behavior

The client subscribes to small tables only: match and lobby headers, bots,
credentials, and our own rack and nonces. It adds bot-scoped participant and
lobby rows, and full rows for each match we are in, dropping them after
writing that match's summary. It processes every queued message before each
decision tick. An earlier version subscribed to every new match. On
Maincloud (about 6 matches/min) it fell behind the message stream and sat out
a match.

`--strategy evolved` calls the same `policy::decide` on the authoritative
snapshot. Words go out first, and all of them are submitted together, since
they never share letters. The bid is sent once the rack reflects those plays,
or immediately if the auction is close to closing. During the final auction
the rack is flushed and the bot bids 0.

## Known limits

- The empirical opponent copies the typical real bot (about 90 points), not
  the very best ones (Bron averages about 196). The hall of fame supplies
  stronger opposition.
- Opponents' racks are hidden. The bag estimate subtracts our starting rack
  and every auctioned letter, then spreads the remaining uncertainty
  proportionally.
- In the simulator each bot acts once per auction, at open. Live bots can act
  at any time, which helps the live bot slightly.

## Reinforcement learning (`rl/`, `rlenv/`)

- `src/sim/stepper.rs`: step-wise engine; a test checks it reproduces `play_match` exactly.
- `src/sim/rl.rs`: RL inputs, the same 90 the GA uses (63 bid + 27 play
  features). Actions are a play threshold (hold, or play words of length 2-8+)
  and a bid drawn from a discretized Gaussian over 28 bins. Also holds the
  Rust inference for exported actors (`*.rl.json`), used in the simulator,
  the panel and the live bot.
- `rlenv/`: pyo3 batched environment (`maturin develop --release`), with
  optional teacher (GA) actions for imitation.
- `rl/ppo.py`: imitation of a GA champion, then PPO. The actor is
  90 -> 12 tanh -> heads (~1.2k params, about the GA's size); the critic is
  used only during training. Supports `--reward score|placement`,
  `--init-actor` for resuming, and a league of its own snapshots.

Lessons from live play:

- v3 was trained on a placement reward against frozen copies of our own
  bots. It looked best in simulation but was worse live (26% vs 40% wins).
  A relative reward teaches the bot to beat whichever opponents it trains
  against.
- v4 (`genomes/v4.rl.json`) is trained on our own points
  (`--reward score --gamma 1.0`), in full rooms only, against "mimic"
  opponents. A mimic is one of our competent policies with its bids scaled to
  a real top bot's mean winning bid (`path@mult` in the pool).
- The simulator can rank models differently from live play. Confirm on
  Maincloud before trusting a simulated gain.
