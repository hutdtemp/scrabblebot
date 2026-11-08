#!/usr/bin/env python3
"""Summarize the Maincloud dump and emit an empirical opponent model.

Usage: python3 -I tools/analyze_maincloud.py [DATA_DIR] [RECENT_MATCHES]

Writes DATA_DIR/opponent_model.json with, per strong real bot and pooled:
  - winning-bid quantiles per letter and game phase,
  - word-length distribution and play timing.
"""

import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

data = sys.argv[1] if len(sys.argv) > 1 else "data/maincloud"
recent_n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000


def rows(name):
    with open(os.path.join(data, f"{name}.csv"), newline="") as f:
        yield from csv.DictReader(f)


bots = {int(r["id"]): r for r in rows("bot")}
sim = {i for i, r in bots.items() if r["is_simulated"] == "True"}
stats = {int(r["bot_id"]): r for r in rows("bot_stats")}

matches = {}
for r in rows("match_state"):
    if r["status"] == "2" and r["auction_type"] == "1":  # Ended, Vickrey
        matches[int(r["id"])] = r
recent = set(sorted(matches)[-recent_n:])
lo = min(recent)

parts = defaultdict(list)
for r in rows("match_participant"):
    m = int(r["match_id"])
    if m in recent:
        parts[m].append((int(r["bot_id"]), int(r["score"]), int(r["balance"])))

top_bid = {}
for r in rows("auction_top_bid_archive"):
    if int(r["match_id"]) >= lo:
        top_bid[int(r["auction_id"])] = int(r["top_bid"])

auctions = defaultdict(list)
for r in rows("auction_result"):
    m = int(r["match_id"])
    if m not in recent:
        continue
    paid = int(r["paid"])
    winner = int(r["winner_bot_id"]) if paid > 0 else None
    auctions[m].append(
        (int(r["auction_id"]), r["letter"], winner, paid, top_bid.get(int(r["auction_id"]), 0))
    )

words = defaultdict(list)
for r in rows("word_play"):
    m = int(r["match_id"])
    if m in recent:
        words[m].append((int(r["bot_id"]), r["word"], int(r["total_reward"]), int(r["played_at"])))

# ---- per-bot summaries over recent matches ----
per_bot = defaultdict(lambda: {"scores": [], "place": [], "lens": Counter(), "spent": 0,
                               "won": 0, "end_bal": [], "matches": 0})
real_counts = Counter()
for m, ps in parts.items():
    real_counts[sum(1 for b, _, _ in ps if b not in sim)] += 1
    for b, s, bal in ps:
        d = per_bot[b]
        d["scores"].append(s)
        d["place"].append(1 + sum(1 for _, s2, _ in ps if s2 > s))
        d["end_bal"].append(bal)
        d["matches"] += 1
    for _, _, w, paid, _ in auctions[m]:
        if w is not None:
            per_bot[w]["spent"] += paid
            per_bot[w]["won"] += 1
    for b, word, _, _ in words[m]:
        per_bot[b]["lens"][min(len(word), 9)] += 1

print(f"recent Vickrey matches: {len(recent)} (ids {lo}..{max(recent)})")
print("real bots per match:", dict(sorted(real_counts.items())))
print()
print(f"{'bot':<22}{'rating':>7}{'n':>6}{'mean':>7}{'p50':>6}{'place':>7}"
      f"{'won/m':>7}{'paid/l':>7}{'endbal':>7}  word lengths 2..9+")
ranked = sorted(
    (b for b in per_bot if per_bot[b]["matches"] >= 30),
    key=lambda b: -statistics.mean(per_bot[b]["scores"]),
)
for b in ranked[:25]:
    d = per_bot[b]
    lens = d["lens"]
    tot = sum(lens.values()) or 1
    dist = " ".join(f"{100 * lens[k] // tot:>2}" for k in range(2, 10))
    print(f"{bots[b]['name'][:21]:<22}{stats.get(b, {}).get('rating', '?'):>7}{d['matches']:>6}"
          f"{statistics.mean(d['scores']):>7.1f}{statistics.median(d['scores']):>6.0f}"
          f"{statistics.mean(d['place']):>7.2f}{d['won'] / d['matches']:>7.1f}"
          f"{d['spent'] / max(d['won'], 1):>7.1f}{statistics.mean(d['end_bal']):>7.1f}  {dist}")

# ---- empirical opponent model from the strongest real bots ----
strong = [b for b in ranked if b not in sim][:8]
phase_of = lambda i: min(i * 3 // 68, 2)  # noqa: E731 - early/mid/late
bid_samples = defaultdict(list)  # (letter, phase) -> winning bids of strong bots
len_counts = Counter()
play_round = []
for m in recent:
    seq = sorted(auctions[m])
    for i, (_, letter, w, _, top) in enumerate(seq):
        if w in strong and top > 0:
            bid_samples[f"{letter}{phase_of(i)}"].append(top)
    if not seq:
        continue
    t0 = min(words[m][0][3] for _ in [0]) if words[m] else 0
    for b, word, _, _ in words[m]:
        if b in strong:
            len_counts[len(word)] += 1


def quantiles(xs):
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(q * len(xs)))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)]


model = {
    "source": "maincloud public tables",
    "recent_matches": len(recent),
    "strong_bots": [bots[b]["name"] for b in strong],
    "winning_bid_quantiles": {k: quantiles(v) for k, v in sorted(bid_samples.items()) if len(v) >= 20},
    "word_length_counts": {str(k): v for k, v in sorted(len_counts.items())},
}
# Market clearing prices (all bots) per letter, for reference.
paid = defaultdict(list)
for m in recent:
    for _, letter, w, p, _ in auctions[m]:
        if w is not None:
            paid[letter].append(p)
model["clearing_price_median"] = {k: statistics.median(v) for k, v in sorted(paid.items())}
model["unsold_fraction"] = sum(1 for m in recent for a in auctions[m] if a[2] is None) / max(
    1, sum(len(auctions[m]) for m in recent))
with open(os.path.join(data, "opponent_model.json"), "w") as f:
    json.dump(model, f, indent=1)
print()
print("strong real bots:", model["strong_bots"])
print("median clearing price:", model["clearing_price_median"])
print("unsold fraction:", round(model["unsold_fraction"], 3))
print("strong-bot word lengths:", model["word_length_counts"])
