#!/usr/bin/env python3
"""Summarize live matches from bot telemetry, grouped by room difficulty.

Usage: python3 -I tools/live_summary.py runs/maincloud/evol.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

rows = []
for line in open(sys.argv[1]):
    d = json.loads(line)
    if d.get("type") != "match_summary":
        continue
    others = [p for p in d["participants"] if p["bot_id"] != d["bot_id"]]
    real = [p for p in others if not p["is_simulated"] and p["name"] != "Afterword"]
    best_other = max(others, key=lambda p: p["score"])
    rows.append((d, len(real), best_other))
    print(
        f"match {d['match_id']}: score {d['final_score']:>4} place {d['placement']} "
        f"endbal {d['budget_end']:>4} real opp {len(real)} | best other "
        f"{best_other['name']} {best_other['score']}"
    )

if rows:
    print()
    by = defaultdict(list)
    for d, n, _ in rows:
        by[n].append(d)
    for n in sorted(by):
        ds = by[n]
        print(
            f"{n} real opponents: n={len(ds)} mean score "
            f"{statistics.mean(x['final_score'] for x in ds):.0f} mean place "
            f"{statistics.mean(x['placement'] for x in ds):.2f}"
        )
    print(
        f"overall: n={len(rows)} mean score "
        f"{statistics.mean(d['final_score'] for d, _, _ in rows):.0f} mean place "
        f"{statistics.mean(d['placement'] for d, _, _ in rows):.2f} wins "
        f"{sum(1 for d, _, _ in rows if d['placement'] == 1)}"
    )
