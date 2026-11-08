#!/usr/bin/env python3
"""Rewrite runs/STATUS.md every minute: RL training progress, snapshot
scores (from the snapshot watcher's output), and live bot results.

Usage: python3 -I tools/status_board.py RUN_DIR_ON_CANIS SNAPSHOT_SCORES LIVE_JSONL
"""

import json
import os
import subprocess
import sys
import time

run, scores_path, live_path = sys.argv[1], sys.argv[2], sys.argv[3]
out = "runs/STATUS.md"


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        return f"(error: {e})"


while True:
    train = sh(f"ssh -o BatchMode=yes syssec-canis \"bash -c 'tail -1 ~/scrabblebot-train/{run}/train.log'\"").strip()
    snaps = open(scores_path).read().strip() if os.path.exists(scores_path) else "(none yet)"
    rows = []
    if os.path.exists(live_path):
        for line in open(live_path):
            d = json.loads(line)
            if d.get("type") == "match_summary" and d["final_score"] > 0:
                best = max(p["score"] for p in d["participants"] if p["bot_id"] != d["bot_id"])
                rows.append((d["match_id"], d["final_score"], d["placement"], best))
    n = len(rows)
    live = "(no matches yet)"
    if n:
        live = (f"{n} matches, {sum(1 for r in rows if r[2] == 1)} wins, "
                f"mean score {sum(r[1] for r in rows) / n:.1f}, mean place {sum(r[2] for r in rows) / n:.2f}\n\n"
                "| match | score | place | best rival |\n|---:|---:|---:|---:|\n"
                + "\n".join(f"| {m} | {s} | {p} | {b} |" for m, s, p, b in rows[-15:]))
    text = (f"# Status ({time.strftime('%H:%M:%S')})\n\n## RL training ({run})\n\n`{train}`\n\n"
            f"## Snapshot scores vs ga7 (full room score/place; lower place is better)\n\n```\n{snaps}\n```\n\n"
            f"## Live bot\n\n{live}\n")
    tmp = out + ".tmp"
    open(tmp, "w").write(text)
    os.replace(tmp, out)
    time.sleep(60)
