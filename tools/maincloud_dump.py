#!/usr/bin/env python3
"""Read-only dump of public ScrabbleBot tables from Maincloud to CSV.

Uses an anonymous SpacetimeDB identity and the HTTP SQL endpoint, which only
exposes `public` tables. Nothing is written to the server.

Usage: python3 -I tools/maincloud_dump.py [OUT_DIR]   (default data/maincloud)

Large tables are fetched in primary-key ranges and appended; re-running
resumes from the highest key already on disk.
"""

import csv
import json
import os
import sys
import time
import urllib.request

HOST = "https://maincloud.spacetimedb.com"
DB = "scrabblebot"
CHUNK = 100_000

# table -> (primary key column, chunked?)
TABLES = {
    "bot": ("id", False),
    "bot_stats": ("bot_id", False),
    "bot_version": ("id", False),
    "match_state": ("id", True),
    "match_participant": ("id", True),
    "word_play": ("id", True),
    "auction_result": ("auction_id", True),
    "auction_top_bid_archive": ("auction_id", True),
}


def post(url, data=None, token=None):
    req = urllib.request.Request(url, data=data, method="POST")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            if attempt == 4:
                raise
            print(f"  retry after {exc}", file=sys.stderr)
            time.sleep(2 + attempt * 3)


def flatten(value):
    """SpacetimeDB JSON encodes options/sums as [tag, payload]."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], int):
        tag, payload = value
        if payload == []:
            return tag  # unit enum variant (or `none` for option tag 1)
        return flatten(payload)
    if isinstance(value, list) and len(value) == 1:
        return flatten(value[0])  # timestamp wrapper
    return value


def sql(token, query):
    result = post(f"{HOST}/v1/database/{DB}/sql", query.encode(), token)[0]
    cols = [e["name"].get("some") for e in result["schema"]["elements"]]
    return cols, result["rows"]


def last_key(path, key):
    if not os.path.exists(path):
        return None
    last = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            last = int(row[key])
    return last


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "data/maincloud"
    os.makedirs(out, exist_ok=True)
    token = post(f"{HOST}/v1/identity")["token"]
    for table, (key, chunked) in TABLES.items():
        path = os.path.join(out, f"{table}.csv")
        if not chunked:
            cols, rows = sql(token, f"SELECT * FROM {table}")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in rows:
                    w.writerow([flatten(v) for v in r])
            print(f"{table}: {len(rows)} rows")
            continue
        start = last_key(path, key)
        new_file = start is None
        # Half-open [lo, hi) ranges; u64 keys reject negative literals.
        start = 0 if start is None else start + 1
        total = 0
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            while True:
                cols, rows = sql(
                    token,
                    f"SELECT * FROM {table} WHERE {key} >= {start} "
                    f"AND {key} < {start + CHUNK}",
                )
                if new_file:
                    w.writerow(cols)
                    new_file = False
                if not rows:
                    # Keys can have gaps; stop only when nothing remains above.
                    _, probe = sql(
                        token, f"SELECT COUNT(*) AS n FROM {table} WHERE {key} >= {start}"
                    )
                    if probe[0][0] == 0:
                        break
                    start += CHUNK
                    continue
                ki = cols.index(key)
                for r in rows:
                    w.writerow([flatten(v) for v in r])
                start = max(r[ki] for r in rows) + 1
                total += len(rows)
                f.flush()
                print(f"{table}: +{len(rows)} (through {key}={start})", flush=True)
        print(f"{table}: {total} new rows")


if __name__ == "__main__":
    main()
