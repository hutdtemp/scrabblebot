# Third-party notices

## Clockwork Labs ScrabbleBot (upstream game)

This bot targets the upstream module and rules at:

- Repository: https://github.com/clockworklabs/scrabblebot
- Revision read for this implementation: `b5ccb36aa238ad16692f8d6af54da15ef9f11324`
- License: MIT, Copyright (c) 2026 Clockwork Labs
- Full license text: `LICENSES/clockworklabs-scrabblebot-MIT.txt`

The schema, reducer names, scoring rules, bag distribution, auction rules, and
the ENABLE wordlist were read from that checkout. The Rust client bindings in
`src/module_bindings/` were generated from the upstream module schema with the
official SpacetimeDB CLI 2.12.0.

## ENABLE wordlist

`data/enable_wordlist.txt` is a byte-for-byte copy of the upstream module's
`spacetimedb/wordlist.txt`:

- SHA-256: `b819e7b67f8f2c80640ae76f0edbe59a4a1befb6de0f348463642ad49bdd2ad3`
- Lines: 172,820
- Source: the public-domain ENABLE wordlist, as described by the upstream
  README (https://en.wikipedia.org/wiki/Moby_Project#ENABLE).
- The upstream project embeds this list in its module; this bot embeds the same
  list for exact local word validity and scoring.

## SpacetimeDB SDK and CLI

The Rust client depends on `spacetimedb-sdk = "=2.12.0"`. The SDK and CLI are
published by Clockwork Laboratories under their own licenses (the SDK crate
ships a Business Source License 1.1 file; `spacetimedb-lib` ships Apache-2.0).
They are used as unmodified dependencies. The generated bindings are a
derived artifact of the upstream module schema.
