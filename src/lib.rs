//! ScrabbleBot Rust bot: pure solver/strategy core plus a SpacetimeDB client.
//!
//! The `module_bindings` module is generated from the upstream module schema
//! with SpacetimeDB CLI 2.12.0. Regenerate it with
//! `scripts/generate-bindings.sh` (the coordinator's Docker route).

pub mod cli;
pub mod core;
pub mod live;
pub mod module_bindings;
