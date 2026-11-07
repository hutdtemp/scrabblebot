//! Pure, testable game logic: letters, racks, bag model, dictionary solver,
//! beam search, bidding/playback policy, and telemetry records.
//!
//! Nothing in this module talks to SpacetimeDB. The live client in
//! `crate::live` feeds authoritative snapshots into these types.

pub mod bag;
pub mod dictionary;
pub mod game;
pub mod letters;
pub mod rack;
pub mod search;
pub mod strategy;
pub mod telemetry;
