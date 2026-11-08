//! Offline match simulator, evolvable policy, and genetic-algorithm trainer.
//!
//! The engine mirrors the pinned upstream module rules so policies can be
//! evaluated over thousands of matches per minute instead of one per 70 s.

pub mod agents;
pub mod engine;
pub mod ga;
pub mod policy;
pub mod rl;
pub mod rng;
pub mod stepper;
pub mod words;
