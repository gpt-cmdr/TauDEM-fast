// SPDX-License-Identifier: GPL-3.0-only
//! Test support for tiled stages: the tiling-boundary adversarial fixtures
//! of `docs/rust-port/STREAMING.md` §6.1 and the parity harness that tiled
//! PitRemove, D8FlowDir and AreaD8 (WP-S1..S3) are gated on.
//!
//! Use it from an integration test with `mod support;`. Fixtures are
//! generated in memory; nothing large is stored in the repository.

#![allow(dead_code)]

pub mod d8_reference;
pub mod fixtures;
pub mod harness;
