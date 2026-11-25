// SPDX-License-Identifier: GPL-3.0-only
//! Shared infrastructure for tiled (out-of-core) execution.
//!
//! The tiled stages of `docs/rust-port/STREAMING.md` share these pieces:
//!
//! - [`TileSpec`]: the partition of a raster into tiles, halo windows,
//!   tile-local `u32` indexing and deterministic tile orders, including the
//!   NW→SE wavefront that D8's `dontCross` rule requires.
//! - [`TileCache`] and [`TileStore`]: tiles kept in RAM, in LZ4-compressed RAM,
//!   or spilled to a scratch file, within one byte budget.
//! - [`BorderTable`]: the perimeter cells of every tile, always in RAM.
//! - [`AsyncTileWriter`]: GeoTIFF output written by a background thread while
//!   compute threads compress tiles.
//! - [`ExecutionOptions`]: `--max-memory`, `--tile-size`, `--scratch-dir` and
//!   `--mode`, and the default memory budget.
//!
//! None of this changes the in-memory kernels.

mod codec;
mod spec;
mod store;

pub use codec::{
    CHUNK_BYTES, Prefilter, TileElement, decode_tile, encode_tile, encoded_header_bytes,
};
pub use spec::{HaloWindow, Rect, TileId, TileSpec, for_each_wavefront};
pub use store::{CacheConfig, StoreOptions, Tier, TileCache, TilePin, TileStats, TileStore};

use std::fmt;

/// Errors of the tiling infrastructure.
#[derive(Debug)]
pub enum Error {
    /// An argument is out of range: a tile size, a tile that does not exist,
    /// a buffer of the wrong length.
    Invalid(String),
    /// A stored tile failed its integrity check.
    Corrupt(String),
    /// Reading or writing the scratch file failed.
    Io(std::io::Error),
    /// The GeoTIFF reader or writer failed.
    Terrio(terrio::Error),
    /// A background writer stopped before its file was finished.
    WriterClosed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "{message}"),
            Self::Corrupt(message) => write!(f, "tile store integrity check failed: {message}"),
            Self::Io(error) => write!(f, "scratch file: {error}"),
            Self::Terrio(error) => write!(f, "{error}"),
            Self::WriterClosed(message) => write!(f, "tile writer stopped: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Terrio(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<terrio::Error> for Error {
    fn from(error: terrio::Error) -> Self {
        Self::Terrio(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
