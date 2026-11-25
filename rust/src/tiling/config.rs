// SPDX-License-Identifier: GPL-3.0-only
//! Command-line options of tiled execution and the default memory budget.

use std::path::PathBuf;

use super::memory::{available_memory, usable_memory};
use super::spec::TileSpec;
use super::store::CacheConfig;

/// The budget used when neither the usable nor the available memory is known.
pub const FALLBACK_BUDGET: u64 = 4 << 30;
/// The smallest and largest `--tile-size`.
pub const MIN_TILE_SIZE: usize = 16;
pub const MAX_TILE_SIZE: usize = 32_768;

/// `--mode`: how a command runs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ExecutionMode {
    /// Tiled when the in-memory estimate exceeds the budget. Until the tiled
    /// stages exist this runs in memory.
    #[default]
    Auto,
    /// Whole grids in RAM (the stage-one kernels).
    InMemory,
    /// The tiled, out-of-core stages.
    Tiled,
}

impl ExecutionMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::InMemory => "in-memory",
            Self::Tiled => "tiled",
        }
    }
}

impl std::str::FromStr for ExecutionMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "in-memory" => Ok(Self::InMemory),
            "tiled" => Ok(Self::Tiled),
            _ => Err(format!("--mode must be auto, in-memory or tiled, not {s}")),
        }
    }
}

/// `--max-memory`, `--tile-size`, `--scratch-dir` and `--mode`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExecutionOptions {
    pub mode: ExecutionMode,
    /// The memory budget in bytes; `None` for [`default_memory_budget`].
    pub max_memory: Option<u64>,
    /// The tile side; `None` for [`auto_tile_size`], aligned to the input's blocks.
    pub tile_size: Option<usize>,
    /// Where tiles spill; `None` for the system temporary directory.
    pub scratch_dir: Option<PathBuf>,
}

impl ExecutionOptions {
    pub const FLAGS: [&'static str; 4] = ["--max-memory", "--tile-size", "--scratch-dir", "--mode"];

    /// Parses the flags' values; `value` returns the value given for a flag.
    pub fn from_flags<'a>(value: impl Fn(&str) -> Option<&'a str>) -> Result<Self, String> {
        let mode = value("--mode").map_or(Ok(ExecutionMode::Auto), str::parse)?;
        let max_memory = value("--max-memory")
            .map(|v| {
                let bytes = parse_bytes(v)?;
                if bytes < 1 << 20 {
                    return Err(format!("--max-memory {v} is below 1 MiB"));
                }
                Ok(bytes)
            })
            .transpose()?;
        let tile_size = value("--tile-size")
            .map(|v| {
                let size: usize = v
                    .parse()
                    .map_err(|_| format!("--tile-size must be a whole number of cells, not {v}"))?;
                if !(MIN_TILE_SIZE..=MAX_TILE_SIZE).contains(&size) {
                    return Err(format!(
                        "--tile-size must be {MIN_TILE_SIZE} to {MAX_TILE_SIZE}, not {size}"
                    ));
                }
                Ok(size)
            })
            .transpose()?;
        let scratch_dir = value("--scratch-dir").map(PathBuf::from);
        Ok(Self {
            mode,
            max_memory,
            tile_size,
            scratch_dir,
        })
    }

    /// `--max-memory`, or [`default_memory_budget`].
    pub fn memory_budget(&self) -> u64 {
        self.max_memory.unwrap_or_else(default_memory_budget)
    }

    /// `--scratch-dir`, or the system temporary directory.
    pub fn scratch_dir(&self) -> PathBuf {
        self.scratch_dir.clone().unwrap_or_else(std::env::temp_dir)
    }

    /// The tiles of a raster: exactly `--tile-size` when given, else
    /// [`auto_tile_size`] for the budget, aligned to the input's native blocks.
    pub fn tile_spec(
        &self,
        width: usize,
        height: usize,
        native_block: Option<(usize, usize)>,
    ) -> super::Result<TileSpec> {
        if let Some(size) = self.tile_size {
            return TileSpec::new(width, height, size, size);
        }
        let cells = width as u64 * height as u64;
        let target = auto_tile_size(cells, self.memory_budget());
        TileSpec::aligned(width, height, target, native_block)
    }

    /// A tile cache bounded by `budget_bytes` (the part of the memory budget
    /// left after working buffers), spilling to the scratch directory.
    pub fn cache_config(&self, budget_bytes: u64) -> CacheConfig {
        CacheConfig::new(budget_bytes, Some(self.scratch_dir()))
    }
}

/// `min(0.65 × usable memory, 0.75 × available memory)` of those known
/// ([`super::memory`]), or 4 GiB when neither is.
pub fn default_memory_budget() -> u64 {
    budget_from(usable_memory(), available_memory())
}

fn budget_from(usable: Option<u64>, available: Option<u64>) -> u64 {
    let usable = usable.map(|b| (b as f64 * 0.65) as u64);
    let available = available.map(|b| (b as f64 * 0.75) as u64);
    match (usable, available) {
        (Some(u), Some(a)) => u.min(a),
        (Some(b), None) | (None, Some(b)) => b,
        (None, None) => FALLBACK_BUDGET,
    }
}

/// The tile side `STREAMING.md` §3.2 recommends: 4096 when the budget is at
/// least 16 GiB and the raster has at least 5·10⁸ cells, else 2048.
pub fn auto_tile_size(cells: u64, budget_bytes: u64) -> usize {
    if budget_bytes >= 16 << 30 && cells >= 500_000_000 {
        4096
    } else {
        2048
    }
}

/// Parses a byte count: a number, optionally fractional, with an optional
/// binary suffix `K`, `M`, `G` or `T` (also `KB`/`KiB`...; all mean powers
/// of 1024), case-insensitive.
pub fn parse_bytes(text: &str) -> Result<u64, String> {
    let s = text.trim();
    let split = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(s.len());
    let (number, suffix) = s.split_at(split);
    let shift = match suffix.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 0,
        "k" | "kb" | "kib" => 10,
        "m" | "mb" | "mib" => 20,
        "g" | "gb" | "gib" => 30,
        "t" | "tb" | "tib" => 40,
        _ => return Err(format!("unknown size suffix in {text}: use K, M, G or T")),
    };
    let value: f64 = number
        .parse()
        .map_err(|_| format!("{text} is not a size such as 4G or 512M"))?;
    let bytes = value * (1u64 << shift) as f64;
    if !bytes.is_finite() || bytes >= u64::MAX as f64 {
        return Err(format!("{text} is too large"));
    }
    Ok(bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn sizes_parse_with_binary_suffixes() {
        assert_eq!(parse_bytes("4G"), Ok(4 << 30));
        assert_eq!(parse_bytes("4GiB"), Ok(4 << 30));
        assert_eq!(parse_bytes("512m"), Ok(512 << 20));
        assert_eq!(parse_bytes("1.5G"), Ok(3 << 29));
        assert_eq!(parse_bytes("64MB"), Ok(64 << 20));
        assert_eq!(parse_bytes("1048576"), Ok(1 << 20));
        assert_eq!(parse_bytes("2 T"), Ok(2 << 40));
        assert!(parse_bytes("4X").is_err());
        assert!(parse_bytes("G").is_err());
        assert!(parse_bytes("-4G").is_err());
        assert!(parse_bytes("99999999999T").is_err());
    }

    #[test]
    fn flags_parse_and_validate() {
        let flags = |pairs: &[(&'static str, &'static str)]| {
            let map: HashMap<&str, &str> = pairs.iter().copied().collect();
            ExecutionOptions::from_flags(|f| map.get(f).copied())
        };
        assert_eq!(flags(&[]), Ok(ExecutionOptions::default()));
        let options = flags(&[
            ("--mode", "tiled"),
            ("--max-memory", "64M"),
            ("--tile-size", "17"),
            ("--scratch-dir", "D:/scratch"),
        ])
        .unwrap();
        assert_eq!(options.mode, ExecutionMode::Tiled);
        assert_eq!(options.memory_budget(), 64 << 20);
        assert_eq!(options.tile_size, Some(17));
        assert_eq!(options.scratch_dir(), PathBuf::from("D:/scratch"));
        assert_eq!(
            options
                .tile_spec(100, 67, Some((256, 256)))
                .unwrap()
                .tile_size(),
            (17, 17)
        );
        assert_eq!(
            flags(&[("--mode", "in-memory")]).unwrap().mode,
            ExecutionMode::InMemory
        );
        assert!(flags(&[("--mode", "stream")]).is_err());
        assert!(flags(&[("--max-memory", "100K")]).is_err());
        assert!(flags(&[("--tile-size", "8")]).is_err());
        assert!(flags(&[("--tile-size", "4096x2")]).is_err());
        let auto = flags(&[("--max-memory", "4G")]).unwrap();
        let spec = auto.tile_spec(36_000, 31_635, Some((256, 256))).unwrap();
        assert_eq!(spec.tile_size(), (2048, 2048));
        let big = flags(&[("--max-memory", "16G")]).unwrap();
        assert_eq!(
            big.tile_spec(36_000, 31_635, Some((256, 256)))
                .unwrap()
                .tile_size(),
            (4096, 4096)
        );
    }

    #[test]
    fn budget_takes_the_smaller_bound() {
        let gib = 1u64 << 30;
        assert_eq!(budget_from(Some(64 * gib), Some(40 * gib)), 30 * gib);
        assert_eq!(
            budget_from(Some(32 * gib), Some(60 * gib)),
            ((32 * gib) as f64 * 0.65) as u64
        );
        assert_eq!(budget_from(None, Some(8 * gib)), 6 * gib);
        assert_eq!(budget_from(None, None), FALLBACK_BUDGET);
        let budget = default_memory_budget();
        assert!(budget >= 64 << 20, "{budget}");
        if let Some(usable) = usable_memory() {
            assert!(budget <= usable);
        }
    }
}
