// SPDX-License-Identifier: GPL-3.0-only
//! The tile store under a forced 64 MiB budget: many times more tile data
//! than the budget passes through it from 8 threads, every tile reads back
//! intact, and the process's peak working set (and heap) stay below the
//! budget plus the fixed per-thread working buffers.
//!
//! This file holds one test so that the process-lifetime peak working set
//! belongs to it. `TAUDEM_STRESS_MIB` sets the tile data volume (default 768).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use rayon::prelude::*;
use taudem_rs::profile::{self, CountingAlloc};
use taudem_rs::tiling::memory::process_memory;
use taudem_rs::tiling::{CacheConfig, StoreOptions, TileCache, TileId, TileStore};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const BUDGET: u64 = 64 << 20;
const SIDE: usize = 512;
const THREADS: usize = 8;

/// A DEM-like tile: a smooth surface quantized to 1/32, with noise in a
/// quarter of the tiles so that some compress poorly and must spill.
fn tile(id: TileId, round: u32) -> Vec<f32> {
    let seed = (id.tx as u64) << 32 | (id.ty as u64) << 8 | round as u64;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let noisy = (id.tx + id.ty + round).is_multiple_of(4);
    (0..SIDE * SIDE)
        .map(|i| {
            let (x, y) = ((i % SIDE) as f32, (i / SIDE) as f32);
            let base = 300.0 + (x * 0.013 + id.tx as f32).sin() * 20.0 + y * 0.05 + round as f32;
            let noise = if noisy {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 52) as f32 * 0.001
            } else {
                0.0
            };
            ((base + noise) * 32.0).round() / 32.0
        })
        .collect()
}

#[test]
fn peak_memory_stays_within_a_forced_64_mib_budget() {
    let mib: usize = std::env::var("TAUDEM_STRESS_MIB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(768);
    let tile_bytes = SIDE * SIDE * 4;
    let tiles = (mib << 20) / tile_bytes;
    let across = 16u32;
    let ids: Vec<TileId> = (0..tiles as u32)
        .map(|i| TileId::new(i % across, i / across))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap();
    let scratch = std::env::temp_dir().join("taudem-rs-stress");

    let baseline = process_memory().map(|m| m.working_set);
    let heap_baseline = profile::heap_current();
    // Sample the working set: the peak counter alone could hide when it was reached.
    let done = AtomicBool::new(false);
    let sampled = AtomicU64::new(0);
    let cache = TileCache::new(CacheConfig::new(BUDGET, Some(scratch)));
    let fel: TileStore<f32> = cache.store("fel", StoreOptions::default());
    std::thread::scope(|s| {
        s.spawn(|| {
            while !done.load(Relaxed) {
                if let Some(m) = process_memory() {
                    sampled.fetch_max(m.working_set, Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        });
        pool.install(|| {
            // Two rounds of writes (the second replaces every tile, freeing
            // spilled extents for reuse) and a verified read-back after each.
            for round in 0..2u32 {
                ids.par_iter()
                    .for_each(|&id| fel.put_owned(id, tile(id, round)).unwrap());
                ids.par_iter().for_each_init(Vec::new, |buffer, &id| {
                    fel.get_into(id, buffer).unwrap();
                    assert!(*buffer == tile(id, round), "tile {id:?} round {round}");
                });
            }
        });
        done.store(true, Relaxed);
    });

    let stats = cache.stats();
    let peak_heap = profile::heap_peak() - heap_baseline.min(profile::heap_peak());
    // Per worker: the tile it builds, the copy it reads back into, the
    // reference it compares with, and a compressed blob or spill read in
    // flight; plus the 2 MiB codec scratch.
    let fixed = THREADS as u64 * (4 * tile_bytes as u64 + (2 << 20)) + (16 << 20);
    let bound = BUDGET + fixed;
    println!(
        "stress: {} tiles x {} KiB = {} MiB through a {} MiB budget on {THREADS} threads",
        tiles,
        tile_bytes >> 10,
        (tiles * tile_bytes) >> 20,
        BUDGET >> 20
    );
    println!("stats: {stats:#?}");
    println!(
        "peak resident tiles {:.1} MiB, peak heap {:.1} MiB, bound {:.1} MiB",
        stats.peak_resident_bytes as f64 / 1048576.0,
        peak_heap as f64 / 1048576.0,
        bound as f64 / 1048576.0
    );
    assert!(
        stats.spilled_tiles > 0 && stats.spill_reads > 0,
        "the stress must spill"
    );
    assert_eq!(stats.peak_over_budget_bytes, 0);
    assert!(
        stats.peak_resident_bytes <= BUDGET + THREADS as u64 * tile_bytes as u64,
        "resident tiles {} over budget plus one in-flight tile per thread",
        stats.peak_resident_bytes
    );
    assert!(peak_heap as u64 <= bound, "peak heap {peak_heap} > {bound}");
    if let (Some(baseline), Some(memory)) = (baseline, process_memory()) {
        let peak = memory.peak_working_set.max(sampled.load(Relaxed));
        let grown = peak.saturating_sub(baseline);
        println!(
            "working set: baseline {:.1} MiB, peak {:.1} MiB, growth {:.1} MiB",
            baseline as f64 / 1048576.0,
            peak as f64 / 1048576.0,
            grown as f64 / 1048576.0
        );
        assert!(
            grown <= bound,
            "working set grew {grown} bytes, over {bound}"
        );
    }
    drop(fel);
    let stats = cache.stats();
    assert_eq!(
        stats.ram_bytes + stats.compressed_bytes + stats.spilled_bytes,
        0
    );
}
