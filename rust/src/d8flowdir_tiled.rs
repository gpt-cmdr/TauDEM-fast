// SPDX-License-Identifier: GPL-3.0-only
//! `d8flowdir --mode tiled`: tiles of the filled DEM are read with a 1-cell
//! halo in parallel, and `-sd8` and `-p` are written by background tile
//! writers while the tiled kernels (`taudem_rs::d8_tiled`) run within
//! `--max-memory`.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use taudem_rs::d8_tiled::{
    D8Outcome, D8TileStats, SteepSchedule, WORK_BYTES_PER_CELL, d8_flow_tiled, tile_size_for_budget,
};
use taudem_rs::tiling::{
    ExecutionMode, ExecutionOptions, HaloWindow, TileCache, TileId, TileSpec, auto_tile_size,
};
use taudem_rs::{DIRECTION_NODATA, SCALAR_NODATA, profile};

use crate::io::{P_NODATA, Raster, Result, tile_writer};

/// Approximate peak bytes per cell of the in-memory d8flowdir: elevations,
/// directions, packed increment levels and the search frontier.
const IN_MEMORY_BYTES_PER_CELL: u64 = 10;

/// Whether d8flowdir runs tiled: `--mode tiled`, or `--mode auto` when the
/// in-memory estimate exceeds the budget or the raster exceeds the
/// in-memory 32-bit index.
pub fn wanted(execution: &ExecutionOptions, raster: &Raster) -> bool {
    match execution.mode {
        ExecutionMode::Tiled => true,
        ExecutionMode::InMemory => false,
        ExecutionMode::Auto => {
            let (w, h) = raster.dimensions();
            let cells = w as u64 * h as u64;
            raster.grid().is_err()
                || cells.saturating_mul(IN_MEMORY_BYTES_PER_CELL) > execution.memory_budget()
        }
    }
}

pub struct TiledRun {
    pub cells: usize,
    /// Time the reader thread spent reading tiles, overlapping `compute`.
    pub read: Duration,
    /// Waiting for the writers to finish.
    pub write: Duration,
    pub compute: Duration,
}

pub fn run(
    mut raster: Raster,
    p_path: &str,
    sd8_path: &str,
    execution: &ExecutionOptions,
) -> Result<TiledRun> {
    let start = Instant::now();
    let (width, height) = raster.dimensions();
    let nodata = raster.nodata();
    let spacing = raster.spacing()?;
    let header = raster.header.clone();
    let threads = rayon::current_num_threads();
    let budget = execution.memory_budget();
    let cells = width as u64 * height as u64;
    let spec = match execution.tile_size {
        Some(t) => TileSpec::new(width, height, t, t)?,
        None => TileSpec::aligned(
            width,
            height,
            tile_size_for_budget(budget, threads, auto_tile_size(cells, budget)),
            raster.block_size(),
        )?,
    };
    let (tw, th) = spec.tile_size();
    // Workers' buffers, the perimeter tables (about 40 bytes per perimeter
    // cell over all tables) and the writers' queues; the rest holds tiles.
    // Each worker's buffers, and the elevations queued for them.
    let workers = threads as u64 * ((tw + 2) * (th + 2)) as u64 * (WORK_BYTES_PER_CELL + 4);
    let borders = spec.tile_count() as u64 * 2 * (tw + th) as u64 * 40;
    let queues = 2 * 4 * threads as u64 * 256 * 256 * 4;
    // The process itself (code, thread stacks, the reader's decode buffers,
    // allocator slack) takes the rest.
    let reserve = (48 << 20) + budget / 16;
    let cache_budget = budget
        .saturating_sub(workers + borders + queues + reserve)
        .max(budget / 8);
    let cache = TileCache::new(execution.cache_config(cache_budget));
    profile::count("d8.tiled.tile_width", tw as u64);
    profile::count("d8.tiled.cache_budget_bytes", cache_budget);

    let reader = raster.halo_reader()?;
    let read_ns = AtomicU64::new(0);
    let read = |_: TileId, halo: &HaloWindow, buffer: &mut Vec<f32>| {
        let t = Instant::now();
        let result = reader.halo(halo, buffer);
        read_ns.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
        result
    };
    let queue = 2 * threads;
    let mut schedule = SteepSchedule::Parallel;
    let (stats, write) = loop {
        let slope_out =
            tile_writer::<f32>(sd8_path, &header, &spec, f64::from(SCALAR_NODATA), queue)
                .map_err(|e| format!("writing {sd8_path}: {e}"))?;
        let dir_out = tile_writer::<i16>(p_path, &header, &spec, f64::from(P_NODATA), queue)
            .map_err(|e| format!("writing {p_path}: {e}"))?;
        let slope_sink = slope_out.sink();
        let dir_sink = dir_out.sink();
        let slope = |id: TileId, data: &[f32]| slope_sink.submit_region(spec.rect(id), data);
        let direction = |id: TileId, data: &[u8]| {
            let codes: Vec<i16> = data
                .iter()
                .map(|&code| {
                    if code == DIRECTION_NODATA {
                        P_NODATA
                    } else {
                        i16::from(code)
                    }
                })
                .collect();
            dir_sink.submit_region(spec.rect(id), &codes)
        };
        let outcome = d8_flow_tiled(
            spec,
            nodata,
            &spacing,
            &cache,
            schedule,
            &read,
            Some(&slope),
            &direction,
        )?;
        match outcome {
            D8Outcome::Done(stats) => {
                drop((slope_sink, dir_sink));
                let finishing = Instant::now();
                let slope_stats = slope_out.finish()?;
                let dir_stats = dir_out.finish()?;
                slope_stats.record_profile("sd8");
                dir_stats.record_profile("p");
                break (stats, finishing.elapsed());
            }
            D8Outcome::CrossingDetected => {
                // Abandon both files and apply the crossing rule exactly.
                drop((slope_sink, dir_sink, slope_out, dir_out));
                profile::count("d8.tiled.wavefront_restarts", 1);
                schedule = SteepSchedule::Wavefront;
            }
        }
    };
    stats.record_profile();
    cache.stats().record_profile();
    report(&spec, &stats, &cache);
    // Tiles are read by one thread (several in the wavefront fallback).
    let read = Duration::from_nanos(read_ns.load(Relaxed)).min(start.elapsed());
    let total = start.elapsed();
    Ok(TiledRun {
        cells: cells as usize,
        read,
        write,
        compute: total.saturating_sub(write),
    })
}

fn report(spec: &TileSpec, stats: &D8TileStats, cache: &TileCache) {
    let (tw, th) = spec.tile_size();
    println!(
        "Tiled D8: {} tiles of {tw} x {th}; {} flat cells, {} routed inside tiles \
         ({} passes), {} in {} edge tiles across {} passes",
        stats.tiles,
        stats.flat_cells,
        stats.interior_cells,
        stats.interior_passes,
        stats.deferred_cells,
        stats.deferred_tiles,
        stats.passes.len()
    );
    let c = &stats.components;
    println!(
        "Tiled D8 edge flats: {} components ({} cells), {} across tiles ({} cells);          largest {} cells, widest over {} tiles ({} x {})",
        c.components,
        c.cells,
        c.cross_tile,
        c.cross_tile_cells,
        c.max_cells,
        c.max_tiles,
        c.max_tiles_across,
        c.max_tiles_down
    );
    for (i, p) in stats.passes.iter().enumerate() {
        println!(
            "Tiled D8 flat pass {}: {} cells in {} tiles, {} rounds, {} tile loads, \
             levels {}/{}, {} left, {:.3} s",
            i + 1,
            p.cells,
            p.tiles,
            p.rounds,
            p.relax_loads,
            p.last_fall,
            p.last_rise,
            p.remaining,
            p.seconds
        );
    }
    let s = cache.stats();
    println!(
        "Tiled D8 tile store: peak {} MiB resident of {} MiB, {} MiB spilled",
        s.peak_resident_bytes >> 20,
        s.budget_bytes >> 20,
        s.peak_spilled_bytes >> 20
    );
}
