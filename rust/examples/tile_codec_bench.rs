// SPDX-License-Identifier: GPL-3.0-only
//! Compression ratio and speed of the tile store's codec on real layers.
//!
//! ```text
//! cargo run --release --example tile_codec_bench -- DEM.tif [TILE]
//! ```
//!
//! Runs the in-memory pipeline on the DEM, cuts `fel` (f32), `p` (u8 as
//! stored, and i16 as written), `ad8` (f32) and a BFS-distance layer (u32,
//! standing in for `fall`/`rise`) into TILE x TILE tiles (default 1024),
//! and for every prefilter reports bytes per cell, ratio, and single-thread
//! compress and decompress speed in MB/s of raw tile data, then the
//! default prefilter's aggregate speed over all threads.

use std::time::Instant;

use rayon::prelude::*;
use taudem_rs::tiling::{Prefilter, TileElement, TileSpec, decode_tile, encode_tile};
use taudem_rs::{DIRECTION_NODATA, FEL_NODATA, Grid, area_d8, d8_flow, pit_remove};
use terrio::{DataType, Window};

fn tiles<T: Copy>(spec: &TileSpec, values: &[T]) -> Vec<Vec<T>> {
    let width = spec.raster_width();
    spec.row_major()
        .map(|id| {
            let r = spec.rect(id);
            (r.y..r.bottom())
                .flat_map(|y| {
                    values[y * width + r.x..y * width + r.right()]
                        .iter()
                        .copied()
                })
                .collect()
        })
        .collect()
}

/// Runs `f` until at least 0.3 s have passed and returns seconds per run.
fn time(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    let mut runs = 0;
    while runs < 3 || start.elapsed().as_secs_f64() < 0.3 {
        f();
        runs += 1;
    }
    start.elapsed().as_secs_f64() / runs as f64
}

fn bench<T: TileElement>(name: &str, tiles: &[Vec<T>]) {
    let cells: usize = tiles.iter().map(Vec::len).sum();
    let raw = (cells * size_of::<T>()) as f64;
    for filter in Prefilter::ALL {
        if size_of::<T>() == 1 && matches!(filter, Prefilter::Shuffle | Prefilter::DeltaShuffle) {
            continue;
        }
        let mut blobs: Vec<Vec<u8>> = tiles.iter().map(|_| Vec::new()).collect();
        let compress = time(|| {
            for (tile, blob) in tiles.iter().zip(&mut blobs) {
                encode_tile(tile, filter, blob);
            }
        });
        let stored: usize = blobs.iter().map(Vec::len).sum();
        let mut out = Vec::new();
        let decompress = time(|| {
            for blob in &blobs {
                decode_tile::<T>(blob, &mut out).unwrap();
            }
        });
        for (tile, blob) in tiles.iter().zip(&blobs) {
            decode_tile::<T>(blob, &mut out).unwrap();
            assert!(
                T::as_bytes(&out) == T::as_bytes(tile),
                "{name} {filter:?} round trip"
            );
        }
        let marker = if filter == T::DEFAULT_PREFILTER {
            " (default)"
        } else {
            ""
        };
        println!(
            "| {name} | {filter:?}{marker} | {:.3} | {:.2} | {:.0} | {:.0} |",
            stored as f64 / cells as f64,
            raw / stored as f64,
            raw / compress / 1e6,
            raw / decompress / 1e6
        );
    }
    // Aggregate over the Rayon pool with the default prefilter.
    let filter = T::DEFAULT_PREFILTER;
    let mut blobs: Vec<Vec<u8>> = tiles.iter().map(|_| Vec::new()).collect();
    let compress = time(|| {
        tiles
            .par_iter()
            .zip(blobs.par_iter_mut())
            .for_each(|(tile, blob)| encode_tile(tile, filter, blob));
    });
    let decompress = time(|| {
        blobs
            .par_iter()
            .for_each_init(Vec::new, |out, blob| decode_tile::<T>(blob, out).unwrap());
    });
    println!(
        "| {name} | {filter:?}, {} threads | | | {:.0} | {:.0} |",
        rayon::current_num_threads(),
        raw / compress / 1e6,
        raw / decompress / 1e6
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: tile_codec_bench DEM.tif [TILE]")?;
    let tile: usize = args.next().map_or(Ok(1024), |t| t.parse())?;
    let raster = terrio::open_raster_lazy(&path)?;
    let (width, height) = (raster.width(), raster.height());
    let nodata = raster.band_nodata(1).unwrap_or(-9999.0) as f32;
    let mut z = vec![0f32; width * height];
    if raster.band_data_type(1) != DataType::F32 {
        return Err("the benchmark reads Float32 DEMs".into());
    }
    raster.read_geotiff_samples_into(Window::new(0, 0, width, height), 1, None, &mut z)?;
    let grid = Grid::new(width, height).ok_or("raster too large")?;
    let [_, dx, _, _, _, dy] = raster.geo_transform();
    let start = Instant::now();
    pit_remove(grid, &mut z, nodata, false, None);
    let flow = d8_flow(grid, &z, FEL_NODATA, &[(dx.abs(), dy.abs())], false);
    let area = area_d8(grid, &flow.direction, true);
    let pipeline = start.elapsed().as_secs_f64();
    let p16: Vec<i16> = flow
        .direction
        .iter()
        .map(|&d| {
            if d == DIRECTION_NODATA {
                i16::MIN
            } else {
                i16::from(d)
            }
        })
        .collect();
    // Breadth-first distance from the nodata edge over valid cells: smooth
    // integer fields like the flat passes' fall and rise.
    let mut distance = vec![u32::MAX; width * height];
    let mut queue = std::collections::VecDeque::new();
    for (i, d) in distance.iter_mut().enumerate() {
        if z[i] == FEL_NODATA || i % width == 0 || i < width {
            *d = 0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for j in [i.wrapping_sub(1), i + 1, i.wrapping_sub(width), i + width] {
            if j < distance.len() && distance[j] == u32::MAX {
                distance[j] = distance[i] + 1;
                queue.push_back(j);
            }
        }
    }
    let spec = TileSpec::new(width, height, tile, tile)?;
    println!(
        "{path}: {width} x {height} ({:.1} M cells), {} tiles of {tile}, pipeline {pipeline:.2} s, {} threads",
        (width * height) as f64 / 1e6,
        spec.tile_count(),
        rayon::current_num_threads()
    );
    println!("\n| layer | prefilter | bytes/cell | ratio | compress MB/s | decompress MB/s |");
    println!("|---|---|---:|---:|---:|---:|");
    bench("fel f32", &tiles(&spec, &z));
    bench("p u8", &tiles(&spec, &flow.direction));
    bench("p i16", &tiles(&spec, &p16));
    bench("ad8 f32", &tiles(&spec, &area));
    bench("distance u32", &tiles(&spec, &distance));
    Ok(())
}
