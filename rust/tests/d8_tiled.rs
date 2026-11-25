// SPDX-License-Identifier: GPL-3.0-only
//! Tiled D8FlowDir (WP-S2) must reproduce the in-memory `d8_flow` bit for
//! bit: directions and sd8, on the §6.1 suite at every tile size and thread
//! count, with spilled tile state, with the wavefront schedule, on unfilled
//! DEMs (undrained flats) and on random quantized terrains.

mod support;

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use support::fixtures::{Fixture, NODATA, THREADS, TILE_SIZES, suite};
use support::harness::{Layer, assert_suite, reference, stage};
use taudem_rs::d8_tiled::{D8Outcome, D8TileStats, SteepSchedule, d8_flow_tiled, halo_from_grid};
use taudem_rs::tiling::{CacheConfig, TileCache, TileSpec};
use taudem_rs::{FEL_NODATA, Grid, d8_flow, pit_remove};

/// The outputs of a tiled run.
struct Tiled {
    p: Vec<u8>,
    sd8: Vec<f32>,
    stats: D8TileStats,
}

/// Runs tiled D8 on a whole grid with a tile cache of `budget` bytes,
/// checking that every cell is delivered exactly once.
fn tiled(
    spec: &TileSpec,
    fel: &[f32],
    nodata: f32,
    spacing: (f64, f64),
    schedule: SteepSchedule,
    budget: u64,
) -> Tiled {
    let cells = fel.len();
    let width = spec.raster_width();
    let run = |schedule| {
        let cache = TileCache::new(CacheConfig::new(
            budget,
            Some(env!("CARGO_TARGET_TMPDIR").into()),
        ));
        let p = Mutex::new(vec![77u8; cells]);
        let sd8 = Mutex::new(vec![f32::NAN; cells]);
        let delivered = AtomicU64::new(0);
        let place = |id, data: &[u8]| {
            let rect = spec.rect(id);
            assert_eq!(data.len(), rect.cells());
            let mut p = p.lock().unwrap();
            for row in 0..rect.height {
                let at = (rect.y + row) * width + rect.x;
                let dst = &mut p[at..at + rect.width];
                assert!(dst.iter().all(|&v| v == 77), "tile {id:?} delivered twice");
                dst.copy_from_slice(&data[row * rect.width..(row + 1) * rect.width]);
            }
            delivered.fetch_add(rect.cells() as u64, Relaxed);
            Ok(())
        };
        let slope = |id, data: &[f32]| {
            let rect = spec.rect(id);
            let mut s = sd8.lock().unwrap();
            for row in 0..rect.height {
                let at = (rect.y + row) * width + rect.x;
                s[at..at + rect.width]
                    .copy_from_slice(&data[row * rect.width..(row + 1) * rect.width]);
            }
            Ok(())
        };
        let read = |id, _: &_, buf: &mut Vec<f32>| {
            halo_from_grid(spec, id, fel, f32::NAN, buf).map(|_| ())
        };
        let outcome = d8_flow_tiled(
            *spec,
            nodata,
            &[spacing],
            &cache,
            schedule,
            &read,
            Some(&slope),
            &place,
        )
        .expect("tiled d8");
        match outcome {
            D8Outcome::Done(stats) => {
                assert_eq!(delivered.load(Relaxed), cells as u64);
                Some(Tiled {
                    p: p.into_inner().unwrap(),
                    sd8: sd8.into_inner().unwrap(),
                    stats,
                })
            }
            D8Outcome::CrossingDetected => None,
        }
    };
    run(schedule).unwrap_or_else(|| run(SteepSchedule::Wavefront).expect("wavefront run"))
}

fn d8_stage(schedule: SteepSchedule, budget: u64) -> impl support::harness::TiledStage {
    stage(
        &format!("d8flowdir tiled ({schedule:?}, {budget} B cache)"),
        |f| {
            let (p, sd8) = reference::d8(f);
            vec![Layer::u8("p", p), Layer::f32("sd8", sd8)]
        },
        move |f, spec| {
            let fel = reference::fel(f);
            let t = tiled(spec, &fel, FEL_NODATA, f.spacing, schedule, budget);
            vec![Layer::u8("p", t.p), Layer::f32("sd8", t.sd8)]
        },
    )
}

#[test]
fn tiled_d8_matches_in_memory_on_the_suite() {
    assert_suite(
        &d8_stage(SteepSchedule::Parallel, 1 << 30),
        &TILE_SIZES,
        &THREADS,
    );
}

#[test]
fn tiled_d8_matches_with_spilled_tiles() {
    // A few kilobytes of cache: every tile is compressed or spilled.
    assert_suite(&d8_stage(SteepSchedule::Parallel, 8 << 10), &[16, 17], &[8]);
}

#[test]
fn wavefront_schedule_matches_in_memory() {
    assert_suite(
        &d8_stage(SteepSchedule::Wavefront, 1 << 30),
        &[16, 17],
        &THREADS,
    );
}

/// D8 on the raw DEM (d8flowdir -fel of an unfilled surface): pits leave
/// flats no drain reaches, which TauDEM marks nodata at a level above every
/// drained one, so the raster-wide highest fall level matters.
#[test]
fn unfilled_dems_match_in_memory() {
    let unfilled = stage(
        "d8flowdir tiled on unfilled DEMs",
        |f: &Fixture| {
            let z = f.dem_values().unwrap();
            let flow = d8_flow(f.grid(), z, f.nodata, &[f.spacing], true);
            vec![
                Layer::u8("p", flow.direction),
                Layer::f32("sd8", flow.slope.unwrap()),
            ]
        },
        |f: &Fixture, spec: &TileSpec| {
            let z = f.dem_values().unwrap();
            let t = tiled(
                spec,
                z,
                f.nodata,
                f.spacing,
                SteepSchedule::Parallel,
                1 << 30,
            );
            vec![Layer::u8("p", t.p), Layer::f32("sd8", t.sd8)]
        },
    );
    assert_suite(&unfilled, &TILE_SIZES, &THREADS);
}

/// Smooth random terrain quantized to 1/32 with plateaus, terraces and
/// nodata blobs: many flats on seams, of every size.
fn random_terrain(w: usize, h: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let waves: Vec<(f64, f64, f64, f64)> = (0..6)
        .map(|_| {
            (
                rnd() * 0.3,
                rnd() * 0.3,
                rnd() * std::f64::consts::TAU,
                0.5 + rnd() * 3.0,
            )
        })
        .collect();
    let step = [1.0 / 32.0, 0.25, 1.0][(rnd() * 3.0) as usize];
    let blobs: Vec<(f64, f64, f64)> = (0..(rnd() * 4.0) as usize)
        .map(|_| (rnd() * w as f64, rnd() * h as f64, 1.0 + rnd() * 4.0))
        .collect();
    let plateau = rnd() * 2.0;
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            if blobs
                .iter()
                .any(|&(bx, by, r)| (x - bx).powi(2) + (y - by).powi(2) < r * r)
            {
                return NODATA;
            }
            let mut z: f64 = waves
                .iter()
                .map(|&(a, b, ph, amp)| amp * (a * x + b * y + ph).sin())
                .sum();
            z = z.max(plateau);
            ((z / step).round() * step) as f32
        })
        .collect()
}

#[test]
fn random_terrains_match_in_memory() {
    let mut runs = 0;
    for seed in 0..60u64 {
        let w = 20 + (seed as usize * 7) % 45;
        let h = 18 + (seed as usize * 11) % 40;
        let dem = random_terrain(w, h, seed);
        let grid = Grid::new(w, h).unwrap();
        let spacing = if seed % 3 == 0 {
            (2.0, 1.0)
        } else {
            (1.0, 1.0)
        };
        let mut fel = dem.clone();
        pit_remove(grid, &mut fel, NODATA, false, None);
        for (input, nodata) in [(&fel, NODATA), (&dem, NODATA)] {
            let expected = d8_flow(grid, input, nodata, &[spacing], true);
            for t in [16, 17, 23] {
                let spec = TileSpec::new(w, h, t, t).unwrap();
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let got = pool.install(|| {
                        tiled(
                            &spec,
                            input,
                            nodata,
                            spacing,
                            SteepSchedule::Parallel,
                            1 << 30,
                        )
                    });
                    runs += 1;
                    assert!(
                        got.p == expected.direction,
                        "p differs: seed {seed}, {w}x{h}, T={t}, {threads} threads"
                    );
                    let slope = expected.slope.as_ref().unwrap();
                    assert!(
                        got.sd8
                            .iter()
                            .zip(slope)
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "sd8 differs: seed {seed}, {w}x{h}, T={t}"
                    );
                }
            }
        }
    }
    assert_eq!(runs, 60 * 2 * 3 * 2);
}

/// A tile-interior flat at 0 holding a patch one subnormal step higher.
/// With 10 m cells the patch's slope toward the flat underflows to 0, so the
/// patch is unresolved and, being higher, unreachable from the flat's drain:
/// a mixed component, routed with the raster-wide highest fall level. A
/// long corridor flat in another tile raises that level far above the
/// interior flat's own.
fn mixed_fixture(t: usize) -> (usize, usize, Vec<f32>) {
    let (w, h) = (4 * t, 2 * t);
    let tiny = f32::from_bits(1);
    let mut z = vec![10.0f32; w * h];
    let mut set = |x: usize, y: usize, v: f32| z[y * w + x] = v;
    // Interior flat of tile (1, 0) with its outlet inside the tile.
    for y in t / 4..3 * t / 4 {
        for x in t + t / 4..t + 3 * t / 4 {
            set(x, y, 0.0);
        }
    }
    set(t + t / 4 - 1, t / 2, -1.0);
    set(t + t / 4 - 2, t / 2, -2.0);
    for y in t / 2 - 1..t / 2 + 2 {
        for x in t + t / 2..t + t / 2 + 3 {
            set(x, y, tiny);
        }
    }
    // A corridor flat across tiles (2, 1) and (3, 1), drained at one end.
    let y = t + t / 2;
    for x in 2 * t + 1..4 * t - 1 {
        set(x, y, 5.0);
    }
    set(2 * t, y, 4.0);
    for x in 0..2 * t + 1 {
        set(x, y, 3.0 - x as f32 * 0.001);
    }
    (w, h, z)
}

#[test]
fn mixed_components_use_the_raster_wide_fall_level() {
    for t in [16, 17, 32] {
        let (w, h, z) = mixed_fixture(t);
        let grid = Grid::new(w, h).unwrap();
        let spacing = (10.0, 10.0);
        let expected = d8_flow(grid, &z, NODATA, &[spacing], true);
        let spec = TileSpec::new(w, h, t, t).unwrap();
        for threads in THREADS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let got = pool
                .install(|| tiled(&spec, &z, NODATA, spacing, SteepSchedule::Parallel, 1 << 30));
            assert!(got.stats.mixed_components > 0, "T={t}: no mixed component");
            let differing = (0..w * h)
                .filter(|&i| got.p[i] != expected.direction[i])
                .count();
            assert_eq!(differing, 0, "T={t}, {threads} threads");
        }
    }
}

/// Every Pass C flat pass is counted, and the counters add up.
#[test]
fn stats_account_for_every_flat_cell() {
    let (mut multi_pass, mut multi_round, mut wide) = (false, false, false);
    for f in suite(16) {
        let Some(dem) = f.dem_values() else {
            continue;
        };
        let spec = TileSpec::new(f.width, f.height, 16, 16).unwrap();
        let t = tiled(
            &spec,
            dem,
            f.nodata,
            f.spacing,
            SteepSchedule::Parallel,
            1 << 30,
        );
        let s = &t.stats;
        assert_eq!(s.tiles, spec.tile_count() as u64, "{}", f.name);
        assert_eq!(
            s.interior_cells + s.deferred_cells,
            s.flat_cells,
            "{}",
            f.name
        );
        if let Some(first) = s.passes.first() {
            assert_eq!(first.cells, s.deferred_cells, "{}", f.name);
        }
        for pair in s.passes.windows(2) {
            assert_eq!(pair[0].remaining, pair[1].cells, "{}", f.name);
        }
        assert_eq!(s.passes.last().map_or(0, |p| p.remaining), 0);
        let c = &s.components;
        assert_eq!(c.cells, s.deferred_cells, "{}", f.name);
        assert!(c.cross_tile <= c.components && c.cross_tile_cells <= c.cells);
        multi_pass |= s.passes.len() >= 3;
        multi_round |= s.passes.iter().any(|p| p.rounds >= 3);
        wide |= c.max_tiles >= 6 && c.max_tiles_across >= 3 && c.max_tiles_down >= 3;
    }
    // The suite exercises several flat passes, several relaxation rounds
    // and components spanning 3 x 3 tiles.
    assert!(multi_pass && multi_round && wide);
}
