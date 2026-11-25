// SPDX-License-Identifier: GPL-3.0-only
//! The fused tiled pipeline (WP-S4) must reproduce the in-memory pipeline
//! bit for bit: fel, p, sd8 and ad8 on the §6.1 suite at every tile size and
//! thread count, with every tile compressed or spilled, with the wavefront
//! schedule, without optional outputs, and delivering every tile once.

mod support;

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use support::fixtures::{Fixture, THREADS, TILE_SIZES, suite};
use support::harness::{Layer, assert_suite, reference, stage};
use taudem_rs::area_d8;
use taudem_rs::area_tiled::AreaOptions;
use taudem_rs::d8_tiled::SteepSchedule;
use taudem_rs::pipeline_tiled::{
    PipelineOptions, PipelineOutcome, PipelinePlan, PipelineSinks, PipelineStats, pipeline_tiled,
    plan,
};
use taudem_rs::tiling::{CacheConfig, ExecutionOptions, HaloWindow, TileId, TileSpec, workers};

/// The outputs of a fused run, and its counters.
struct Fused {
    fel: Vec<f32>,
    p: Vec<u8>,
    sd8: Vec<f32>,
    ad8: Vec<f32>,
    stats: Box<PipelineStats>,
}

/// A plan for `spec` with `cache_bytes` of cache on the current pool.
fn plan_for(spec: &TileSpec, cache_bytes: u64) -> PipelinePlan {
    PipelinePlan {
        spec: *spec,
        workers: workers::threads(),
        budget: cache_bytes,
        worker_bytes: 0,
        perimeter_bytes: 0,
        writer_bytes: 0,
        reserve_bytes: 0,
        cache_bytes,
    }
}

/// Runs the fused pipeline on a fixture, asserting every output cell is
/// written exactly once; `outputs` false leaves out fel, sd8 and p.
fn fused(
    f: &Fixture,
    plan: &PipelinePlan,
    schedule: SteepSchedule,
    outputs: bool,
) -> Option<Fused> {
    let spec = plan.spec;
    let dem = f.dem_values().expect("a DEM fixture");
    let width = f.width;
    let cells = f.cells();
    let read = |halo: HaloWindow, z: &mut [f32], mask: Option<&mut [u8]>| {
        for row in 0..halo.window.height {
            let src = (halo.window.y + row) * width + halo.window.x;
            let dst = (row + halo.pad_top) * halo.buffer_width() + halo.pad_left;
            z[dst..dst + halo.window.width].copy_from_slice(&dem[src..src + halo.window.width]);
        }
        if let Some(mask) = mask {
            let raw = f.mask.as_ref().expect("a mask");
            for (row, dst) in mask.chunks_exact_mut(halo.core.width).enumerate() {
                let src = (halo.core.y + row) * width + halo.core.x;
                dst.copy_from_slice(&raw[src..src + halo.core.width]);
            }
        }
        Ok(())
    };
    let fel = Mutex::new(vec![f32::NAN; cells]);
    let sd8 = Mutex::new(vec![f32::NAN; cells]);
    let ad8 = Mutex::new(vec![f32::NAN; cells]);
    let p = Mutex::new(vec![77u8; cells]);
    let delivered = AtomicU64::new(0);
    fn place<T: Copy + PartialEq>(
        out: &Mutex<Vec<T>>,
        spec: &TileSpec,
        id: TileId,
        data: &[T],
        unset: impl Fn(T) -> bool,
    ) {
        let rect = spec.rect(id);
        assert_eq!(data.len(), rect.cells(), "tile {id:?}");
        let width = spec.raster_width();
        let mut out = out.lock().unwrap();
        for row in 0..rect.height {
            let at = (rect.y + row) * width + rect.x;
            let dst = &mut out[at..at + rect.width];
            assert!(dst.iter().all(|&v| unset(v)), "tile {id:?} written twice");
            dst.copy_from_slice(&data[row * rect.width..(row + 1) * rect.width]);
        }
    }
    let fel_sink = |id, data: &[f32]| {
        place(&fel, &spec, id, data, f32::is_nan);
        Ok(())
    };
    let sd8_sink = |id, data: &[f32]| {
        place(&sd8, &spec, id, data, f32::is_nan);
        Ok(())
    };
    let p_sink = |id, data: &[u8]| {
        place(&p, &spec, id, data, |v| v == 77);
        delivered.fetch_add(data.len() as u64, Relaxed);
        Ok(())
    };
    let ad8_sink = |id, rect: taudem_rs::tiling::Rect, data: &[f32]| {
        assert_eq!(rect, spec.rect(id));
        place(&ad8, &spec, id, data, f32::is_nan);
        Ok(())
    };
    let sinks = PipelineSinks {
        fel: outputs.then_some(&fel_sink as _),
        sd8: outputs.then_some(&sd8_sink as _),
        p: outputs.then_some(&p_sink as _),
        ad8: Some(&ad8_sink),
    };
    let options = PipelineOptions {
        nodata: f.nodata,
        four_way: f.four_way,
        has_mask: f.mask.is_some(),
        area: AreaOptions::new(f.check_contamination),
        schedule,
    };
    let cache = CacheConfig::new(plan.cache_bytes, Some(env!("CARGO_TARGET_TMPDIR").into()));
    match pipeline_tiled(plan, &options, &[f.spacing], cache, &read, &sinks).unwrap() {
        PipelineOutcome::Done(stats) => {
            if outputs {
                assert_eq!(delivered.load(Relaxed), cells as u64, "{}", f.name);
            }
            Some(Fused {
                fel: fel.into_inner().unwrap(),
                p: p.into_inner().unwrap(),
                sd8: sd8.into_inner().unwrap(),
                ad8: ad8.into_inner().unwrap(),
                stats,
            })
        }
        PipelineOutcome::CrossingDetected => None,
    }
}

fn in_memory(f: &Fixture) -> Vec<Layer> {
    let fel = reference::fel(f);
    let (p, sd8) = reference::d8(f);
    let ad8 = area_d8(f.grid(), &p, f.check_contamination);
    vec![
        Layer::f32("fel", fel),
        Layer::u8("p", p),
        Layer::f32("sd8", sd8),
        Layer::f32("ad8", ad8),
    ]
}

fn layers(run: Fused) -> Vec<Layer> {
    vec![
        Layer::f32("fel", run.fel),
        Layer::u8("p", run.p),
        Layer::f32("sd8", run.sd8),
        Layer::f32("ad8", run.ad8),
    ]
}

/// The parallel schedule, falling back to the wavefront as the CLI does.
fn fused_or_wavefront(f: &Fixture, plan: &PipelinePlan) -> Fused {
    fused(f, plan, SteepSchedule::Parallel, true)
        .or_else(|| fused(f, plan, SteepSchedule::Wavefront, true))
        .expect("the wavefront schedule always completes")
}

#[test]
fn fused_pipeline_matches_in_memory_on_the_adversarial_suite() {
    let pipeline = stage("fused tiled pipeline", in_memory, |f, spec| {
        layers(fused_or_wavefront(f, &plan_for(spec, 1 << 30)))
    });
    assert_suite(&pipeline, &TILE_SIZES, &THREADS);
}

#[test]
fn fused_pipeline_matches_with_every_tile_compressed_or_spilled() {
    let pipeline = stage("fused tiled pipeline, spilling", in_memory, |f, spec| {
        let run = fused_or_wavefront(f, &plan_for(spec, 4 << 10));
        // Raw tiles of more than 2 KiB cannot all fit 4 KiB.
        if f.cells() > 4096 {
            let c = &run.stats.cache;
            assert!(
                c.compressions > 0 && c.spill_writes > 0,
                "{}: {c:?}",
                f.name
            );
        }
        layers(run)
    });
    assert_suite(&pipeline, &[17], &THREADS);
}

#[test]
fn fused_pipeline_matches_with_the_wavefront_schedule() {
    let pipeline = stage("fused tiled pipeline, wavefront", in_memory, |f, spec| {
        layers(fused(f, &plan_for(spec, 1 << 30), SteepSchedule::Wavefront, true).unwrap())
    });
    assert_suite(&pipeline, &[16, 32], &THREADS);
}

#[test]
fn fused_pipeline_without_optional_outputs_still_computes_areas() {
    for f in suite(16).iter().filter(|f| f.dem_values().is_some()) {
        let spec = TileSpec::new(f.width, f.height, 16, 16).unwrap();
        let run = fused(f, &plan_for(&spec, 1 << 30), SteepSchedule::Parallel, false)
            .expect("no crossing on the suite");
        let expected = reference::ad8(f);
        assert!(
            run.ad8
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{}",
            f.name
        );
        assert_eq!(run.stats.passes.len(), 6, "{}", f.name);
    }
}

#[test]
fn planner_fits_the_budget_and_shrinks_tiles() {
    let area = AreaOptions::new(true);
    for (budget, cells) in [
        (512u64 << 20, 36_000_000usize),
        (2 << 30, 83_000_000),
        (2 << 30, 1_139_000_000),
        (4 << 30, 1_139_000_000),
    ] {
        let execution = ExecutionOptions {
            max_memory: Some(budget),
            ..Default::default()
        };
        let width = 36_000.min(cells);
        let height = cells / width;
        let p = plan(width, height, &execution, Some((256, 256)), 8, &area, 4).unwrap();
        let used = p.reserve_bytes
            + p.perimeter_bytes
            + p.writer_bytes
            + p.workers as u64 * p.worker_bytes
            + p.cache_bytes;
        assert!(used <= budget, "{budget}: {p:?}");
        assert!(p.cache_bytes <= budget / 2, "{p:?}");
        assert_eq!(p.workers, 8, "{p:?}");
        let (tw, th) = p.spec.tile_size();
        assert!(tw >= 256 && th >= 256 && tw % 256 == 0, "{p:?}");
        assert!(p.cache_bytes >= 64 << 20, "{p:?}");
    }
    // An explicit tile size is kept.
    let execution = ExecutionOptions {
        max_memory: Some(1 << 30),
        tile_size: Some(17),
        ..Default::default()
    };
    let p = plan(100, 67, &execution, None, 8, &area, 4).unwrap();
    assert_eq!(p.spec.tile_size(), (17, 17));
}
