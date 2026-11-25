// SPDX-License-Identifier: GPL-3.0-only
//! `pipeline --mode tiled`: the fused tiled pipeline
//! ([`taudem_rs::pipeline_tiled`]) from a DEM GeoTIFF, with each requested
//! output written by a background tile writer, within `--max-memory`.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use taudem_rs::area_tiled::AreaOptions;
use taudem_rs::d8_tiled::SteepSchedule;
use taudem_rs::pipeline_tiled::{
    PipelineOptions, PipelineOutcome, PipelinePlan, PipelineSinks, PipelineStats, pipeline_tiled,
    plan,
};
use taudem_rs::tiling::{AsyncTileWriter, ExecutionOptions, HaloWindow, TileId, workers};
use taudem_rs::{DIRECTION_NODATA, FEL_NODATA, SCALAR_NODATA, profile};
use terrio::{RasterDataset, RasterElement};

use crate::io::{P_NODATA, Raster, Result, band_bytes, tile_writer};

/// The output flags, in the order outputs are reported.
const OUTPUTS: [&str; 4] = ["-fel", "-sd8", "-p", "-ad8"];

pub struct Run {
    pub cells: u64,
    /// Opening the inputs and planning.
    pub open: Duration,
    /// Reading the DEM's tiles, summed over the workers and divided by
    /// their count (it overlaps compute).
    pub read: Duration,
    pub compute: Duration,
    /// Finishing the writers after the last tile.
    pub write: Duration,
}

/// An optional background writer and its sink.
struct Output<T> {
    path: String,
    writer: AsyncTileWriter<T>,
}

fn writer<T: RasterElement + Default>(
    flags: &HashMap<&str, &str>,
    flag: &str,
    header: &RasterDataset,
    plan: &PipelinePlan,
    nodata: f64,
) -> Result<Option<Output<T>>> {
    let Some(&path) = flags.get(flag) else {
        return Ok(None);
    };
    let queue = 2 * plan.workers;
    let writer = tile_writer::<T>(path, header, &plan.spec, nodata, queue)
        .map_err(|error| format!("writing {path}: {error}"))?;
    Ok(Some(Output {
        path: path.to_owned(),
        writer,
    }))
}

fn finish<T: RasterElement + Default>(output: Option<Output<T>>, name: &str) -> Result<()> {
    if let Some(Output { path, writer }) = output {
        let stats = writer
            .finish()
            .map_err(|error| format!("writing {path}: {error}"))?;
        stats.record_profile(name);
    }
    Ok(())
}

pub fn run(
    flags: &HashMap<&str, &str>,
    execution: &ExecutionOptions,
    check_contamination: bool,
) -> Result<Run> {
    let start = Instant::now();
    let mut raster = Raster::open(flags["-z"])?;
    let mut mask = flags
        .get("-depmask")
        .map(|path| Raster::open(path))
        .transpose()?;
    if let Some(mask) = &mask {
        raster.check_mask(mask)?;
    }
    let (width, height) = raster.dimensions();
    let spacing = raster.spacing()?;
    let threads = workers::threads();
    let area = AreaOptions::new(check_contamination);
    let outputs = OUTPUTS.iter().filter(|f| flags.contains_key(*f)).count();
    let layout = |execution: &ExecutionOptions| {
        plan(
            width,
            height,
            execution,
            raster.block_size(),
            threads,
            &area,
            outputs,
        )
    };
    let mut plan = layout(execution)?;
    // A striped DEM's halos come from full-width bands the reader keeps.
    let (tw, th) = plan.spec.tile_size();
    let bands = band_bytes(&raster, tw, th);
    if bands > 0 {
        let budget = execution.memory_budget();
        let reduced = ExecutionOptions {
            max_memory: Some(budget.saturating_sub(bands).max(1 << 20)),
            ..execution.clone()
        };
        plan = layout(&reduced)?;
        plan.budget = budget;
        plan.reserve_bytes += bands;
    }
    report_plan(&plan);
    let mut options = PipelineOptions {
        nodata: raster.nodata(),
        four_way: flags.contains_key("-4"),
        has_mask: mask.is_some(),
        area,
        schedule: SteepSchedule::Parallel,
    };
    let header = raster.header.clone();
    let dem = raster.tile_reader()?;
    let mask_reader = mask.as_mut().map(Raster::tile_reader).transpose()?;
    let read_ns = AtomicU64::new(0);
    let read = |halo: HaloWindow, z: &mut [f32], cells: Option<&mut [u8]>| {
        let start = Instant::now();
        dem.halo(&halo, z)?;
        if let (Some(reader), Some(cells)) = (&mask_reader, cells) {
            reader.mask(halo.core, cells)?;
        }
        read_ns.fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
        Ok(())
    };
    let open = start.elapsed();

    let start = Instant::now();
    let spec = plan.spec;
    let codes: Mutex<Vec<Vec<i16>>> = Mutex::new(Vec::new());
    let (stats, write) = loop {
        let fel = writer::<f32>(flags, "-fel", &header, &plan, f64::from(FEL_NODATA))?;
        let sd8 = writer::<f32>(flags, "-sd8", &header, &plan, f64::from(SCALAR_NODATA))?;
        let p = writer::<i16>(flags, "-p", &header, &plan, f64::from(P_NODATA))?;
        let ad8 = writer::<f32>(flags, "-ad8", &header, &plan, f64::from(SCALAR_NODATA))?;
        let fel_sink = fel.as_ref().map(|o| o.writer.sink());
        let sd8_sink = sd8.as_ref().map(|o| o.writer.sink());
        let p_sink = p.as_ref().map(|o| o.writer.sink());
        let ad8_sink = ad8.as_ref().map(|o| o.writer.sink());
        let fel_out = |id: TileId, data: &[f32]| {
            fel_sink
                .as_ref()
                .unwrap()
                .submit_region(spec.rect(id), data)
        };
        let sd8_out = |id: TileId, data: &[f32]| {
            sd8_sink
                .as_ref()
                .unwrap()
                .submit_region(spec.rect(id), data)
        };
        let p_out = |id: TileId, data: &[u8]| {
            let mut buffer = codes
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .pop()
                .unwrap_or_default();
            buffer.clear();
            buffer.extend(data.iter().map(|&code| {
                if code == DIRECTION_NODATA {
                    P_NODATA
                } else {
                    i16::from(code)
                }
            }));
            let result = p_sink
                .as_ref()
                .unwrap()
                .submit_region(spec.rect(id), &buffer);
            codes.lock().unwrap_or_else(|p| p.into_inner()).push(buffer);
            result
        };
        let ad8_out =
            |_: TileId, rect, data: &[f32]| ad8_sink.as_ref().unwrap().submit_region(rect, data);
        let sinks = PipelineSinks {
            fel: fel.is_some().then_some(&fel_out as _),
            sd8: sd8.is_some().then_some(&sd8_out as _),
            p: p.is_some().then_some(&p_out as _),
            ad8: ad8.is_some().then_some(&ad8_out as _),
        };
        let outcome = pipeline_tiled(
            &plan,
            &options,
            &spacing,
            execution.cache_config(plan.cache_bytes),
            &read,
            &sinks,
        )?;
        drop((fel_sink, sd8_sink, p_sink, ad8_sink));
        match outcome {
            PipelineOutcome::Done(stats) => {
                let finishing = Instant::now();
                finish(fel, "fel")?;
                finish(sd8, "sd8")?;
                finish(p, "p")?;
                finish(ad8, "ad8")?;
                break (stats, finishing.elapsed());
            }
            PipelineOutcome::CrossingDetected => {
                // Abandon every output and apply the crossing rule exactly.
                drop((fel, sd8, p, ad8));
                profile::count("pipeline.wavefront_restarts", 1);
                println!("Pipeline: crossing candidate found; rerunning in wavefront order");
                options.schedule = SteepSchedule::Wavefront;
            }
        }
    };
    let elapsed = start.elapsed();
    stats.record_profile();
    report(&stats);
    let read =
        Duration::from_nanos(read_ns.load(Relaxed) / plan.workers.max(1) as u64).min(elapsed);
    Ok(Run {
        cells: width as u64 * height as u64,
        open,
        read,
        compute: elapsed.saturating_sub(write),
        write,
    })
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn report_plan(plan: &PipelinePlan) {
    let (tw, th) = plan.spec.tile_size();
    println!(
        "Tiled pipeline: {} tiles of {tw} x {th}, {} workers; budget {:.0} MiB = cache {:.0} + \
         workers {:.0} ({:.0} each) + perimeters {:.0} + writers {:.0} + reserve {:.0}",
        plan.spec.tile_count(),
        plan.workers,
        mib(plan.budget),
        mib(plan.cache_bytes),
        mib(plan.workers as u64 * plan.worker_bytes),
        mib(plan.worker_bytes),
        mib(plan.perimeter_bytes),
        mib(plan.writer_bytes),
        mib(plan.reserve_bytes),
    );
}

fn report(stats: &PipelineStats) {
    let describe = |name: &str| match name {
        "pass_a" => "Pass A (PitRemove Pass 1, spill graph)",
        "pass_b" => "Pass B (PitRemove Pass 2 + D8 steep, sd8, tile flats)",
        "pass_c" => "Pass C (D8 flats across tiles)",
        "pass_d" => "Pass D (AreaD8 Pass 1)",
        "solve_area" => "AreaD8 perimeter solve",
        "pass_e" => "Pass E (AreaD8 Pass 2)",
        _ => "pass",
    };
    for pass in &stats.passes {
        let c = &pass.cache;
        println!(
            "{} time: {:.6}  [heap peak {:.0} MiB; codec {:.0} MiB in, {:.1} s compress, \
             {:.1} s decompress; spill {:.0} MiB out, {:.0} MiB in]",
            describe(pass.name),
            pass.seconds,
            mib(pass.heap_peak),
            mib(c.compress_in_bytes),
            c.compress_seconds,
            c.decompress_seconds,
            mib(c.spill_write_bytes),
            mib(c.spill_read_bytes),
        );
    }
    let d8 = &stats.d8;
    println!(
        "Tiled D8: {} flat cells, {} routed inside tiles, {} in {} edge tiles across {} passes",
        d8.flat_cells,
        d8.interior_cells,
        d8.deferred_cells,
        d8.deferred_tiles,
        d8.passes.len()
    );
    for (i, p) in d8.passes.iter().enumerate() {
        println!(
            "Tiled D8 flat pass {}: {} cells in {} tiles, {} rounds, {} tile loads, {:.3} s",
            i + 1,
            p.cells,
            p.tiles,
            p.rounds,
            p.relax_loads,
            p.seconds
        );
    }
    let g = &stats.area;
    println!(
        "Tiled AreaD8: {} exits, max area {}, {} exits replayed in {} tiles",
        g.nodes, g.max_area, g.replay_nodes, g.replay_tiles
    );
    let c = &stats.cache;
    println!(
        "Tile cache: budget {:.0} MiB, peak resident {:.0} MiB (raw {:.0}, compressed {:.0}), \
         peak spilled {:.0} MiB",
        mib(c.budget_bytes),
        mib(c.peak_resident_bytes),
        mib(c.peak_ram_bytes),
        mib(c.peak_compressed_bytes),
        mib(c.peak_spilled_bytes)
    );
}
