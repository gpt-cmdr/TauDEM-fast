// SPDX-License-Identifier: GPL-3.0-only
//! The fused, bounded-memory pipeline: PitRemove → D8FlowDir → AreaD8 on one
//! tile grid, one tile cache and one memory budget, reproducing the
//! in-memory pipeline bit for bit (`docs/rust-port/STREAMING.md` §2.5).
//!
//! # Passes
//!
//! - **A** (DEM read once): PitRemove Pass 1 on every tile
//!   ([`PitRemovePlan::solve_in`]); local fills and labels go to the shared
//!   [`TileCache`]. The perimeter graph then gives every watershed its spill
//!   elevation.
//! - **B**: PitRemove Pass 2 raises each tile and hands its filled core and
//!   exact one-cell halo straight to D8 Pass B ([`TiledD8::process_tile`]):
//!   `fel` is never read back. `fel` and `sd8` go to their sinks; a tile whose
//!   flats stay inside it delivers its final directions at once.
//! - **C**: D8 Pass C routes the flats that cross tiles ([`TiledD8::finish`])
//!   and delivers the remaining tiles' directions.
//! - **D**: AreaD8 Pass 1 on the delivered directions, read from the cache
//!   with their halo from a border table ([`StoredDirections`]); the
//!   perimeter graph is solved (with TauDEM's `f32` replay above 2^24).
//! - **E**: AreaD8 Pass 2 writes `ad8`.
//!
//! Directions are delivered exactly once per tile (in B or C); each delivery
//! writes `p`, stores the tile's codes and their perimeter. All tile loops
//! run on plain worker threads ([`workers`]).
//!
//! # Memory
//!
//! [`plan`] splits `--max-memory` into a process reserve, the perimeter
//! structures (the largest of the passes: PitRemove's graph, D8's border
//! tables, AreaD8's graph), the writers' queues, the workers' buffers (the
//! largest working set of any pass) and the tile cache, which holds every
//! stage's tiles under one budget: uncompressed while they fit, compressed
//! by least recent use beyond, spilled to `--scratch-dir` beyond that.
//! Passes that sweep the cache alternate direction (B runs in reverse of A,
//! E in reverse of D) so the tiles the cache still holds uncompressed are
//! used first.
//!
//! # Exactness
//!
//! Each stage is the exact tiled stage of WP-S1..S3 on the same tiles: D8
//! receives the halos PitRemove's in-memory fill would give (WP-S1 proves the
//! filled halo exact), AreaD8 the final directions. The crossing rule is
//! handled as in standalone D8: when the parallel steep pass finds a
//! crossing candidate the run returns [`PipelineOutcome::CrossingDetected`]
//! and the caller reruns with [`SteepSchedule::Wavefront`], which consumes
//! PitRemove's tiles in wavefront order.

use std::sync::Mutex;
use std::time::Instant;

use crate::area_tiled::{
    AreaOptions, AreaScratch, AreaSink, AreaTiles, DirectionSource, GraphStats, StoredDirections,
    worker_bytes as area_worker_bytes,
};
use crate::d8_tiled::{D8Outcome, D8TileStats, SteepSchedule, TileOutput, TiledD8};
use crate::pitremove_tiled::{EmitOrder, PitRemovePlan, PitRemoveStats};
use crate::tiling::{
    BorderTable, CacheConfig, Error, ExecutionOptions, HaloWindow, Result, StoreOptions, TileCache,
    TileId, TileSpec, TileStats, auto_tile_size, workers,
};
use crate::{DIRECTION_NODATA, FEL_NODATA, MAX_TILE, profile};

/// Working bytes per halo cell of one worker in Pass B, the largest pass:
/// PitRemove's output buffers (raised core, labels, halo), D8's scratch
/// ([`crate::d8_tiled::WORK_BYTES_PER_CELL`]), the `sd8` tile and the
/// converted `p` codes.
pub const WORKER_BYTES_PER_CELL: u64 = 56;

/// Perimeter bytes per perimeter cell outside PitRemove's graph: D8's border
/// tables (directions, elevations, two level fields, labels and the next
/// pass's tables), the delivered directions' perimeters, and AreaD8's graph
/// (about 32 B per perimeter cell; the two never coexist, so the larger).
const PERIMETER_BYTES_PER_CELL: u64 = 40;

/// Held back for the process itself: its image, thread stacks, the reader's
/// block tables and decode buffers (at most an eighth of the budget), plus a
/// sixteenth of the budget for allocator growth.
const BASELINE_BYTES: u64 = 48 << 20;

/// The largest share of the budget the cache gets. The tile state of a large
/// raster exceeds any practical budget, so every layer is compressed or
/// spilled at some point; on ELC (1.1e9 cells) a 2.8 GB cache at 4G was only
/// 2.5% faster than 0.9 GB at 2G, at 1.9 GiB more peak memory. The rest of
/// the budget stays free for the operating system's file cache, which holds
/// the input and the outputs being written.
const CACHE_SHARE: f64 = 0.5;

/// The smallest cache a plan gives: below it the run spills almost every
/// tile but still runs.
const FLOOR_CACHE_BYTES: u64 = 4 << 20;

/// The smallest cache the planner accepts before it shrinks tiles.
const MIN_CACHE_BYTES: u64 = 64 << 20;

/// Tiles per thread the planner aims for, so that no pass ends with most
/// workers idle behind a few large tiles.
pub const TILES_PER_WORKER: usize = 8;

/// One output block of the writers (256 x 256 `f32`).
const OUTPUT_BLOCK_BYTES: u64 = 256 * 256 * 4;

/// How a fused run divides its memory budget ([`plan`]).
#[derive(Clone, Debug)]
pub struct PipelinePlan {
    pub spec: TileSpec,
    /// Worker threads of every pass.
    pub workers: usize,
    pub budget: u64,
    /// One worker's buffers in the largest pass.
    pub worker_bytes: u64,
    /// The perimeter structures' peak.
    pub perimeter_bytes: u64,
    /// The output writers' queues and partly assembled blocks.
    pub writer_bytes: u64,
    /// The process reserve.
    pub reserve_bytes: u64,
    /// The tile cache's budget: what is left.
    pub cache_bytes: u64,
}

/// Bytes of the fused run's working buffers per worker for `w x h` tiles.
pub fn worker_bytes(w: usize, h: usize, area: &AreaOptions) -> u64 {
    let halo = ((w + 2) * (h + 2)) as u64;
    let pass_b = WORKER_BYTES_PER_CELL * halo + (2 << 20);
    crate::pitremove_tiled::worker_bytes(w, h)
        .max(pass_b)
        .max(area_worker_bytes(w, h, area) + halo)
}

/// The perimeter structures' peak over the passes for `spec`.
pub fn perimeter_bytes(spec: &TileSpec) -> u64 {
    let (tw, th) = spec.tile_size();
    let perimeter = spec.tile_count() as u64 * 2 * (tw + th + 2) as u64;
    crate::pitremove_tiled::graph_bytes(spec).max(PERIMETER_BYTES_PER_CELL * perimeter)
}

fn writer_bytes(spec: &TileSpec, threads: usize, outputs: usize) -> u64 {
    let (tw, th) = spec.tile_size();
    let block = 256;
    // Tiles not made of whole 256-blocks leave blocks straddling seams in the
    // writer until their neighbors arrive (each output's block side is 256
    // or a divisor of the tile side, see the CLI's `block_size_for`).
    let unaligned = (spec.tiles_x() > 1 && !tw.is_multiple_of(block))
        || (spec.tiles_y() > 1 && !th.is_multiple_of(block));
    let partial = if unaligned {
        ((spec.raster_width() + (tw + th) * 2 * threads) * block * 4) as u64
    } else {
        0
    };
    outputs as u64 * (2 * threads as u64 * OUTPUT_BLOCK_BYTES + partial)
}

/// Plans a fused run of a `width x height` raster on `threads` threads
/// within `execution`'s budget, writing `outputs` rasters. An explicit
/// `--tile-size` is used as given; otherwise the automatic side (2048, or
/// 4096 for large rasters under large budgets; aligned to the input's
/// blocks) is halved down to 256 until the workers' buffers take at most
/// 40% of what the reserve, perimeter structures and writers leave, the
/// cache keeps at least 64 MiB, and there are at least
/// [`TILES_PER_WORKER`] tiles per thread (or the side is down to 512). The
/// cache gets the rest, at most [`CACHE_SHARE`] of the budget. One tile size
/// serves every stage.
pub fn plan(
    width: usize,
    height: usize,
    execution: &ExecutionOptions,
    native_block: Option<(usize, usize)>,
    threads: usize,
    area: &AreaOptions,
    outputs: usize,
) -> Result<PipelinePlan> {
    let threads = threads.max(1);
    let budget = execution.memory_budget();
    let reserve = BASELINE_BYTES.min(budget / 8) + budget / 16;
    let usable = budget.saturating_sub(reserve);
    let cells = width as u64 * height as u64;
    let layout = |spec: TileSpec| {
        let (tw, th) = spec.tile_size();
        let worker = worker_bytes(tw, th, area);
        let perimeter = perimeter_bytes(&spec);
        let writer = writer_bytes(&spec, threads, outputs);
        let room = usable.saturating_sub(perimeter + writer);
        let workers = ((room * 2 / 5) / worker.max(1)).clamp(1, threads as u64) as usize;
        let workers = workers.min(spec.tile_count());
        let cache = room
            .saturating_sub(workers as u64 * worker)
            .min((budget as f64 * CACHE_SHARE) as u64)
            .max(FLOOR_CACHE_BYTES);
        PipelinePlan {
            spec,
            workers,
            budget,
            worker_bytes: worker,
            perimeter_bytes: perimeter,
            writer_bytes: writer,
            reserve_bytes: reserve,
            cache_bytes: cache,
        }
    };
    // Enough tiles to keep every worker busy to the end of a pass, unless
    // that needs tiles under 512 cells.
    let balanced = |p: &PipelinePlan| {
        let (tw, th) = p.spec.tile_size();
        p.spec.tile_count() >= TILES_PER_WORKER * threads || tw.min(th) <= 512
    };
    let fits = |p: &PipelinePlan| {
        p.workers >= threads.min(p.spec.tile_count())
            && p.cache_bytes >= MIN_CACHE_BYTES
            && balanced(p)
    };
    let plan = match execution.tile_size {
        Some(size) => {
            if size > MAX_TILE {
                return Err(Error::Invalid(format!(
                    "the pipeline's tile side must be at most {MAX_TILE}"
                )));
            }
            layout(TileSpec::new(width, height, size, size)?)
        }
        None => {
            let mut target = auto_tile_size(cells, budget).min(MAX_TILE);
            let mut best: Option<PipelinePlan> = None;
            loop {
                let mut spec = TileSpec::aligned(width, height, target, native_block)?;
                let (tw, th) = spec.tile_size();
                // Strips or odd blocks can stretch an aligned tile; keep the
                // side near the target and within PitRemove's limit.
                if tw.max(th) > (2 * target).min(MAX_TILE) {
                    spec = TileSpec::new(width, height, target, target)?;
                }
                let candidate = layout(spec);
                if fits(&candidate) {
                    break candidate;
                }
                let better = best.as_ref().is_none_or(|b| {
                    (candidate.workers, candidate.cache_bytes) > (b.workers, b.cache_bytes)
                });
                if better {
                    best = Some(candidate);
                }
                if target <= 256 {
                    break best.expect("a candidate");
                }
                target /= 2;
            }
        }
    };
    Ok(plan)
}

/// The fixed inputs of a fused run.
#[derive(Clone, Debug)]
pub struct PipelineOptions {
    /// The DEM's nodata value.
    pub nodata: f32,
    /// `-4`: four-way PitRemove.
    pub four_way: bool,
    /// `-depmask`: the reader fills each tile's mask.
    pub has_mask: bool,
    pub area: AreaOptions,
    pub schedule: SteepSchedule,
}

/// Where the outputs go; a missing sink skips that output (AreaD8 still
/// runs, as the in-memory pipeline computes it without `-ad8`).
#[derive(Default)]
pub struct PipelineSinks<'a> {
    pub fel: Option<&'a TileOutput<'a, f32>>,
    pub sd8: Option<&'a TileOutput<'a, f32>>,
    /// Final directions, row-major cores, 255 for nodata.
    pub p: Option<&'a TileOutput<'a, u8>>,
    pub ad8: Option<&'a AreaSink<'a>>,
}

/// One pass of a fused run: its wall time and what went through the cache.
#[derive(Clone, Debug)]
pub struct PassRecord {
    pub name: &'static str,
    pub seconds: f64,
    /// Cache counters accumulated during the pass (peaks are the run's).
    pub cache: TileStats,
    /// The heap's high-water mark during the pass.
    pub heap_peak: u64,
}

/// Counters of a fused run.
#[derive(Clone, Debug)]
pub struct PipelineStats {
    pub plan: PipelinePlan,
    pub pitremove: PitRemoveStats,
    pub d8: D8TileStats,
    pub area: GraphStats,
    pub area_cells: u64,
    pub passes: Vec<PassRecord>,
    /// The whole run's cache counters.
    pub cache: TileStats,
}

/// The result of [`pipeline_tiled`].
#[derive(Debug)]
pub enum PipelineOutcome {
    Done(Box<PipelineStats>),
    /// D8's parallel steep pass found a crossing candidate: everything the
    /// sinks received is invalid. Rerun with [`SteepSchedule::Wavefront`].
    CrossingDetected,
}

/// The counters accumulated between two snapshots of one cache; the peaks
/// and current levels are taken from `after`.
pub fn stats_since(after: &TileStats, before: &TileStats) -> TileStats {
    TileStats {
        compressions: after.compressions - before.compressions,
        compress_in_bytes: after.compress_in_bytes - before.compress_in_bytes,
        compress_out_bytes: after.compress_out_bytes - before.compress_out_bytes,
        compress_seconds: after.compress_seconds - before.compress_seconds,
        decompressions: after.decompressions - before.decompressions,
        decompress_seconds: after.decompress_seconds - before.decompress_seconds,
        spill_writes: after.spill_writes - before.spill_writes,
        spill_write_bytes: after.spill_write_bytes - before.spill_write_bytes,
        spill_write_seconds: after.spill_write_seconds - before.spill_write_seconds,
        spill_reads: after.spill_reads - before.spill_reads,
        spill_read_bytes: after.spill_read_bytes - before.spill_read_bytes,
        spill_read_seconds: after.spill_read_seconds - before.spill_read_seconds,
        ram_hits: after.ram_hits - before.ram_hits,
        compressed_hits: after.compressed_hits - before.compressed_hits,
        spilled_hits: after.spilled_hits - before.spilled_hits,
        misses: after.misses - before.misses,
        ..after.clone()
    }
}

impl PipelineStats {
    /// Records the run for `--profile-json`: the plan, every pass's time,
    /// heap peak and cache traffic (`pipeline.<pass>.*`), and the stages'
    /// own counters.
    pub fn record_profile(&self) {
        use profile::{count, count_named};
        let p = &self.plan;
        let (tw, th) = p.spec.tile_size();
        count("pipeline.tile_width", tw as u64);
        count("pipeline.tile_height", th as u64);
        count("pipeline.tiles", p.spec.tile_count() as u64);
        count("pipeline.workers", p.workers as u64);
        count("pipeline.plan.budget_bytes", p.budget);
        count("pipeline.plan.worker_bytes", p.worker_bytes);
        count("pipeline.plan.perimeter_bytes", p.perimeter_bytes);
        count("pipeline.plan.writer_bytes", p.writer_bytes);
        count("pipeline.plan.reserve_bytes", p.reserve_bytes);
        count("pipeline.plan.cache_bytes", p.cache_bytes);
        for pass in &self.passes {
            let n = |field: &str| format!("pipeline.{}.{field}", pass.name);
            let c = &pass.cache;
            count_named(&n("us"), (pass.seconds * 1e6) as u64);
            count_named(&n("heap_peak_bytes"), pass.heap_peak);
            count_named(&n("compressions"), c.compressions);
            count_named(&n("compress_in_bytes"), c.compress_in_bytes);
            count_named(&n("compress_out_bytes"), c.compress_out_bytes);
            count_named(&n("compress_us"), (c.compress_seconds * 1e6) as u64);
            count_named(&n("decompressions"), c.decompressions);
            count_named(&n("decompress_us"), (c.decompress_seconds * 1e6) as u64);
            count_named(&n("spill_write_bytes"), c.spill_write_bytes);
            count_named(&n("spill_read_bytes"), c.spill_read_bytes);
            count_named(
                &n("spill_us"),
                ((c.spill_write_seconds + c.spill_read_seconds) * 1e6) as u64,
            );
            count_named(&n("ram_hits"), c.ram_hits);
            count_named(&n("compressed_hits"), c.compressed_hits);
            count_named(&n("spilled_hits"), c.spilled_hits);
        }
        // PitRemove's emit records its tile counts itself.
        self.d8.record_profile();
        let g = &self.area;
        count("aread8.cells_accumulated", self.area_cells);
        count("aread8.graph.nodes", g.nodes);
        count("aread8.graph.entries", g.entries);
        count("aread8.graph.edges", g.edges);
        count("aread8.graph.inflow_nodes", g.inflow_nodes);
        count("aread8.graph.poisoned_nodes", g.poisoned_nodes);
        count("aread8.graph.cycle_nodes", g.cycle_nodes);
        count("aread8.graph.max_area", g.max_area);
        count("aread8.graph.bytes", g.bytes);
        count("aread8.replay.nodes", g.replay_nodes);
        count("aread8.replay.tiles", g.replay_tiles);
        count("aread8.replay.steps", g.replay_steps);
        self.cache.record_profile();
    }
}

/// Times passes and takes the cache's counters at their ends.
struct Passes<'a> {
    cache: &'a TileCache,
    records: Vec<PassRecord>,
    last: TileStats,
}

impl<'a> Passes<'a> {
    fn new(cache: &'a TileCache) -> Self {
        Self {
            cache,
            records: Vec::new(),
            last: cache.stats(),
        }
    }

    fn run<T>(&mut self, name: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let span = profile::span(format!("pipeline.{name}"));
        let start = Instant::now();
        let value = f()?;
        let seconds = start.elapsed().as_secs_f64();
        // Inside the span the heap peak is the pass's own.
        let heap_peak = profile::heap_peak() as u64;
        drop(span);
        let now = self.cache.stats();
        self.records.push(PassRecord {
            name,
            seconds,
            cache: stats_since(&now, &self.last),
            heap_peak,
        });
        self.last = now;
        Ok(value)
    }
}

/// Fills a tile's radius-1 halo buffer of elevations (cells outside the
/// raster are ignored) and, with `-depmask`, its row-major core mask.
pub type ReadDem<'a> = dyn Fn(HaloWindow, &mut [f32], Option<&mut [u8]>) -> Result<()> + Sync + 'a;

/// Runs the fused pipeline over `plan`'s tiles. `read` is called once per
/// tile, from the worker threads. `spacing` is one `(dx, dy)` or one per
/// raster row; `cache` configures the shared tile cache (its budget should
/// be `plan.cache_bytes`).
pub fn pipeline_tiled(
    plan: &PipelinePlan,
    options: &PipelineOptions,
    spacing: &[(f64, f64)],
    cache: CacheConfig,
    read: &ReadDem<'_>,
    sinks: &PipelineSinks<'_>,
) -> Result<PipelineOutcome> {
    let spec = plan.spec;
    let workers = plan.workers;
    let cache = TileCache::new(cache);
    let mut passes = Passes::new(&cache);

    // Pass A: the only read of the DEM.
    let pit = PitRemovePlan::with_workers(
        spec,
        workers,
        options.nodata,
        options.four_way,
        options.has_mask,
    )?
    .keep_raw();
    let solved = passes.run("pass_a", || pit.solve_in(&cache, read))?;

    // Pass B: filled tiles straight into D8.
    let mut d8 = TiledD8::new(spec, FEL_NODATA, spacing, &cache, options.schedule)?;
    d8.set_workers(workers);
    let codes = cache.store::<u8>("pipeline.p", StoreOptions::default());
    let code_borders = BorderTable::new(spec, DIRECTION_NODATA);
    let deliver = |id: TileId, dirs: &[u8]| -> Result<()> {
        if let Some(p) = sinks.p {
            p(id, dirs)?;
        }
        code_borders.set_from_tile(id, dirs)?;
        codes.put(id, dirs)
    };
    let slopes: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
    let order = match options.schedule {
        SteepSchedule::Parallel => EmitOrder::Reverse,
        SteepSchedule::Wavefront => EmitOrder::Wavefront,
    };
    let pitremove = passes.run("pass_b", || {
        solved.emit_in(order, |tile| {
            if let Some(fel) = sinks.fel {
                fel(tile.id, tile.core)?;
            }
            if d8.crossing_detected() {
                // The run will be discarded; skip the rest of the work.
                return Ok(());
            }
            let mut sd8 = slopes
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .pop()
                .unwrap_or_default();
            d8.process_tile(
                tile.id,
                tile.values,
                sinks.sd8.is_some().then_some(&mut sd8),
                |dirs| deliver(tile.id, dirs),
            )?;
            if let Some(sink) = sinks.sd8 {
                sink(tile.id, &sd8)?;
            }
            slopes.lock().unwrap_or_else(|p| p.into_inner()).push(sd8);
            Ok(())
        })
    })?;
    drop(slopes);

    // Pass C: flats across tiles.
    let d8_stats = match passes.run("pass_c", || d8.finish(&deliver))? {
        D8Outcome::Done(stats) => stats,
        D8Outcome::CrossingDetected => return Ok(PipelineOutcome::CrossingDetected),
    };

    // Pass D: AreaD8 Pass 1 and the perimeter solve.
    let count = spec.tile_count();
    let tiles = AreaTiles::new(spec, options.area)?;
    let stored = StoredDirections::new(&codes, &code_borders, false);
    let mut buffers: Vec<(Vec<u8>, AreaScratch)> = (0..workers.min(count))
        .map(|_| (Vec::new(), AreaScratch::new()))
        .collect();
    passes.run("pass_d", || {
        workers::run_with(&mut buffers, count, |(dir, scratch), index| {
            let id = spec.id_at(index);
            stored.read_halo(id, &spec.halo_window(id), dir)?;
            tiles.pass1(id, dir, scratch)
        })
    })?;
    let solved_area = passes.run("solve_area", || tiles.solve(&stored, workers))?;

    // Pass E: final areas, in reverse of Pass D's order (see the module
    // documentation), each tile's codes dropped after their last use.
    let last = StoredDirections::new(&codes, &code_borders, true);
    let area_cells = passes.run("pass_e", || {
        workers::run_with(&mut buffers, count, |(dir, scratch), index| {
            let id = spec.id_at(count - 1 - index);
            let halo = spec.halo_window(id);
            last.read_halo(id, &halo, dir)?;
            let (area, cells) = solved_area.pass2(id, dir, scratch)?;
            match sinks.ad8 {
                Some(sink) => sink(id, halo.core, area)?,
                None => {
                    std::hint::black_box(area);
                }
            }
            Ok(cells)
        })
    })?;
    let stats = PipelineStats {
        plan: plan.clone(),
        pitremove,
        d8: d8_stats,
        area: solved_area.stats().clone(),
        area_cells,
        passes: passes.records,
        cache: cache.stats(),
    };
    Ok(PipelineOutcome::Done(Box::new(stats)))
}
