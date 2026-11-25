// SPDX-License-Identifier: GPL-3.0-only
//! Out-of-core Barnes (2016) Priority-Flood. Pass 1 stores local fills and
//! their watershed labels. The perimeter graph determines each watershed's
//! global spill level. Pass 2 raises the stored cells and supplies their halo.

use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::flood::{FloodQueue, minimax, priority};
use crate::tiling::{
    BorderTable, Error, ExecutionOptions, HaloWindow, Rect, Result, StoreOptions, Tier, TileCache,
    TileId, TileSpec, TileStats,
};
use crate::{DX, DY, FEL_NODATA, Grid, MAX_TILE, missing, profile};

const NONE: u16 = u16::MAX;
const UNSET: u16 = NONE - 1;
const SETTLED: u16 = 0x8000;
const ORIGINAL_ZERO: u16 = 0x4000;
const ORIGINAL_NEGATIVE: u16 = 0x8000;
const LABEL: u16 = 0x3fff;
const ZERO_OUTLET: u16 = LABEL - 1;

#[derive(Clone, Copy)]
struct Seed {
    label: u16,
    height: f32,
}

const VOID: Seed = Seed {
    label: NONE,
    height: f32::NAN,
};

#[derive(Default)]
struct TileGraph {
    labels: u16,
    has_zero: bool,
    edges: Vec<(u16, u16, f32)>,
}

struct Graphs {
    borders: BorderTable<Seed>,
    tiles: Vec<OnceLock<TileGraph>>,
}

struct SpillLevels {
    base: Vec<u32>,
    levels: Vec<f32>,
}

impl SpillLevels {
    #[inline]
    fn tile(&self, index: usize) -> &[f32] {
        &self.levels[self.base[index] as usize..self.base[index + 1] as usize]
    }
}

/// Bounded execution parameters for a tiled PitRemove. The caller can choose
/// a smaller tile size or a larger budget if construction fails.
#[derive(Clone)]
pub struct PitRemovePlan {
    spec: TileSpec,
    workers: usize,
    cache_bytes: u64,
    options: ExecutionOptions,
    nodata: f32,
    step: usize,
    has_mask: bool,
}

impl PitRemovePlan {
    pub fn new(
        spec: TileSpec,
        options: &ExecutionOptions,
        nodata: f32,
        four_way: bool,
        has_mask: bool,
    ) -> Result<Self> {
        let (w, h) = spec.tile_size();
        if w.max(h) > MAX_TILE {
            return Err(Error::Invalid(format!(
                "PitRemove tile sides must be at most {MAX_TILE}"
            )));
        }
        let perimeter = perimeter_capacity(spec);
        if perimeter >= u32::MAX as u64 {
            return Err(Error::Invalid(
                "PitRemove needs fewer than 2^32 perimeter nodes; increase --tile-size".into(),
            ));
        }
        // Includes local forests, the global edge list and union-find, the
        // temporary signed-zero graph, and the final border table.
        let graph_bytes = 112 * perimeter + 768 * spec.tile_count() as u64;
        // Includes I/O and codec buffers, local flood queues, and Pass 2's
        // windows. Reserve GeoTIFF block scratch even for small compute tiles.
        let worker_bytes = 48 * ((w + 2) * (h + 2)) as u64 + (2 << 20);
        let available = options.memory_budget().saturating_sub(graph_bytes);
        let workers = (available / worker_bytes)
            .min(rayon::current_num_threads() as u64)
            .min(spec.tile_count() as u64) as usize;
        if workers == 0 {
            return Err(Error::Invalid(format!(
                "PitRemove needs about {} MiB for this tile layout; increase --max-memory or change --tile-size",
                (graph_bytes + worker_bytes).div_ceil(1 << 20)
            )));
        }
        Ok(Self {
            spec,
            workers,
            cache_bytes: available - workers as u64 * worker_bytes,
            options: options.clone(),
            nodata,
            step: if four_way { 2 } else { 1 },
            has_mask,
        })
    }

    pub fn spec(&self) -> TileSpec {
        self.spec
    }

    pub fn workers(&self) -> usize {
        self.workers
    }

    /// Runs Pass 1 and solves the perimeter graph. The reader fills the
    /// padded radius-one elevation buffer and, if enabled, the row-major
    /// core mask. Its calls may overlap. It is no longer needed after this
    /// function returns. Cells outside the raster are ignored.
    pub fn solve(
        &self,
        read: impl Fn(HaloWindow, &mut [f32], Option<&mut [u8]>) -> Result<()> + Sync,
    ) -> Result<SolvedPitRemove> {
        let cache = TileCache::new(self.options.cache_config(self.cache_bytes));
        let store_options = StoreOptions {
            initial: Tier::Compressed,
            ..Default::default()
        };
        let elevations = cache.store("pitremove.local_fel", store_options);
        let labels = cache.store("pitremove.labels", store_options);
        let graphs = Graphs::new(self.spec);
        let negative_zero = AtomicBool::new(false);
        let pass = profile::span("pitremove.tiled.pass1");
        let start = Instant::now();
        self.for_each_tile::<LocalScratch>(|index, scratch| {
            let id = self.spec.id_at(index);
            let halo = self.spec.halo_window(id);
            scratch.raw.resize(halo.buffer_cells(), f32::NAN);
            if self.has_mask {
                scratch.mask.resize(halo.core.cells(), 0);
            }
            read(
                halo,
                &mut scratch.raw,
                self.has_mask.then_some(scratch.mask.as_mut_slice()),
            )?;
            blank_padding(halo, &mut scratch.raw);
            let graph = scratch.flood(halo, self.nodata, self.step, &negative_zero);
            if let Some(mut graph) = graph {
                graphs.publish(index, halo, &scratch.labels, &scratch.filled);
                halo.core_from_buffer(&scratch.filled, &mut scratch.core)?;
                graph.has_zero = scratch.core.contains(&0.0);
                elevations.put(id, &scratch.core)?;
                scratch.pack_labels(halo, self.nodata, self.step);
                labels.put(id, &scratch.packed)?;
                graphs.tiles[index].set(graph).ok().unwrap();
            }
            Ok(())
        })?;
        let pass1 = start.elapsed();
        drop(pass);
        let empty_tiles = (self.spec.tile_count() - elevations.len()) as u64;
        let solve = profile::span("pitremove.tiled.solve");
        let start = Instant::now();
        let has_negative_zero = negative_zero.load(Relaxed);
        let spill = graphs.solve(self.step);
        let borders = graphs.numeric_borders(&spill, has_negative_zero);
        let zero_candidates = has_negative_zero.then(|| {
            graphs
                .tiles
                .iter()
                .enumerate()
                .map(|(index, tile)| {
                    tile.get()
                        .is_some_and(|g| g.has_zero || spill.tile(index).contains(&0.0))
                })
                .collect::<Vec<_>>()
        });
        drop(graphs);
        let mut result = SolvedPitRemove {
            plan: self.clone(),
            elevations,
            labels,
            spill,
            borders,
            zeros: None,
            cache,
            pass1,
            graph: Duration::ZERO,
            empty_tiles,
        };
        if let Some(candidates) = zero_candidates {
            result.fix_zero_signs(store_options, &candidates)?;
        }
        result.graph = start.elapsed();
        drop(solve);
        Ok(result)
    }

    fn for_each_tile<S: Default>(
        &self,
        f: impl Fn(usize, &mut S) -> Result<()> + Sync,
    ) -> Result<()> {
        let next = AtomicUsize::new(0);
        let failed = AtomicBool::new(false);
        (0..self.workers).into_par_iter().try_for_each(|_| {
            let mut scratch = S::default();
            while !failed.load(Relaxed) {
                let index = next.fetch_add(1, Relaxed);
                if index >= self.spec.tile_count() {
                    break;
                }
                if let Err(error) = f(index, &mut scratch) {
                    failed.store(true, Relaxed);
                    return Err(error);
                }
            }
            Ok(())
        })
    }
}

/// Pass timing and cache counters. A tile carries two independently cached
/// layers (elevation and labels), plus zero labels on signed-zero terrains.
pub struct PitRemoveStats {
    pub pass1: Duration,
    pub graph: Duration,
    pub pass2: Duration,
    pub empty_tiles: u64,
    pub spilled_tiles: u64,
    pub cache: TileStats,
}

/// A borrowed radius-one filled tile, available until its consumer returns.
pub struct FilledTile<'a> {
    pub id: TileId,
    pub halo: HaloWindow,
    /// Row-major values including the radius-one halo.
    pub values: &'a [f32],
    /// Contiguous row-major core, for consumers that write or store it.
    pub core: &'a [f32],
    pub all_nodata: bool,
}

/// A solved flood, with its intermediate tiles and final perimeter elevations.
pub struct SolvedPitRemove {
    plan: PitRemovePlan,
    elevations: crate::tiling::TileStore<f32>,
    labels: crate::tiling::TileStore<u16>,
    spill: SpillLevels,
    borders: BorderTable<f32>,
    zeros: Option<(crate::tiling::TileStore<u16>, SpillLevels)>,
    cache: TileCache,
    pass1: Duration,
    graph: Duration,
    empty_tiles: u64,
}

#[derive(Default)]
struct OutputScratch {
    core: Vec<f32>,
    labels: Vec<u16>,
    buffer: Vec<f32>,
    zeros: Vec<u16>,
}

impl SolvedPitRemove {
    /// Pass 2 raises each local tile to the global spill elevations, then
    /// passes its core and one-cell halo to the consumer. The consumer may
    /// run concurrently on several workers. The raster is never read again.
    pub fn emit(
        self,
        consume: impl Fn(FilledTile<'_>) -> Result<()> + Sync,
    ) -> Result<PitRemoveStats> {
        let spilled_tiles = if self.cache.stats().spill_writes == 0 {
            0
        } else {
            self.plan
                .spec
                .row_major()
                .filter(|&id| {
                    self.elevations.tier(id) == Some(Tier::Spilled)
                        || self.labels.tier(id) == Some(Tier::Spilled)
                        || self
                            .zeros
                            .as_ref()
                            .is_some_and(|(store, _)| store.tier(id) == Some(Tier::Spilled))
                })
                .count() as u64
        };
        profile::count("pitremove.tiled.spilled_tiles", spilled_tiles);
        profile::count("pitremove.tiled.empty_tiles", self.empty_tiles);
        let pass = profile::span("pitremove.tiled.pass2");
        let start = Instant::now();
        self.plan.for_each_tile::<OutputScratch>(|index, scratch| {
            let id = self.plan.spec.id_at(index);
            let halo = self.plan.spec.halo_window(id);
            let populated = self.numeric_tile(index, scratch)?;
            if let Some((store, levels)) = &self.zeros
                && store.try_get_into(id, &mut scratch.zeros)?
            {
                let spill = levels.tile(index);
                for (z, &label) in scratch.core.iter_mut().zip(&scratch.zeros) {
                    if label == 0 || (label < UNSET && spill[label as usize - 1] == 0.0) {
                        *z = -0.0;
                    }
                }
                store.remove(id);
            }
            if !populated {
                scratch.core.resize(halo.core.cells(), FEL_NODATA);
                scratch.core.fill(FEL_NODATA);
            }
            scratch.buffer.resize(halo.buffer_cells(), FEL_NODATA);
            self.borders
                .fill_halo_ring(id, &halo, &mut scratch.buffer, FEL_NODATA)?;
            for (dest, src) in scratch
                .buffer
                .chunks_exact_mut(halo.buffer_width())
                .skip(1)
                .zip(scratch.core.chunks_exact(halo.core.width))
            {
                dest[1..=halo.core.width].copy_from_slice(src);
            }
            self.elevations.remove(id);
            self.labels.remove(id);
            consume(FilledTile {
                id,
                halo,
                values: &scratch.buffer,
                core: &scratch.core,
                all_nodata: !populated,
            })
        })?;
        drop(pass);
        let cache = self.cache.stats();
        cache.record_profile();
        Ok(PitRemoveStats {
            pass1: self.pass1,
            graph: self.graph,
            pass2: start.elapsed(),
            empty_tiles: self.empty_tiles,
            spilled_tiles,
            cache,
        })
    }

    fn numeric_tile(&self, index: usize, scratch: &mut OutputScratch) -> Result<bool> {
        let id = self.plan.spec.id_at(index);
        if !self.elevations.try_get_into(id, &mut scratch.core)? {
            return Ok(false);
        }
        self.labels.get_into(id, &mut scratch.labels)?;
        let spill = self.spill.tile(index);
        for (z, &packed) in scratch.core.iter_mut().zip(&scratch.labels) {
            if packed == NONE {
                continue;
            }
            *z = raised(*z, packed & LABEL, spill);
        }
        Ok(true)
    }

    fn fix_zero_signs(&mut self, store_options: StoreOptions, candidates: &[bool]) -> Result<()> {
        let span = profile::span("pitremove.tiled.zero_signs");
        let store = self.cache.store("pitremove.zero_labels", store_options);
        let graphs = Graphs::new(self.plan.spec);
        self.plan.for_each_tile::<ZeroScratch>(|index, scratch| {
            let id = self.plan.spec.id_at(index);
            if !candidates[index]
                || !self.numeric_tile(index, &mut scratch.output)?
                || !scratch.output.core.contains(&0.0)
            {
                return Ok(());
            }
            let halo = self.plan.spec.halo_window(id);
            let graph = scratch.label(halo, &self.borders, id, self.plan.step)?;
            graphs.publish(index, halo, &scratch.labels, &scratch.output.buffer);
            halo.core_from_buffer(&scratch.labels, &mut scratch.output.zeros)?;
            store.put(id, &scratch.output.zeros)?;
            graphs.tiles[index].set(graph).ok().unwrap();
            Ok(())
        })?;
        let levels = graphs.solve(self.plan.step);
        for index in 0..self.plan.spec.tile_count() {
            let id = self.plan.spec.id_at(index);
            let labels = graphs.borders.read(id);
            let spill = levels.tile(index);
            self.borders.update(id, |border| {
                perimeter(self.plan.spec.rect(id), |x, y| {
                    let Seed { label, .. } = labels.get(x, y).unwrap();
                    let z = border.get(x, y).unwrap();
                    let negative =
                        label == 0 || (label < UNSET && spill[label as usize - 1] == 0.0);
                    border.set(
                        x,
                        y,
                        if z.is_nan() {
                            FEL_NODATA
                        } else if negative {
                            -0.0
                        } else {
                            z
                        },
                    );
                });
            });
        }
        self.zeros = Some((store, levels));
        drop(span);
        Ok(())
    }
}

#[inline]
fn raised(z: f32, label: u16, levels: &[f32]) -> f32 {
    let level = if label == 0 || label == ZERO_OUTLET {
        z
    } else {
        levels[label as usize - 1]
    };
    let z = if level > z { level } else { z };
    if z == 0.0 { 0.0 } else { z }
}

fn perimeter(rect: Rect, mut visit: impl FnMut(usize, usize)) {
    let (w, h) = (rect.width, rect.height);
    for x in 0..w {
        visit(x, 0);
        if h > 1 {
            visit(x, h - 1);
        }
    }
    for y in 1..h - 1 {
        visit(0, y);
        if w > 1 {
            visit(w - 1, y);
        }
    }
}

fn perimeter_capacity(spec: TileSpec) -> u64 {
    let horizontal = (spec.raster_width() as u64).saturating_mul(u64::from(spec.tiles_y()));
    let vertical = (spec.raster_height() as u64).saturating_mul(u64::from(spec.tiles_x()));
    horizontal.saturating_add(vertical).saturating_mul(2)
}

fn blank_padding(halo: HaloWindow, values: &mut [f32]) {
    let stride = halo.buffer_width();
    for (y, row) in values.chunks_exact_mut(stride).enumerate() {
        if y < halo.pad_top || y >= halo.buffer_height() - halo.pad_bottom {
            row.fill(f32::NAN);
        } else {
            row[..halo.pad_left].fill(f32::NAN);
            row[stride - halo.pad_right..].fill(f32::NAN);
        }
    }
}

impl Graphs {
    fn new(spec: TileSpec) -> Self {
        Self {
            borders: BorderTable::new(spec, VOID),
            tiles: (0..spec.tile_count()).map(|_| OnceLock::new()).collect(),
        }
    }

    fn publish(&self, index: usize, halo: HaloWindow, labels: &[u16], z: &[f32]) {
        let id = self.borders.spec().id_at(index);
        self.borders.update(id, |border| {
            perimeter(halo.core, |x, y| {
                let i = halo.index(x, y) as usize;
                let label = labels[i];
                if label < UNSET {
                    border.set(
                        x,
                        y,
                        Seed {
                            label,
                            height: z[i],
                        },
                    );
                }
            });
        });
    }

    fn solve(&self, step: usize) -> SpillLevels {
        let spec = *self.borders.spec();
        let mut base = Vec::with_capacity(self.tiles.len() + 1);
        let mut nodes = 1u32;
        for tile in &self.tiles {
            base.push(nodes);
            nodes += tile.get().map_or(0, |g| u32::from(g.labels));
        }
        base.push(nodes);
        let node = |index: usize, label: u16| {
            if label == 0 {
                0
            } else {
                base[index] + u32::from(label) - 1
            }
        };
        let occupied_perimeter: usize = self
            .tiles
            .iter()
            .enumerate()
            .filter(|(_, graph)| graph.get().is_some())
            .map(|(index, _)| {
                let r = spec.rect(spec.id_at(index));
                2 * (r.width + r.height)
            })
            .sum();
        let mut edges = Vec::with_capacity(5 * occupied_perimeter);
        for (index, graph) in self.tiles.iter().enumerate() {
            let Some(graph) = graph.get() else { continue };
            edges.extend(
                graph
                    .edges
                    .iter()
                    .map(|&(a, b, z)| (node(index, a), node(index, b), z)),
            );
            let id = spec.id_at(index);
            let rect = spec.rect(id);
            let border = self.borders.read(id);
            perimeter(rect, |x, y| {
                let seed = border.get(x, y).unwrap();
                if seed.label == NONE {
                    return;
                }
                let from = node(index, seed.label);
                for k in (1..9).step_by(step) {
                    let gx = (rect.x + x).wrapping_add_signed(DX[k]);
                    let gy = (rect.y + y).wrapping_add_signed(DY[k]);
                    if gx >= spec.raster_width()
                        || gy >= spec.raster_height()
                        || rect.contains(gx, gy)
                    {
                        continue;
                    }
                    let other = spec.index(spec.tile_of(gx, gy));
                    if other <= index {
                        continue;
                    }
                    let dest = self.borders.cell(gx, gy).unwrap();
                    if dest.label != NONE {
                        let to = node(other, dest.label);
                        if from != to {
                            edges.push((from, to, seed.height.max(dest.height)));
                        }
                    }
                }
            });
        }
        profile::count("pitremove.tiled.graph_nodes", u64::from(nodes));
        profile::count("pitremove.tiled.graph_edges", edges.len() as u64);
        SpillLevels {
            base,
            levels: minimax(nodes as usize, edges),
        }
    }

    fn numeric_borders(&self, levels: &SpillLevels, has_negative_zero: bool) -> BorderTable<f32> {
        let spec = *self.borders.spec();
        let missing = if has_negative_zero {
            f32::NAN
        } else {
            FEL_NODATA
        };
        let borders = BorderTable::new(spec, missing);
        for id in spec.row_major() {
            let seeds = self.borders.read(id);
            let spill = levels.tile(spec.index(id));
            borders.update(id, |border| {
                perimeter(spec.rect(id), |x, y| {
                    let Seed { label, height } = seeds.get(x, y).unwrap();
                    if label != NONE {
                        border.set(x, y, raised(height, label, spill));
                    }
                });
            });
        }
        borders
    }
}

struct LocalScratch {
    raw: Vec<f32>,
    mask: Vec<u8>,
    filled: Vec<f32>,
    labels: Vec<u16>,
    void: Vec<u64>,
    heap: FloodQueue,
    pit: VecDeque<u32>,
    parent: Vec<u16>,
    ranks: Vec<u8>,
    core: Vec<f32>,
    packed: Vec<u16>,
}

impl Default for LocalScratch {
    fn default() -> Self {
        Self {
            raw: Vec::new(),
            mask: Vec::new(),
            filled: Vec::new(),
            labels: Vec::new(),
            void: Vec::new(),
            heap: FloodQueue::bounded(),
            pit: VecDeque::new(),
            parent: Vec::new(),
            ranks: Vec::new(),
            core: Vec::new(),
            packed: Vec::new(),
        }
    }
}

impl LocalScratch {
    fn flood(
        &mut self,
        halo: HaloWindow,
        nodata: f32,
        step: usize,
        negative_zero: &AtomicBool,
    ) -> Option<TileGraph> {
        let stride = halo.buffer_width();
        let grid = Grid::new(stride, halo.buffer_height()).unwrap();
        self.labels.clear();
        self.labels.resize(grid.cells(), NONE);
        self.filled.clear();
        self.filled.resize(grid.cells(), FEL_NODATA);
        self.void.clear();
        self.void.resize(grid.cells().div_ceil(64), 0);
        self.heap.reset();
        self.pit.clear();
        let mut valid = false;
        let mut negative = false;
        for (y, row) in self.raw.chunks_exact(stride).enumerate() {
            for (x, &z) in row.iter().enumerate() {
                let i = y * stride + x;
                if missing(z, nodata) {
                    self.void[i / 64] |= 1 << (i % 64);
                } else if y > 0 && y <= halo.core.height && x > 0 && x <= halo.core.width {
                    self.filled[i] = z;
                    self.labels[i] = UNSET;
                    valid = true;
                    negative |= z.to_bits() == (-0.0f32).to_bits();
                }
            }
        }
        if !valid {
            return None;
        }
        if negative {
            negative_zero.store(true, Relaxed);
        }

        // Only missing samples seed their neighbors, including missing cells
        // across a seam and raster padding. Valid halo cells never enter the queue.
        for (w, &bits) in self.void.iter().enumerate() {
            let mut remaining = bits;
            while remaining != 0 {
                let i = w * 64 + remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                grid.for_neighbors(i, step, |_, j| {
                    seed(j, 0, &mut self.labels, &self.filled, &mut self.heap);
                });
            }
        }
        for (i, &flag) in self.mask.iter().enumerate() {
            if flag == 1 {
                let width = halo.core.width;
                seed(
                    halo.index(i % width, i / width) as usize,
                    0,
                    &mut self.labels,
                    &self.filled,
                    &mut self.heap,
                );
            }
        }
        let mut next = 1u16;
        perimeter(halo.core, |x, y| {
            let i = halo.index(x, y) as usize;
            if seed(i, next, &mut self.labels, &self.filled, &mut self.heap) {
                next += 1;
            }
        });
        self.parent.clear();
        self.parent.extend(0..next);
        self.ranks.clear();
        self.ranks.resize(next as usize, 0);
        let mut edges = Vec::new();
        let Self {
            filled,
            labels,
            heap,
            pit,
            parent,
            ranks,
            ..
        } = self;
        while let Some(cell) = pit.pop_front().or_else(|| heap.pop()) {
            let i = cell as usize;
            let level = filled[i];
            let label = labels[i];
            labels[i] |= SETTLED;
            // Core cells have a full halo; its labels stop the flood.
            for k in (1..9).step_by(step) {
                let j = grid.offset(i, k);
                let other = labels[j];
                if other == UNSET {
                    labels[j] = label;
                    if filled[j] <= level {
                        filled[j] = level;
                        pit.push_back(j as u32);
                    } else {
                        heap.push(priority(filled[j], j));
                    }
                } else if other != NONE && other & SETTLED != 0 {
                    let other = other & LABEL;
                    // Only the later settled cell sees this adjacency. Its
                    // level is the larger of the two and pops in ascending
                    // order, allowing Kruskal to discard cycle edges here.
                    if other != label && join(parent, ranks, label, other) {
                        edges.push((label, other, level));
                    }
                }
            }
        }
        for label in &mut self.labels {
            if *label != NONE {
                *label &= LABEL;
            }
        }
        Some(TileGraph {
            labels: next - 1,
            has_zero: false,
            edges,
        })
    }

    fn pack_labels(&mut self, halo: HaloWindow, nodata: f32, step: usize) {
        let grid = Grid::new(halo.buffer_width(), halo.buffer_height()).unwrap();
        self.packed.clear();
        self.packed.reserve(halo.core.cells());
        for y in 0..halo.core.height {
            let start = halo.index(0, y) as usize;
            for x in 0..halo.core.width {
                let i = start + x;
                let label = self.labels[i];
                let z = self.raw[i];
                let packed = if label == NONE || z != 0.0 {
                    label
                } else {
                    let mask_index = y * halo.core.width + x;
                    let outlet = self.mask.get(mask_index) == Some(&1)
                        || (1..9)
                            .step_by(step)
                            .any(|k| missing(self.raw[grid.offset(i, k)], nodata));
                    let source = if outlet { ZERO_OUTLET } else { label };
                    source
                        | ORIGINAL_ZERO
                        | if z.is_sign_negative() {
                            ORIGINAL_NEGATIVE
                        } else {
                            0
                        }
                };
                self.packed.push(packed);
            }
        }
    }
}

#[inline]
fn seed(i: usize, label: u16, labels: &mut [u16], filled: &[f32], heap: &mut FloodQueue) -> bool {
    if labels[i] != UNSET {
        return false;
    }
    labels[i] = label;
    heap.push(priority(filled[i], i));
    true
}

#[inline]
fn join(parent: &mut [u16], ranks: &mut [u8], a: u16, b: u16) -> bool {
    let root = |parent: &mut [u16], mut id: u16| {
        while parent[id as usize] != id {
            let up = parent[parent[id as usize] as usize];
            parent[id as usize] = up;
            id = up;
        }
        id as usize
    };
    let (mut a, mut b) = (root(parent, a), root(parent, b));
    if a == b {
        return false;
    }
    if ranks[a] < ranks[b] {
        std::mem::swap(&mut a, &mut b);
    }
    parent[b] = a as u16;
    ranks[a] += u8::from(ranks[a] == ranks[b]);
    true
}

#[derive(Default)]
struct ZeroScratch {
    output: OutputScratch,
    labels: Vec<u16>,
    queue: VecDeque<u32>,
}

impl ZeroScratch {
    // Zero outlets and raw zeros reached from below keep their input sign.
    // Every other filled zero takes the sign of its component: negative if
    // linked to a negative source without crossing a fixed positive source.
    fn label(
        &mut self,
        halo: HaloWindow,
        borders: &BorderTable<f32>,
        id: TileId,
        step: usize,
    ) -> Result<TileGraph> {
        let stride = halo.buffer_width();
        let grid = Grid::new(stride, halo.buffer_height()).unwrap();
        self.output.buffer.resize(halo.buffer_cells(), f32::NAN);
        borders.fill_halo_ring(id, &halo, &mut self.output.buffer, f32::NAN)?;
        self.labels.clear();
        self.labels.resize(halo.buffer_cells(), NONE);
        self.queue.clear();
        for y in 0..halo.core.height {
            let start = halo.index(0, y) as usize;
            for x in 0..halo.core.width {
                let core = y * halo.core.width + x;
                self.output.buffer[start + x] = if self.output.labels[core] == NONE {
                    f32::NAN
                } else {
                    self.output.core[core]
                };
            }
        }
        for y in 0..halo.core.height {
            let start = halo.index(0, y) as usize;
            for x in 0..halo.core.width {
                let i = start + x;
                if self.output.buffer[i] != 0.0 {
                    continue;
                }
                let packed = self.output.labels[y * halo.core.width + x];
                let fixed = packed & LABEL == ZERO_OUTLET
                    || (packed & ORIGINAL_ZERO != 0
                        && (1..9)
                            .step_by(step)
                            .any(|k| self.output.buffer[grid.offset(i, k)] < 0.0));
                if !fixed {
                    self.labels[i] = UNSET;
                } else if packed & ORIGINAL_NEGATIVE != 0 {
                    self.labels[i] = 0;
                    self.queue.push_back(i as u32);
                }
            }
        }
        self.flood_component(grid, step, 0);
        let mut next = 1;
        perimeter(halo.core, |x, y| {
            let i = halo.index(x, y) as usize;
            if self.labels[i] == UNSET {
                self.labels[i] = next;
                self.queue.push_back(i as u32);
                self.flood_component(grid, step, next);
                next += 1;
            }
        });
        Ok(TileGraph {
            labels: next - 1,
            has_zero: false,
            edges: Vec::new(),
        })
    }

    fn flood_component(&mut self, grid: Grid, step: usize, label: u16) {
        while let Some(i) = self.queue.pop_front() {
            for k in (1..9).step_by(step) {
                let j = grid.offset(i as usize, k);
                if self.labels[j] == UNSET {
                    self.labels[j] = label;
                    self.queue.push_back(j as u32);
                }
            }
        }
    }
}
