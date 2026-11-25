// SPDX-License-Identifier: GPL-3.0-only
//! Tiled, bounded-memory D8FlowDir that reproduces [`crate::d8_flow`] bit for
//! bit, including TauDEM's multi-pass Garbrecht–Martz flat routing across
//! tiles (`docs/rust-port/STREAMING.md` §2.2–2.3).
//!
//! # Passes
//!
//! **Pass B** ([`TiledD8::process_tile`], tiles in any order and in
//! parallel). A tile's filled elevations with a 1-cell halo give its steep
//! directions and `sd8`. Its unresolved cells (direction 0) form 8-connected
//! components:
//!
//! - A component that touches no tile edge has all of its neighbors inside
//!   the tile and is routed there through every flat pass it needs.
//! - A component that touches a tile edge is *deferred*. The tile computes
//!   the first-pass increment levels its own seeds and steps give, keeps its
//!   state in the [`TileStore`] and publishes its perimeter in
//!   [`BorderTable`]s.
//!
//! **Pass C** ([`TiledD8::finish`]) routes the deferred components pass by
//! pass, as the in-memory resolver routes every flat. A pass first completes
//! TauDEM's two breadth-first increment fields (`incfall`, `incrise`) by
//! label-correcting relaxation: a tile whose perimeter can be lowered through
//! a neighbor's perimeter is loaded, lowered and searched again, until no
//! perimeter changes. Both fields are then the exact multi-source shortest
//! path lengths of the in-memory searches. The highest levels of the pass
//! over the whole raster (`last_fall`, `last_rise`) come from these tiles and
//! from the interior components of Pass B; each tile then routes its cells
//! and builds the next pass's artificial surface, as `resolve_flats` does.
//!
//! # Exactness
//!
//! Flat components evolve independently: every neighbor of a cell of
//! component `C` is in `C` or was never unresolved, so the seeds, steps and
//! routing of `C` read only `C` and fixed cells, and in later passes the
//! artificial surface is 1 outside the cells unresolved in the previous pass.
//! Every pass resolves at least one cell of each component (a seeded cell
//! drains, an unseeded one becomes nodata), so TauDEM's no-progress stop
//! never fires and a component's passes do not depend on the others.
//!
//! The only coupling is through two raster-wide constants of each pass.
//! `last_rise` adds the same offset to every rised cell of a component, and
//! a component is rised everywhere or nowhere (the rise search crosses all
//! of it), so the offset cancels in every difference routing takes and in
//! every comparison of the next surface (exact while levels stay below
//! 2^24, which [`TiledD8::finish`] checks); against cells outside the
//! component the surface is at least 3 > 1 either way. `last_fall` enters
//! only through the level of cells no drain reaches; it cancels unless a
//! component has both kinds of cell (*mixed*). Interior components are
//! therefore routed with tile-local constants, except mixed ones, which
//! Pass B detects and defers. Pass C uses the exact raster-wide constants.
//!
//! The steep pass applies TauDEM's row-major crossing rule. It cannot fire
//! in exact arithmetic (STREAMING.md §2.2), so tiles run in parallel and
//! record crossing candidates; [`TiledD8::finish`] then reports
//! [`D8Outcome::CrossingDetected`] and the caller reruns with
//! [`SteepSchedule::Wavefront`], which visits tiles after their west and
//! north neighbors and applies the rule exactly.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use rayon::prelude::*;

use crate::tiling::{
    Border, BorderTable, Error, HaloWindow, Rect, Result, Ring, StoreOptions, TileCache, TileId,
    TileSpec, TileStore, for_each_wavefront,
};
use crate::{DIRECTION_NODATA, DX, DY, ORDER, SCALAR_NODATA, missing, profile};

/// How the steep pass meets TauDEM's row-major crossing rule.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum SteepSchedule {
    /// Tiles in any order and in parallel. A crossing candidate makes
    /// [`TiledD8::finish`] return [`D8Outcome::CrossingDetected`].
    #[default]
    Parallel,
    /// Each tile after its west and north neighbors
    /// ([`for_each_wavefront`]), applying the crossing rule exactly.
    Wavefront,
}

/// The result of [`TiledD8::finish`].
#[derive(Debug)]
pub enum D8Outcome {
    /// Every tile's directions were delivered.
    Done(D8TileStats),
    /// The parallel steep pass found a crossing candidate; nothing it
    /// delivered is valid. Rerun with [`SteepSchedule::Wavefront`].
    CrossingDetected,
}

/// Counters of one tiled run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct D8TileStats {
    pub tiles: u64,
    /// Cells unresolved after the steep pass.
    pub flat_cells: u64,
    /// Flat cells of components inside one tile, routed in Pass B.
    pub interior_cells: u64,
    /// The most flat passes an interior component needed.
    pub interior_passes: u64,
    /// Flat cells deferred to Pass C, and the tiles holding them.
    pub deferred_cells: u64,
    pub deferred_tiles: u64,
    /// Interior components deferred because they mixed drained and
    /// undrained cells.
    pub mixed_components: u64,
    /// Directions changed by the crossing rule (wavefront schedule).
    pub crossing_fixes: u64,
    /// Pass C, one entry per flat pass.
    pub passes: Vec<PassStats>,
}

impl D8TileStats {
    /// Records the counters for `--profile-json`.
    pub fn record_profile(&self) {
        use profile::count;
        count("d8.tiled.tiles", self.tiles);
        count("d8.flat_cells", self.flat_cells);
        count("d8.tiled.interior_cells", self.interior_cells);
        count("d8.tiled.interior_passes", self.interior_passes);
        count("d8.tiled.deferred_cells", self.deferred_cells);
        count("d8.tiled.deferred_tiles", self.deferred_tiles);
        count("d8.tiled.mixed_components", self.mixed_components);
        count("d8.crossing_fixes", self.crossing_fixes);
        count("d8.tiled.passes", self.passes.len() as u64);
        let total = |f: fn(&PassStats) -> u64| self.passes.iter().map(f).sum::<u64>();
        count("d8.tiled.rounds", total(|p| p.rounds));
        count("d8.tiled.relax_loads", total(|p| p.relax_loads));
        count("d8.tiled.relax_checks", total(|p| p.relax_checks));
        let names: [[&'static str; 6]; 3] = [
            [
                "d8.tiled.pass1.tiles",
                "d8.tiled.pass1.cells",
                "d8.tiled.pass1.rounds",
                "d8.tiled.pass1.relax_loads",
                "d8.tiled.pass1.max_fall",
                "d8.tiled.pass1.max_rise",
            ],
            [
                "d8.tiled.pass2.tiles",
                "d8.tiled.pass2.cells",
                "d8.tiled.pass2.rounds",
                "d8.tiled.pass2.relax_loads",
                "d8.tiled.pass2.max_fall",
                "d8.tiled.pass2.max_rise",
            ],
            [
                "d8.tiled.pass3.tiles",
                "d8.tiled.pass3.cells",
                "d8.tiled.pass3.rounds",
                "d8.tiled.pass3.relax_loads",
                "d8.tiled.pass3.max_fall",
                "d8.tiled.pass3.max_rise",
            ],
        ];
        for (n, p) in names.iter().zip(&self.passes) {
            count(n[0], p.tiles);
            count(n[1], p.cells);
            count(n[2], p.rounds);
            count(n[3], p.relax_loads);
            count(n[4], u64::from(p.last_fall));
            count(n[5], u64::from(p.last_rise));
        }
    }
}

/// One Pass C flat pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PassStats {
    /// Tiles with unresolved cells, and those cells.
    pub tiles: u64,
    pub cells: u64,
    /// Relaxation rounds until no perimeter changed.
    pub rounds: u64,
    /// Tiles whose perimeter was checked, and tiles loaded to lower levels.
    pub relax_checks: u64,
    pub relax_loads: u64,
    /// The raster-wide highest levels of the pass.
    pub last_fall: u32,
    pub last_rise: u32,
    /// Cells still unresolved after the pass.
    pub remaining: u64,
    pub seconds: f64,
}

/// Approximate working bytes per tile cell of one [`TiledD8`] worker: the
/// halo elevations, directions, increment fields and queues.
pub const WORK_BYTES_PER_CELL: u64 = 32;

/// The tile side for a memory budget: the largest multiple of 256 (at most
/// `ceiling`) whose workers' buffers take at most 40% of the budget.
pub fn tile_size_for_budget(budget: u64, threads: usize, ceiling: usize) -> usize {
    let per_worker = budget as f64 * 0.4 / threads.max(1) as f64;
    let side = (per_worker / WORK_BYTES_PER_CELL as f64).sqrt() as usize;
    (side / 256 * 256).clamp(256, ceiling.max(256))
}

const INTERIOR: u8 = 1;
const DEFERRED: u8 = 2;

const SEED_DRAIN: u8 = 1;
const SEED_DEFAULT: u8 = 2;
const SEED_RISE: u8 = 4;

/// Flat levels must stay exact in `f32`, the type of TauDEM's artificial surface.
const F32_EXACT: i64 = 1 << 24;

/// The halo buffer of a tile: core `w × h` inside a 1-cell ring.
#[derive(Clone, Copy, Debug)]
struct Frame {
    w: usize,
    h: usize,
    bw: usize,
    off: [isize; 9],
}

impl Frame {
    fn new(w: usize, h: usize) -> Self {
        let bw = w + 2;
        Self {
            w,
            h,
            bw,
            off: std::array::from_fn(|k| DX[k] + DY[k] * bw as isize),
        }
    }

    fn cells(&self) -> usize {
        self.bw * (self.h + 2)
    }

    #[inline]
    fn at(&self, lx: usize, ly: usize) -> usize {
        (ly + 1) * self.bw + lx + 1
    }

    #[inline]
    fn nb(&self, i: usize, k: usize) -> usize {
        i.wrapping_add_signed(self.off[k])
    }

    #[inline]
    fn on_edge(&self, lx: usize, ly: usize) -> bool {
        lx == 0 || ly == 0 || lx + 1 == self.w || ly + 1 == self.h
    }

    /// Whether neighbor `k` of core cell `(lx, ly)` is a core cell.
    #[inline]
    fn inside(&self, lx: usize, ly: usize, k: usize) -> bool {
        let x = lx as isize + DX[k];
        let y = ly as isize + DY[k];
        x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h
    }

    /// The core row of buffer index `i`.
    #[inline]
    fn row(&self, i: usize) -> usize {
        i / self.bw - 1
    }

    /// Core cells on the tile's edge, each once.
    fn perimeter(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (w, h) = (self.w, self.h);
        let top = (0..w).map(|x| (x, 0));
        let bottom_width = if h > 1 { w } else { 0 };
        let bottom = (0..bottom_width).map(move |x| (x, h - 1));
        let sides = (1..h.saturating_sub(1))
            .flat_map(move |y| std::iter::once((0, y)).chain((w > 1).then_some((w - 1, y))));
        top.chain(bottom).chain(sides)
    }
}

/// Per-row D8 distance factors, as `d8_steep` builds them.
struct Factors(Vec<[f64; 9]>);

impl Factors {
    fn new(spacing: &[(f64, f64)], rows: usize) -> Result<Self> {
        if spacing.len() != 1 && spacing.len() != rows {
            return Err(Error::Invalid(format!(
                "{} cell sizes for {rows} rows",
                spacing.len()
            )));
        }
        let mut factors = Vec::with_capacity(spacing.len());
        for &(dx, dy) in spacing {
            if !(dx.is_finite() && dy.is_finite() && dx > 0.0 && dy > 0.0) {
                return Err(Error::Invalid(format!("invalid cell size {dx} x {dy}")));
            }
            let diag = 1.0 / (dx * dx + dy * dy).sqrt();
            factors.push([
                0.0,
                1.0 / dx,
                diag,
                1.0 / dy,
                diag,
                1.0 / dx,
                diag,
                1.0 / dy,
                diag,
            ]);
        }
        Ok(Self(factors))
    }

    #[inline]
    fn row(&self, y: usize) -> &[f64; 9] {
        &self.0[if self.0.len() == 1 { 0 } else { y }]
    }
}

#[inline]
fn scaled(delta: f32, factor: f64) -> f32 {
    (f64::from(delta) * factor) as f32
}

#[inline]
fn steepest(heights: &[f32; 9], scale: &[f64; 9], allowed: impl Fn(usize) -> bool) -> u8 {
    let mut best = 0.0;
    let mut direction = 0;
    for k in ORDER {
        let slope = scaled(heights[0] - heights[k], scale[k]);
        if slope > best && allowed(k) {
            best = slope;
            direction = k as u8;
        }
    }
    direction
}

/// The neighbors forming the other diagonal of a diagonal step `k` and the
/// directions that would cross it.
#[inline]
fn other_diagonal(k: usize) -> Option<(usize, u8, usize, u8)> {
    match k {
        2 => Some((1, 4, 3, 8)),
        4 => Some((3, 6, 5, 2)),
        6 => Some((7, 4, 5, 8)),
        8 => Some((1, 6, 7, 2)),
        _ => None,
    }
}

/// TauDEM's crossing test against every assigned direction (flat passes).
#[inline]
fn crosses(frame: &Frame, dirs: &[u8], i: usize, k: usize) -> bool {
    other_diagonal(k)
        .is_some_and(|(a, ac, b, bc)| dirs[frame.nb(i, a)] == ac || dirs[frame.nb(i, b)] == bc)
}

/// TauDEM's crossing test against the directions assigned before `i` in
/// row-major order (the steep pass): the north and west neighbors.
#[inline]
fn crosses_before(frame: &Frame, dirs: &[u8], i: usize, k: usize) -> bool {
    other_diagonal(k).is_some_and(|(a, ac, b, bc)| {
        let (a, b) = (frame.nb(i, a), frame.nb(i, b));
        (a < i && dirs[a] == ac) || (b < i && dirs[b] == bc)
    })
}

/// TauDEM's combined increment level `elev2` (`d8.rs::surface`).
#[inline]
fn surface(fall: u32, rise: u32, last_rise: u32) -> i64 {
    let level = i64::from(fall.max(1));
    if rise > 0 {
        level + i64::from(last_rise) + 2 - i64::from(rise)
    } else {
        level
    }
}

/// `a` lowered to `b`, where 0 means unreached.
#[inline]
fn lower(a: u32, b: u32) -> u32 {
    if a == 0 { b } else { a.min(b) }
}

/// Multi-source breadth-first levels with sources at any level. `values`
/// holds each cell's level (0 unreached); every seed `(level, cell)` has
/// already been applied. A cell one admissible step (`mask`) from a cell at
/// level `l` is lowered to `l + 1`. Levels are expanded in increasing order,
/// so each lowered cell is expanded once per level it takes.
fn propagate(
    frame: &Frame,
    values: &mut [u32],
    seeds: &mut [(u32, u32)],
    mask: impl Fn(usize) -> u8,
    cur: &mut Vec<u32>,
    next: &mut Vec<u32>,
) {
    if seeds.is_empty() {
        return;
    }
    seeds.sort_unstable();
    let mut taken = 0;
    let mut level = seeds[0].0;
    cur.clear();
    loop {
        while taken < seeds.len() && seeds[taken].0 == level {
            let i = seeds[taken].1;
            if values[i as usize] == level {
                cur.push(i);
            }
            taken += 1;
        }
        if cur.is_empty() {
            if taken == seeds.len() {
                break;
            }
            level = seeds[taken].0;
            continue;
        }
        let up = level + 1;
        next.clear();
        for &c in cur.iter() {
            let i = c as usize;
            let mut m = mask(i);
            while m != 0 {
                let k = m.trailing_zeros() as usize + 1;
                m &= m - 1;
                let j = frame.nb(i, k);
                let v = values[j];
                if v == 0 || v > up {
                    values[j] = up;
                    next.push(j as u32);
                }
            }
        }
        std::mem::swap(cur, next);
        level = up;
    }
}

/// Reusable per-worker buffers, all of halo size except the lists.
#[derive(Default)]
struct Scratch {
    dirs: Vec<u8>,
    class: Vec<u8>,
    steps: Vec<u8>,
    fall: Vec<u32>,
    rise: Vec<u32>,
    elev: Vec<f32>,
    set: Vec<u32>,
    prev: Vec<u32>,
    cur: Vec<u32>,
    next: Vec<u32>,
    seeds: Vec<(u32, u32)>,
    seeds2: Vec<(u32, u32)>,
    routed: Vec<u8>,
    core: Vec<u8>,
}

struct ScratchGuard<'a> {
    pool: &'a Mutex<Vec<Scratch>>,
    scratch: Option<Scratch>,
}

impl std::ops::Deref for ScratchGuard<'_> {
    type Target = Scratch;
    fn deref(&self) -> &Scratch {
        self.scratch.as_ref().expect("scratch")
    }
}

impl std::ops::DerefMut for ScratchGuard<'_> {
    fn deref_mut(&mut self) -> &mut Scratch {
        self.scratch.as_mut().expect("scratch")
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        if let Some(s) = self.scratch.take() {
            self.pool.lock().unwrap_or_else(|p| p.into_inner()).push(s);
        }
    }
}

/// A tile's perimeter cells and its ring, from a [`BorderTable`], addressed
/// by tile-local coordinates from `-1` to `w` and `h`.
struct Around<T> {
    border: Border<T>,
    ring: Ring<T>,
    w: isize,
    h: isize,
}

impl<T: Copy + Send + Sync> Around<T> {
    fn new(table: &BorderTable<T>, id: TileId, outside: T) -> Self {
        let border = table.read(id).clone();
        let ring = table.ring(id, outside);
        Self {
            w: border.width() as isize,
            h: border.height() as isize,
            border,
            ring,
        }
    }

    #[inline]
    fn at(&self, x: isize, y: isize) -> T {
        if x >= 0 && y >= 0 && x < self.w && y < self.h {
            self.border
                .get(x as usize, y as usize)
                .expect("a perimeter cell")
        } else if y < 0 {
            self.ring.north[(x + 1) as usize]
        } else if y >= self.h {
            self.ring.south[(x + 1) as usize]
        } else if x < 0 {
            self.ring.west[y as usize]
        } else {
            self.ring.east[y as usize]
        }
    }
}

/// What a perimeter cell sees across the tile edge in one pass: its steps
/// to unresolved cells of equal height (`fall`), its neighbors that are
/// unresolved at all (`rise`) and its seeds there.
#[derive(Clone, Copy, Debug)]
struct EdgeCell {
    x: u32,
    y: u32,
    fall: u8,
    rise: u8,
    seed: u8,
}

/// Bits `k - 1` of a step mask as directions `k`.
#[inline]
fn bits(mut m: u8) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (m != 0).then(|| {
            let k = m.trailing_zeros() as usize + 1;
            m &= m - 1;
            k
        })
    })
}

struct Layers {
    dir: TileStore<u8>,
    elev: TileStore<f32>,
    adj: TileStore<u16>,
    fall: TileStore<u32>,
    rise: TileStore<u32>,
}

fn missing_tile(id: TileId, layer: &str) -> Error {
    Error::Invalid(format!("tile {id:?} has no {layer} layer"))
}

#[derive(Default)]
struct Counters {
    tiles: AtomicU64,
    flat_cells: AtomicU64,
    interior_cells: AtomicU64,
    interior_passes: AtomicU64,
    deferred_cells: AtomicU64,
    deferred_tiles: AtomicU64,
    mixed: AtomicU64,
    crossing_fixes: AtomicU64,
}

/// Tiled D8FlowDir state shared by every tile of one raster.
///
/// Feed every tile to [`Self::process_tile`] (in any order and in parallel
/// for [`SteepSchedule::Parallel`], or from [`for_each_wavefront`] for
/// [`SteepSchedule::Wavefront`]), then call [`Self::finish`]. Each tile's
/// final directions are delivered exactly once, from `process_tile` when
/// its flats are routed inside it and from `finish` otherwise.
pub struct TiledD8 {
    spec: TileSpec,
    nodata: f32,
    factors: Factors,
    schedule: SteepSchedule,
    dirs: BorderTable<u8>,
    elev: BorderTable<f32>,
    fall: BorderTable<u32>,
    rise: BorderTable<u32>,
    layers: Layers,
    /// Unresolved cells each tile leaves to Pass C.
    deferred: Vec<AtomicU64>,
    /// Each tile's highest fall and rise levels in the current pass.
    tile_max: Vec<AtomicU64>,
    /// The highest fall and rise levels of interior components, per pass.
    interior: Mutex<Vec<[u32; 2]>>,
    crossing: AtomicBool,
    scratch: Mutex<Vec<Scratch>>,
    counters: Counters,
}

impl TiledD8 {
    /// `spacing` is one `(dx, dy)` or one per raster row; `nodata` marks
    /// missing elevations; tile state lives in `cache`.
    pub fn new(
        spec: TileSpec,
        nodata: f32,
        spacing: &[(f64, f64)],
        cache: &TileCache,
        schedule: SteepSchedule,
    ) -> Result<Self> {
        let factors = Factors::new(spacing, spec.raster_height())?;
        let tiles = spec.tile_count();
        Ok(Self {
            spec,
            nodata,
            factors,
            schedule,
            dirs: BorderTable::new(spec, DIRECTION_NODATA),
            elev: BorderTable::new(spec, f32::NAN),
            fall: BorderTable::new(spec, 0),
            rise: BorderTable::new(spec, 0),
            layers: Layers {
                dir: cache.store("d8.dir", StoreOptions::default()),
                elev: cache.store("d8.elev", StoreOptions::default()),
                adj: cache.store("d8.adj", StoreOptions::default()),
                fall: cache.store("d8.fall", StoreOptions::default()),
                rise: cache.store("d8.rise", StoreOptions::default()),
            },
            deferred: (0..tiles).map(|_| AtomicU64::new(0)).collect(),
            tile_max: (0..tiles).map(|_| AtomicU64::new(0)).collect(),
            interior: Mutex::new(Vec::new()),
            crossing: AtomicBool::new(false),
            scratch: Mutex::new(Vec::new()),
            counters: Counters::default(),
        })
    }

    pub fn spec(&self) -> &TileSpec {
        &self.spec
    }

    pub fn schedule(&self) -> SteepSchedule {
        self.schedule
    }

    /// Whether the parallel steep pass has met a crossing candidate inside a
    /// tile; the run's results will be discarded.
    pub fn crossing_detected(&self) -> bool {
        self.crossing.load(Relaxed)
    }

    fn scratch(&self) -> ScratchGuard<'_> {
        let scratch = self
            .scratch
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_default();
        ScratchGuard {
            pool: &self.scratch,
            scratch: Some(scratch),
        }
    }

    /// Pass B for tile `id`. `fel` is its radius-1 halo buffer
    /// ([`TileSpec::halo_window`]); cells outside the raster are never read.
    /// Writes the tile's `sd8` (row-major core) into `slope` when given and
    /// calls `finished` with its final directions (row-major core, 255 for
    /// nodata) when no flat of the tile reaches its edge.
    pub fn process_tile(
        &self,
        id: TileId,
        fel: &[f32],
        slope: Option<&mut Vec<f32>>,
        finished: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let halo = self.spec.halo_window(id);
        if fel.len() != halo.buffer_cells() {
            return Err(Error::Invalid(format!(
                "{} elevations for the halo of tile {id:?} ({} cells)",
                fel.len(),
                halo.buffer_cells()
            )));
        }
        let rect = halo.core;
        let frame = Frame::new(rect.width, rect.height);
        let mut s = self.scratch();
        let s = &mut *s;
        self.steep(&frame, rect, fel, &mut s.dirs);
        match self.schedule {
            SteepSchedule::Parallel => {
                if inside_crossing(&frame, &s.dirs) {
                    self.crossing.store(true, Relaxed);
                }
            }
            SteepSchedule::Wavefront => {
                // The west and north neighbors are final.
                self.dirs
                    .fill_halo_ring(id, &halo, &mut s.dirs, DIRECTION_NODATA)?;
                let fixes = self.fix_crossings(&frame, rect, fel, &mut s.dirs);
                self.counters.crossing_fixes.fetch_add(fixes, Relaxed);
                clear_ring(&frame, &mut s.dirs, DIRECTION_NODATA);
            }
        }
        // Flat routing in Pass B never changes a perimeter cell.
        self.dirs.set_from_halo(id, &halo, &s.dirs)?;
        self.elev.set_from_halo(id, &halo, fel)?;
        if let Some(out) = slope {
            self.slope(&frame, rect, fel, &s.dirs, out);
        }
        self.counters.tiles.fetch_add(1, Relaxed);

        let (flats, edge) = classify(&frame, &s.dirs, &mut s.class, &mut s.cur);
        self.counters.flat_cells.fetch_add(flats, Relaxed);
        let mut deferred = edge;
        if flats > edge {
            let interior = resolve_interior(&frame, rect, fel, &self.factors, s);
            deferred += interior.deferred;
            self.counters.mixed.fetch_add(interior.mixed, Relaxed);
            self.counters
                .interior_cells
                .fetch_add(flats - deferred, Relaxed);
            self.counters
                .interior_passes
                .fetch_max(interior.maxima.len() as u64, Relaxed);
            let mut all = self.interior.lock().unwrap_or_else(|p| p.into_inner());
            if all.len() < interior.maxima.len() {
                all.resize(interior.maxima.len(), [0, 0]);
            }
            for (a, m) in all.iter_mut().zip(&interior.maxima) {
                a[0] = a[0].max(m[0]);
                a[1] = a[1].max(m[1]);
            }
        }
        if deferred == 0 {
            halo.core_from_buffer(&s.dirs, &mut s.core)?;
            return finished(&s.core);
        }

        // Deferred flats: the first pass's levels from inside the tile.
        self.counters.deferred_cells.fetch_add(deferred, Relaxed);
        self.counters.deferred_tiles.fetch_add(1, Relaxed);
        let dirs = std::mem::take(&mut s.dirs);
        let mut adj = Vec::new();
        let (max_fall, max_rise, cells) = init_levels(&frame, &dirs, fel, &mut adj, s);
        debug_assert_eq!(cells, deferred);
        self.deferred[self.spec.index(id)].store(deferred, Relaxed);
        self.tile_max[self.spec.index(id)].store(pack(max_fall, max_rise), Relaxed);
        self.fall.set_from_halo(id, &halo, &s.fall)?;
        self.rise.set_from_halo(id, &halo, &s.rise)?;
        self.layers.dir.put_owned(id, dirs)?;
        self.layers.elev.put(id, fel)?;
        self.layers.adj.put_owned(id, adj)?;
        self.layers
            .fall
            .put_owned(id, std::mem::take(&mut s.fall))?;
        self.layers
            .rise
            .put_owned(id, std::mem::take(&mut s.rise))?;
        Ok(())
    }

    /// Steepest-descent directions without the crossing rule.
    fn steep(&self, frame: &Frame, rect: Rect, fel: &[f32], dirs: &mut Vec<u8>) {
        dirs.clear();
        dirs.resize(frame.cells(), DIRECTION_NODATA);
        let (width, height) = (self.spec.raster_width(), self.spec.raster_height());
        if width < 3 {
            return;
        }
        for ly in 0..frame.h {
            let y = rect.y + ly;
            if y == 0 || y + 1 == height {
                continue;
            }
            let scale = self.factors.row(y);
            for lx in 0..frame.w {
                let x = rect.x + lx;
                if x == 0 || x + 1 == width {
                    continue;
                }
                let i = frame.at(lx, ly);
                let heights: [f32; 9] = std::array::from_fn(|k| fel[frame.nb(i, k)]);
                if heights.iter().any(|&z| missing(z, self.nodata)) {
                    continue;
                }
                dirs[i] = steepest(&heights, scale, |_| true);
            }
        }
    }

    /// TauDEM's row-major crossing rule, given final north and west rings.
    fn fix_crossings(&self, frame: &Frame, rect: Rect, fel: &[f32], dirs: &mut [u8]) -> u64 {
        let mut fixes = 0;
        for ly in 0..frame.h {
            let scale = self.factors.row(rect.y + ly);
            for lx in 0..frame.w {
                let i = frame.at(lx, ly);
                let k = dirs[i] as usize;
                if matches!(k, 2 | 4 | 6 | 8) && crosses_before(frame, dirs, i, k) {
                    let heights: [f32; 9] = std::array::from_fn(|k| fel[frame.nb(i, k)]);
                    let view: &[u8] = dirs;
                    let d = steepest(&heights, scale, |k| !crosses_before(frame, view, i, k));
                    dirs[i] = d;
                    fixes += 1;
                }
            }
        }
        fixes
    }

    fn slope(&self, frame: &Frame, rect: Rect, fel: &[f32], dirs: &[u8], out: &mut Vec<f32>) {
        out.clear();
        out.reserve(rect.cells());
        for ly in 0..frame.h {
            let scale = self.factors.row(rect.y + ly);
            for lx in 0..frame.w {
                let i = frame.at(lx, ly);
                let k = dirs[i] as usize;
                out.push(if k <= 8 {
                    scaled(fel[i] - fel[frame.nb(i, k)], scale[k])
                } else {
                    SCALAR_NODATA
                });
            }
        }
    }

    /// Whether a perimeter cell's unrestricted direction crosses that of its
    /// north or west neighbor in the adjacent tile.
    fn seam_crossing(&self) -> bool {
        (0..self.spec.tile_count()).into_par_iter().any(|index| {
            let id = self.spec.id_at(index);
            let border = self.dirs.read(id).clone();
            let ring = self.dirs.ring(id, DIRECTION_NODATA);
            let top = border.top().iter().enumerate().any(|(x, &d)| {
                let n = ring.north[x + 1];
                (d == 2 && n == 8) || (d == 4 && n == 6)
            });
            let left = border.left().iter().enumerate().any(|(y, &d)| {
                let w = ring.west[y];
                (d == 4 && w == 2) || (d == 6 && w == 8)
            });
            top || left
        })
    }

    /// Pass C: routes the deferred flats and delivers the directions of
    /// every tile `process_tile` did not finish.
    pub fn finish(self, finished: &TileOutput<'_, u8>) -> Result<D8Outcome> {
        if self.schedule == SteepSchedule::Parallel
            && (self.crossing.load(Relaxed) || self.seam_crossing())
        {
            return Ok(D8Outcome::CrossingDetected);
        }
        let _span = profile::span("d8.tiled.flats");
        let c = &self.counters;
        let mut stats = D8TileStats {
            tiles: c.tiles.load(Relaxed),
            flat_cells: c.flat_cells.load(Relaxed),
            interior_cells: c.interior_cells.load(Relaxed),
            interior_passes: c.interior_passes.load(Relaxed),
            deferred_cells: c.deferred_cells.load(Relaxed),
            deferred_tiles: c.deferred_tiles.load(Relaxed),
            mixed_components: c.mixed.load(Relaxed),
            crossing_fixes: c.crossing_fixes.load(Relaxed),
            passes: Vec::new(),
        };
        let interior = self
            .interior
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        let spec = self.spec;
        let mut active: Vec<TileId> = spec
            .row_major()
            .filter(|&id| self.deferred[spec.index(id)].load(Relaxed) > 0)
            .collect();
        let mut owned: Option<OwnedTables> = None;
        let mut pass = 0usize;
        while !active.is_empty() {
            pass += 1;
            let start = Instant::now();
            let _pass_span = profile::span(format!("d8.tiled.pass{pass}"));
            let mut p = PassStats {
                tiles: active.len() as u64,
                cells: active
                    .iter()
                    .map(|&id| self.deferred[spec.index(id)].load(Relaxed))
                    .sum(),
                ..Default::default()
            };
            // Pass 1 starts from Pass B's perimeters.
            let tables = owned.as_ref().map_or(
                Tables {
                    fall: &self.fall,
                    rise: &self.rise,
                    elev: &self.elev,
                },
                OwnedTables::view,
            );
            let edges: Vec<Vec<EdgeCell>> = active
                .par_iter()
                .map(|&id| self.edge_cells(id, tables.elev))
                .collect();
            self.relax(&active, &edges, tables, &mut p)?;

            let (pass_fall, pass_rise) = active
                .iter()
                .map(|&id| unpack(self.tile_max[spec.index(id)].load(Relaxed)))
                .fold((0, 0), |(f, r), (a, b)| (f.max(a), r.max(b)));
            let inner = interior.get(pass - 1).copied().unwrap_or([0, 0]);
            let last_fall = pass_fall.max(inner[0]);
            let last_rise = pass_rise.max(inner[1]);
            check_exact(pass, last_fall, last_rise)?;
            p.last_fall = last_fall;
            p.last_rise = last_rise;

            let next = OwnedTables {
                fall: BorderTable::new(spec, 0),
                rise: BorderTable::new(spec, 0),
                elev: BorderTable::new(spec, 1.0),
            };
            let constants = Constants {
                unresolved: last_fall.max(1) + 2,
                last_rise,
            };
            let remaining: Vec<u64> = active
                .par_iter()
                .map(|&id| self.route_tile(id, tables, next.view(), constants, finished))
                .collect::<Result<_>>()?;
            p.remaining = remaining.iter().sum();
            p.seconds = start.elapsed().as_secs_f64();
            active = active
                .into_iter()
                .zip(remaining)
                .filter_map(|(id, r)| (r > 0).then_some(id))
                .collect();
            stats.passes.push(p);
            owned = Some(next);
        }
        for (t, m) in interior.iter().enumerate().skip(pass) {
            check_exact(t + 1, m[0], m[1])?;
        }
        Ok(D8Outcome::Done(stats))
    }

    /// What tile `id`'s unresolved perimeter cells see across its edges.
    fn edge_cells(&self, id: TileId, elev: &BorderTable<f32>) -> Vec<EdgeCell> {
        let dirs = Around::new(&self.dirs, id, DIRECTION_NODATA);
        let z = Around::new(elev, id, f32::NAN);
        let rect = self.spec.rect(id);
        let frame = Frame::new(rect.width, rect.height);
        let mut out = Vec::new();
        for (lx, ly) in frame.perimeter() {
            let (x, y) = (lx as isize, ly as isize);
            if dirs.at(x, y) != 0 {
                continue;
            }
            let here = z.at(x, y);
            let open = |k| {
                !other_diagonal(k).is_some_and(|(a, ac, b, bc)| {
                    dirs.at(x + DX[a], y + DY[a]) == ac || dirs.at(x + DX[b], y + DY[b]) == bc
                })
            };
            let mut cell = EdgeCell {
                x: lx as u32,
                y: ly as u32,
                fall: 0,
                rise: 0,
                seed: 0,
            };
            for k in 1..9 {
                if frame.inside(lx, ly, k) {
                    continue;
                }
                let (nx, ny) = (x + DX[k], y + DY[k]);
                let (d, there) = (dirs.at(nx, ny), z.at(nx, ny));
                let bit = 1 << (k - 1);
                if d == 0 {
                    cell.rise |= bit;
                    if here == there && open(k) {
                        cell.fall |= bit;
                    }
                } else if d <= 8 {
                    if here >= there && open(k) {
                        cell.seed |= SEED_DRAIN;
                    }
                } else if here == there && open(k) {
                    cell.seed |= SEED_DEFAULT;
                }
                if there > here {
                    cell.seed |= SEED_RISE;
                }
            }
            if cell.fall | cell.rise | cell.seed != 0 {
                out.push(cell);
            }
        }
        out
    }

    /// Label-correcting rounds until no tile's perimeter levels change.
    fn relax(
        &self,
        active: &[TileId],
        edges: &[Vec<EdgeCell>],
        tables: Tables<'_>,
        p: &mut PassStats,
    ) -> Result<()> {
        let spec = &self.spec;
        let mut is_active = vec![false; spec.tile_count()];
        for &id in active {
            is_active[spec.index(id)] = true;
        }
        let dirty: Vec<AtomicBool> = is_active.iter().map(|&a| AtomicBool::new(a)).collect();
        let loads = AtomicU64::new(0);
        loop {
            let pending: Vec<usize> = (0..active.len())
                .filter(|&a| dirty[spec.index(active[a])].swap(false, Relaxed))
                .collect();
            if pending.is_empty() {
                break;
            }
            p.rounds += 1;
            p.relax_checks += pending.len() as u64;
            pending.par_iter().try_for_each(|&a| {
                if self.relax_tile(active[a], &edges[a], tables, &dirty, &is_active)? {
                    loads.fetch_add(1, Relaxed);
                }
                Ok::<(), Error>(())
            })?;
        }
        p.relax_loads = loads.load(Relaxed);
        Ok(())
    }

    /// Lowers tile `id`'s levels through its edges; `false` when nothing
    /// could be lowered (the tile is not loaded).
    fn relax_tile(
        &self,
        id: TileId,
        edges: &[EdgeCell],
        tables: Tables<'_>,
        dirty: &[AtomicBool],
        is_active: &[bool],
    ) -> Result<bool> {
        if edges.is_empty() {
            return Ok(false);
        }
        let fall_around = Around::new(tables.fall, id, 0);
        let rise_around = Around::new(tables.rise, id, 0);
        let rect = self.spec.rect(id);
        let frame = Frame::new(rect.width, rect.height);
        let mut s = self.scratch();
        let s = &mut *s;
        s.seeds.clear();
        s.seeds2.clear();
        for e in edges {
            let (x, y) = (e.x as isize, e.y as isize);
            let i = frame.at(e.x as usize, e.y as usize) as u32;
            let current = fall_around.at(x, y);
            let mut best = current;
            if e.seed & SEED_DRAIN != 0 {
                best = lower(best, 1);
            }
            if e.seed & SEED_DEFAULT != 0 {
                best = lower(best, 2);
            }
            for k in bits(e.fall) {
                let v = fall_around.at(x + DX[k], y + DY[k]);
                if v > 0 {
                    best = lower(best, v + 1);
                }
            }
            if best != current {
                s.seeds.push((best, i));
            }
            let current = rise_around.at(x, y);
            let mut best = current;
            if e.seed & SEED_RISE != 0 {
                best = lower(best, 1);
            }
            for k in bits(e.rise) {
                let v = rise_around.at(x + DX[k], y + DY[k]);
                if v > 0 {
                    best = lower(best, v + 1);
                }
            }
            if best != current {
                s.seeds2.push((best, i));
            }
        }
        if s.seeds.is_empty() && s.seeds2.is_empty() {
            return Ok(false);
        }
        let layers = &self.layers;
        let adj = layers
            .adj
            .take(id)?
            .ok_or_else(|| missing_tile(id, "adj"))?;
        let mut fall = layers
            .fall
            .take(id)?
            .ok_or_else(|| missing_tile(id, "fall"))?;
        let mut rise = layers
            .rise
            .take(id)?
            .ok_or_else(|| missing_tile(id, "rise"))?;
        for &(v, i) in s.seeds.iter() {
            fall[i as usize] = v;
        }
        for &(v, i) in s.seeds2.iter() {
            rise[i as usize] = v;
        }
        propagate(
            &frame,
            &mut fall,
            &mut s.seeds,
            |i| adj[i] as u8,
            &mut s.cur,
            &mut s.next,
        );
        propagate(
            &frame,
            &mut rise,
            &mut s.seeds2,
            |i| (adj[i] >> 8) as u8,
            &mut s.cur,
            &mut s.next,
        );
        let (max_fall, max_rise) = core_max(&frame, &fall, &rise);
        self.tile_max[self.spec.index(id)].store(pack(max_fall, max_rise), Relaxed);
        let halo = self.spec.halo_window(id);
        tables.fall.set_from_halo(id, &halo, &fall)?;
        tables.rise.set_from_halo(id, &halo, &rise)?;
        let changed = *tables.fall.read(id) != fall_around.border
            || *tables.rise.read(id) != rise_around.border;
        layers.adj.put_owned(id, adj)?;
        layers.fall.put_owned(id, fall)?;
        layers.rise.put_owned(id, rise)?;
        if changed {
            for k in 1..=8 {
                if let Some(n) = self.spec.neighbor(id, k) {
                    let n = self.spec.index(n);
                    if is_active[n] {
                        dirty[n].store(true, Relaxed);
                    }
                }
            }
        }
        Ok(true)
    }

    /// Routes tile `id`'s cells for the pass of `tables` and prepares its
    /// next pass in `next`; returns the cells left unresolved.
    fn route_tile(
        &self,
        id: TileId,
        tables: Tables<'_>,
        next: Tables<'_>,
        constants: Constants,
        finished: &TileOutput<'_, u8>,
    ) -> Result<u64> {
        let halo = self.spec.halo_window(id);
        let rect = halo.core;
        let frame = Frame::new(rect.width, rect.height);
        let layers = &self.layers;
        let mut dirs = layers
            .dir
            .take(id)?
            .ok_or_else(|| missing_tile(id, "dir"))?;
        let mut elev = layers
            .elev
            .take(id)?
            .ok_or_else(|| missing_tile(id, "elev"))?;
        let mut fall = layers
            .fall
            .take(id)?
            .ok_or_else(|| missing_tile(id, "fall"))?;
        let mut rise = layers
            .rise
            .take(id)?
            .ok_or_else(|| missing_tile(id, "rise"))?;
        layers.adj.remove(id);
        tables.fall.fill_halo_ring(id, &halo, &mut fall, 0)?;
        tables.rise.fill_halo_ring(id, &halo, &mut rise, 0)?;
        tables.elev.fill_halo_ring(id, &halo, &mut elev, f32::NAN)?;

        let mut s = self.scratch();
        let s = &mut *s;
        let Constants {
            unresolved,
            last_rise,
        } = constants;
        let level = |i: usize| {
            let f = fall[i];
            surface(if f == 0 { unresolved } else { f }, rise[i], last_rise)
        };
        // Route every unresolved cell on the pass's surface, then replace
        // the surface with the pass's combined levels.
        s.set.clear();
        s.routed.clear();
        for ly in 0..frame.h {
            let scale = self.factors.row(rect.y + ly);
            let start = frame.at(0, ly);
            for (i, _) in (start..)
                .zip(&dirs[start..start + frame.w])
                .filter(|(_, d)| **d == 0)
            {
                s.set.push(i as u32);
                s.routed
                    .push(route(&frame, i, &elev, &fall, &rise, scale, &level));
            }
        }
        for ly in 0..frame.h {
            let start = frame.at(0, ly);
            elev[start..start + frame.w].fill(1.0);
        }
        for &c in &s.set {
            elev[c as usize] = level(c as usize) as f32;
        }
        let mut remaining = 0;
        for (&c, &d) in s.set.iter().zip(&s.routed) {
            dirs[c as usize] = d;
            remaining += u64::from(d == 0);
        }
        self.dirs.set_from_halo(id, &halo, &dirs)?;
        next.elev.set_from_halo(id, &halo, &elev)?;
        self.deferred[self.spec.index(id)].store(remaining, Relaxed);
        if remaining == 0 {
            halo.core_from_buffer(&dirs, &mut s.core)?;
            return finished(id, &s.core).map(|()| 0);
        }

        // The next pass's levels from inside the tile.
        s.fall = fall;
        s.rise = rise;
        let mut adj = Vec::new();
        let (max_fall, max_rise, cells) = init_levels(&frame, &dirs, &elev, &mut adj, s);
        debug_assert_eq!(cells, remaining);
        self.tile_max[self.spec.index(id)].store(pack(max_fall, max_rise), Relaxed);
        next.fall.set_from_halo(id, &halo, &s.fall)?;
        next.rise.set_from_halo(id, &halo, &s.rise)?;
        layers.dir.put_owned(id, dirs)?;
        layers.elev.put_owned(id, elev)?;
        layers.adj.put_owned(id, adj)?;
        layers.fall.put_owned(id, std::mem::take(&mut s.fall))?;
        layers.rise.put_owned(id, std::mem::take(&mut s.rise))?;
        Ok(remaining)
    }
}

/// The perimeter tables of one Pass C flat pass.
#[derive(Clone, Copy)]
struct Tables<'a> {
    fall: &'a BorderTable<u32>,
    rise: &'a BorderTable<u32>,
    elev: &'a BorderTable<f32>,
}

struct OwnedTables {
    fall: BorderTable<u32>,
    rise: BorderTable<u32>,
    elev: BorderTable<f32>,
}

impl OwnedTables {
    fn view(&self) -> Tables<'_> {
        Tables {
            fall: &self.fall,
            rise: &self.rise,
            elev: &self.elev,
        }
    }
}

#[derive(Clone, Copy)]
struct Constants {
    unresolved: u32,
    last_rise: u32,
}

fn check_exact(pass: usize, last_fall: u32, last_rise: u32) -> Result<()> {
    let top = i64::from(last_fall.max(1)) + 2 + i64::from(last_rise) + 2;
    if top >= F32_EXACT {
        return Err(Error::Invalid(format!(
            "flat pass {pass} reaches increment level {top}, beyond the exact f32 range \
             tiled flat routing relies on; use --mode in-memory"
        )));
    }
    Ok(())
}

#[inline]
fn pack(fall: u32, rise: u32) -> u64 {
    u64::from(fall) << 32 | u64::from(rise)
}

#[inline]
fn unpack(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, v as u32)
}

/// The highest fall and rise levels of the core.
fn core_max(frame: &Frame, fall: &[u32], rise: &[u32]) -> (u32, u32) {
    let mut m = (0, 0);
    for ly in 0..frame.h {
        let start = frame.at(0, ly);
        let r = start..start + frame.w;
        m.0 = fall[r.clone()].iter().fold(m.0, |a, &b| a.max(b));
        m.1 = rise[r].iter().fold(m.1, |a, &b| a.max(b));
    }
    m
}

fn clear_ring<T: Copy>(frame: &Frame, buffer: &mut [T], value: T) {
    let bw = frame.bw;
    let bh = frame.h + 2;
    buffer[..bw].fill(value);
    buffer[(bh - 1) * bw..].fill(value);
    for r in 1..bh - 1 {
        buffer[r * bw] = value;
        buffer[r * bw + bw - 1] = value;
    }
}

/// Whether a cell's unrestricted direction crosses its north or west
/// neighbor's inside the tile (TauDEM's initial crossing candidates).
fn inside_crossing(frame: &Frame, dirs: &[u8]) -> bool {
    let bw = frame.bw;
    (0..frame.h).any(|ly| {
        let start = frame.at(0, ly);
        (0..frame.w).any(|lx| {
            let i = start + lx;
            let north = || ly > 0 && dirs[i - bw] == 8;
            match dirs[i] {
                2 => north(),
                4 => (ly > 0 && dirs[i - bw] == 6) || (lx > 0 && dirs[i - 1] == 2),
                6 => lx > 0 && dirs[i - 1] == 8,
                _ => false,
            }
        })
    })
}

/// Marks the core's unresolved cells: [`DEFERRED`] for components that
/// reach the tile edge, [`INTERIOR`] for the rest. Returns the unresolved
/// and deferred cell counts.
fn classify(frame: &Frame, dirs: &[u8], class: &mut Vec<u8>, stack: &mut Vec<u32>) -> (u64, u64) {
    class.clear();
    class.resize(frame.cells(), 0);
    let mut flats = 0;
    for ly in 0..frame.h {
        let start = frame.at(0, ly);
        for i in start..start + frame.w {
            if dirs[i] == 0 {
                class[i] = INTERIOR;
                flats += 1;
            }
        }
    }
    stack.clear();
    for (lx, ly) in frame.perimeter() {
        let i = frame.at(lx, ly);
        if class[i] == INTERIOR {
            class[i] = DEFERRED;
            stack.push(i as u32);
        }
    }
    let deferred = flood(frame, class, stack);
    (flats, deferred)
}

/// Marks every [`INTERIOR`] cell 8-connected to the stacked (already
/// marked) cells [`DEFERRED`]; returns how many cells are marked, stacked
/// ones included.
fn flood(frame: &Frame, class: &mut [u8], stack: &mut Vec<u32>) -> u64 {
    let mut marked = stack.len() as u64;
    while let Some(c) = stack.pop() {
        let i = c as usize;
        for k in 1..9 {
            let j = frame.nb(i, k);
            if class[j] == INTERIOR {
                class[j] = DEFERRED;
                marked += 1;
                stack.push(j as u32);
            }
        }
    }
    marked
}

/// TauDEM's `setFlow2` for unresolved cell `i`: the steepest descent on the
/// combined levels toward rised neighbors, or the first non-rised neighbor
/// no higher than `i`.
#[inline]
fn route(
    frame: &Frame,
    i: usize,
    elev: &[f32],
    fall: &[u32],
    rise: &[u32],
    scale: &[f64; 9],
    level: &impl Fn(usize) -> i64,
) -> u8 {
    let here = level(i);
    let mut d = if fall[i] == 0 { DIRECTION_NODATA } else { 0 };
    let mut best = 0.0;
    for k in ORDER {
        let j = frame.nb(i, k);
        if rise[j] > 0 {
            let slope = scaled((here - level(j)) as f32, scale[k]);
            if slope > best {
                best = slope;
                d = k as u8;
            }
        } else if elev[i] >= elev[j] {
            d = k as u8;
            break;
        }
    }
    d
}

struct Interior {
    /// The highest fall and rise levels of each pass.
    maxima: Vec<[u32; 2]>,
    /// Components deferred because they were mixed, and their cells.
    mixed: u64,
    deferred: u64,
}

/// Routes every [`INTERIOR`] component through TauDEM's flat passes inside
/// the tile, with the tile's own highest levels in place of the raster's.
/// A mixed component (drained and undrained cells side by side), whose
/// routing depends on the raster-wide `last_fall`, is marked [`DEFERRED`]
/// and the tile's interior routed again without it.
fn resolve_interior(
    frame: &Frame,
    rect: Rect,
    fel: &[f32],
    factors: &Factors,
    s: &mut Scratch,
) -> Interior {
    let cells = frame.cells();
    let mut mixed = 0;
    let mut deferred = 0;
    'restart: loop {
        let Scratch {
            dirs,
            class,
            steps,
            fall,
            rise,
            elev,
            set,
            prev,
            cur,
            next,
            seeds,
            routed,
            ..
        } = s;
        set.clear();
        for ly in 0..frame.h {
            let start = frame.at(0, ly);
            for i in start..start + frame.w {
                // Mixed components deferred on a restart start over too.
                if class[i] != 0 {
                    dirs[i] = 0;
                }
                if class[i] == INTERIOR {
                    set.push(i as u32);
                }
            }
        }
        if set.is_empty() {
            return Interior {
                maxima: Vec::new(),
                mixed,
                deferred,
            };
        }
        fall.clear();
        fall.resize(cells, 0);
        rise.clear();
        rise.resize(cells, 0);
        steps.clear();
        steps.resize(cells, 0);
        prev.clear();
        let mut maxima = Vec::new();
        let mut pass = 1;
        loop {
            let e: &[f32] = if pass == 1 { fel } else { elev.as_slice() };
            // incfall seeds and steps.
            seeds.clear();
            for &c in set.iter() {
                let i = c as usize;
                let z = e[i];
                let (mut drains, mut default, mut m) = (false, false, 0u8);
                for k in 1..9 {
                    let j = frame.nb(i, k);
                    let (d, zj) = (dirs[j], e[j]);
                    let open = || !crosses(frame, dirs, i, k);
                    if d == 0 {
                        if z == zj && open() {
                            m |= 1 << (k - 1);
                        }
                    } else if d <= 8 {
                        drains |= z >= zj && open();
                    } else {
                        default |= z == zj && open();
                    }
                }
                steps[i] = m;
                let level = if drains {
                    1
                } else if default {
                    2
                } else {
                    0
                };
                fall[i] = level;
                if level > 0 {
                    seeds.push((level, c));
                }
            }
            propagate(frame, fall, seeds, |i| steps[i], cur, next);
            let last_fall = set.iter().map(|&c| fall[c as usize]).max().unwrap_or(0);
            // A mixed component needs the raster's last_fall: defer it.
            let mut found = false;
            for &c in set.iter() {
                let i = c as usize;
                if fall[i] == 0
                    && class[i] == INTERIOR
                    && (1..9).any(|k| {
                        let j = frame.nb(i, k);
                        dirs[j] == 0 && fall[j] > 0
                    })
                {
                    class[i] = DEFERRED;
                    cur.clear();
                    cur.push(c);
                    deferred += flood(frame, class, cur);
                    mixed += 1;
                    found = true;
                }
            }
            if found {
                continue 'restart;
            }
            let unresolved = last_fall.max(1) + 2;
            // incrise over every unresolved cell.
            seeds.clear();
            for &c in set.iter() {
                let i = c as usize;
                if (1..9).any(|k| e[frame.nb(i, k)] > e[i]) {
                    rise[i] = 1;
                    seeds.push((1, c));
                }
            }
            let open_mask = |i: usize| {
                let mut m = 0u8;
                for k in 1..9 {
                    if dirs[frame.nb(i, k)] == 0 {
                        m |= 1 << (k - 1);
                    }
                }
                m
            };
            propagate(frame, rise, seeds, open_mask, cur, next);
            let last_rise = set.iter().map(|&c| rise[c as usize]).max().unwrap_or(0);
            maxima.push([last_fall, last_rise]);

            let level = |i: usize| {
                let f = fall[i];
                surface(if f == 0 { unresolved } else { f }, rise[i], last_rise)
            };
            routed.clear();
            for &c in set.iter() {
                let i = c as usize;
                let scale = factors.row(rect.y + frame.row(i));
                routed.push(route(frame, i, e, fall, rise, scale, &level));
            }
            let remaining = routed.iter().filter(|&&d| d == 0).count();
            if remaining > 0 {
                // The next pass's artificial surface: 1 except the cells of
                // this pass.
                if pass == 1 {
                    elev.clear();
                    elev.resize(cells, 1.0);
                } else {
                    for &c in prev.iter() {
                        elev[c as usize] = 1.0;
                    }
                }
                for &c in set.iter() {
                    elev[c as usize] = level(c as usize) as f32;
                }
            }
            for (&c, &d) in set.iter().zip(routed.iter()) {
                dirs[c as usize] = d;
            }
            if remaining == 0 {
                return Interior {
                    maxima,
                    mixed,
                    deferred,
                };
            }
            for &c in set.iter() {
                fall[c as usize] = 0;
                rise[c as usize] = 0;
            }
            std::mem::swap(prev, set);
            set.clear();
            set.extend(prev.iter().copied().filter(|&c| dirs[c as usize] == 0));
            pass += 1;
        }
    }
}

/// The levels of a tile's unresolved cells (`dirs == 0`) from the seeds and
/// steps inside the tile: fills `s.fall`, `s.rise` and `adj` (bits `k - 1`
/// of the low byte are fall steps, of the high byte rise steps, both to core
/// cells only). Returns the highest fall and rise levels and the cells.
fn init_levels(
    frame: &Frame,
    dirs: &[u8],
    e: &[f32],
    adj: &mut Vec<u16>,
    s: &mut Scratch,
) -> (u32, u32, u64) {
    let cells = frame.cells();
    adj.clear();
    adj.resize(cells, 0);
    s.fall.clear();
    s.fall.resize(cells, 0);
    s.rise.clear();
    s.rise.resize(cells, 0);
    s.seeds.clear();
    s.seeds2.clear();
    let mut count = 0;
    for ly in 0..frame.h {
        for lx in 0..frame.w {
            let i = frame.at(lx, ly);
            if dirs[i] != 0 {
                continue;
            }
            count += 1;
            let edge = frame.on_edge(lx, ly);
            let z = e[i];
            let (mut drains, mut default, mut higher) = (false, false, false);
            let (mut fall_mask, mut rise_mask) = (0u8, 0u8);
            for k in 1..9 {
                if edge && !frame.inside(lx, ly, k) {
                    continue;
                }
                let j = frame.nb(i, k);
                let (d, zj) = (dirs[j], e[j]);
                higher |= zj > z;
                let open = || !crosses(frame, dirs, i, k);
                if d == 0 {
                    rise_mask |= 1 << (k - 1);
                    if z == zj && open() {
                        fall_mask |= 1 << (k - 1);
                    }
                } else if d <= 8 {
                    drains |= z >= zj && open();
                } else {
                    default |= z == zj && open();
                }
            }
            adj[i] = u16::from(fall_mask) | u16::from(rise_mask) << 8;
            let level = if drains {
                1
            } else if default {
                2
            } else {
                0
            };
            if level > 0 {
                s.fall[i] = level;
                s.seeds.push((level, i as u32));
            }
            if higher {
                s.rise[i] = 1;
                s.seeds2.push((1, i as u32));
            }
        }
    }
    propagate(
        frame,
        &mut s.fall,
        &mut s.seeds,
        |i| adj[i] as u8,
        &mut s.cur,
        &mut s.next,
    );
    propagate(
        frame,
        &mut s.rise,
        &mut s.seeds2,
        |i| (adj[i] >> 8) as u8,
        &mut s.cur,
        &mut s.next,
    );
    let (f, r) = core_max(frame, &s.fall, &s.rise);
    (f, r, count)
}

/// Fills `buffer` with tile `id`'s radius-1 halo of a whole row-major
/// raster, `outside` beyond its edges (for tests and in-memory callers).
pub fn halo_from_grid<T: Copy>(
    spec: &TileSpec,
    id: TileId,
    grid: &[T],
    outside: T,
    buffer: &mut Vec<T>,
) -> Result<HaloWindow> {
    let halo = spec.halo_window(id);
    let w = spec.raster_width();
    if grid.len() as u64 != spec.cells() {
        return Err(Error::Invalid(format!(
            "{} values for a raster of {} cells",
            grid.len(),
            spec.cells()
        )));
    }
    let bw = halo.buffer_width();
    buffer.clear();
    buffer.resize(halo.buffer_cells(), outside);
    let win = halo.window;
    for row in 0..win.height {
        let src = (win.y + row) * w + win.x;
        let dst = (row + halo.pad_top) * bw + halo.pad_left;
        buffer[dst..dst + win.width].copy_from_slice(&grid[src..src + win.width]);
    }
    Ok(halo)
}

/// Fills a tile's radius-1 halo buffer of filled elevations.
pub type ReadTile<'a> = dyn Fn(TileId, &HaloWindow, &mut Vec<f32>) -> Result<()> + Sync + 'a;
/// Receives a tile's row-major core values.
pub type TileOutput<'a, T> = dyn Fn(TileId, &[T]) -> Result<()> + Sync + 'a;

/// Runs tiled D8 over every tile of `spec`: `read` fills a tile's radius-1
/// halo of filled elevations, `slope` receives each tile's `sd8` and
/// `direction` its final directions (row-major cores of [`TileSpec::rect`]).
/// The parallel schedule returns [`D8Outcome::CrossingDetected`] when the
/// crossing rule might apply; the caller then reruns with
/// [`SteepSchedule::Wavefront`] after discarding what the sinks received.
#[allow(clippy::too_many_arguments)]
pub fn d8_flow_tiled(
    spec: TileSpec,
    nodata: f32,
    spacing: &[(f64, f64)],
    cache: &TileCache,
    schedule: SteepSchedule,
    read: &ReadTile<'_>,
    slope: Option<&TileOutput<'_, f32>>,
    direction: &TileOutput<'_, u8>,
) -> Result<D8Outcome> {
    let d8 = TiledD8::new(spec, nodata, spacing, cache, schedule)?;
    let pool: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
    let buffer = || {
        pool.lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_default()
    };
    let give_back = |b: Vec<f32>| pool.lock().unwrap_or_else(|p| p.into_inner()).push(b);
    let load = |id: TileId, fel: &mut Vec<f32>| {
        let halo = spec.halo_window(id);
        fel.clear();
        fel.resize(halo.buffer_cells(), f32::NAN);
        read(id, &halo, fel)
    };
    let compute = |id: TileId, fel: &[f32]| -> Result<()> {
        if d8.crossing_detected() {
            return Ok(());
        }
        let mut sd8 = buffer();
        d8.process_tile(id, fel, slope.is_some().then_some(&mut sd8), |dirs| {
            direction(id, dirs)
        })?;
        if let Some(sink) = slope {
            sink(id, &sd8)?;
        }
        give_back(sd8);
        Ok(())
    };
    {
        let _span = profile::span("d8.tiled.tiles");
        match schedule {
            SteepSchedule::Parallel => {
                // One plain thread reads tiles ahead of the workers. A worker
                // waiting on a read that runs on another Rayon pool would
                // steal and nest further tiles, unbounding the tiles in
                // flight; the channel bounds them instead.
                let threads = rayon::current_num_threads();
                let (sender, receiver) = std::sync::mpsc::sync_channel(threads);
                let stop = AtomicBool::new(false);
                let read_ns = AtomicU64::new(0);
                std::thread::scope(|scope| -> Result<()> {
                    let (stop, read_ns, d8, buffer, load) = (&stop, &read_ns, &d8, &buffer, &load);
                    let producer = scope.spawn(move || -> Result<()> {
                        for id in spec.row_major() {
                            if stop.load(Relaxed) || d8.crossing_detected() {
                                break;
                            }
                            let mut fel = buffer();
                            let start = Instant::now();
                            load(id, &mut fel)?;
                            read_ns.fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                            if sender.send((id, fel)).is_err() {
                                break;
                            }
                        }
                        Ok(())
                    });
                    let consumed = receiver.into_iter().par_bridge().try_for_each(
                        |(id, fel): (TileId, Vec<f32>)| {
                            let result = compute(id, &fel);
                            give_back(fel);
                            if result.is_err() {
                                stop.store(true, Relaxed);
                            }
                            result
                        },
                    );
                    stop.store(true, Relaxed);
                    let produced = producer
                        .join()
                        .unwrap_or_else(|_| Err(Error::Invalid("the tile reader panicked".into())));
                    consumed.and(produced)
                })?;
                profile::count("d8.tiled.read_us", read_ns.load(Relaxed) / 1000);
            }
            SteepSchedule::Wavefront => for_each_wavefront(&spec, |id| {
                let mut fel = buffer();
                load(id, &mut fel)?;
                let result = compute(id, &fel);
                give_back(fel);
                result
            })?,
        }
    }
    drop(pool);
    d8.finish(direction)
}
