// SPDX-License-Identifier: GPL-3.0-only
//! Tiled, bounded-memory AreaD8: Barnes (2017) two-pass flow accumulation
//! that reproduces the in-memory [`crate::area_d8`] bit for bit, including
//! TauDEM's `f32` rounding above 2^24 cells.
//!
//! # Passes
//!
//! 1. **Pass 1** ([`tile_area_pass1`], parallel over tiles): each tile's D8
//!    codes with a 1-cell halo are accumulated on their own, with every
//!    inflow from outside the tile taken as zero. The result is a
//!    [`TileSummary`]: for every *exit* (a perimeter cell flowing into an
//!    active cell of another tile) its local area, exact count and whether
//!    it is already nodata (contaminated or blocked), and for every *entry*
//!    (a perimeter cell receiving flow from another tile) the exit its flow
//!    path leaves through, if any.
//! 2. **Perimeter solve** ([`AreaTiles::solve`]): exits are the nodes of a
//!    graph with out-degree at most one (exit → the exit its downstream
//!    neighbor's path reaches). The graph is acyclic unless the D8 codes
//!    themselves contain a cycle, even when the tile-to-tile graph has
//!    cycles (a river A → B → C → A → B). Kahn's algorithm propagates exact
//!    `u64` counts and the nodata state; nodes left on or below a cycle are
//!    blocked, as cycles are in memory.
//! 3. **Replay** (only when some exit with an inflow exceeds 2^24 cells):
//!    the few tiles holding such exits are read again and their f32 values
//!    are recomputed by small programs, in the perimeter graph's
//!    topological order (see below).
//! 4. **Pass 2** ([`SolvedArea::pass2`], parallel over tiles): each tile is
//!    accumulated again with the solved f32 area of every halo cell that
//!    flows into it, which gives every cell its final value.
//!
//! # Why the result is bit-exact
//!
//! The in-memory kernel sets every active, non-nodata cell to
//! `((1 + a_1) + a_2) + ...`, summed in `f32` over the upstream neighbors in
//! fixed `k = 1..8` order. That is a function of the neighbors' final `f32`
//! values only, not of the order cells are visited. Pass 2 evaluates the
//! same expression with the same neighbor order, so it is exact as soon as
//! the injected halo values equal the in-memory values of those cells.
//! Every halo cell flowing into a tile is an exit of its own tile, so the
//! perimeter solve must give each exit `q` its in-memory value `v(q)`:
//!
//! * **No inflow** reaches `q` (no edge into it): `v(q)` is Pass 1's local
//!   value, computed by the same expression from the same inputs.
//! * **Exact count `n(q) <= 2^24`**: every upstream cell has a smaller count
//!   and, by induction, an exact value, and every partial sum of `q`'s
//!   expression is an integer at most `n(q)`, hence representable, so each
//!   `f32` addition is exact: `v(q) = n(q)`. Counts superpose exactly in
//!   integers, so Kahn's `u64` propagation gives `n(q)`.
//! * **Otherwise** (a mainstem past 2^24): `q`'s tile is read again with the
//!   exact `u64` inflows injected, which gives the exact count of every cell
//!   in it. The cells draining to `q` with counts above 2^24 form an in-tree
//!   (D8 is a tree); for each, in upstream-first order, the program records
//!   its expression with each upstream neighbor as a constant (an in-tile
//!   cell with count at most 2^24, whose value is that count), a slot (an
//!   earlier cell of the program) or an external value (a halo cell: an
//!   exit of the perimeter graph with an edge into `q`, so already solved
//!   in topological order). Running the program in `f32` evaluates the
//!   in-memory expressions with the in-memory operands, so it yields `v(q)`.
//!   Constants that open an expression are folded in the same order.
//!
//! Nodata propagates identically: in memory a contaminated cell completes
//! without an area and contaminates its downstream cells, and a blocked
//! cell (the `k == 4 && other == 0` quirk, or a cycle) never completes, so
//! its downstream cells never do. Both leave every downstream cell nodata,
//! so a single "poisoned" bit per exit suffices; Pass 2 injects it as a
//! nodata inflow that contaminates the receiving cell even under `-nc`.
//! The quirk itself, edge contamination and `-nc` are evaluated from the
//! halo exactly as in memory: halo cells outside the raster are
//! `DIRECTION_NODATA`, which contaminates exactly the raster's edge cells
//! (as `!interior` does) and never counts as an inflow or as a code-0 NW
//! neighbor.
//!
//! # Extending to `-o` and `-wg`
//!
//! The kernel is generic over the accumulated value ([`Value`] here is
//! private: `f32` and `u64`). Weights replace the `1` that starts each
//! expression with the cell's weight; then counts no longer bound the
//! rounding, so the replay must cover every exit with an inflow
//! ([`AreaOptions::replay_above`] `= 0` already does, slowly). Outlets
//! restrict accumulation to cells upstream of them, a mask applied before
//! Pass 1 that the perimeter graph carries unchanged.

// The neighbor loops index by D8 direction, as TauDEM's code does.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use rayon::prelude::*;

use crate::tiling::{
    BorderTable, Error, HaloWindow, Rect, Result, StoreOptions, Tier, TileCache, TileId, TileSpec,
    TileStats, TileStore,
};
use crate::{DIRECTION_NODATA, DX, DY, OPPOSITE, SCALAR_NODATA, profile};

/// The largest count whose every `f32` partial sum is exact: 2^24.
pub const F32_EXACT_LIMIT: u64 = 1 << 24;

const COUNT: u8 = 15;
const CONTAMINATED: u8 = 16;
const INACTIVE: u8 = 32;
const HEAD: u8 = 64;
/// Halo cells; core states never reach 128.
const RING: u8 = 128;
/// Core cells on the flow path being traced, and cells whose exit is known.
const TRACING: u8 = 0xFE;
const TRACED: u8 = 0xFF;
const NONE: u32 = u32::MAX;

/// How areas are summed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum Arithmetic {
    /// TauDEM's `f32` sums in TauDEM's order: bit for bit the in-memory
    /// [`crate::area_d8`] and C++ AreaD8, which stop counting single cells
    /// once a mainstem passes 2^24.
    #[default]
    TauDem,
    /// Exact integer counts rounded to `f32` once per cell. Equal to
    /// [`Self::TauDem`] up to 2^24 cells, closer to the true count above.
    Exact,
}

/// AreaD8 options shared by every pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AreaOptions {
    /// Edge contamination (`false` for `-nc`).
    pub check_contamination: bool,
    pub arithmetic: Arithmetic,
    /// Exits with an inflow whose exact area exceeds this are replayed in
    /// `f32` (at most [`F32_EXACT_LIMIT`], the default). Lower values only
    /// replay more; tests use them to exercise the replay on small rasters.
    pub replay_above: u64,
}

impl AreaOptions {
    pub fn new(check_contamination: bool) -> Self {
        Self {
            check_contamination,
            arithmetic: Arithmetic::TauDem,
            replay_above: F32_EXACT_LIMIT,
        }
    }

    pub fn with_arithmetic(mut self, arithmetic: Arithmetic) -> Self {
        self.arithmetic = arithmetic;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.replay_above > F32_EXACT_LIMIT {
            return Err(Error::Invalid(format!(
                "replay_above {} exceeds 2^24",
                self.replay_above
            )));
        }
        Ok(())
    }
}

/// An accumulated quantity: TauDEM's `f32` area or an exact count.
trait Value: Copy + PartialEq + Send + Sync + 'static {
    const ZERO: Self;
    const ONE: Self;
    const NODATA: Self;
    fn add(self, other: Self) -> Self;
}

impl Value for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const NODATA: Self = SCALAR_NODATA;
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
}

impl Value for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const NODATA: Self = u64::MAX;
    #[inline]
    fn add(self, other: Self) -> Self {
        self + other
    }
}

/// A radius-1 halo buffer's layout.
#[derive(Clone, Copy, Debug)]
struct Geom {
    bw: usize,
    w: usize,
    h: usize,
    off: [isize; 9],
}

impl Geom {
    fn new(halo: &HaloWindow) -> Result<Self> {
        if halo.radius != 1 {
            return Err(Error::Invalid(format!(
                "AreaD8 tiles need a 1-cell halo, not {}",
                halo.radius
            )));
        }
        let bw = halo.buffer_width();
        Ok(Self {
            bw,
            w: halo.core.width,
            h: halo.core.height,
            off: std::array::from_fn(|k| DX[k] + DY[k] * bw as isize),
        })
    }

    fn cells(&self) -> usize {
        self.bw * (self.h + 2)
    }

    #[inline]
    fn at(&self, lx: usize, ly: usize) -> usize {
        (ly + 1) * self.bw + lx + 1
    }

    #[inline]
    fn step(&self, i: usize, k: usize) -> usize {
        i.wrapping_add_signed(self.off[k])
    }

    /// Whether the neighbor of core cell `(lx, ly)` in direction `k` is a halo cell.
    #[inline]
    fn outside(&self, lx: usize, ly: usize, k: usize) -> bool {
        let x = lx as isize + DX[k];
        let y = ly as isize + DY[k];
        x < 0 || y < 0 || x >= self.w as isize || y >= self.h as isize
    }

    /// The core-local, row-major index of buffer cell `i`.
    fn local(&self, i: usize) -> u32 {
        ((i / self.bw - 1) * self.w + i % self.bw - 1) as u32
    }

    /// Visits the halo ring's buffer coordinates.
    fn for_ring(&self, mut visit: impl FnMut(usize, usize)) {
        let bh = self.h + 2;
        for bx in 0..self.bw {
            visit(bx, 0);
        }
        for by in 1..bh - 1 {
            visit(0, by);
            visit(self.bw - 1, by);
        }
        for bx in 0..self.bw {
            visit(bx, bh - 1);
        }
    }

    /// Visits the core's perimeter cells `(lx, ly)` in row-major order.
    fn for_perimeter(&self, mut visit: impl FnMut(usize, usize)) {
        for lx in 0..self.w {
            visit(lx, 0);
        }
        for ly in 1..self.h.saturating_sub(1) {
            visit(0, ly);
            if self.w > 1 {
                visit(self.w - 1, ly);
            }
        }
        if self.h > 1 {
            for lx in 0..self.w {
                visit(lx, self.h - 1);
            }
        }
    }
}

/// Accumulates the core of a halo buffer. Halo cells of `vals` hold the
/// value flowing in from each halo cell that drains into the core (zero for
/// none, `NODATA` for a poisoned inflow); the core's values are computed as
/// in [`crate::area_d8`]. Returns the cells accumulated.
fn accumulate<V: Value>(
    g: &Geom,
    dir: &[u8],
    state: &mut [u8],
    vals: &mut [V],
    check_contamination: bool,
) -> u64 {
    debug_assert!(dir.len() == g.cells() && state.len() == g.cells() && vals.len() == g.cells());
    g.for_ring(|bx, by| state[by * g.bw + bx] = RING);
    for ly in 0..g.h {
        let edge_row = ly == 0 || ly + 1 == g.h;
        for lx in 0..g.w {
            let i = g.at(lx, ly);
            vals[i] = V::NODATA;
            if dir[i] > 8 {
                state[i] = INACTIVE;
                continue;
            }
            let edge = edge_row || lx == 0 || lx + 1 == g.w;
            let mut incoming = 0u8;
            let mut nodata_neighbor = false;
            let mut poisoned = false;
            for k in 1..9 {
                let j = g.step(i, k);
                let other = dir[j];
                if other == OPPOSITE[k] {
                    if edge && g.outside(lx, ly, k) {
                        poisoned |= vals[j] == V::NODATA;
                    } else {
                        incoming += 1;
                    }
                } else if k == 4 && other == 0 {
                    // TauDEM counts a northwest neighbor with direction 0
                    // (in this tile or the next) as flowing in; such a cell
                    // never completes.
                    incoming += 1;
                }
                nodata_neighbor |= other == DIRECTION_NODATA;
            }
            let contaminated = (check_contamination && nodata_neighbor) || poisoned;
            state[i] = incoming
                | (u8::from(contaminated) * CONTAMINATED)
                | (u8::from(incoming == 0) * HEAD);
        }
    }
    let mut visited = 0;
    for ly in 0..g.h {
        for lx in 0..g.w {
            let start = g.at(lx, ly);
            // Heads have no upstream cell in the core, so no walk reaches them.
            if state[start] & HEAD != 0 {
                visited += walk(g, dir, state, vals, start);
            }
        }
    }
    visited
}

/// Accumulates `start` and continues downstream through every cell whose
/// last upstream neighbor this walk completes.
#[inline]
fn walk<V: Value>(g: &Geom, dir: &[u8], state: &mut [u8], vals: &mut [V], start: usize) -> u64 {
    let mut i = start;
    let mut contaminated = state[i] & CONTAMINATED != 0;
    let mut visited = 0;
    loop {
        visited += 1;
        if !contaminated {
            let mut sum = V::ONE;
            for k in 1..9 {
                let j = g.step(i, k);
                if dir[j] == OPPOSITE[k] {
                    sum = sum.add(vals[j]);
                }
            }
            vals[i] = sum;
        }
        state[i] = INACTIVE;
        let k = dir[i] as usize;
        if k == 0 {
            break;
        }
        let j = g.step(i, k);
        let mut s = state[j];
        // Halo cells are another tile's; inactive cells never accumulate.
        if s & (RING | INACTIVE) != 0 {
            break;
        }
        if contaminated {
            s |= CONTAMINATED;
        }
        s -= 1;
        state[j] = s;
        if s & COUNT != 0 {
            break;
        }
        contaminated = s & CONTAMINATED != 0;
        i = j;
    }
    visited
}

/// Reusable per-worker buffers of the tile kernels.
#[derive(Debug, Default)]
pub struct AreaScratch {
    state: Vec<u8>,
    area: Vec<f32>,
    counts: Vec<u64>,
    path: Vec<usize>,
}

impl AreaScratch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes allocated.
    pub fn bytes(&self) -> usize {
        self.state.capacity()
            + self.area.capacity() * 4
            + self.counts.capacity() * 8
            + self.path.capacity() * 8
    }

    fn prepare(&mut self, cells: usize, counts: bool) {
        self.state.resize(cells, 0);
        self.area.resize(cells, 0.0);
        if counts {
            self.counts.resize(cells, 0);
        }
    }
}

/// Upper bound of [`AreaScratch`] plus the direction buffer of one worker,
/// in bytes per halo cell (excluding the exact counts that tiles of more
/// than 2^24 cells and [`Arithmetic::Exact`] add, see [`worker_bytes`]).
pub const WORKER_BYTES_PER_CELL: u64 = 1 + 1 + 4;

/// Bytes of one worker's buffers for `tile_w x tile_h` tiles.
pub fn worker_bytes(tile_w: usize, tile_h: usize, options: &AreaOptions) -> u64 {
    let cells = (tile_w as u64 + 2) * (tile_h as u64 + 2);
    let counts =
        tile_w as u64 * tile_h as u64 > F32_EXACT_LIMIT || options.arithmetic == Arithmetic::Exact;
    cells * (WORKER_BYTES_PER_CELL + if counts { 8 } else { 0 })
}

/// An exit: a perimeter cell flowing into an active cell of another tile.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Exit {
    /// Core-local index.
    local: u32,
    /// Direction to the downstream cell.
    k: u8,
    /// Pass 1's `f32` value (`SCALAR_NODATA` when contaminated or blocked).
    value: f32,
    /// The exact count of the tile's cells draining to it.
    count: u64,
}

/// An entry: a perimeter cell receiving flow from another tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    local: u32,
    /// The index of the exit its flow path leaves the tile through, or `NONE`.
    terminal: u32,
}

/// What Pass 1 reports about one tile: its exits and entries.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileSummary {
    exits: Vec<Exit>,
    entries: Vec<Entry>,
    cells: u64,
}

impl TileSummary {
    pub fn exits(&self) -> usize {
        self.exits.len()
    }

    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// Cells accumulated by Pass 1.
    pub fn cells(&self) -> u64 {
        self.cells
    }

    fn exit_index(&self, local: u32) -> Option<u32> {
        self.exits
            .binary_search_by_key(&local, |e| e.local)
            .ok()
            .map(|i| i as u32)
    }
}

/// Pass 1 on one tile: `dir` is a radius-1 halo buffer of D8 codes
/// (`DIRECTION_NODATA` outside the raster).
pub fn tile_area_pass1(
    halo: &HaloWindow,
    dir: &[u8],
    options: &AreaOptions,
    scratch: &mut AreaScratch,
) -> Result<TileSummary> {
    let g = Geom::new(halo)?;
    check_buffer(&g, dir)?;
    let exact_counts = (g.w * g.h) as u64 > F32_EXACT_LIMIT;
    scratch.prepare(g.cells(), exact_counts);
    let check = options.check_contamination;
    g.for_ring(|bx, by| scratch.area[by * g.bw + bx] = 0.0);
    let cells = accumulate(&g, dir, &mut scratch.state, &mut scratch.area, check);
    if exact_counts {
        g.for_ring(|bx, by| scratch.counts[by * g.bw + bx] = 0);
        accumulate(&g, dir, &mut scratch.state, &mut scratch.counts, check);
    }

    let mut exits = Vec::new();
    g.for_perimeter(|lx, ly| {
        let i = g.at(lx, ly);
        let k = dir[i] as usize;
        if (1..=8).contains(&k) && g.outside(lx, ly, k) && dir[g.step(i, k)] <= 8 {
            let value = scratch.area[i];
            let count = if value == SCALAR_NODATA {
                0
            } else if exact_counts {
                scratch.counts[i]
            } else {
                value as u64
            };
            exits.push(Exit {
                local: (ly * g.w + lx) as u32,
                k: k as u8,
                value,
                count,
            });
        }
    });

    // Trace each entry's flow path to its exit. The state and area buffers
    // are free now: state marks traced cells and area holds their exits.
    let summary = TileSummary {
        exits,
        entries: Vec::new(),
        cells,
    };
    let AreaScratch {
        state, area, path, ..
    } = scratch;
    let labels: &mut [u32] = bytes_as_u32(area);
    let mut entries = Vec::new();
    g.for_perimeter(|lx, ly| {
        let i = g.at(lx, ly);
        if dir[i] > 8 {
            return;
        }
        let entered = (1..9).any(|k| g.outside(lx, ly, k) && dir[g.step(i, k)] == OPPOSITE[k]);
        if entered {
            let terminal = trace(&g, dir, state, labels, path, &summary, i);
            entries.push(Entry {
                local: (ly * g.w + lx) as u32,
                terminal,
            });
        }
    });
    Ok(TileSummary { entries, ..summary })
}

fn bytes_as_u32(values: &mut [f32]) -> &mut [u32] {
    // SAFETY: f32 and u32 have the same size and alignment and every bit
    // pattern is valid for both; the borrow is exclusive.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u32>(), values.len()) }
}

/// The exit the flow path from core cell `start` leaves through, or `NONE`
/// when it ends in the tile (a code 0, an inactive cell, a cycle).
fn trace(
    g: &Geom,
    dir: &[u8],
    state: &mut [u8],
    labels: &mut [u32],
    path: &mut Vec<usize>,
    summary: &TileSummary,
    start: usize,
) -> u32 {
    path.clear();
    let mut c = start;
    let terminal = loop {
        match state[c] {
            TRACED => break labels[c],
            TRACING => break NONE,
            _ => {}
        }
        state[c] = TRACING;
        path.push(c);
        let k = dir[c] as usize;
        if k == 0 {
            break NONE;
        }
        let j = g.step(c, k);
        if state[j] == RING {
            break if dir[j] <= 8 {
                summary.exit_index(g.local(c)).expect("an exit")
            } else {
                NONE
            };
        }
        if dir[j] > 8 {
            break NONE;
        }
        c = j;
    };
    for &p in path.iter() {
        state[p] = TRACED;
        labels[p] = terminal;
    }
    terminal
}

fn check_buffer(g: &Geom, dir: &[u8]) -> Result<()> {
    if dir.len() != g.cells() {
        return Err(Error::Invalid(format!(
            "{} direction codes for a halo buffer of {} cells",
            dir.len(),
            g.cells()
        )));
    }
    Ok(())
}

/// Reads tiles of D8 codes for the passes that need them again.
pub trait DirectionSource: Sync {
    /// Fills `buffer` with tile `id`'s codes and a 1-cell halo
    /// (`halo`'s buffer), `DIRECTION_NODATA` outside the raster.
    fn read_halo(&self, id: TileId, halo: &HaloWindow, buffer: &mut Vec<u8>) -> Result<()>;
}

/// D8 codes held in one row-major grid.
#[derive(Clone, Copy, Debug)]
pub struct GridDirections<'a> {
    width: usize,
    height: usize,
    data: &'a [u8],
}

impl<'a> GridDirections<'a> {
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Result<Self> {
        if width.checked_mul(height) != Some(data.len()) {
            return Err(Error::Invalid(format!(
                "{} codes for a {width} x {height} grid",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

impl DirectionSource for GridDirections<'_> {
    fn read_halo(&self, _id: TileId, halo: &HaloWindow, buffer: &mut Vec<u8>) -> Result<()> {
        let window = halo.window;
        if window.right() > self.width || window.bottom() > self.height {
            return Err(Error::Invalid(format!("{window:?} is outside the grid")));
        }
        let bw = halo.buffer_width();
        buffer.clear();
        buffer.resize(halo.buffer_cells(), DIRECTION_NODATA);
        for row in 0..window.height {
            let src = (window.y + row) * self.width + window.x;
            let dst = (row + halo.pad_top) * bw + halo.pad_left;
            buffer[dst..dst + window.width].copy_from_slice(&self.data[src..src + window.width]);
        }
        Ok(())
    }
}

/// D8 codes in a [`TileStore`] (each tile's core) with their perimeters in a
/// [`BorderTable`]: how the fused pipeline hands D8's tiles to AreaD8
/// without reading the `p` raster.
pub struct StoredDirections<'a> {
    store: &'a TileStore<u8>,
    borders: &'a BorderTable<u8>,
    consume: bool,
}

impl<'a> StoredDirections<'a> {
    /// `consume` removes each tile from the store once read (its last use).
    pub fn new(store: &'a TileStore<u8>, borders: &'a BorderTable<u8>, consume: bool) -> Self {
        Self {
            store,
            borders,
            consume,
        }
    }
}

impl DirectionSource for StoredDirections<'_> {
    fn read_halo(&self, id: TileId, halo: &HaloWindow, buffer: &mut Vec<u8>) -> Result<()> {
        self.store.get_into(id, buffer)?;
        let (w, h) = (halo.core.width, halo.core.height);
        if buffer.len() != w * h {
            return Err(Error::Invalid(format!(
                "stored tile {id:?} has {} codes for {w} x {h}",
                buffer.len()
            )));
        }
        let bw = halo.buffer_width();
        buffer.resize(halo.buffer_cells(), DIRECTION_NODATA);
        // Spread the rows into the halo layout, last row first: each row's
        // destination lies past every earlier row's source.
        for ly in (0..h).rev() {
            buffer.copy_within(ly * w..ly * w + w, (ly + 1) * bw + 1);
        }
        self.borders
            .fill_halo_ring(id, halo, buffer, DIRECTION_NODATA)?;
        if self.consume {
            self.store.remove(id);
        }
        Ok(())
    }
}

/// One step of a replay program: `init` plus its operands, in order.
#[derive(Clone, Copy, Debug)]
struct Step {
    init: f32,
    end: u32,
}

#[derive(Clone, Copy, Debug)]
enum Operand {
    /// An in-tile cell whose count (at most 2^24) is its value.
    Const(f32),
    /// An earlier step's value.
    Slot(u32),
    /// A solved exit of the perimeter graph.
    External(u32),
}

/// The `f32` expressions of the cells above 2^24 draining to one exit,
/// upstream first; the last step is the exit.
#[derive(Clone, Debug, Default)]
struct Program {
    steps: Vec<Step>,
    ops: Vec<Operand>,
}

impl Program {
    fn run(&self, solved: &[f32], slots: &mut Vec<f32>) -> f32 {
        slots.clear();
        let mut start = 0;
        for step in &self.steps {
            let mut sum = step.init;
            for op in &self.ops[start..step.end as usize] {
                sum += match *op {
                    Operand::Const(v) => v,
                    Operand::Slot(s) => slots[s as usize],
                    Operand::External(n) => solved[n as usize],
                };
            }
            slots.push(sum);
            start = step.end as usize;
        }
        *slots.last().expect("a program has a step")
    }
}

/// Counters of the perimeter solve.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphStats {
    /// Exits (graph nodes) and entries over all tiles.
    pub nodes: u64,
    pub entries: u64,
    /// Nodes whose path continues to another exit.
    pub edges: u64,
    /// Nodes with an inflow from another tile.
    pub inflow_nodes: u64,
    /// Nodes nodata after the solve, and those on or below a D8 cycle.
    pub poisoned_nodes: u64,
    pub cycle_nodes: u64,
    /// The largest exact area at an exit.
    pub max_area: u64,
    /// Exits whose `f32` value was replayed, the tiles read again for them,
    /// and the programs' steps and operands.
    pub replay_nodes: u64,
    pub replay_tiles: u64,
    pub replay_steps: u64,
    pub replay_operands: u64,
    /// Peak bytes of the graph and summaries.
    pub bytes: u64,
    pub solve_seconds: f64,
    pub replay_seconds: f64,
}

/// Pass 1 results of every tile, before the perimeter solve.
pub struct AreaTiles {
    spec: TileSpec,
    options: AreaOptions,
    summaries: Vec<OnceLock<TileSummary>>,
}

impl AreaTiles {
    pub fn new(spec: TileSpec, options: AreaOptions) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            spec,
            options,
            summaries: (0..spec.tile_count()).map(|_| OnceLock::new()).collect(),
        })
    }

    pub fn spec(&self) -> &TileSpec {
        &self.spec
    }

    pub fn options(&self) -> &AreaOptions {
        &self.options
    }

    /// Pass 1 on tile `id` (its radius-1 halo buffer of codes). Callable
    /// from many threads, once per tile. Returns the cells accumulated.
    pub fn pass1(&self, id: TileId, dir: &[u8], scratch: &mut AreaScratch) -> Result<u64> {
        let halo = self.spec.halo_window(id);
        self.record(id, tile_area_pass1(&halo, dir, &self.options, scratch)?)
    }

    /// Records a summary computed by [`tile_area_pass1`].
    pub fn record(&self, id: TileId, summary: TileSummary) -> Result<u64> {
        let cells = summary.cells;
        self.summaries[self.spec.index(id)]
            .set(summary)
            .map_err(|_| Error::Invalid(format!("tile {id:?} has two Pass 1 summaries")))?;
        Ok(cells)
    }

    /// The perimeter solve. `source` gives the codes of the few tiles whose
    /// exits pass 2^24 cells (only read under [`Arithmetic::TauDem`]), at
    /// most `workers` at a time.
    pub fn solve(self, source: &dyn DirectionSource, workers: usize) -> Result<SolvedArea> {
        let mut buffers: Vec<Worker> = (0..workers.max(1)).map(|_| Worker::default()).collect();
        self.solve_with(source, &mut buffers)
    }

    fn solve_with(
        self,
        source: &dyn DirectionSource,
        buffers: &mut [Worker],
    ) -> Result<SolvedArea> {
        let start = Instant::now();
        let spec = self.spec;
        let options = self.options;
        let mut summaries = Vec::with_capacity(self.summaries.len());
        for (index, cell) in self.summaries.into_iter().enumerate() {
            summaries.push(cell.into_inner().ok_or_else(|| {
                Error::Invalid(format!(
                    "tile {:?} has no Pass 1 summary",
                    spec.id_at(index)
                ))
            })?);
        }
        let mut offsets = Vec::with_capacity(summaries.len() + 1);
        offsets.push(0u32);
        let mut total = 0u64;
        for s in &summaries {
            total += s.exits.len() as u64;
            offsets.push(
                u32::try_from(total)
                    .map_err(|_| Error::Invalid("more than 2^32 perimeter exits".into()))?,
            );
        }
        let n = total as usize;
        let mut stats = GraphStats {
            nodes: total,
            entries: summaries.iter().map(|s| s.entries.len() as u64).sum(),
            ..GraphStats::default()
        };
        let graph = Graph::build(&spec, &summaries, &offsets)?;
        stats.bytes = summaries
            .iter()
            .map(|s| (s.exits.capacity() * size_of::<Exit>() + s.entries.capacity() * 8) as u64)
            .sum::<u64>()
            + graph.bytes();
        drop(summaries);
        let Graph {
            local,
            value: local_value,
            count: mut exact,
            target,
            preds,
        } = graph;

        // Kahn's algorithm over exact counts and the nodata state.
        let mut poisoned: Vec<bool> = local_value.iter().map(|&v| v == SCALAR_NODATA).collect();
        let mut indegree = preds.clone();
        let mut order = Vec::with_capacity(n);
        let mut stack: Vec<u32> = (0..n as u32)
            .filter(|&i| indegree[i as usize] == 0)
            .collect();
        while let Some(i) = stack.pop() {
            order.push(i);
            let m = target[i as usize];
            if m == NONE {
                continue;
            }
            let (i, m) = (i as usize, m as usize);
            if poisoned[i] {
                poisoned[m] = true;
            } else if !poisoned[m] {
                exact[m] += exact[i];
            }
            indegree[m] -= 1;
            if indegree[m] == 0 {
                stack.push(m as u32);
            }
        }
        drop(indegree);
        if order.len() < n {
            let mut done = vec![false; n];
            for &i in &order {
                done[i as usize] = true;
            }
            for i in 0..n {
                if !done[i] {
                    poisoned[i] = true;
                    stats.cycle_nodes += 1;
                }
            }
        }
        stats.edges = target.iter().filter(|&&t| t != NONE).count() as u64;
        stats.inflow_nodes = preds.iter().filter(|&&p| p > 0).count() as u64;
        drop(target);
        stats.poisoned_nodes = poisoned.iter().filter(|&&p| p).count() as u64;
        stats.max_area = (0..n)
            .filter(|&i| !poisoned[i])
            .map(|i| exact[i])
            .max()
            .unwrap_or(0);

        let mut solved = SolvedArea {
            spec,
            options,
            offsets,
            local,
            value: Vec::new(),
            exact: Vec::new(),
            stats,
        };
        if options.arithmetic == Arithmetic::Exact {
            solved.exact = (0..n)
                .map(|i| if poisoned[i] { u64::MAX } else { exact[i] })
                .collect();
            solved.stats.solve_seconds = start.elapsed().as_secs_f64();
            return Ok(solved);
        }

        // Exits past the replay threshold that an inflow reaches.
        let replay: Vec<u32> = (0..n as u32)
            .filter(|&i| {
                let i = i as usize;
                !poisoned[i] && preds[i] > 0 && exact[i] > options.replay_above
            })
            .collect();
        let programs = if replay.is_empty() {
            HashMap::new()
        } else {
            let replay_start = Instant::now();
            // The exact u64 graph values, nodata where poisoned, for the
            // inflows injected into the tiles read again.
            solved.exact = (0..n)
                .map(|i| if poisoned[i] { u64::MAX } else { exact[i] })
                .collect();
            let (programs, tiles) = solved.programs(&replay, source, buffers)?;
            for buffer in buffers.iter_mut() {
                // Pass 2 sums f32 only.
                buffer.scratch.counts = Vec::new();
            }
            solved.exact = Vec::new();
            solved.stats.replay_tiles = tiles;
            solved.stats.replay_seconds = replay_start.elapsed().as_secs_f64();
            programs
        };
        solved.stats.replay_nodes = replay.len() as u64;
        solved.stats.replay_steps = programs.values().map(|p| p.steps.len() as u64).sum();
        solved.stats.replay_operands = programs.values().map(|p| p.ops.len() as u64).sum();

        // f32 values in topological order.
        let mut value = vec![SCALAR_NODATA; n];
        let mut slots = Vec::new();
        for &i in &order {
            let i = i as usize;
            value[i] = if poisoned[i] {
                SCALAR_NODATA
            } else if preds[i] == 0 {
                local_value[i]
            } else if exact[i] <= options.replay_above {
                exact[i] as f32
            } else {
                programs[&(i as u32)].run(&value, &mut slots)
            };
        }
        solved.value = value;
        solved.stats.solve_seconds = start.elapsed().as_secs_f64();
        Ok(solved)
    }
}

/// The perimeter graph: one node per exit, numbered tile by tile.
struct Graph {
    local: Vec<u32>,
    value: Vec<f32>,
    count: Vec<u64>,
    target: Vec<u32>,
    preds: Vec<u32>,
}

impl Graph {
    fn build(spec: &TileSpec, summaries: &[TileSummary], offsets: &[u32]) -> Result<Self> {
        let n = *offsets.last().expect("offsets") as usize;
        let mut local = Vec::with_capacity(n);
        let mut value = Vec::with_capacity(n);
        let mut count = Vec::with_capacity(n);
        for s in summaries {
            for e in &s.exits {
                local.push(e.local);
                value.push(e.value);
                count.push(e.count);
            }
        }
        let targets: Vec<Vec<u32>> = summaries
            .par_iter()
            .enumerate()
            .map(|(t, s)| {
                let id = spec.id_at(t);
                let rect = spec.rect(id);
                s.exits
                    .iter()
                    .map(|e| {
                        let k = e.k as usize;
                        let x = rect.x + e.local as usize % rect.width;
                        let y = rect.y + e.local as usize / rect.width;
                        let x2 = x.wrapping_add_signed(DX[k]);
                        let y2 = y.wrapping_add_signed(DY[k]);
                        let (t2, l2) = spec.locate(x2, y2);
                        let down = &summaries[spec.index(t2)];
                        let entry = down
                            .entries
                            .binary_search_by_key(&l2, |e| e.local)
                            .map_err(|_| {
                                Error::Invalid(format!(
                                    "cell ({x2}, {y2}) receives flow from tile {id:?} but is no entry of tile {t2:?}"
                                ))
                            })?;
                        let terminal = down.entries[entry].terminal;
                        Ok(if terminal == NONE {
                            NONE
                        } else {
                            offsets[spec.index(t2)] + terminal
                        })
                    })
                    .collect::<Result<Vec<u32>>>()
            })
            .collect::<Result<_>>()?;
        let target: Vec<u32> = targets.into_iter().flatten().collect();
        let mut preds = vec![0u32; n];
        for &t in &target {
            if t != NONE {
                preds[t as usize] += 1;
            }
        }
        Ok(Self {
            local,
            value,
            count,
            target,
            preds,
        })
    }

    fn bytes(&self) -> u64 {
        // Plus the solve's exact, poisoned, indegree and order arrays.
        (self.local.capacity() * (4 + 4 + 8 + 4 + 4 + 8 + 1 + 4 + 4)) as u64
    }
}

/// The solved perimeter: every exit's final value, for Pass 2.
pub struct SolvedArea {
    spec: TileSpec,
    options: AreaOptions,
    offsets: Vec<u32>,
    local: Vec<u32>,
    /// Final `f32` value of each exit ([`Arithmetic::TauDem`]).
    value: Vec<f32>,
    /// Exact count of each exit, `u64::MAX` when nodata ([`Arithmetic::Exact`],
    /// and during the replay).
    exact: Vec<u64>,
    stats: GraphStats,
}

impl SolvedArea {
    pub fn stats(&self) -> &GraphStats {
        &self.stats
    }

    pub fn spec(&self) -> &TileSpec {
        &self.spec
    }

    /// Bytes held for Pass 2.
    pub fn bytes(&self) -> u64 {
        (self.offsets.capacity() * 4
            + self.local.capacity() * 4
            + self.value.capacity() * 4
            + self.exact.capacity() * 8) as u64
    }

    /// The node of the exit at raster cell `(x, y)`.
    pub fn node_at(&self, x: usize, y: usize) -> Option<u32> {
        let (id, local) = self.spec.locate(x, y);
        let t = self.spec.index(id);
        let (a, b) = (self.offsets[t] as usize, self.offsets[t + 1] as usize);
        self.local[a..b]
            .binary_search(&local)
            .ok()
            .map(|i| (a + i) as u32)
    }

    /// Sets the halo cells of `vals` that drain into the core to their
    /// exit's value (`value(node)`), and the others to zero.
    fn fill_ring<V: Value>(
        &self,
        halo: &HaloWindow,
        g: &Geom,
        dir: &[u8],
        vals: &mut [V],
        value: impl Fn(usize) -> V,
    ) -> Result<()> {
        let mut missing = None;
        g.for_ring(|bx, by| {
            let j = by * g.bw + bx;
            vals[j] = V::ZERO;
            let k = dir[j] as usize;
            if !(1..=8).contains(&k) {
                return;
            }
            let tx = bx as isize + DX[k];
            let ty = by as isize + DY[k];
            if tx < 1 || ty < 1 || tx > g.w as isize || ty > g.h as isize {
                return;
            }
            if dir[g.step(j, k)] > 8 {
                return;
            }
            // A code inside the raster: the halo cell is a raster cell.
            let x = (halo.core.x + bx).wrapping_sub(1);
            let y = (halo.core.y + by).wrapping_sub(1);
            match self.node_at(x, y) {
                Some(node) => vals[j] = value(node as usize),
                None => missing = Some((x, y)),
            }
        });
        match missing {
            Some((x, y)) => Err(Error::Invalid(format!(
                "cell ({x}, {y}) flows into tile at {:?} but is no exit: the codes changed between passes",
                halo.core
            ))),
            None => Ok(()),
        }
    }

    /// Reads the tiles holding `replay` exits again and builds their programs.
    fn programs(
        &self,
        replay: &[u32],
        source: &dyn DirectionSource,
        buffers: &mut [Worker],
    ) -> Result<(HashMap<u32, Program>, u64)> {
        let mut by_tile: Vec<(usize, Vec<u32>)> = Vec::new();
        for &node in replay {
            let t = self.offsets.partition_point(|&o| o <= node) - 1;
            match by_tile.last_mut() {
                Some((last, nodes)) if *last == t => nodes.push(node),
                _ => by_tile.push((t, vec![node])),
            }
        }
        let programs = Mutex::new(HashMap::new());
        run_tiles(&by_tile, buffers, |worker, (t, nodes)| {
            let built = self.tile_programs(*t, nodes, source, worker)?;
            programs
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend(built);
            Ok(0)
        })?;
        let programs = programs.into_inner().unwrap_or_else(|p| p.into_inner());
        Ok((programs, by_tile.len() as u64))
    }

    fn tile_programs(
        &self,
        t: usize,
        nodes: &[u32],
        source: &dyn DirectionSource,
        worker: &mut Worker,
    ) -> Result<Vec<(u32, Program)>> {
        let id = self.spec.id_at(t);
        let halo = self.spec.halo_window(id);
        let g = Geom::new(&halo)?;
        let Worker { dir, scratch } = worker;
        source.read_halo(id, &halo, dir)?;
        check_buffer(&g, dir)?;
        scratch.prepare(g.cells(), true);
        let (state, counts) = (&mut scratch.state, &mut scratch.counts);
        self.fill_ring(&halo, &g, dir, counts, |n| self.exact[n])?;
        accumulate(&g, dir, state, counts, self.options.check_contamination);
        let limit = self.options.replay_above;
        let rect = halo.core;
        let mut out = Vec::with_capacity(nodes.len());
        for &node in nodes {
            let local = self.local[node as usize] as usize;
            let q = g.at(local % g.w, local / g.w);
            if counts[q] != self.exact[node as usize] {
                return Err(Error::Invalid(format!(
                    "tile {id:?} exit {local}: replayed count {} differs from the solve's {}",
                    counts[q], self.exact[node as usize]
                )));
            }
            // Post-order over the in-tree of cells above the limit.
            let mut order = Vec::new();
            let mut slot = HashMap::new();
            let mut stack = vec![(q, 1usize)];
            while let Some(top) = stack.last_mut() {
                let (c, k) = *top;
                if k > 8 {
                    slot.insert(c, order.len() as u32);
                    order.push(c);
                    stack.pop();
                    continue;
                }
                top.1 += 1;
                let j = g.step(c, k);
                if dir[j] == OPPOSITE[k] && state[j] != RING && counts[j] > limit {
                    stack.push((j, 1));
                }
            }
            let mut program = Program::default();
            for &c in &order {
                let mut init = 1.0f32;
                let mut leading = true;
                for k in 1..9 {
                    let j = g.step(c, k);
                    if dir[j] != OPPOSITE[k] {
                        continue;
                    }
                    let op = if state[j] == RING {
                        let bx = j % g.bw;
                        let by = j / g.bw;
                        let x = (rect.x + bx).wrapping_sub(1);
                        let y = (rect.y + by).wrapping_sub(1);
                        Operand::External(self.node_at(x, y).ok_or_else(|| {
                            Error::Invalid(format!("halo cell ({x}, {y}) is no exit"))
                        })?)
                    } else if counts[j] > limit {
                        Operand::Slot(slot[&j])
                    } else {
                        // Counts at most 2^24 are exact in f32.
                        Operand::Const(counts[j] as f32)
                    };
                    match op {
                        Operand::Const(v) if leading => init += v,
                        _ => {
                            leading = false;
                            program.ops.push(op);
                        }
                    }
                }
                program.steps.push(Step {
                    init,
                    end: program.ops.len() as u32,
                });
            }
            out.push((node, program));
        }
        Ok(out)
    }

    /// Pass 2 on tile `id` (its radius-1 halo buffer of codes): the final
    /// areas of the tile's cells, row-major, `SCALAR_NODATA` for nodata.
    pub fn pass2<'s>(
        &self,
        id: TileId,
        dir: &[u8],
        scratch: &'s mut AreaScratch,
    ) -> Result<(&'s [f32], u64)> {
        let halo = self.spec.halo_window(id);
        let g = Geom::new(&halo)?;
        check_buffer(&g, dir)?;
        let check = self.options.check_contamination;
        let cells;
        if self.options.arithmetic == Arithmetic::Exact {
            scratch.prepare(g.cells(), true);
            self.fill_ring(&halo, &g, dir, &mut scratch.counts, |n| self.exact[n])?;
            cells = accumulate(&g, dir, &mut scratch.state, &mut scratch.counts, check);
            for ly in 0..g.h {
                for lx in 0..g.w {
                    let c = scratch.counts[g.at(lx, ly)];
                    scratch.area[ly * g.w + lx] = if c == u64::MAX {
                        SCALAR_NODATA
                    } else {
                        c as f32
                    };
                }
            }
        } else {
            scratch.prepare(g.cells(), false);
            self.fill_ring(&halo, &g, dir, &mut scratch.area, |n| self.value[n])?;
            cells = accumulate(&g, dir, &mut scratch.state, &mut scratch.area, check);
            // Compact the core rows to the front, in place.
            for ly in 0..g.h {
                let from = g.at(0, ly);
                scratch.area.copy_within(from..from + g.w, ly * g.w);
            }
        }
        Ok((&scratch.area[..g.w * g.h], cells))
    }
}

/// Options of [`area_d8_tiled`].
#[derive(Clone, Debug)]
pub struct TiledAreaOptions {
    pub area: AreaOptions,
    /// Tiles processed at once (each with its own buffers); 0 for the
    /// current Rayon pool's thread count.
    pub workers: usize,
    /// When set, Pass 1 keeps each tile's codes in this cache (compressed,
    /// spilling under its budget) for the later passes instead of reading
    /// `source` again.
    pub cache: Option<TileCache>,
}

impl TiledAreaOptions {
    pub fn new(area: AreaOptions) -> Self {
        Self {
            area,
            workers: 0,
            cache: None,
        }
    }
}

/// Counters of [`area_d8_tiled`].
#[derive(Clone, Debug, Default)]
pub struct TiledAreaStats {
    pub tiles: usize,
    pub tile_size: (usize, usize),
    pub workers: usize,
    pub graph: GraphStats,
    /// Cells accumulated by Pass 1 and by Pass 2.
    pub pass1_cells: u64,
    pub pass2_cells: u64,
    pub pass1_seconds: f64,
    pub solve_seconds: f64,
    pub pass2_seconds: f64,
    /// Bytes of one worker's buffers, after the run.
    pub worker_bytes: u64,
    pub border_bytes: u64,
    /// The code cache, when one was used.
    pub store: Option<TileStats>,
}

impl TiledAreaStats {
    /// Records the counters for `--profile-json`.
    pub fn record_profile(&self) {
        use profile::count;
        count("aread8.tiles", self.tiles as u64);
        count("aread8.workers", self.workers as u64);
        count("aread8.worker_bytes", self.worker_bytes);
        count("aread8.border_bytes", self.border_bytes);
        count("aread8.cells_accumulated", self.pass2_cells);
        count("aread8.pass1_cells", self.pass1_cells);
        count("aread8.graph.nodes", self.graph.nodes);
        count("aread8.graph.entries", self.graph.entries);
        count("aread8.graph.edges", self.graph.edges);
        count("aread8.graph.inflow_nodes", self.graph.inflow_nodes);
        count("aread8.graph.poisoned_nodes", self.graph.poisoned_nodes);
        count("aread8.graph.cycle_nodes", self.graph.cycle_nodes);
        count("aread8.graph.max_area", self.graph.max_area);
        count("aread8.graph.bytes", self.graph.bytes);
        count("aread8.replay.nodes", self.graph.replay_nodes);
        count("aread8.replay.tiles", self.graph.replay_tiles);
        count("aread8.replay.steps", self.graph.replay_steps);
        count("aread8.replay.operands", self.graph.replay_operands);
        if let Some(store) = &self.store {
            store.record_profile();
        }
    }
}

/// A worker's buffers.
#[derive(Default)]
struct Worker {
    dir: Vec<u8>,
    scratch: AreaScratch,
}

impl Worker {
    fn bytes(&self) -> u64 {
        (self.dir.capacity() + self.scratch.bytes()) as u64
    }
}

/// Runs `visit` on every item, one thread per worker's buffers, items
/// taken in order. The threads are not Rayon's: a Rayon worker waiting on
/// a GeoTIFF read (which terrio parallelizes) runs other queued jobs, so
/// it would start other tiles while holding a tile's buffers. Returns the
/// sum of `visit`'s counts, or the error of the earliest failed item.
fn run_tiles<T: Sync>(
    items: &[T],
    workers: &mut [Worker],
    visit: impl Fn(&mut Worker, &T) -> Result<u64> + Sync,
) -> Result<u64> {
    let next = AtomicUsize::new(0);
    let total = AtomicU64::new(0);
    let failure: Mutex<Option<(usize, Error)>> = Mutex::new(None);
    let failed = AtomicBool::new(false);
    let threads = workers.len().min(items.len());
    std::thread::scope(|scope| {
        for worker in workers.iter_mut().take(threads) {
            let (next, total, failure, failed, visit) = (&next, &total, &failure, &failed, &visit);
            let spawned = std::thread::Builder::new()
                .name("taudem-aread8".into())
                .stack_size(8 << 20)
                .spawn_scoped(scope, move || {
                    while !failed.load(Relaxed) {
                        let i = next.fetch_add(1, Relaxed);
                        let Some(item) = items.get(i) else { break };
                        match visit(worker, item) {
                            Ok(count) => {
                                total.fetch_add(count, Relaxed);
                            }
                            Err(error) => {
                                failed.store(true, Relaxed);
                                let mut slot = failure.lock().unwrap_or_else(|p| p.into_inner());
                                if slot.as_ref().is_none_or(|(j, _)| i < *j) {
                                    *slot = Some((i, error));
                                }
                            }
                        }
                    }
                });
            if let Err(error) = spawned {
                failed.store(true, Relaxed);
                let mut slot = failure.lock().unwrap_or_else(|p| p.into_inner());
                slot.get_or_insert((usize::MAX, Error::Io(error)));
                break;
            }
        }
    });
    match failure.into_inner().unwrap_or_else(|p| p.into_inner()) {
        Some((_, error)) => Err(error),
        None => Ok(total.into_inner()),
    }
}

/// Receives a tile's final areas: its core rectangle, row-major.
pub type AreaSink<'a> = dyn Fn(TileId, Rect, &[f32]) -> Result<()> + Sync + 'a;

/// Tiled AreaD8 of the codes `source` gives: Pass 1, the perimeter solve
/// and Pass 2, with `sink` receiving each tile's final areas (its core
/// rectangle, row-major) from the worker threads in any order.
pub fn area_d8_tiled(
    spec: &TileSpec,
    source: &dyn DirectionSource,
    options: &TiledAreaOptions,
    sink: &AreaSink<'_>,
) -> Result<TiledAreaStats> {
    let workers = if options.workers == 0 {
        rayon::current_num_threads()
    } else {
        options.workers
    };
    let ids: Vec<TileId> = spec.row_major().collect();
    let mut buffers: Vec<Worker> = (0..workers.min(ids.len()).max(1))
        .map(|_| Worker::default())
        .collect();
    let tiles = AreaTiles::new(*spec, options.area)?;
    let cached = options.cache.as_ref().map(|cache| {
        (
            cache.store::<u8>(
                "aread8.p",
                StoreOptions {
                    prefilter: None,
                    initial: Tier::Compressed,
                },
            ),
            BorderTable::new(*spec, DIRECTION_NODATA),
        )
    });

    let start = Instant::now();
    let pass1 = profile::span("aread8.pass1");
    let pass1_cells = run_tiles(&ids, &mut buffers, |worker, &id| {
        let halo = spec.halo_window(id);
        source.read_halo(id, &halo, &mut worker.dir)?;
        let cells = tiles.pass1(id, &worker.dir, &mut worker.scratch)?;
        if let Some((store, borders)) = &cached {
            borders.set_from_halo(id, &halo, &worker.dir)?;
            let mut core = Vec::new();
            halo.core_from_buffer(&worker.dir, &mut core)?;
            store.put_owned(id, core)?;
        }
        Ok(cells)
    })?;
    drop(pass1);
    let pass1_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let solve = profile::span("aread8.solve");
    let stored = cached
        .as_ref()
        .map(|(store, borders)| StoredDirections::new(store, borders, false));
    let reread: &dyn DirectionSource = match &stored {
        Some(s) => s,
        None => source,
    };
    let solved = tiles.solve_with(reread, &mut buffers)?;
    drop(solve);
    let solve_seconds = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let pass2 = profile::span("aread8.pass2");
    let consumed = cached
        .as_ref()
        .map(|(store, borders)| StoredDirections::new(store, borders, true));
    let last: &dyn DirectionSource = match &consumed {
        Some(s) => s,
        None => source,
    };
    let pass2_cells = run_tiles(&ids, &mut buffers, |worker, &id| {
        let Worker { dir, scratch } = worker;
        let halo = spec.halo_window(id);
        last.read_halo(id, &halo, dir)?;
        let (area, cells) = solved.pass2(id, dir, scratch)?;
        sink(id, halo.core, area)?;
        Ok(cells)
    })?;
    drop(pass2);
    let pass2_seconds = start.elapsed().as_secs_f64();

    Ok(TiledAreaStats {
        tiles: ids.len(),
        tile_size: spec.tile_size(),
        workers: buffers.len(),
        graph: solved.stats.clone(),
        pass1_cells,
        pass2_cells,
        pass1_seconds,
        solve_seconds,
        pass2_seconds,
        worker_bytes: buffers.iter().map(Worker::bytes).max().unwrap_or(0),
        border_bytes: cached.as_ref().map_or(0, |(_, b)| b.bytes() as u64),
        store: cached.as_ref().map(|(store, _)| store.cache().stats()),
    })
}

/// How a tiled AreaD8 run divides its memory budget ([`plan`]).
#[derive(Clone, Debug)]
pub struct AreaPlan {
    pub spec: TileSpec,
    /// Tiles processed at once.
    pub workers: usize,
    pub budget: u64,
    /// One worker's buffers, including the source's read buffers.
    pub worker_bytes: u64,
    /// Estimated peak of the perimeter graph and the tiles' summaries.
    pub graph_bytes: u64,
    /// The output writer's queue and partly written blocks.
    pub writer_bytes: u64,
    /// What is left for the tile cache of codes (it spills beyond this).
    pub cache_bytes: u64,
}

/// Perimeter cells over all tiles, the bound of the graph's size.
fn perimeter_cells(spec: &TileSpec) -> u64 {
    let (tw, th) = spec.tile_size();
    spec.tile_count() as u64 * 2 * (tw as u64 + th as u64)
}

/// Graph and summary bytes per perimeter cell. An exit costs about 80 B
/// over the summaries (24 B), the graph (24 B) and the solve (~32 B), an
/// entry 8 B; on the 1e8 synthetic a quarter of the perimeter cells are
/// exits, about 20 B per perimeter cell.
const GRAPH_BYTES_PER_PERIMETER_CELL: u64 = 32;

/// Held back from the budget for the process itself: its image, the
/// allocator, the reader's block tables and the threads' stacks.
pub const BASELINE_BYTES: u64 = 24 << 20;

/// Plans a tiled AreaD8 run of a `width x height` raster on `threads`
/// threads within `execution`'s memory budget (less [`BASELINE_BYTES`]).
/// An explicit `--tile-size` is used as given. Otherwise the automatic size
/// is halved (down to 256) until the workers' buffers fit half of what the
/// perimeter graph and the writer leave; when no size fits, the one needing
/// least memory with a single worker is used. `source_bytes_per_cell` is
/// what the source's reads add per halo cell (2 for `i16` GeoTIFF reads).
/// Workers are then limited to that half, and the rest goes to the cache.
pub fn plan(
    width: usize,
    height: usize,
    execution: &crate::tiling::ExecutionOptions,
    native_block: Option<(usize, usize)>,
    threads: usize,
    options: &AreaOptions,
    source_bytes_per_cell: u64,
) -> Result<AreaPlan> {
    const MIN_CACHE: u64 = 8 << 20;
    const OUTPUT_BLOCK: u64 = 256 * 256 * 4;
    let threads = threads.max(1);
    let budget = execution.memory_budget();
    let usable = budget.saturating_sub(BASELINE_BYTES);
    let cells = width as u64 * height as u64;
    let estimate = |spec: &TileSpec| {
        let (tw, th) = spec.tile_size();
        let halo_cells = (tw as u64 + 2) * (th as u64 + 2);
        // A raster past 2^24 cells may need the f32 replay, whose workers
        // hold exact counts (8 B per cell) on top of their usual buffers.
        let replay = if cells > F32_EXACT_LIMIT && options.arithmetic == Arithmetic::TauDem {
            8 * halo_cells
        } else {
            0
        };
        let worker =
            worker_bytes(tw, th, options) + halo_cells * (source_bytes_per_cell + 1) + replay;
        let graph = perimeter_cells(spec) * GRAPH_BYTES_PER_PERIMETER_CELL;
        // Blocks straddling tile seams wait for their neighbors.
        let partial = if spec.is_aligned_to(256, 256) {
            0
        } else {
            2 * (width as u64).div_ceil(256) * threads as u64 * OUTPUT_BLOCK
        };
        let writer = 2 * threads as u64 * OUTPUT_BLOCK + partial;
        (worker, graph, writer)
    };
    let spec = match execution.tile_size {
        Some(size) => TileSpec::new(width, height, size, size)?,
        None => {
            let mut target = crate::tiling::auto_tile_size(cells, budget);
            let mut smallest: Option<(u64, TileSpec)> = None;
            loop {
                let spec = TileSpec::aligned(width, height, target, native_block)?;
                let (worker, graph, writer) = estimate(&spec);
                let room = usable.saturating_sub(graph + writer) / 2;
                if threads as u64 * worker <= room {
                    break spec;
                }
                let need = worker + graph + writer;
                if smallest.is_none_or(|(least, _)| need < least) {
                    smallest = Some((need, spec));
                }
                if target <= 256 {
                    break smallest.expect("a candidate").1;
                }
                target /= 2;
            }
        }
    };
    let (worker, graph, writer) = estimate(&spec);
    let room = usable.saturating_sub(graph + writer) / 2;
    let workers = ((room / worker.max(1)) as usize).clamp(1, threads);
    let cache_bytes = usable
        .saturating_sub(graph + writer + workers as u64 * worker)
        .max(MIN_CACHE);
    Ok(AreaPlan {
        spec,
        workers,
        budget,
        worker_bytes: worker,
        graph_bytes: graph,
        writer_bytes: writer,
        cache_bytes,
    })
}

/// Bytes the in-memory `area_d8` needs: codes, state and areas.
pub fn in_memory_bytes(width: usize, height: usize) -> u64 {
    width as u64 * height as u64 * (1 + 1 + 4)
}
