// SPDX-License-Identifier: GPL-3.0-only
//! Improved Priority-Flood, after Barnes, Lehman & Mulla (2014), run on tiles
//! in parallel and joined through a perimeter spill graph, after Barnes (2016),
//! "Parallel Priority-Flood depression filling for trillion cell digital
//! elevation models", doi:10.1016/j.cageo.2016.07.001.
//!
//! Filling to the lowest spill elevation has a unique result: every cell ends
//! at the smallest, over all paths to an outlet, of the highest elevation on
//! the path. The tiled and whole-grid floods therefore agree on every value.
//! Only the sign of a filled zero can differ, because -0.0 and 0.0 compare
//! equal; [`fix_zero_signs`] restores the whole-grid flood's signs.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::{DX, DY, FEL_NODATA, Grid, missing, profile};

/// Fills depressions in place to their lowest spill elevation. Raster edges,
/// cells adjacent to nodata, and cells with mask value 1 are fixed outlets.
pub fn pit_remove(
    grid: Grid,
    elevations: &mut [f32],
    nodata: f32,
    four_way: bool,
    mask: Option<&[u8]>,
) {
    let threads = rayon::current_num_threads();
    // One worker gains nothing from tiles but pays for the stitching.
    let tile = if grid.cells < 1 << 21 || threads == 1 {
        0
    } else {
        // Several tiles per thread balance tiles that are mostly nodata, and
        // a 512-cell tile's scratch (1.5 MB) keeps every thread's working set
        // in a shared L3 cache. Smaller tiles only enlarge the spill graph.
        let target = grid.cells / (8 * threads);
        (target as f64).sqrt().clamp(256.0, DEFAULT_TILE as f64) as usize
    };
    pit_remove_with_tiles(grid, elevations, nodata, four_way, mask, tile);
}

/// The largest tile side. A tile's perimeter labels must fit in 16 bits.
pub const MAX_TILE: usize = 4096;
const DEFAULT_TILE: usize = 512;
/// Tile scratch labels: cells outside the tile or nodata, and cells not yet
/// reached. Perimeter labels stay below both.
const NONE: u16 = u16::MAX;
const UNSET: u16 = u16::MAX - 1;

/// [`pit_remove`] with an explicit square tile side; 0, or a side at least
/// as large as the grid, floods the whole grid on one thread.
pub fn pit_remove_with_tiles(
    grid: Grid,
    elevations: &mut [f32],
    nodata: f32,
    four_way: bool,
    mask: Option<&[u8]>,
    tile: usize,
) {
    assert_eq!(elevations.len(), grid.cells);
    assert!(mask.is_none_or(|mask| mask.len() == grid.cells));
    assert!(tile <= MAX_TILE);
    let step = if four_way { 2 } else { 1 };
    if tile == 0 || (tile >= grid.width && tile >= grid.height) {
        flood_serial(grid, elevations, nodata, step, mask);
        return;
    }
    let prepass = profile::span("pitremove.prepass");
    let (void, negative_zero) = void_cells(elevations, nodata);
    profile::buffer("pitremove.void", void.len() * 8);
    // Without -0.0 in the input every filled zero is +0.0 in either flood.
    let zeros = negative_zero.then(|| ZeroInputs::new(elevations, &void));
    drop(prepass);
    let layout = Layout::new(grid, tile);
    profile::count("pitremove.tiles", layout.count() as u64);

    let tiles = profile::span("pitremove.tiles");
    let mut labels = vec![0u16; grid.cells];
    profile::buffer("pitremove.labels", labels.len() * 2);
    let graphs: Vec<TileGraph> = layout
        .views(elevations, &mut labels)
        .into_par_iter()
        .map_init(Scratch::default, |scratch, mut view| {
            scratch.flood(grid, &mut view, &void, mask, step)
        })
        .collect();
    drop(tiles);

    let stitch = profile::span("pitremove.stitch");
    let mut base = Vec::with_capacity(graphs.len() + 1);
    let mut nodes = 1usize;
    for graph in &graphs {
        base.push(nodes);
        nodes += graph.labels as usize;
    }
    let node = |t: usize, label: u16| {
        if label == 0 {
            0
        } else {
            (base[t] + label as usize - 1) as u32
        }
    };
    let elevations_ref = &*elevations;
    let labels_ref = &labels;
    let cross: Vec<(u32, u32, f32)> = (0..layout.count())
        .into_par_iter()
        .flat_map_iter(|t| {
            let mut edges = Vec::new();
            layout.perimeter(t, |r, c| {
                let i = r * grid.width + c;
                if bit(&void, i) {
                    return;
                }
                let from = node(t, labels_ref[i]);
                for k in (1..9).step_by(step) {
                    let (Some(qr), Some(qc)) = (
                        r.checked_add_signed(DY[k]).filter(|&v| v < grid.height),
                        c.checked_add_signed(DX[k]).filter(|&v| v < grid.width),
                    ) else {
                        continue;
                    };
                    let u = layout.tile_of(qr, qc);
                    let q = qr * grid.width + qc;
                    // Each pair of tiles contributes its seam edges once.
                    if u <= t || bit(&void, q) {
                        continue;
                    }
                    let to = node(u, labels_ref[q]);
                    if from != to {
                        // Perimeter cells are seeds, so they keep their elevation.
                        let (a, b) = (elevations_ref[i], elevations_ref[q]);
                        edges.push((from, to, if b > a { b } else { a }));
                    }
                }
            });
            edges
        })
        .collect();
    let mut edges: Vec<(u32, u32, f32)> = cross;
    for (t, graph) in graphs.iter().enumerate() {
        edges.extend(
            graph
                .edges
                .iter()
                .map(|&(a, b, w)| (node(t, a), node(t, b), w)),
        );
    }
    drop(graphs);
    profile::count("pitremove.graph_nodes", nodes as u64);
    profile::count("pitremove.graph_edges", edges.len() as u64);
    let spill = minimax(nodes, edges);
    drop(stitch);

    let finish = profile::span("pitremove.finish");
    elevations
        .par_chunks_mut(grid.width)
        .zip(labels.par_chunks(grid.width))
        .enumerate()
        .for_each(|(row, (z, labels))| {
            let ty = row / layout.tile_h;
            for (c, (z, &label)) in z.iter_mut().zip(labels).enumerate() {
                if label != 0 {
                    let level = spill[node(ty * layout.across + c / layout.tile_w, label) as usize];
                    if level > *z {
                        *z = level;
                    }
                }
            }
        });
    drop(labels);
    drop(finish);
    if let Some(zeros) = zeros {
        let _span = profile::span("pitremove.zero_signs");
        fix_zero_signs(grid, elevations, &void, mask, step, &zeros);
    }
}

/// Which valid input cells were zero, and which of those -0.0.
struct ZeroInputs {
    zero: Vec<u64>,
    negative: Vec<u64>,
}

impl ZeroInputs {
    fn new(elevations: &[f32], void: &[u64]) -> Self {
        let words = void.len();
        let mut zero = vec![0u64; words];
        let mut negative = vec![0u64; words];
        profile::buffer("pitremove.zero_inputs", words * 16);
        zero.par_iter_mut()
            .zip(negative.par_iter_mut())
            .zip(elevations.par_chunks(64).zip(void))
            .for_each(|((zero, negative), (chunk, &void))| {
                for (b, &z) in chunk.iter().enumerate() {
                    if void >> b & 1 == 0 && z == 0.0 {
                        *zero |= 1 << b;
                        if z.is_sign_negative() {
                            *negative |= 1 << b;
                        }
                    }
                }
            });
        Self { zero, negative }
    }
}

/// Gives every filled zero the sign the whole-grid flood writes. That flood
/// pops -0.0 before 0.0 and pushes no zero-valued entry while at level zero,
/// so the outcome does not depend on visit order:
/// * a zero cell that is an outlet, or was zero and has a neighbor filled
///   below zero, entered the queue with its own value and keeps its sign;
/// * every other zero cell was reached at level zero and takes the sign of
///   the first such source reaching it: -0.0 if a -0.0 source connects to it
///   through other such cells, otherwise 0.0.
fn fix_zero_signs(
    grid: Grid,
    elevations: &mut [f32],
    void: &[u64],
    mask: Option<&[u8]>,
    step: usize,
    inputs: &ZeroInputs,
) {
    let level: Vec<u32> = elevations
        .par_chunks(BLOCK)
        .enumerate()
        .flat_map_iter(|(b, chunk)| {
            chunk.iter().enumerate().filter_map(move |(o, &z)| {
                let i = b * BLOCK + o;
                (z == 0.0 && !bit(void, i)).then_some(i as u32)
            })
        })
        .collect();
    profile::count("pitremove.zero_level_cells", level.len() as u64);
    if level.is_empty() {
        return;
    }
    const PIT: u8 = 0;
    const NEGATIVE: u8 = 1;
    const POSITIVE: u8 = 2;
    let z = &*elevations;
    // Sources carry their input sign; the rest start unassigned.
    let mut sign: Vec<u8> = level
        .par_iter()
        .map(|&cell| {
            let i = cell as usize;
            let col = i % grid.width;
            let mut outlet = !grid.interior(i, col) || mask.is_some_and(|m| m[i] == 1);
            let mut below = false;
            grid.for_neighbors(i, step, |_, j| {
                outlet |= bit(void, j);
                below |= !bit(void, j) && z[j] < 0.0;
            });
            if outlet || (bit(&inputs.zero, i) && below) {
                if bit(&inputs.negative, i) {
                    NEGATIVE
                } else {
                    POSITIVE
                }
            } else {
                PIT
            }
        })
        .collect();
    let mut queue = VecDeque::new();
    for wave in [NEGATIVE, POSITIVE] {
        queue.extend((0..level.len()).filter(|&n| sign[n] == wave));
        while let Some(n) = queue.pop_front() {
            grid.for_neighbors(level[n] as usize, step, |_, j| {
                if let Ok(m) = level.binary_search(&(j as u32))
                    && sign[m] == PIT
                {
                    sign[m] = wave;
                    queue.push_back(m);
                }
            });
        }
    }
    for (&cell, &s) in level.iter().zip(&sign) {
        if s != PIT {
            elevations[cell as usize] = if s == NEGATIVE { -0.0 } else { 0.0 };
        }
    }
}

const BLOCK: usize = 1 << 16;

/// Marks missing cells, 64 to a word, and reports whether any valid cell is -0.0.
fn void_cells(elevations: &[f32], nodata: f32) -> (Vec<u64>, bool) {
    let mut void = vec![0u64; elevations.len().div_ceil(64)];
    let negative_zero = void
        .par_iter_mut()
        .zip(elevations.par_chunks(64))
        .map(|(word, chunk)| {
            let mut negative_zero = false;
            for (b, &z) in chunk.iter().enumerate() {
                if missing(z, nodata) {
                    *word |= 1 << b;
                } else {
                    negative_zero |= z.to_bits() == 0x8000_0000;
                }
            }
            negative_zero
        })
        .reduce(|| false, |a, b| a | b);
    (void, negative_zero)
}

#[inline]
fn bit(bits: &[u64], i: usize) -> bool {
    bits[i / 64] >> (i % 64) & 1 != 0
}

/// Square tiles covering the grid in row-major tile order; the last tile in
/// each row and column may be smaller.
struct Layout {
    grid: Grid,
    tile_w: usize,
    tile_h: usize,
    across: usize,
    down: usize,
}

/// One tile's rows of the elevation and label grids.
struct TileView<'a> {
    row: usize,
    col: usize,
    z: Vec<&'a mut [f32]>,
    labels: Vec<&'a mut [u16]>,
}

impl Layout {
    fn new(grid: Grid, tile: usize) -> Self {
        let tile_w = tile.min(grid.width);
        let tile_h = tile.min(grid.height);
        Self {
            grid,
            tile_w,
            tile_h,
            across: grid.width.div_ceil(tile_w),
            down: grid.height.div_ceil(tile_h),
        }
    }

    fn count(&self) -> usize {
        self.across * self.down
    }

    fn tile_of(&self, row: usize, col: usize) -> usize {
        row / self.tile_h * self.across + col / self.tile_w
    }

    /// Splits both grids into disjoint per-tile row segments.
    fn views<'a>(&self, z: &'a mut [f32], labels: &'a mut [u16]) -> Vec<TileView<'a>> {
        let width = self.grid.width;
        let mut views: Vec<TileView<'a>> = (0..self.count())
            .map(|t| TileView {
                row: t / self.across * self.tile_h,
                col: t % self.across * self.tile_w,
                z: Vec::new(),
                labels: Vec::new(),
            })
            .collect();
        for (r, (z, labels)) in z
            .chunks_mut(width)
            .zip(labels.chunks_mut(width))
            .enumerate()
        {
            let first = r / self.tile_h * self.across;
            for (tx, (z, labels)) in z
                .chunks_mut(self.tile_w)
                .zip(labels.chunks_mut(self.tile_w))
                .enumerate()
            {
                views[first + tx].z.push(z);
                views[first + tx].labels.push(labels);
            }
        }
        views
    }

    /// Visits each perimeter cell of tile `t` once, as (row, col).
    fn perimeter(&self, t: usize, mut visit: impl FnMut(usize, usize)) {
        let r0 = t / self.across * self.tile_h;
        let c0 = t % self.across * self.tile_w;
        let r1 = (r0 + self.tile_h).min(self.grid.height) - 1;
        let c1 = (c0 + self.tile_w).min(self.grid.width) - 1;
        for c in c0..=c1 {
            visit(r0, c);
            if r1 > r0 {
                visit(r1, c);
            }
        }
        for r in r0 + 1..r1 {
            visit(r, c0);
            if c1 > c0 {
                visit(r, c1);
            }
        }
    }
}

/// A tile's spill graph: perimeter labels 1..=`labels`, label 0 for the
/// outlets, and a minimum spanning forest of the label adjacencies weighted
/// by spill elevation.
struct TileGraph {
    labels: u16,
    edges: Vec<(u16, u16, f32)>,
}

/// Per-thread buffers for flooding one tile with a one-cell halo.
#[derive(Default)]
struct Scratch {
    z: Vec<f32>,
    labels: Vec<u16>,
    void: Vec<u64>,
    heap: FloodQueue,
    pit: VecDeque<u32>,
    spills: SpillTable,
}

impl Scratch {
    /// Floods one tile from its outlets and perimeter cells. Each cell takes
    /// the label of the seed that reaches it; where two labels meet, the
    /// higher of the two filled cells is a spill elevation between them.
    fn flood(
        &mut self,
        grid: Grid,
        view: &mut TileView<'_>,
        void: &[u64],
        mask: Option<&[u8]>,
        step: usize,
    ) -> TileGraph {
        let th = view.z.len();
        let tw = view.z[0].len();
        let (lw, lh) = (tw + 2, th + 2);
        let local = Grid::new(lw, lh).expect("tile fits the cell index");
        let cells = lw * lh;
        self.z.clear();
        self.z.resize(cells, FEL_NODATA);
        self.labels.clear();
        self.labels.resize(cells, UNSET);
        self.void.clear();
        self.void.resize(cells.div_ceil(64), 0);
        self.heap.reset();
        self.pit.clear();
        self.spills.clear();

        let t0 = std::time::Instant::now();
        // The halo holds no cells of this tile. Outside the raster it counts
        // as nodata, so edge cells become outlets as in the whole-grid flood.
        for lr in 0..lh {
            let gr = (view.row + lr).checked_sub(1).filter(|&r| r < grid.height);
            for lc in 0..lw {
                let li = lr * lw + lc;
                let gc = (view.col + lc).checked_sub(1).filter(|&c| c < grid.width);
                let is_void = match (gr, gc) {
                    (Some(r), Some(c)) => bit(void, r * grid.width + c),
                    _ => true,
                };
                if is_void {
                    set(&mut self.void, li);
                }
                let halo = lr == 0 || lc == 0 || lr == lh - 1 || lc == lw - 1;
                if halo || is_void {
                    self.labels[li] = NONE;
                } else {
                    self.z[li] = view.z[lr - 1][lc - 1];
                }
            }
        }

        let t1 = std::time::Instant::now();
        // Cells next to nodata or the raster edge drain freely (label 0).
        for w in 0..self.void.len() {
            let mut word = self.void[w];
            while word != 0 {
                let v = w * 64 + word.trailing_zeros() as usize;
                word &= word - 1;
                local.for_neighbors(v, step, |_, n| {
                    if self.labels[n] == UNSET {
                        self.labels[n] = 0;
                        self.heap.push(priority(self.z[n], n));
                    }
                });
            }
        }
        let fixed = |lr: usize, lc: usize| {
            mask.is_some_and(|m| m[(view.row + lr - 1) * grid.width + view.col + lc - 1] == 1)
        };
        // The remaining perimeter cells stand in for the outside of the tile.
        let mut next = 1u16;
        let mut perimeter = |lr: usize, lc: usize, this: &mut Self| {
            let li = lr * lw + lc;
            if this.labels[li] == UNSET {
                this.labels[li] = if fixed(lr, lc) {
                    0
                } else {
                    next += 1;
                    next - 1
                };
                this.heap.push(priority(this.z[li], li));
            }
        };
        for lc in 1..=tw {
            perimeter(1, lc, self);
            perimeter(th, lc, self);
        }
        for lr in 2..th {
            perimeter(lr, 1, self);
            perimeter(lr, tw, self);
        }
        if mask.is_some() {
            for lr in 2..th {
                for lc in 2..tw {
                    let li = lr * lw + lc;
                    if fixed(lr, lc) && self.labels[li] == UNSET {
                        self.labels[li] = 0;
                        self.heap.push(priority(self.z[li], li));
                    }
                }
            }
        }

        let t2 = std::time::Instant::now();
        let mut inserts = 0u64;
        let Self {
            z,
            labels,
            heap,
            pit,
            spills,
            ..
        } = self;
        while let Some(cell) = pit.pop_front().or_else(|| heap.pop()) {
            let i = cell as usize;
            let height = z[i];
            let label = labels[i];
            local.for_neighbors(i, step, |_, j| {
                let other = labels[j];
                if other == UNSET {
                    labels[j] = label;
                    if z[j] <= height {
                        z[j] = height;
                        pit.push_back(j as u32);
                    } else {
                        heap.push(priority(z[j], j));
                    }
                } else if other != label && other != NONE {
                    inserts += 1;
                    spills.insert(label, other, if z[j] > height { z[j] } else { height });
                }
            });
        }

        let t3 = std::time::Instant::now();
        for (lr, (z_row, label_row)) in view.z.iter_mut().zip(&mut view.labels).enumerate() {
            let start = (lr + 1) * lw + 1;
            z_row.copy_from_slice(&z[start..start + tw]);
            for (out, &label) in label_row.iter_mut().zip(&labels[start..start + tw]) {
                *out = if label == NONE { 0 } else { label };
            }
        }
        let t4 = std::time::Instant::now();
        let edges = spills.spanning_forest(next);
        let t5 = std::time::Instant::now();
        let us = |a: std::time::Instant, b: std::time::Instant| (b - a).as_micros() as u64;
        profile::count("pitremove.tile.load_us", us(t0, t1));
        profile::count("pitremove.tile.seed_us", us(t1, t2));
        profile::count("pitremove.tile.flood_us", us(t2, t3));
        profile::count("pitremove.tile.writeback_us", us(t3, t4));
        profile::count("pitremove.tile.forest_us", us(t4, t5));
        profile::count("pitremove.tile.spill_checks", inserts);
        profile::count("pitremove.tile.spill_pairs", spills.len as u64);
        TileGraph {
            labels: next - 1,
            edges,
        }
    }
}

#[inline]
fn set(bits: &mut [u64], i: usize) {
    bits[i / 64] |= 1 << (i % 64);
}

/// Lowest spill elevation between each pair of labels in a tile.
#[derive(Default)]
struct SpillTable {
    keys: Vec<u32>,
    weights: Vec<f32>,
    len: usize,
    last: (u32, f32),
}

impl SpillTable {
    const EMPTY: u32 = u32::MAX;

    fn clear(&mut self) {
        if self.keys.is_empty() {
            self.keys = vec![Self::EMPTY; 1024];
            self.weights = vec![0.0; 1024];
        } else {
            self.keys.fill(Self::EMPTY);
        }
        self.len = 0;
        self.last = (Self::EMPTY, 0.0);
    }

    #[inline]
    fn insert(&mut self, a: u16, b: u16, weight: f32) {
        let key = if a < b {
            u32::from(a) << 16 | u32::from(b)
        } else {
            u32::from(b) << 16 | u32::from(a)
        };
        // Region boundaries repeat the same pair many times in a row.
        if self.last.0 == key && weight >= self.last.1 {
            return;
        }
        let mask = self.keys.len() - 1;
        let mut slot = (key.wrapping_mul(0x9e37_79b1) >> 7) as usize & mask;
        loop {
            match self.keys[slot] {
                k if k == key => {
                    if weight < self.weights[slot] {
                        self.weights[slot] = weight;
                    }
                    self.last = (key, self.weights[slot]);
                    return;
                }
                Self::EMPTY => break,
                _ => slot = (slot + 1) & mask,
            }
        }
        self.keys[slot] = key;
        self.weights[slot] = weight;
        self.last = (key, weight);
        self.len += 1;
        if self.len * 2 > self.keys.len() {
            self.grow();
        }
    }

    #[cold]
    fn grow(&mut self) {
        let size = self.keys.len() * 2;
        let keys = std::mem::replace(&mut self.keys, vec![Self::EMPTY; size]);
        let weights = std::mem::replace(&mut self.weights, vec![0.0; size]);
        self.len = 0;
        self.last = (Self::EMPTY, 0.0);
        for (key, weight) in keys.into_iter().zip(weights) {
            if key != Self::EMPTY {
                self.insert((key >> 16) as u16, key as u16, weight);
            }
        }
    }

    /// A minimum spanning forest over labels `0..labels`. Lowest spill
    /// elevations between any two labels are unchanged by dropping the rest.
    fn spanning_forest(&self, labels: u16) -> Vec<(u16, u16, f32)> {
        let mut edges: Vec<(f32, u16, u16)> = self
            .keys
            .iter()
            .zip(&self.weights)
            .filter(|&(&k, _)| k != Self::EMPTY)
            .map(|(&k, &w)| (w, (k >> 16) as u16, k as u16))
            .collect();
        edges.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut parent: Vec<u16> = (0..labels).collect();
        let find = |parent: &mut Vec<u16>, mut x: u16| {
            while parent[x as usize] != x {
                let up = parent[parent[x as usize] as usize];
                parent[x as usize] = up;
                x = up;
            }
            x
        };
        let mut forest = Vec::new();
        for (w, a, b) in edges {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra as usize] = rb;
                forest.push((a, b, w));
            }
        }
        forest
    }
}

/// The lowest elevation at which water from each node can reach node 0, the
/// outlet: the minimum over paths of the highest edge on the path. Edges are
/// joined in rising order (Kruskal); a component's level is the weight of the
/// edge that first joins it to the outlet's component.
pub(crate) fn minimax(nodes: usize, mut edges: Vec<(u32, u32, f32)>) -> Vec<f32> {
    edges.par_sort_unstable_by_key(|&(_, _, w)| sort_key(w));
    let mut parent: Vec<u32> = (0..nodes as u32).collect();
    // Members of each component as a linked list from its root.
    let mut next = vec![u32::MAX; nodes];
    let mut tail: Vec<u32> = (0..nodes as u32).collect();
    let mut size = vec![1u32; nodes];
    let mut level = vec![f32::NEG_INFINITY; nodes];
    let find = |parent: &mut [u32], mut x: u32| {
        while parent[x as usize] != x {
            let up = parent[parent[x as usize] as usize];
            parent[x as usize] = up;
            x = up;
        }
        x
    };
    let outlet_root = |parent: &mut [u32]| find(parent, 0);
    for (a, b, w) in edges {
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra == rb {
            continue;
        }
        let outlet = outlet_root(&mut parent);
        if ra == outlet || rb == outlet {
            let mut n = if ra == outlet { rb } else { ra };
            while n != u32::MAX {
                level[n as usize] = w;
                n = next[n as usize];
            }
        }
        let (big, small) = if size[ra as usize] >= size[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[small as usize] = big;
        size[big as usize] += size[small as usize];
        next[tail[big as usize] as usize] = small;
        tail[big as usize] = tail[small as usize];
    }
    // Node 0 and any node without a path to an outlet are never raised.
    level[0] = f32::NEG_INFINITY;
    level
}

/// The whole-grid Improved Priority-Flood on one thread.
fn flood_serial(grid: Grid, elevations: &mut [f32], nodata: f32, step: usize, mask: Option<&[u8]>) {
    let seeding = profile::span("pitremove.seed");
    let mut visited = vec![0u64; grid.cells.div_ceil(64)];
    profile::buffer("pitremove.visited", visited.len() * 8);
    let mut heap = FloodQueue::default();
    let mut pit = VecDeque::new();

    // Mark holes first, seeding their valid neighbors without inspecting all
    // eight neighbors of every land cell.
    for i in 0..grid.cells {
        if missing(elevations[i], nodata) {
            elevations[i] = FEL_NODATA;
            mark(&mut visited, i);
            grid.for_neighbors(i, step, |_, j| {
                if !missing(elevations[j], nodata) {
                    seed(elevations, &mut visited, &mut heap, j);
                }
            });
        }
    }
    for col in 0..grid.width {
        seed(elevations, &mut visited, &mut heap, col);
        seed(
            elevations,
            &mut visited,
            &mut heap,
            grid.cells - grid.width + col,
        );
    }
    for row in 1..grid.height.saturating_sub(1) {
        seed(elevations, &mut visited, &mut heap, row * grid.width);
        seed(
            elevations,
            &mut visited,
            &mut heap,
            (row + 1) * grid.width - 1,
        );
    }
    if let Some(mask) = mask {
        for (i, &value) in mask.iter().enumerate() {
            if value == 1 {
                seed(elevations, &mut visited, &mut heap, i);
            }
        }
    }
    drop(seeding);

    let _flooding = profile::span("pitremove.flood");
    while let Some(cell) = pit.pop_front().or_else(|| heap.pop()) {
        let i = cell as usize;
        let height = elevations[i];
        grid.for_neighbors(i, step, |_, j| {
            if mark(&mut visited, j) {
                if elevations[j] <= height {
                    elevations[j] = height;
                    pit.push_back(j as u32);
                } else {
                    heap.push(priority(elevations[j], j));
                }
            }
        });
    }
}

#[inline]
fn mark(visited: &mut [u64], i: usize) -> bool {
    let bit = 1 << (i % 64);
    let word = &mut visited[i / 64];
    let fresh = *word & bit == 0;
    *word |= bit;
    fresh
}

#[inline]
fn seed(elevations: &[f32], visited: &mut [u64], heap: &mut FloodQueue, i: usize) {
    if mark(visited, i) {
        heap.push(priority(elevations[i], i));
    }
}

/// An unsigned key with the order of the float elevation.
#[inline]
fn sort_key(height: f32) -> u32 {
    let bits = height.to_bits();
    bits ^ (((bits as i32 >> 31) as u32) | 0x8000_0000)
}

#[inline]
pub(crate) fn priority(height: f32, i: usize) -> u64 {
    (u64::from(sort_key(height)) << 32) | i as u64
}

/// Monotone radix heap. New priority elevations cannot be lower than the
/// last one removed, so only the populated bucket nearest that key needs
/// redistribution. Each entry packs a float sort key and a cell index.
pub(crate) struct FloodQueue {
    bounded: bool,
    buckets: [Vec<u64>; 33],
    used: u64,
    last: u32,
}

impl Default for FloodQueue {
    fn default() -> Self {
        Self {
            bounded: false,
            buckets: std::array::from_fn(|_| Vec::new()),
            used: 0,
            last: 0,
        }
    }
}

impl FloodQueue {
    /// The streaming worker can drop drained allocations to keep the total
    /// capacity proportional to the active frontier.
    pub(crate) fn bounded() -> Self {
        Self {
            bounded: true,
            ..Self::default()
        }
    }

    pub(crate) fn reset(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.used = 0;
        self.last = 0;
    }

    #[inline]
    pub(crate) fn push(&mut self, entry: u64) {
        let key = (entry >> 32) as u32;
        debug_assert!(key >= self.last);
        let bucket = 32 - (key ^ self.last).leading_zeros();
        self.buckets[bucket as usize].push(entry);
        self.used |= 1 << bucket;
    }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<u32> {
        if self.used & 1 == 0 {
            if self.used == 0 {
                return None;
            }
            self.refill();
        }
        let entry = self.buckets[0].pop()?;
        if self.buckets[0].is_empty() {
            self.used &= !1;
            if self.bounded && self.buckets[0].capacity() > 1024 {
                self.buckets[0] = Vec::new();
            }
        }
        Some(entry as u32)
    }

    #[cold]
    fn refill(&mut self) {
        let bucket = self.used.trailing_zeros() as usize;
        self.used &= !(1 << bucket);
        let mut entries = std::mem::take(&mut self.buckets[bucket]);
        self.last = (entries.iter().min().expect("nonempty bucket") >> 32) as u32;
        for entry in entries.drain(..) {
            self.push(entry);
        }
        if !self.bounded || entries.capacity() <= 1024 {
            self.buckets[bucket] = entries;
        }
    }
}
