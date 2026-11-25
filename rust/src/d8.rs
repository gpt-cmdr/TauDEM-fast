// SPDX-License-Identifier: GPL-3.0-only
//! D8 direction codes and the TauDEM implementation of Garbrecht & Martz.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

use rayon::iter::Either;
use rayon::prelude::*;

use crate::{DIRECTION_NODATA, Grid, ORDER, SCALAR_NODATA, missing, profile};

#[derive(Debug)]
pub struct D8Result {
    pub direction: Vec<u8>,
    pub slope: Option<Vec<f32>>,
}

/// Computes D8 flow and, optionally, the slope on the original elevation surface.
/// Spacing is a single (width, height) pair or one pair per raster row.
pub fn d8_flow(
    grid: Grid,
    elevations: &[f32],
    nodata: f32,
    spacing: &[(f64, f64)],
    with_slope: bool,
) -> D8Result {
    let steep = d8_steep(grid, elevations, nodata, spacing);
    let slope = with_slope.then(|| steep.slope(elevations));
    let direction = steep.resolve_flats(Surface::Borrowed(elevations));
    D8Result { direction, slope }
}

/// The elevation surface handed to flat resolution. TauDEM routes the second
/// and later flat passes over an artificial surface; an owned buffer is
/// overwritten with it instead of allocating another grid.
pub enum Surface<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

/// D8 directions before flat routing: 1–8 for a downslope neighbor, 0 for an
/// unresolved flat and 255 for nodata or the raster edge.
pub struct SteepFlow {
    grid: Grid,
    factors: Vec<[f64; 9]>,
    direction: Vec<u8>,
    flats: usize,
}

/// Assigns steepest-descent directions with TauDEM's crossing rule.
pub fn d8_steep(grid: Grid, elevations: &[f32], nodata: f32, spacing: &[(f64, f64)]) -> SteepFlow {
    assert_eq!(elevations.len(), grid.cells);
    assert!(spacing.len() == 1 || spacing.len() == grid.height);
    let factors: Vec<_> = spacing
        .iter()
        .map(|&(dx, dy)| {
            assert!(dx.is_finite() && dy.is_finite() && dx > 0.0 && dy > 0.0);
            let diag = 1.0 / (dx * dx + dy * dy).sqrt();
            [
                0.0,
                1.0 / dx,
                diag,
                1.0 / dy,
                diag,
                1.0 / dx,
                diag,
                1.0 / dy,
                diag,
            ]
        })
        .collect();
    let row_factors = |row| &factors[if factors.len() == 1 { 0 } else { row }];
    let steep = profile::span("d8.steep");
    let mut direction = vec![DIRECTION_NODATA; grid.cells];
    profile::buffer("d8.direction", direction.len());
    direction
        .par_chunks_mut(grid.width)
        .enumerate()
        .for_each(|(row, output)| {
            if row == 0 || row + 1 == grid.height || grid.width < 3 {
                return;
            }
            let scale = row_factors(row);
            // Slice three rows once. The windows give the compiler fixed-size
            // stencils and keep bounds checks out of individual neighbor accesses.
            let top = &elevations[(row - 1) * grid.width..row * grid.width];
            let middle = &elevations[row * grid.width..(row + 1) * grid.width];
            let bottom = &elevations[(row + 1) * grid.width..(row + 2) * grid.width];
            for ((([nw, n, ne], [w, center, e]), [sw, s, se]), cell) in top
                .array_windows::<3>()
                .zip(middle.array_windows::<3>())
                .zip(bottom.array_windows::<3>())
                .zip(&mut output[1..])
            {
                let heights = [*center, *e, *ne, *n, *nw, *w, *sw, *s, *se];
                if heights.iter().any(|&z| missing(z, nodata)) {
                    continue;
                }
                *cell = steepest(&heights, scale, |_| true);
            }
        });
    drop(steep);

    let crossing = profile::span("d8.crossing_and_flat_scan");
    fix_crossings(grid, elevations, &mut direction, &row_factors);
    let flats = direction
        .par_chunks(BLOCK)
        .map(|chunk| chunk.iter().filter(|&&d| d == 0).count())
        .sum();
    profile::count("d8.flat_cells", flats as u64);
    drop(crossing);
    SteepFlow {
        grid,
        factors,
        direction,
        flats,
    }
}

/// Cells per parallel work item in whole-grid scans.
const BLOCK: usize = 1 << 16;

/// TauDEM tests a diagonal for crossing against the directions it has already
/// assigned in row-major order, that is the final north and west neighbors.
/// The unrestricted winner only needs reevaluation when it crosses, which is
/// rare, so candidates are found in parallel and only they, and cells whose
/// north or west neighbor then changes, are revisited in index order.
fn fix_crossings<'a>(
    grid: Grid,
    elevations: &[f32],
    direction: &mut [u8],
    row_factors: &(impl Fn(usize) -> &'a [f64; 9] + Sync),
) {
    let initial: &[u8] = direction;
    let mut pending: BinaryHeap<Reverse<u32>> = initial
        .par_chunks(BLOCK)
        .enumerate()
        .flat_map_iter(|(b, chunk)| {
            chunk.iter().enumerate().filter_map(move |(o, &d)| {
                let i = b * BLOCK + o;
                (matches!(d, 2 | 4 | 6) && crosses(grid, initial, i, d as usize, i))
                    .then_some(Reverse(i as u32))
            })
        })
        .collect::<Vec<_>>()
        .into();
    let mut last = None;
    let mut fixed = 0u64;
    while let Some(Reverse(cell)) = pending.pop() {
        if last == Some(cell) {
            continue;
        }
        last = Some(cell);
        let i = cell as usize;
        // A cell is visited after its north and west neighbors are final and
        // before any change to itself, so `direction[i]` is still the winner.
        let k = direction[i] as usize;
        if matches!(k, 2 | 4 | 6 | 8) && crosses(grid, direction, i, k, i) {
            let heights = std::array::from_fn(|k| elevations[grid.offset(i, k)]);
            direction[i] = steepest(&heights, row_factors(i / grid.width), |k| {
                !crosses(grid, direction, i, k, i)
            });
            fixed += 1;
            for j in [i + 1, i + grid.width] {
                if j < grid.cells {
                    pending.push(Reverse(j as u32));
                }
            }
        }
    }
    profile::count("d8.crossing_fixes", fixed);
}

impl SteepFlow {
    pub fn grid(&self) -> Grid {
        self.grid
    }

    /// Directions before flat routing.
    pub fn direction(&self) -> &[u8] {
        &self.direction
    }

    /// Number of cells with no downslope neighbor.
    pub fn flat_cells(&self) -> usize {
        self.flats
    }

    fn row_factors(&self, row: usize) -> &[f64; 9] {
        &self.factors[if self.factors.len() == 1 { 0 } else { row }]
    }

    /// Writes sd8 for whole rows starting at `first_row`; `out` holds a whole
    /// number of rows. TauDEM writes sd8 before routing across artificial flat
    /// elevations, so flats have slope 0.
    pub fn slope_rows(&self, elevations: &[f32], first_row: usize, out: &mut [f32]) {
        let grid = self.grid;
        assert_eq!(elevations.len(), grid.cells);
        assert_eq!(out.len() % grid.width, 0);
        assert!(first_row * grid.width + out.len() <= grid.cells);
        out.par_chunks_mut(grid.width)
            .enumerate()
            .for_each(|(offset, output)| {
                let row = first_row + offset;
                let scale = self.row_factors(row);
                let start = row * grid.width;
                for (col, value) in output.iter_mut().enumerate() {
                    let i = start + col;
                    let k = self.direction[i] as usize;
                    *value = if k <= 8 {
                        scaled(elevations[i] - elevations[grid.offset(i, k)], scale[k])
                    } else {
                        SCALAR_NODATA
                    };
                }
            });
    }

    /// sd8 for the whole grid.
    pub fn slope(&self, elevations: &[f32]) -> Vec<f32> {
        let _span = profile::span("d8.slope");
        let mut slope = vec![SCALAR_NODATA; self.grid.cells];
        profile::buffer("d8.slope", slope.len() * 4);
        self.slope_rows(elevations, 0, &mut slope);
        slope
    }

    /// Routes flats with TauDEM's Garbrecht–Martz scheme and returns the
    /// final directions. `surface` must be the elevations given to
    /// [`d8_steep`]; an owned buffer is consumed as scratch.
    pub fn resolve_flats(self, surface: Surface<'_>) -> Vec<u8> {
        let SteepFlow {
            grid,
            factors,
            mut direction,
            flats,
        } = self;
        if flats > 0 {
            let _span = profile::span("d8.flats");
            let row_factors = |row| &factors[if factors.len() == 1 { 0 } else { row }];
            resolve_flats(grid, surface, &mut direction, flats, &row_factors);
        }
        direction
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

#[inline]
fn crosses(grid: Grid, dirs: &[u8], i: usize, k: usize, before: usize) -> bool {
    let (a, ac, b, bc) = match k {
        2 => (1, 4, 3, 8),
        4 => (3, 6, 5, 2),
        6 => (7, 4, 5, 8),
        8 => (1, 6, 7, 2),
        _ => return false,
    };
    let a = grid.offset(i, a);
    let b = grid.offset(i, b);
    (a < before && dirs[a] == ac) || (b < before && dirs[b] == bc)
}

/// Both TauDEM increment fields of a cell in one word, fall in the low half
/// and rise in the high half, at the narrowest width that holds them; a pass
/// that would overflow 16-bit halves is repeated with 32-bit halves. While the
/// fall search runs, the high half instead holds the cell's admissible fall
/// steps, so each search step reads a single array.
trait Packed: Sync {
    type Word: Sync + Send;
    const HALF: u32;
    /// The largest level a pass may assign, leaving room for the level of
    /// unresolved cells two above the highest draining one.
    const LIMIT: u64;
    fn zeroed(cells: usize) -> Vec<Self::Word>;
    fn load(word: &Self::Word) -> u64;
    fn store(word: &Self::Word, value: u64);

    #[inline]
    fn low(value: u64) -> u64 {
        value & ((1 << Self::HALF) - 1)
    }

    #[inline]
    fn high(value: u64) -> u64 {
        value >> Self::HALF
    }
}

struct Narrow;
struct Wide;

impl Packed for Narrow {
    type Word = AtomicU32;
    const HALF: u32 = 16;
    const LIMIT: u64 = u16::MAX as u64 - 2;

    fn zeroed(cells: usize) -> Vec<AtomicU32> {
        let mut zeros = std::mem::ManuallyDrop::new(vec![0u32; cells]);
        // SAFETY: AtomicU32 has the size, alignment and bit validity of u32,
        // so the allocation is reinterpreted with an identical layout.
        unsafe {
            Vec::from_raw_parts(
                zeros.as_mut_ptr().cast::<AtomicU32>(),
                zeros.len(),
                zeros.capacity(),
            )
        }
    }

    #[inline]
    fn load(word: &AtomicU32) -> u64 {
        u64::from(word.load(Relaxed))
    }

    #[inline]
    fn store(word: &AtomicU32, value: u64) {
        word.store(value as u32, Relaxed);
    }
}

impl Packed for Wide {
    type Word = AtomicU64;
    const HALF: u32 = 32;
    const LIMIT: u64 = u32::MAX as u64 - 2;

    fn zeroed(cells: usize) -> Vec<AtomicU64> {
        let mut zeros = std::mem::ManuallyDrop::new(vec![0u64; cells]);
        // SAFETY: AtomicU64 has the size, alignment and bit validity of u64.
        unsafe {
            Vec::from_raw_parts(
                zeros.as_mut_ptr().cast::<AtomicU64>(),
                zeros.len(),
                zeros.capacity(),
            )
        }
    }

    #[inline]
    fn load(word: &AtomicU64) -> u64 {
        word.load(Relaxed)
    }

    #[inline]
    fn store(word: &AtomicU64, value: u64) {
        word.store(value, Relaxed);
    }
}

struct Overflow;

/// The unresolved cells of one pass: either every cell whose direction is 0,
/// found by scanning, or an explicit list once few remain.
enum Flats {
    Scan,
    List(Vec<u32>),
}

impl Flats {
    fn indices<'a>(&'a self, dirs: &'a [u8]) -> impl ParallelIterator<Item = usize> + 'a {
        match self {
            Flats::Scan => Either::Left(dirs.par_chunks(BLOCK).enumerate().flat_map_iter(
                |(b, chunk)| {
                    chunk
                        .iter()
                        .enumerate()
                        .filter(|&(_, &d)| d == 0)
                        .map(move |(o, _)| b * BLOCK + o)
                },
            )),
            Flats::List(list) => Either::Right(list.par_iter().map(|&i| i as usize)),
        }
    }

    fn collect(dirs: &[u8]) -> Self {
        Flats::List(
            Flats::Scan
                .indices(dirs)
                .map(|i| i as u32)
                .collect::<Vec<_>>(),
        )
    }
}

/// Lists of the first two fall levels: cells next to a draining neighbor, and
/// cells that only see the default level of an equal non-flat neighbor.
type Seeds = (Vec<u32>, Vec<u32>);

/// Fewer cells than this in a frontier are expanded on the calling thread.
const SERIAL_FRONTIER: usize = 4096;
const FRONTIER_CHUNK: usize = 2048;

/// Level-synchronous breadth-first search. `first` holds level-1 cells and
/// `second` cells already at level 2; `expand(i, next, out)` assigns level
/// `next` to the unassigned cells one step from `i` and lists them. The
/// levels are breadth-first distances and so do not depend on the order in
/// which threads reach cells. Returns the highest level assigned.
///
/// Every thread that finds an unassigned cell stores the same level, so a
/// plain store suffices. Two threads racing on one cell both list it, and
/// expanding it twice changes nothing; a compare-and-swap would serialize
/// the memory accesses that dominate the search.
fn bfs<P: Packed>(
    first: Vec<u32>,
    second: Vec<u32>,
    expand: impl Fn(usize, u64, &mut Vec<u32>) + Sync,
    peak: &mut usize,
) -> Result<u64, Overflow> {
    let mut second = Some(second);
    let mut frontier = first;
    let mut level = 1u64;
    let mut last = 0;
    loop {
        if level == 2
            && let Some(mut late) = second.take()
        {
            frontier.append(&mut late);
        }
        if frontier.is_empty() {
            if level == 1 && second.as_ref().is_some_and(|late| !late.is_empty()) {
                level = 2;
                continue;
            }
            break;
        }
        last = level;
        let next = level + 1;
        if next > P::LIMIT {
            return Err(Overflow);
        }
        let expanded = if frontier.len() < SERIAL_FRONTIER {
            let mut out = Vec::new();
            for &i in &frontier {
                expand(i as usize, next, &mut out);
            }
            out
        } else {
            frontier
                .par_chunks(FRONTIER_CHUNK)
                .flat_map_iter(|chunk| {
                    let mut out = Vec::with_capacity(chunk.len());
                    for &i in chunk {
                        expand(i as usize, next, &mut out);
                    }
                    out
                })
                .collect()
        };
        *peak = (*peak).max(frontier.capacity() + expanded.capacity());
        frontier = expanded;
        level = next;
    }
    Ok(last)
}

struct PassResult {
    remaining: usize,
    last_rise: u64,
}

/// TauDEM's combined increment level `elev2` for one cell, including the
/// constant offset of the rise field.
#[inline]
fn surface(fall: u64, rise: u64, last_rise: u64) -> i64 {
    let level = fall.max(1) as i64;
    if rise > 0 {
        level + last_rise as i64 + 2 - rise as i64
    } else {
        level
    }
}

/// One TauDEM `resolveflats` call over the current surface. Directions are
/// only changed once both increment fields are complete, so an overflow of
/// the level width leaves `dirs` untouched.
#[allow(clippy::too_many_arguments)]
fn flat_pass<'a, P: Packed>(
    grid: Grid,
    pass: usize,
    elevations: &[f32],
    dirs: &mut [u8],
    flats: &Flats,
    words: &[P::Word],
    row_factors: &(impl Fn(usize) -> &'a [f64; 9] + Sync),
) -> Result<PassResult, Overflow> {
    let mut peak_frontier = 0;
    let fall_span = profile::span(format!("d8.flats.pass{pass}.fall"));
    let seed_span = profile::span(format!("d8.flats.pass{pass}.fall.seeds"));
    // incfall: BFS from draining flat boundaries across equal-height cells,
    // excluding edges blocked by an existing diagonal. At stage two, incfall
    // can also see the default level of a same-height neighbor outside the
    // unresolved set. The admissible steps are recorded per cell.
    let (first, second): Seeds = {
        let dirs = &*dirs;
        flats
            .indices(dirs)
            .fold(
                || (Vec::new(), Vec::new()),
                |(mut first, mut second), i| {
                    let z = elevations[i];
                    let (mut drains, mut default, mut steps) = (false, false, 0u64);
                    for k in 1..9 {
                        let j = grid.offset(i, k);
                        let (d, zj) = (dirs[j], elevations[j]);
                        let open = || !crosses(grid, dirs, i, k, grid.cells);
                        if d == 0 {
                            if z == zj && open() {
                                steps |= 1 << (k - 1);
                            }
                        } else if d <= 8 {
                            // An equal draining neighbor also meets stage two,
                            // but stage one takes precedence.
                            drains |= z >= zj && open();
                        } else {
                            default |= z == zj && open();
                        }
                    }
                    let level = if drains {
                        first.push(i as u32);
                        1
                    } else if default {
                        second.push(i as u32);
                        2
                    } else {
                        0
                    };
                    P::store(&words[i], steps << P::HALF | level);
                    (first, second)
                },
            )
            .reduce(
                || (Vec::new(), Vec::new()),
                |(mut a, mut b), (mut c, mut d)| {
                    a.append(&mut c);
                    b.append(&mut d);
                    (a, b)
                },
            )
    };
    profile::count("d8.flats.fall_seeds", (first.len() + second.len()) as u64);
    drop(seed_span);
    let bfs_span = profile::span(format!("d8.flats.pass{pass}.fall.bfs"));
    let last_fall = bfs::<P>(
        first,
        second,
        |i, next, out| {
            let mut steps = P::high(P::load(&words[i]));
            while steps != 0 {
                let k = steps.trailing_zeros() as usize + 1;
                steps &= steps - 1;
                let j = grid.offset(i, k);
                let word = P::load(&words[j]);
                if P::low(word) == 0 {
                    P::store(&words[j], word | next);
                    out.push(j as u32);
                }
            }
        },
        &mut peak_frontier,
    )?;
    drop(bfs_span);
    drop(fall_span);

    // incrise: BFS from higher terrain across the entire unresolved set. The
    // original scan increments each marked cell on every remaining pass,
    // including the final pass with no newly marked cells.
    let rise_span = profile::span(format!("d8.flats.pass{pass}.rise"));
    // Cells no drain reaches are pits. They take the level two above the
    // highest draining one, and are marked nodata before routing. Every
    // unresolved cell then has a nonzero fall level, which marks the set.
    let unresolved = last_fall.max(1) + 2;
    let seeds: Vec<u32> = {
        let dirs = &*dirs;
        flats
            .indices(dirs)
            .filter_map(|i| {
                let fall = match P::low(P::load(&words[i])) {
                    0 => unresolved,
                    fall => fall,
                };
                let higher = (1..9).any(|k| elevations[grid.offset(i, k)] > elevations[i]);
                P::store(&words[i], fall | u64::from(higher) << P::HALF);
                higher.then_some(i as u32)
            })
            .collect()
    };
    profile::count("d8.flats.rise_seeds", seeds.len() as u64);
    let last_rise = bfs::<P>(
        seeds,
        Vec::new(),
        |i, next, out| {
            for k in 1..9 {
                let j = grid.offset(i, k);
                let word = P::load(&words[j]);
                if P::low(word) != 0 && P::high(word) == 0 {
                    P::store(&words[j], word | next << P::HALF);
                    out.push(j as u32);
                }
            }
        },
        &mut peak_frontier,
    )?;
    drop(rise_span);
    profile::buffer("d8.flats.frontier_peak", peak_frontier * 4);

    let set_span = profile::span(format!("d8.flats.pass{pass}.directions"));
    let route = |i: usize| {
        let word = P::load(&words[i]);
        let here = surface(P::low(word), P::high(word), last_rise);
        let mut d = if P::low(word) == unresolved {
            DIRECTION_NODATA
        } else {
            0
        };
        let scale = row_factors(i / grid.width);
        let mut best = 0.0;
        for k in ORDER {
            let j = grid.offset(i, k);
            let other = P::load(&words[j]);
            if P::high(other) > 0 {
                let there = surface(P::low(other), P::high(other), last_rise);
                let slope = scaled((here - there) as f32, scale[k]);
                if slope > best {
                    best = slope;
                    d = k as u8;
                }
            } else if elevations[i] >= elevations[j] {
                d = k as u8;
                break;
            }
        }
        d
    };
    let remaining = match flats {
        // Every unresolved cell still has direction 0 here.
        Flats::Scan => dirs
            .par_chunks_mut(BLOCK)
            .enumerate()
            .map(|(b, chunk)| {
                let mut zeros = 0;
                for (o, d) in chunk.iter_mut().enumerate() {
                    if *d == 0 {
                        *d = route(b * BLOCK + o);
                        zeros += usize::from(*d == 0);
                    }
                }
                zeros
            })
            .sum(),
        Flats::List(list) => {
            let routed: Vec<u8> = list.par_iter().map(|&i| route(i as usize)).collect();
            let mut zeros = 0;
            for (&i, &d) in list.iter().zip(&routed) {
                dirs[i as usize] = d;
                zeros += usize::from(d == 0);
            }
            zeros
        }
    };
    drop(set_span);
    profile::count("d8.flats.max_fall", last_fall);
    profile::count("d8.flats.max_rise", last_rise);
    Ok(PassResult {
        remaining,
        last_rise,
    })
}

/// Builds the artificial surface of a finished pass for the next pass: the
/// combined increment level of each flat cell and level 1 everywhere else.
fn artificial<P: Packed>(out: &mut [f32], words: &[P::Word], last_rise: u64) {
    out.par_chunks_mut(BLOCK)
        .enumerate()
        .for_each(|(b, chunk)| {
            for (o, z) in chunk.iter_mut().enumerate() {
                let word = P::load(&words[b * BLOCK + o]);
                *z = surface(P::low(word), P::high(word), last_rise) as f32;
            }
        });
}

enum Increments {
    Narrow(Vec<AtomicU32>),
    Wide(Vec<AtomicU64>),
}

impl Increments {
    fn new(cells: usize, wide: bool) -> Self {
        profile::buffer("d8.flats.levels", cells * if wide { 8 } else { 4 });
        if wide {
            Increments::Wide(Wide::zeroed(cells))
        } else {
            Increments::Narrow(Narrow::zeroed(cells))
        }
    }

    fn is_wide(&self) -> bool {
        matches!(self, Increments::Wide(..))
    }

    fn clear(&self, list: &[u32]) {
        match self {
            Increments::Narrow(w) => list
                .par_iter()
                .for_each(|&i| Narrow::store(&w[i as usize], 0)),
            Increments::Wide(w) => list
                .par_iter()
                .for_each(|&i| Wide::store(&w[i as usize], 0)),
        }
    }
}

fn resolve_flats<'a>(
    grid: Grid,
    surface: Surface<'_>,
    dirs: &mut [u8],
    flats: usize,
    row_factors: &(impl Fn(usize) -> &'a [f64; 9] + Sync),
) {
    let (original, mut scratch) = match surface {
        Surface::Borrowed(z) => (Some(z), Vec::new()),
        Surface::Owned(z) => (None, z),
    };
    assert_eq!(original.map_or(scratch.len(), <[f32]>::len), grid.cells);
    let mut set = if flats <= grid.cells / 32 {
        Flats::collect(dirs)
    } else {
        Flats::Scan
    };
    let mut count = flats;
    let mut increments = Increments::new(grid.cells, false);
    let mut pass = 1;
    loop {
        // TauDEM only replaces the input surface for subsequent flat passes.
        let elevations = match original {
            Some(z) if pass == 1 => z,
            _ => &scratch[..],
        };
        let outcome = match &increments {
            Increments::Narrow(words) => {
                flat_pass::<Narrow>(grid, pass, elevations, dirs, &set, words, row_factors)
            }
            Increments::Wide(words) => {
                flat_pass::<Wide>(grid, pass, elevations, dirs, &set, words, row_factors)
            }
        };
        let Ok(PassResult {
            remaining,
            last_rise,
        }) = outcome
        else {
            // Directions are unchanged; repeat the pass with wider levels.
            assert!(!increments.is_wide(), "flat increment exceeds 32 bits");
            profile::count("d8.flats.wide_restarts", 1);
            drop(increments);
            increments = Increments::new(grid.cells, true);
            continue;
        };
        profile::count("d8.flats.passes", 1);
        profile::count(
            match pass {
                1 => "d8.flats.pass1.cells",
                2 => "d8.flats.pass2.cells",
                3 => "d8.flats.pass3.cells",
                _ => "d8.flats.later_pass.cells",
            },
            count as u64,
        );
        if remaining == 0 || remaining == count {
            profile::count("d8.flats.unresolved_cells", remaining as u64);
            break;
        }
        {
            let _span = profile::span(format!("d8.flats.pass{pass}.artificial"));
            if scratch.is_empty() {
                profile::buffer("d8.flats.artificial", grid.cells * 4);
                scratch = vec![0.0; grid.cells];
            }
            match &increments {
                Increments::Narrow(words) => artificial::<Narrow>(&mut scratch, words, last_rise),
                Increments::Wide(words) => artificial::<Wide>(&mut scratch, words, last_rise),
            }
            if let Flats::List(list) = &set {
                increments.clear(list);
            } else {
                // Fresh zeroed levels are cheaper than clearing a mostly
                // untouched grid; pages are only committed where flats remain.
                let wide = increments.is_wide();
                drop(increments);
                increments = Increments::new(grid.cells, wide);
            }
            set = match set {
                Flats::List(mut list) => {
                    list.retain(|&i| dirs[i as usize] == 0);
                    Flats::List(list)
                }
                Flats::Scan if remaining <= grid.cells / 32 => Flats::collect(dirs),
                Flats::Scan => Flats::Scan,
            };
        }
        count = remaining;
        pass += 1;
    }
}
