// SPDX-License-Identifier: GPL-3.0-only
//! D8 direction codes and the TauDEM implementation of Garbrecht & Martz.

use std::collections::VecDeque;

use rayon::prelude::*;

use crate::{DIRECTION_NODATA, Grid, ORDER, SCALAR_NODATA, missing};

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
    let mut direction = vec![DIRECTION_NODATA; grid.cells];
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

    let mut flats = Vec::new();
    // TauDEM tests crossing against directions it has already assigned.
    // The unrestricted winner only needs reevaluation when it crosses.
    for i in 0..grid.cells {
        let k = direction[i] as usize;
        if matches!(k, 2 | 4 | 6 | 8) && crosses(grid, &direction, i, k, i) {
            let heights = std::array::from_fn(|k| elevations[grid.offset(i, k)]);
            direction[i] = steepest(&heights, row_factors(i / grid.width), |k| {
                !crosses(grid, &direction, i, k, i)
            });
        }
        if direction[i] == 0 {
            flats.push(i as u32);
        }
    }

    // TauDEM writes sd8 before routing across artificial flat elevations.
    let slope = with_slope.then(|| {
        let mut slope = vec![SCALAR_NODATA; grid.cells];
        slope
            .par_chunks_mut(grid.width)
            .enumerate()
            .for_each(|(row, output)| {
                let scale = row_factors(row);
                let start = row * grid.width;
                for (col, value) in output.iter_mut().enumerate() {
                    let i = start + col;
                    let k = direction[i] as usize;
                    if k <= 8 {
                        *value = scaled(elevations[i] - elevations[grid.offset(i, k)], scale[k]);
                    }
                }
            });
        slope
    });

    if !flats.is_empty() {
        resolve_flats(grid, elevations, &mut direction, flats, &row_factors);
    }
    D8Result { direction, slope }
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

fn resolve_flats<'a>(
    grid: Grid,
    elevations: &[f32],
    dirs: &mut [u8],
    mut flats: Vec<u32>,
    row_factors: &impl Fn(usize) -> &'a [f64; 9],
) {
    let mut artificial = Vec::new();
    let mut fall = vec![0u32; grid.cells];
    let mut rise = vec![0u32; grid.cells];
    let mut frontier = VecDeque::new();

    loop {
        let elevations = if artificial.is_empty() {
            elevations
        } else {
            &artificial
        };
        // incfall: BFS from draining flat boundaries across equal-height
        // cells, excluding edges blocked by an existing diagonal.
        for &i in &flats {
            let i = i as usize;
            if (1..9).any(|k| {
                let j = grid.offset(i, k);
                (1..=8).contains(&dirs[j])
                    && elevations[i] >= elevations[j]
                    && !crosses(grid, dirs, i, k, grid.cells)
            }) {
                fall[i] = 1;
                frontier.push_back(i as u32);
            }
        }
        // At stage two, incfall can also see the default level of a
        // same-height neighbor outside the unresolved set.
        for &i in &flats {
            let i = i as usize;
            if fall[i] == 0
                && (1..9).any(|k| {
                    let j = grid.offset(i, k);
                    dirs[j] != 0
                        && elevations[i] == elevations[j]
                        && !crosses(grid, dirs, i, k, grid.cells)
                })
            {
                fall[i] = 2;
                frontier.push_back(i as u32);
            }
        }
        let mut last_fall = 0;
        while let Some(i) = frontier.pop_front() {
            let i = i as usize;
            last_fall = fall[i];
            for k in 1..9 {
                let j = grid.offset(i, k);
                if dirs[j] == 0
                    && fall[j] == 0
                    && elevations[i] == elevations[j]
                    && !crosses(grid, dirs, i, k, grid.cells)
                {
                    fall[j] = last_fall + 1;
                    frontier.push_back(j as u32);
                }
            }
        }

        // incrise: BFS from higher terrain across the entire unresolved set.
        // The original scan increments each marked cell on every remaining
        // pass, including the final pass with no newly marked cells.
        for &i in &flats {
            let i = i as usize;
            if (1..9).any(|k| elevations[grid.offset(i, k)] > elevations[i]) {
                rise[i] = 1;
                frontier.push_back(i as u32);
            }
        }
        let mut last_rise = 0;
        while let Some(i) = frontier.pop_front() {
            let i = i as usize;
            last_rise = rise[i];
            for k in 1..9 {
                let j = grid.offset(i, k);
                if dirs[j] == 0 && rise[j] == 0 {
                    rise[j] = last_rise + 1;
                    frontier.push_back(j as u32);
                }
            }
        }

        for &i in &flats {
            let i = i as usize;
            if fall[i] == 0 {
                dirs[i] = DIRECTION_NODATA;
                fall[i] = last_fall.max(1).saturating_add(2);
            }
        }
        let surface = |i: usize| {
            let level = fall[i].max(1) as i64;
            if rise[i] > 0 {
                level + last_rise as i64 + 2 - rise[i] as i64
            } else {
                level
            }
        };
        for &i in &flats {
            let i = i as usize;
            let scale = row_factors(i / grid.width);
            let mut best = 0.0;
            for k in ORDER {
                let j = grid.offset(i, k);
                if rise[j] > 0 {
                    let slope = scaled((surface(i) - surface(j)) as f32, scale[k]);
                    if slope > best {
                        best = slope;
                        dirs[i] = k as u8;
                    }
                } else if elevations[i] >= elevations[j] {
                    dirs[i] = k as u8;
                    break;
                }
            }
        }

        let previous = flats.len();
        flats.retain(|&i| dirs[i as usize] == 0);
        if flats.is_empty() || flats.len() == previous {
            break;
        }
        // TauDEM only replaces the input surface for subsequent flat passes.
        artificial.resize(grid.cells, 0.0);
        for (i, z) in artificial.iter_mut().enumerate() {
            *z = surface(i) as f32;
        }
        fall.fill(0);
        rise.fill(0);
    }
}
