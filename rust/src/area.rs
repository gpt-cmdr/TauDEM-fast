// SPDX-License-Identifier: GPL-3.0-only

use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use rayon::prelude::*;

use crate::{DIRECTION_NODATA, Grid, OPPOSITE, SCALAR_NODATA, profile};

/// The low bits of a cell's state count upstream neighbors not yet summed.
const COUNT: u8 = 15;
const CONTAMINATED: u8 = 16;
const INACTIVE: u8 = 32;
/// Set once, before traversal, on cells that start with no upstream neighbor.
const HEAD: u8 = 64;

const BLOCK: usize = 1 << 14;

const _: () = assert!(align_of::<f32>() == align_of::<AtomicU32>());

/// Accumulates one unit per cell in TauDEM's D8 summation order. The input
/// directions use 1–8, 0 for unresolved flow and 255 for nodata.
///
/// Each cell's area is one plus the areas of its upstream neighbors, added in
/// fixed neighbor order once all of them are final, so the result does not
/// depend on which thread visits which cell.
pub fn area_d8(grid: Grid, direction: &[u8], check_contamination: bool) -> Vec<f32> {
    assert_eq!(direction.len(), grid.cells);
    let indegree = profile::span("area.indegree");
    let mut state = vec![INACTIVE; grid.cells];
    profile::buffer("area.state", state.len());
    state
        .par_chunks_mut(grid.width)
        .enumerate()
        .for_each(|(row, output)| {
            for (col, cell) in output.iter_mut().enumerate() {
                let i = row * grid.width + col;
                if direction[i] > 8 {
                    continue;
                }
                let mut incoming = 0;
                let mut contaminated = !grid.interior(i, col);
                grid.for_neighbors(i, 1, |k, j| {
                    let other = direction[j];
                    // TauDEM counts a northwest neighbor with direction 0 as
                    // flowing in; such a cell never completes.
                    incoming += u8::from(other == OPPOSITE[k] || (k == 4 && other == 0));
                    contaminated |= other == DIRECTION_NODATA;
                });
                *cell = incoming
                    | (u8::from(check_contamination && contaminated) * CONTAMINATED)
                    | (u8::from(incoming == 0) * HEAD);
            }
        });
    drop(indegree);

    let _traverse = profile::span("area.traverse");
    let mut area = vec![SCALAR_NODATA; grid.cells];
    profile::buffer("area.area", area.len() * 4);
    // SAFETY: Both slices are exclusively borrowed for the views' lifetime,
    // and the atomic types have the size, alignment and bit validity of u8
    // and of the f32 bit patterns stored as u32.
    let shared_state = unsafe { &*(state.as_mut_slice() as *mut [u8] as *const [AtomicU8]) };
    let shared_area = unsafe { &*(area.as_mut_slice() as *mut [f32] as *const [AtomicU32]) };
    let total = (0..grid.cells.div_ceil(BLOCK))
        .into_par_iter()
        .map(|block| {
            let mut visited = 0u64;
            let end = ((block + 1) * BLOCK).min(grid.cells);
            for start in block * BLOCK..end {
                // Heads are never decremented, so this test cannot race.
                if shared_state[start].load(Ordering::Relaxed) & HEAD != 0 {
                    visited += walk(grid, direction, shared_state, shared_area, start);
                }
            }
            visited
        })
        .sum::<u64>();
    profile::count("area.cells_accumulated", total);
    area
}

/// Accumulates `start` and continues downstream through every cell whose
/// last upstream neighbor this walk completes. Returns the cells visited.
fn walk(grid: Grid, direction: &[u8], state: &[AtomicU8], area: &[AtomicU32], start: usize) -> u64 {
    let mut i = start;
    let mut contaminated = state[i].load(Ordering::Relaxed) & CONTAMINATED != 0;
    let mut visited = 0;
    loop {
        visited += 1;
        if !contaminated {
            let mut sum = 1.0f32;
            grid.for_neighbors(i, 1, |k, j| {
                if direction[j] == OPPOSITE[k] {
                    sum += f32::from_bits(area[j].load(Ordering::Relaxed));
                }
            });
            area[i].store(sum.to_bits(), Ordering::Relaxed);
        }
        state[i].store(INACTIVE, Ordering::Relaxed);
        let k = direction[i] as usize;
        if k == 0 {
            break;
        }
        let col = i % grid.width;
        let next = if grid.interior(i, col) {
            Some(grid.offset(i, k))
        } else {
            grid.neighbor(i, col, k)
        };
        let Some(j) = next else { break };
        // Only cells that never accumulate (nodata, invalid codes) are
        // inactive here: a cell downstream of `i` cannot have completed.
        if state[j].load(Ordering::Relaxed) == INACTIVE {
            break;
        }
        if contaminated {
            state[j].fetch_or(CONTAMINATED, Ordering::Relaxed);
        }
        // Release publishes this cell's area; the walk that completes `j`
        // acquires every upstream area along the decrement chain.
        let before = state[j].fetch_sub(1, Ordering::AcqRel);
        if before & COUNT != 1 {
            break;
        }
        contaminated = before & CONTAMINATED != 0;
        i = j;
    }
    visited
}
