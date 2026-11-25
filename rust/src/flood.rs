// SPDX-License-Identifier: GPL-3.0-only
//! Improved Priority-Flood, after Barnes, Lehman & Mulla (2014).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use crate::{FEL_NODATA, Grid, missing};

/// Fills depressions in place to their lowest spill elevation. Raster edges,
/// cells adjacent to nodata, and cells with mask value 1 are fixed outlets.
pub fn pit_remove(
    grid: Grid,
    elevations: &mut [f32],
    nodata: f32,
    four_way: bool,
    mask: Option<&[u8]>,
) {
    assert_eq!(elevations.len(), grid.cells);
    assert!(mask.is_none_or(|mask| mask.len() == grid.cells));
    let step = if four_way { 2 } else { 1 };
    let mut visited = vec![0u64; grid.cells.div_ceil(64)];
    let mut heap = BinaryHeap::new();
    let mut pit = VecDeque::new();

    // Mark holes first, seeding their valid neighbors without inspecting all
    // eight neighbors of every land cell.
    for i in 0..grid.cells {
        if missing(elevations[i], nodata) {
            elevations[i] = FEL_NODATA;
            mark(&mut visited, i);
            grid.for_neighbors(i, step, |_, j| {
                if !missing(elevations[j], nodata) && elevations[j] != FEL_NODATA {
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

    while let Some(cell) = pit
        .pop_front()
        .or_else(|| heap.pop().map(|key| key.0 as u32))
    {
        let i = cell as usize;
        let height = elevations[i];
        grid.for_neighbors(i, step, |_, j| {
            if mark(&mut visited, j) {
                if elevations[j] <= height {
                    elevations[j] = height;
                    pit.push_back(j as u32);
                } else {
                    heap.push(Reverse(priority(elevations[j], j)));
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
fn seed(elevations: &[f32], visited: &mut [u64], heap: &mut BinaryHeap<Reverse<u64>>, i: usize) {
    if mark(visited, i) {
        heap.push(Reverse(priority(elevations[i], i)));
    }
}

#[inline]
fn priority(height: f32, i: usize) -> u64 {
    let bits = height.to_bits();
    let sign = ((bits as i32 >> 31) as u32) | 0x8000_0000;
    (u64::from(bits ^ sign) << 32) | i as u64
}
