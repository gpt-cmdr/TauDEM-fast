// SPDX-License-Identifier: GPL-3.0-only
//! Improved Priority-Flood, after Barnes, Lehman & Mulla (2014).

use std::collections::VecDeque;

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
    let mut heap = FloodQueue::new();
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

#[inline]
fn priority(height: f32, i: usize) -> u64 {
    let bits = height.to_bits();
    let sign = ((bits as i32 >> 31) as u32) | 0x8000_0000;
    (u64::from(bits ^ sign) << 32) | i as u64
}

/// Monotone radix heap. New priority elevations cannot be lower than the
/// last one removed, so only the populated bucket nearest that key needs
/// redistribution. Each entry packs a float sort key and a cell index.
struct FloodQueue {
    buckets: [Vec<u64>; 33],
    used: u64,
    last: u32,
}

impl FloodQueue {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Vec::new()),
            used: 0,
            last: 0,
        }
    }

    #[inline]
    fn push(&mut self, entry: u64) {
        let key = (entry >> 32) as u32;
        debug_assert!(key >= self.last);
        let bucket = 32 - (key ^ self.last).leading_zeros();
        self.buckets[bucket as usize].push(entry);
        self.used |= 1 << bucket;
    }

    #[inline]
    fn pop(&mut self) -> Option<u32> {
        if self.used & 1 == 0 {
            if self.used == 0 {
                return None;
            }
            self.refill();
        }
        let entry = self.buckets[0].pop()?;
        if self.buckets[0].is_empty() {
            self.used &= !1;
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
        self.buckets[bucket] = entries;
    }
}
