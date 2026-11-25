// SPDX-License-Identifier: GPL-3.0-only

use rayon::prelude::*;

use crate::{DIRECTION_NODATA, Grid, OPPOSITE, SCALAR_NODATA};

const CONTAMINATED: u8 = 16;
const INACTIVE: u8 = 32;

/// Accumulates one unit per cell in TauDEM's D8 summation order. The input
/// directions use 1–8, 0 for unresolved flow and 255 for nodata.
pub fn area_d8(grid: Grid, direction: &[u8], check_contamination: bool) -> Vec<f32> {
    assert_eq!(direction.len(), grid.cells);
    let mut state = vec![INACTIVE; grid.cells];
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
                    incoming += u8::from(other == OPPOSITE[k] || (k == 4 && other == 0));
                    contaminated |= other == DIRECTION_NODATA;
                });
                *cell = incoming | (u8::from(check_contamination && contaminated) * CONTAMINATED);
            }
        });
    let mut area = vec![SCALAR_NODATA; grid.cells];
    for start in 0..grid.cells {
        let mut i = start;
        // There is only one downstream edge. Follow it immediately whenever
        // its last dependency arrives; no queue of headwater cells is needed.
        while state[i] & !CONTAMINATED == 0 {
            let col = i % grid.width;
            if state[i] == 0 {
                let mut sum = 1.0;
                grid.for_neighbors(i, 1, |k, j| {
                    if direction[j] == OPPOSITE[k] {
                        sum += area[j];
                    }
                });
                area[i] = sum;
            }
            let contaminated = state[i];
            state[i] = INACTIVE;
            let k = direction[i] as usize;
            if k == 0 {
                break;
            }
            let next = if grid.interior(i, col) {
                Some(grid.offset(i, k))
            } else {
                grid.neighbor(i, col, k)
            };
            let Some(j) = next else { break };
            if state[j] == INACTIVE {
                break;
            }
            state[j] = (state[j] - 1) | contaminated;
            i = j;
        }
    }
    area
}
