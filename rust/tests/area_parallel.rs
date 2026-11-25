// SPDX-License-Identifier: GPL-3.0-only
//! Parallel AreaD8 must equal the sequential TauDEM-order accumulation.
// The reference loops mirror the original direction-indexed code.
#![allow(clippy::needless_range_loop)]

use taudem_rs::{DIRECTION_NODATA, FEL_NODATA, Grid, area_d8, d8_flow, pit_remove_with_tiles};

const DX: [isize; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];
const DY: [isize; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];
const OPPOSITE: [u8; 9] = [0, 5, 6, 7, 8, 1, 2, 3, 4];

/// The single-threaded accumulation the parallel walk replaced.
fn reference(width: usize, height: usize, p: &[u8], check: bool) -> Vec<f32> {
    const CONTAMINATED: u8 = 16;
    const INACTIVE: u8 = 32;
    let cells = width * height;
    let at = |i: usize, k: usize| {
        let (x, y) = ((i % width) as isize + DX[k], (i / width) as isize + DY[k]);
        (x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height)
            .then(|| y as usize * width + x as usize)
    };
    let mut state = vec![INACTIVE; cells];
    for i in 0..cells {
        if p[i] > 8 {
            continue;
        }
        let (x, y) = (i % width, i / width);
        let mut incoming = 0;
        let mut contaminated = x == 0 || y == 0 || x + 1 == width || y + 1 == height;
        for k in 1..9 {
            if let Some(j) = at(i, k) {
                incoming += u8::from(p[j] == OPPOSITE[k] || (k == 4 && p[j] == 0));
                contaminated |= p[j] == DIRECTION_NODATA;
            }
        }
        state[i] = incoming | (u8::from(check && contaminated) * CONTAMINATED);
    }
    let mut area = vec![-1.0f32; cells];
    for start in 0..cells {
        let mut i = start;
        while state[i] & !CONTAMINATED == 0 {
            if state[i] == 0 {
                let mut sum = 1.0;
                for k in 1..9 {
                    if let Some(j) = at(i, k)
                        && p[j] == OPPOSITE[k]
                    {
                        sum += area[j];
                    }
                }
                area[i] = sum;
            }
            let contaminated = state[i];
            state[i] = INACTIVE;
            let k = p[i] as usize;
            if k == 0 {
                break;
            }
            let Some(j) = at(i, k) else { break };
            if state[j] == INACTIVE {
                break;
            }
            state[j] = (state[j] - 1) | contaminated;
            i = j;
        }
    }
    area
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        (z ^ (z >> 31)) % n
    }
}

fn check_all(width: usize, height: usize, p: &[u8]) {
    let grid = Grid::new(width, height).unwrap();
    for check in [false, true] {
        let expected: Vec<u32> = reference(width, height, p, check)
            .iter()
            .map(|a| a.to_bits())
            .collect();
        for threads in [1, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let got: Vec<u32> = pool
                .install(|| area_d8(grid, p, check))
                .iter()
                .map(|a| a.to_bits())
                .collect();
            assert_eq!(
                got, expected,
                "{width}x{height} check {check} threads {threads}"
            );
        }
    }
}

#[test]
fn random_codes_with_cycles_nodata_and_flats() {
    let mut rng = Rng(3);
    for (width, height) in [(1, 1), (5, 1), (1, 7), (19, 23), (64, 40), (300, 211)] {
        let p: Vec<u8> = (0..width * height)
            .map(|_| match rng.below(20) {
                0 => DIRECTION_NODATA,
                1 => 0,
                2 => 254,
                r => (r % 8 + 1) as u8,
            })
            .collect();
        check_all(width, height, &p);
    }
}

#[test]
fn directions_from_random_terrain_form_long_chains() {
    let mut rng = Rng(11);
    for (width, height) in [(97, 61), (400, 333)] {
        let mut z: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                if rng.below(200) == 0 {
                    -9999.0
                } else {
                    (x * 0.05).sin() * 20.0 + (y * 0.031).cos() * 15.0 + rng.below(8) as f32 * 0.5
                }
            })
            .collect();
        let grid = Grid::new(width, height).unwrap();
        pit_remove_with_tiles(grid, &mut z, -9999.0, false, None, 37);
        let p = d8_flow(grid, &z, FEL_NODATA, &[(1.0, 1.0)], false).direction;
        check_all(width, height, &p);
    }
}
