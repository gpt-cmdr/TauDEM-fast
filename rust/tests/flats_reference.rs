// SPDX-License-Identifier: GPL-3.0-only
//! The parallel flat resolver must reproduce the sequential TauDEM-order
//! implementation it replaced, direction for direction.
// The reference loops mirror the original direction-indexed code.
#![allow(clippy::needless_range_loop)]

use std::collections::VecDeque;

use taudem_rs::{DIRECTION_NODATA, Grid, d8_flow, pit_remove_with_tiles};

const DX: [isize; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];
const DY: [isize; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];
const ORDER: [usize; 8] = [1, 3, 5, 7, 2, 4, 6, 8];
const ND: f32 = -9999.0;

fn missing(z: f32) -> bool {
    !z.is_finite() || (z - ND).abs() < 1.0e-5
}

fn scaled(delta: f32, factor: f64) -> f32 {
    (f64::from(delta) * factor) as f32
}

/// The single-threaded D8 with flat resolution, as it was before the
/// parallel rewrite, for a grid with unit spacing.
fn reference(width: usize, height: usize, z: &[f32]) -> Vec<u8> {
    let cells = width * height;
    let off = |i: usize, k: usize| i.wrapping_add_signed(DX[k] + DY[k] * width as isize);
    let diag = 1.0 / 2f64.sqrt();
    let scale = [0.0, 1.0, diag, 1.0, diag, 1.0, diag, 1.0, diag];
    let steepest = |h: &[f32; 9], allowed: &dyn Fn(usize) -> bool| {
        let (mut best, mut d) = (0.0, 0);
        for k in ORDER {
            let s = scaled(h[0] - h[k], scale[k]);
            if s > best && allowed(k) {
                best = s;
                d = k as u8;
            }
        }
        d
    };
    let crosses = |dirs: &[u8], i: usize, k: usize, before: usize| {
        let (a, ac, b, bc) = match k {
            2 => (1, 4, 3, 8),
            4 => (3, 6, 5, 2),
            6 => (7, 4, 5, 8),
            8 => (1, 6, 7, 2),
            _ => return false,
        };
        let (a, b) = (off(i, a), off(i, b));
        (a < before && dirs[a] == ac) || (b < before && dirs[b] == bc)
    };
    let mut dirs = vec![DIRECTION_NODATA; cells];
    for i in 0..cells {
        let (x, y) = (i % width, i / width);
        if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
            continue;
        }
        let h: [f32; 9] = std::array::from_fn(|k| z[off(i, k)]);
        if h.iter().any(|&v| missing(v)) {
            continue;
        }
        dirs[i] = steepest(&h, &|_| true);
    }
    let mut flats = Vec::new();
    for i in 0..cells {
        let k = dirs[i] as usize;
        if matches!(k, 2 | 4 | 6 | 8) && crosses(&dirs, i, k, i) {
            let h: [f32; 9] = std::array::from_fn(|k| z[off(i, k)]);
            let fixed = steepest(&h, &|k| !crosses(&dirs, i, k, i));
            dirs[i] = fixed;
        }
        if dirs[i] == 0 {
            flats.push(i);
        }
    }
    if flats.is_empty() {
        return dirs;
    }
    let mut artificial = Vec::new();
    let mut fall = vec![0u32; cells];
    let mut rise = vec![0u32; cells];
    let mut frontier = VecDeque::new();
    loop {
        let e: &[f32] = if artificial.is_empty() {
            z
        } else {
            &artificial
        };
        for &i in &flats {
            if (1..9).any(|k| {
                let j = off(i, k);
                (1..=8).contains(&dirs[j]) && e[i] >= e[j] && !crosses(&dirs, i, k, cells)
            }) {
                fall[i] = 1;
                frontier.push_back(i);
            }
        }
        for &i in &flats {
            if fall[i] == 0
                && (1..9).any(|k| {
                    let j = off(i, k);
                    dirs[j] != 0 && e[i] == e[j] && !crosses(&dirs, i, k, cells)
                })
            {
                fall[i] = 2;
                frontier.push_back(i);
            }
        }
        let mut last_fall = 0;
        while let Some(i) = frontier.pop_front() {
            last_fall = fall[i];
            for k in 1..9 {
                let j = off(i, k);
                if dirs[j] == 0 && fall[j] == 0 && e[i] == e[j] && !crosses(&dirs, i, k, cells) {
                    fall[j] = last_fall + 1;
                    frontier.push_back(j);
                }
            }
        }
        for &i in &flats {
            if (1..9).any(|k| e[off(i, k)] > e[i]) {
                rise[i] = 1;
                frontier.push_back(i);
            }
        }
        let mut last_rise = 0;
        while let Some(i) = frontier.pop_front() {
            last_rise = rise[i];
            for k in 1..9 {
                let j = off(i, k);
                if dirs[j] == 0 && rise[j] == 0 {
                    rise[j] = last_rise + 1;
                    frontier.push_back(j);
                }
            }
        }
        for &i in &flats {
            if fall[i] == 0 {
                dirs[i] = DIRECTION_NODATA;
                fall[i] = last_fall.max(1) + 2;
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
            let mut best = 0.0;
            for k in ORDER {
                let j = off(i, k);
                if rise[j] > 0 {
                    let s = scaled((surface(i) - surface(j)) as f32, scale[k]);
                    if s > best {
                        best = s;
                        dirs[i] = k as u8;
                    }
                } else if e[i] >= e[j] {
                    dirs[i] = k as u8;
                    break;
                }
            }
        }
        let previous = flats.len();
        flats.retain(|&i| dirs[i] == 0);
        if flats.is_empty() || flats.len() == previous {
            break;
        }
        artificial = (0..cells).map(|i| surface(i) as f32).collect();
        fall.fill(0);
        rise.fill(0);
    }
    dirs
}

fn check(name: &str, width: usize, height: usize, z: &[f32]) {
    let grid = Grid::new(width, height).unwrap();
    let expected = reference(width, height, z);
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let got = pool.install(|| d8_flow(grid, z, ND, &[(1.0, 1.0)], false).direction);
        if got != expected {
            let i = (0..z.len()).find(|&i| got[i] != expected[i]).unwrap();
            panic!(
                "{name} {width}x{height} threads {threads}: cell {i} (row {}, col {}) got {} expected {}",
                i / width,
                i % width,
                got[i],
                expected[i]
            );
        }
    }
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

/// Fills a terrain the way the pipeline does, so that flats are level planes.
fn filled(width: usize, height: usize, mut z: Vec<f32>) -> Vec<f32> {
    let grid = Grid::new(width, height).unwrap();
    pit_remove_with_tiles(grid, &mut z, ND, false, None, 0);
    z.iter()
        .map(|&v| if v == taudem_rs::FEL_NODATA { ND } else { v })
        .collect()
}

#[test]
fn quantized_terrain_has_many_small_flats() {
    let mut rng = Rng(7);
    for (width, height) in [(30, 20), (64, 64), (120, 45)] {
        let z: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                ((x * 0.21).sin() * 3.0 + (y * 0.17).cos() * 3.0 + rng.below(3) as f32).round()
            })
            .collect();
        check("quantized raw", width, height, &z);
        check("quantized filled", width, height, &filled(width, height, z));
    }
}

#[test]
fn large_and_nested_flats_with_nodata_need_several_passes() {
    let mut rng = Rng(17);
    for (width, height) in [(90, 70), (200, 150)] {
        let z: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                let (cx, cy) = (width as f32 / 2.0, height as f32 / 2.0);
                let r = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                if rng.below(150) == 0 {
                    ND
                } else if r < 10.0 {
                    3.0 // an inner plateau
                } else if r < 30.0 {
                    2.0 // a ring flat around it
                } else if (x + y) % 37 == 0 {
                    1.5 // thin dikes splitting the outer flat
                } else {
                    1.0 + (x as f32) * 0.0001 * (rng.below(2) as f32)
                }
            })
            .collect();
        check("nested", width, height, &filled(width, height, z));
    }
    assert!(counter("d8.flats.pass2.cells") > 0);
}

#[test]
fn serpentine_flat_and_long_flat_overflow_sixteen_bits() {
    // A winding one-cell channel: long fall distances in a small grid.
    let (width, height) = (41, 41);
    let mut z = vec![9.0f32; width * height];
    for y in (2..height - 2).step_by(4) {
        for x in 2..width - 2 {
            z[y * width + x] = 1.0;
        }
        let link = if (y / 4) % 2 == 0 { width - 3 } else { 2 };
        if y + 4 < height - 2 {
            for dy in 1..4 {
                z[(y + dy) * width + link] = 1.0;
            }
        }
    }
    z[2 * width + 1] = 0.5; // the only exit, at the channel's start
    check("serpentine", width, height, &z);

    // A flat longer than 65,533 cells forces 32-bit levels.
    let (width, height) = (70_005, 3);
    let mut z = vec![5.0f32; width * height];
    for x in 1..width - 1 {
        z[width + x] = 1.0;
    }
    z[width] = 0.0; // drains at the west edge only
    check("long flat", width, height, &z);
    assert!(counter("d8.flats.wide_restarts") >= 1);
}

/// A process-wide profile counter (tests share it, so only lower bounds hold).
fn counter(name: &str) -> u64 {
    taudem_rs::profile::report()
        .counters
        .iter()
        .find(|(n, _)| *n == name)
        .map_or(0, |(_, v)| *v)
}

#[test]
fn sparse_and_dense_flat_sets() {
    let mut rng = Rng(29);
    // Dense: most cells flat (scanned); sparse: a few flats (listed).
    for (width, height, flat_per_mille) in [(150, 100, 900), (150, 100, 10)] {
        let z: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = (i % width, i / width);
                if rng.below(1000) < flat_per_mille {
                    (2 + (x / 25 + y / 20) % 3) as f32
                } else {
                    (x + 2 * y) as f32 * 0.01 + rng.below(100) as f32 * 0.001
                }
            })
            .collect();
        check("flat density", width, height, &filled(width, height, z));
    }
}

#[test]
fn many_small_random_grids_with_unresolved_flats() {
    let mut rng = Rng(41);
    for case in 0..600 {
        let (width, height) = (5 + rng.below(20) as usize, 5 + rng.below(20) as usize);
        let levels = 2 + rng.below(3);
        let z: Vec<f32> = (0..width * height)
            .map(|_| {
                if rng.below(40) == 0 {
                    ND
                } else {
                    rng.below(levels) as f32
                }
            })
            .collect();
        // Unfilled grids keep closed flats that no drain reaches.
        if case % 2 == 0 {
            check("random raw", width, height, &z);
        } else {
            check("random filled", width, height, &filled(width, height, z));
        }
    }
}
