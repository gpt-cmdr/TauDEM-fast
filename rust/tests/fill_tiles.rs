// SPDX-License-Identifier: GPL-3.0-only
//! The tiled depression fill must equal the whole-grid flood bit for bit.

use taudem_rs::{Grid, pit_remove_with_tiles};

/// SplitMix64, so the cases are reproducible without dependencies.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const ND: f32 = -9999.0;

fn bits(z: &[f32]) -> Vec<u32> {
    z.iter().map(|z| z.to_bits()).collect()
}

/// Fills `z` serially and with each tile side, for both connectivities, with
/// and without `mask`, and requires identical bits.
fn check(name: &str, width: usize, height: usize, z: &[f32], mask: Option<&[u8]>) {
    let grid = Grid::new(width, height).unwrap();
    for four_way in [false, true] {
        let mut reference = z.to_vec();
        pit_remove_with_tiles(grid, &mut reference, ND, four_way, mask, 0);
        for tile in [1, 2, 3, 4, 5, 7, 9] {
            let mut tiled = z.to_vec();
            pit_remove_with_tiles(grid, &mut tiled, ND, four_way, mask, tile);
            if bits(&tiled) != bits(&reference) {
                let i = (0..z.len())
                    .find(|&i| tiled[i].to_bits() != reference[i].to_bits())
                    .unwrap();
                panic!(
                    "{name} {width}x{height} tile {tile} four_way {four_way}: cell {i} \
                     (row {}, col {}) tiled {} serial {} input {}",
                    i / width,
                    i % width,
                    tiled[i],
                    reference[i],
                    z[i]
                );
            }
        }
    }
}

fn random_mask(rng: &mut Rng, cells: usize, per_mille: u64) -> Vec<u8> {
    (0..cells)
        .map(|_| u8::from(rng.below(1000) < per_mille))
        .collect()
}

#[test]
fn quantized_noise_with_many_ties() {
    let mut rng = Rng(1);
    for (width, height) in [(13, 11), (20, 17), (31, 8), (8, 29), (40, 40)] {
        for levels in [3, 10, 1000] {
            let z: Vec<f32> = (0..width * height)
                .map(|_| rng.below(levels) as f32)
                .collect();
            check("noise", width, height, &z, None);
            let mask = random_mask(&mut rng, z.len(), 20);
            check("noise+mask", width, height, &z, Some(&mask));
        }
    }
}

#[test]
fn nested_pits_cross_tile_boundaries() {
    // Concentric rings with alternating heights: each ring is a sill for the
    // pit inside it, and the rings span many tiles.
    for (width, height) in [(37, 33), (24, 41)] {
        let (cx, cy) = (width as f32 / 2.3, height as f32 / 1.9);
        let z: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as f32, (i / width) as f32);
                let r = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                let ring = (r / 3.0) as i32;
                if ring % 2 == 0 {
                    10.0 + ring as f32 * 0.5
                } else {
                    1.0 + (r * 7.3).sin()
                }
            })
            .collect();
        check("rings", width, height, &z, None);
    }
}

#[test]
fn spiral_maze_drains_through_one_long_channel() {
    // A spiral wall with one gap at the outside; the channel floor descends
    // towards the centre, so the whole channel fills to the gap's sill.
    let n = 31;
    let mut z = vec![50.0f32; n * n];
    let (mut x, mut y) = (1usize, 1usize);
    let (mut dx, mut dy) = (1isize, 0isize);
    let (mut lo, mut hi) = (1usize, n - 2);
    let mut floor = 20.0;
    for _ in 0..n * n {
        z[y * n + x] = floor;
        floor -= 0.01;
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        let turn = nx < lo as isize || nx > hi as isize || ny < lo as isize || ny > hi as isize;
        if turn {
            (dx, dy) = (-dy, dx);
            if dx == 1 {
                lo += 2;
                hi -= 2;
                if lo > hi {
                    break;
                }
            }
        }
        x = (x as isize + dx) as usize;
        y = (y as isize + dy) as usize;
    }
    z[n] = 25.0; // the gap in the outer wall, at the raster edge
    check("spiral", n, n, &z, None);
}

#[test]
fn huge_flat_and_level_planes() {
    let (width, height) = (45, 38);
    let mut z = vec![7.0f32; width * height];
    for i in 0..width {
        z[i] = 9.0;
        z[(height - 1) * width + i] = 9.0;
    }
    z[20 * width + 17] = 6.5; // a single interior pit below the plane
    z[(height - 1) * width + 30] = 3.0; // one low edge outlet
    check("flat", width, height, &z, None);
    let mut rng = Rng(5);
    let mask = random_mask(&mut rng, z.len(), 5);
    check("flat+mask", width, height, &z, Some(&mask));
}

#[test]
fn nodata_holes_irregular_edges_and_nan() {
    let mut rng = Rng(9);
    for (width, height) in [(29, 23), (16, 50)] {
        let mut z: Vec<f32> = (0..width * height)
            .map(|_| rng.below(20) as f32 * 0.25)
            .collect();
        // Irregular nodata edge and interior holes, including NaN.
        for row in 0..height {
            let cut = rng.below(width as u64 / 3) as usize;
            for col in 0..cut {
                z[row * width + col] = ND;
            }
        }
        for _ in 0..z.len() / 12 {
            let i = rng.below(z.len() as u64) as usize;
            z[i] = if rng.below(2) == 0 { ND } else { f32::NAN };
        }
        check("holes", width, height, &z, None);
        let mask = random_mask(&mut rng, z.len(), 30);
        check("holes+mask", width, height, &z, Some(&mask));
    }
    // An all-nodata tile row and a raster that is mostly void.
    let (width, height) = (18, 18);
    let z: Vec<f32> = (0..width * height)
        .map(|i| {
            if (i / width) % 7 < 4 {
                ND
            } else {
                (i % 5) as f32
            }
        })
        .collect();
    check("void bands", width, height, &z, None);
}

#[test]
fn degenerate_shapes_and_signed_zero() {
    let mut rng = Rng(13);
    for (width, height) in [(1, 9), (9, 1), (2, 2), (3, 2), (1, 1), (6, 3)] {
        let z: Vec<f32> = (0..width * height).map(|_| rng.below(4) as f32).collect();
        check("small", width, height, &z, None);
    }
    // -0.0 and 0.0 compare equal, so the sign written over a filled zero
    // depends on which source reaches it; the tiled path must agree.
    let (width, height) = (12, 10);
    let z: Vec<f32> = (0..width * height)
        .map(|i| match i % 3 {
            0 => -0.0,
            1 => 0.0,
            _ => 1.0,
        })
        .collect();
    check("signed zero", width, height, &z, None);
}

#[test]
fn signed_zero_planes_sills_and_basins() {
    let mut rng = Rng(21);
    for (width, height) in [(23, 19), (40, 13), (17, 44)] {
        for case in 0..6 {
            let z: Vec<f32> = (0..width * height)
                .map(|i| {
                    let (x, y) = (i % width, i / width);
                    let r = rng.below(100);
                    match case {
                        // Mostly signed zeros with scattered negative pits.
                        0 => match r {
                            0..40 => -0.0,
                            40..80 => 0.0,
                            80..95 => -(rng.below(5) as f32),
                            _ => rng.below(3) as f32,
                        },
                        // Below-zero basins whose sills are signed zeros.
                        1 => {
                            if x % 6 == 0 || y % 5 == 0 {
                                if r < 50 { -0.0 } else { 0.0 }
                            } else {
                                -1.0 - (r % 4) as f32
                            }
                        }
                        // Negative plains broken by zero ridges and nodata.
                        2 => match r {
                            0..10 => ND,
                            10..30 => -0.0,
                            30..50 => 0.0,
                            _ => -0.5,
                        },
                        // Large -0.0 and 0.0 regions meeting along a seam.
                        3 => {
                            if r < 8 {
                                -2.0
                            } else if (x + y / 3) % 11 < 5 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        // A zero rim around negative terrain with one +0.0 gap.
                        4 => {
                            if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
                                if x == 3 && y == 0 { 0.0 } else { 1.0 }
                            } else if r < 30 {
                                -0.0
                            } else {
                                -((r % 7) as f32) * 0.25
                            }
                        }
                        // Sparse zeros in rough terrain.
                        _ => match r {
                            0..5 => -0.0,
                            5..10 => 0.0,
                            _ => (r as f32 - 60.0) * 0.1,
                        },
                    }
                })
                .collect();
            check("signed zero", width, height, &z, None);
            let mask = random_mask(&mut rng, z.len(), 15);
            check("signed zero+mask", width, height, &z, Some(&mask));
        }
    }
}
