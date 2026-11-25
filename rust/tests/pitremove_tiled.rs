// SPDX-License-Identifier: GPL-3.0-only

mod support;

use std::sync::Mutex;
use support::fixtures::{Category, Fixture, Input, NODATA, THREADS, TILE_SIZES};
use support::harness::{
    Layer, TiledStage, assert_suite, assert_tiled_matches_in_memory, reference, stage,
};
use taudem_rs::FEL_NODATA;
use taudem_rs::pitremove_tiled::PitRemovePlan;
use taudem_rs::tiling::ExecutionOptions;

fn pitremove() -> impl TiledStage {
    stage(
        "out-of-core PitRemove with final halos",
        |fixture| vec![Layer::f32("fel", reference::fel(fixture))],
        |fixture, spec| {
            let dem = fixture.dem_values().unwrap();
            let output = Mutex::new(vec![0.0; fixture.cells()]);
            let expected = reference::fel(fixture);
            let options = ExecutionOptions {
                max_memory: Some(48 << 20),
                ..Default::default()
            };
            let solved = PitRemovePlan::new(
                *spec,
                &options,
                fixture.nodata,
                fixture.four_way,
                fixture.mask.is_some(),
            )
            .unwrap()
            .solve(|halo, elevations, mask| {
                for row in 0..halo.window.height {
                    let start = (halo.window.y + row) * fixture.width + halo.window.x;
                    let dest = (row + halo.pad_top) * halo.buffer_width() + halo.pad_left;
                    elevations[dest..dest + halo.window.width]
                        .copy_from_slice(&dem[start..start + halo.window.width]);
                }
                if let Some(mask) = mask {
                    let raw = fixture.mask.as_ref().unwrap();
                    for (row, dest) in mask.chunks_exact_mut(halo.core.width).enumerate() {
                        let start = (halo.core.y + row) * fixture.width + halo.core.x;
                        dest.copy_from_slice(&raw[start..start + halo.core.width]);
                    }
                }
                Ok(())
            })
            .unwrap();
            solved
                .emit(|tile| {
                    let r = tile.halo.core;
                    let mut output = output.lock().unwrap();
                    for y in 0..tile.halo.buffer_height() {
                        for x in 0..tile.halo.buffer_width() {
                            let i = y * tile.halo.buffer_width() + x;
                            let gx = (r.x + x).wrapping_sub(1);
                            let gy = (r.y + y).wrapping_sub(1);
                            let expected = if gx < fixture.width && gy < fixture.height {
                                expected[gy * fixture.width + gx]
                            } else {
                                FEL_NODATA
                            };
                            assert_eq!(
                                tile.values[i].to_bits(),
                                expected.to_bits(),
                                "{} tile {:?} halo ({x}, {y})",
                                fixture.name,
                                tile.id
                            );
                        }
                    }
                    for row in 0..r.height {
                        let start = (r.y + row) * fixture.width + r.x;
                        let src = tile.halo.index(0, row) as usize;
                        output[start..start + r.width]
                            .copy_from_slice(&tile.values[src..src + r.width]);
                    }
                    Ok(())
                })
                .unwrap();
            vec![Layer::f32("fel", output.into_inner().unwrap())]
        },
    )
}

#[test]
fn all_adversarial_dem_fixtures() {
    assert_suite(&pitremove(), &TILE_SIZES, &THREADS);
}

#[test]
fn signed_zero_spills_and_fixed_outlets_across_seams() {
    let stage = pitremove();
    for tile in TILE_SIZES {
        let width = 4 * tile + 1;
        let height = 3 * tile + 3;
        let mut dem = vec![12.0; width * height];
        let mut mask = vec![0; dem.len()];
        let idx = |x, y| y * width + x;
        // A positive zero reached from below keeps its own sign even when
        // the surrounding zero spill component has a negative outlet.
        for y in 3..2 * tile + 2 {
            for x in 4..3 * tile {
                dem[idx(x, y)] = -3.0;
            }
        }
        for x in 0..tile {
            dem[idx(x, tile)] = -5.0;
        }
        dem[idx(tile, tile)] = 0.0;
        dem[idx(2 * tile, 2 * tile)] = -0.0;
        mask[idx(2 * tile, 2 * tile)] = 1;
        dem[idx(2 * tile + 2, 2 * tile + 4)] = f32::NAN;
        dem[idx(2 * tile + 3, 2 * tile + 4)] = -0.0;

        for four_way in [false, true] {
            let fixture = Fixture {
                name: format!("signed-zero-fill-4way-{four_way}"),
                category: Category::CrossSeamPits,
                width,
                height,
                input: Input::Dem(dem.clone()),
                nodata: NODATA,
                mask: Some(mask.clone()),
                four_way,
                check_contamination: true,
                spacing: (1.0, 1.0),
            };
            for threads in THREADS {
                assert_tiled_matches_in_memory(&stage, &fixture, tile, threads);
            }
        }
    }
}
