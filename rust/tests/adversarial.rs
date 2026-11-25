// SPDX-License-Identifier: GPL-3.0-only
//! Checks of the §6.1 adversarial fixtures and the parity harness that
//! WP-S1..S3 use. No tiled stage exists yet, so the harness is exercised
//! with stages that route every tile through the tile store and back
//! (which must pass) and with naive tile-local stages (which must fail).

mod support;

use rayon::prelude::*;
use support::d8_reference::d8_reference;
use support::fixtures::{
    Category, Fixture, Input, NODATA, THREADS, TILE_SIZES, large_suite, suite,
};
use support::harness::{Layer, TiledStage, assert_suite, compare, reference, stage};
use taudem_rs::tiling::{
    BorderTable, CacheConfig, StoreOptions, Tier, TileCache, TileId, TileSpec,
};
use taudem_rs::{DIRECTION_NODATA, FEL_NODATA, Grid, SCALAR_NODATA, area_d8, d8_flow, pit_remove};

fn cell(f: &Fixture, values: &[f32], x: usize, y: usize) -> f32 {
    values[y * f.width + x]
}

#[test]
fn fixtures_carry_their_adversarial_features() {
    for t in TILE_SIZES {
        let fixtures = suite(t);
        let by_name = |name: &str| {
            fixtures
                .iter()
                .find(|f| f.name == name)
                .unwrap_or_else(|| panic!("no fixture {name}"))
        };
        for category in [
            Category::CrossSeamPits,
            Category::NodataAtSeams,
            Category::DontCross,
            Category::MultiTileFlats,
            Category::RiversAndQuirks,
            Category::Geometry,
        ] {
            assert!(
                fixtures.iter().any(|f| f.category == category),
                "{category:?} at T={t}"
            );
        }
        for f in &fixtures {
            let len = match &f.input {
                Input::Dem(z) => z.len(),
                Input::Directions(p) => p.len(),
            };
            assert_eq!(len, f.cells(), "{}", f.name);
            assert!(f.mask.as_ref().is_none_or(|m| m.len() == f.cells()));
        }

        // 1. The corner pit spills over its saddle three tiles away in 8-way
        // flooding, and only over the plateau with -4.
        let corner = 2 * t;
        let fel = reference::fel(by_name("c1-corner-pit-winding-spill"));
        assert_eq!(
            cell(by_name("c1-corner-pit-winding-spill"), &fel, corner, corner),
            70.0
        );
        let fel4 = reference::fel(by_name("c1-corner-pit-winding-spill-4way"));
        assert!(
            cell(
                by_name("c1-corner-pit-winding-spill-4way"),
                &fel4,
                corner,
                corner
            ) > 95.0
        );
        for name in [
            "c1-pit-drained-by-depmask",
            "c1-pit-drained-by-depmask-4way",
        ] {
            let f = by_name(name);
            assert_eq!(
                cell(f, &reference::fel(f), t + t / 2, corner),
                30.0,
                "{name}"
            );
        }

        // 2. A pit whose outlet is nodata diagonally across a tile corner.
        let f = by_name("c2-nodata-on-seams-and-corners");
        assert_eq!(cell(f, &reference::fel(f), 3 * t, t), 10.0);
        let f = by_name("c2-nodata-on-seams-and-corners-4way");
        assert_eq!(cell(f, &reference::fel(f), 3 * t, t), 120.0);

        // 4. Secondary flats need more than one Garbrecht-Martz pass.
        for name in ["c4-plateau-with-secondary-flats", "c4-twin-outlet-plateau"] {
            let f = by_name(name);
            let trace = d8_reference(
                f.width,
                f.height,
                &reference::fel(f),
                FEL_NODATA,
                f.spacing,
                true,
            );
            assert!(
                trace.passes >= 2,
                "{name} at T={t}: {} passes",
                trace.passes
            );
        }

        // 5. The k == 4 quirk blocks the cell SE of each zero, in the
        // diagonally adjacent tile; the river crosses tiles and has area.
        let f = by_name("c5-corner-zero-blocks-diagonal-tile");
        let area = reference::ad8(f);
        assert_eq!(cell(f, &area, t, t), SCALAR_NODATA);
        assert_eq!(cell(f, &area, t - 2, t), 1.0 + cell(f, &area, t - 3, t));
        let f = by_name("c5-meandering-river-tile-cycle-nc");
        let area = reference::ad8(f);
        let max = area.iter().copied().fold(0.0f32, f32::max);
        assert!(max > (2 * t) as f32 * 4.0, "river area {max}");

        // 6. Whole tiles of nodata.
        let f = by_name("c6-sparse-nodata-tiles");
        let z = f.dem_values().unwrap();
        let spec = TileSpec::new(f.width, f.height, t, t).unwrap();
        let empty = spec
            .row_major()
            .filter(|&id| {
                let r = spec.rect(id);
                (r.y..r.bottom()).all(|y| (r.x..r.right()).all(|x| z[y * f.width + x] == NODATA))
            })
            .count();
        assert!(empty >= 4, "{empty} empty tiles");
    }
}

/// Cells whose steepest D8 slope is reached by more than one direction, so
/// that TauDEM's `ORDER` tie-break decides them.
fn tie_cells(f: &Fixture, fel: &[f32]) -> Vec<usize> {
    const DX: [isize; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];
    const DY: [isize; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];
    let (dx, dy) = f.spacing;
    let diag = 1.0 / (dx * dx + dy * dy).sqrt();
    let scale = [
        0.0,
        1.0 / dx,
        diag,
        1.0 / dy,
        diag,
        1.0 / dx,
        diag,
        1.0 / dy,
        diag,
    ];
    let w = f.width as isize;
    (0..f.cells())
        .filter(|&i| {
            let (x, y) = ((i % f.width) as isize, (i / f.width) as isize);
            if x == 0 || y == 0 || x + 1 >= w || y + 1 >= f.height as isize {
                return false;
            }
            let slopes: Vec<f32> = (1..9)
                .map(|k| {
                    let j = ((y + DY[k]) * w + x + DX[k]) as usize;
                    ((fel[i] - fel[j]) as f64 * scale[k]) as f32
                })
                .collect();
            let best = slopes.iter().copied().fold(0.0f32, f32::max);
            best > 0.0 && slopes.iter().filter(|&&s| s == best).count() > 1
        })
        .collect()
}

/// The transcription of `src/d8.rs` agrees with `d8_flow` on every DEM
/// fixture, and the `dontCross` fixtures put diagonal ties and flat cells on
/// tile edges for every tile size.
///
/// The crossing rule itself never fires on the suite, nor on 200,000
/// random and 3,000 smooth quantized terrains tried while building it: two
/// steepest-descent directions of vertically or horizontally adjacent cells
/// cannot cross (the slopes each needs contradict each other), and no flat
/// pass produced a crossing either. The count is printed so that a change
/// of the kernel that makes it fire is noticed.
#[test]
fn dont_cross_fixtures_put_diagonal_ties_on_seams() {
    for t in TILE_SIZES {
        for f in suite(t).iter().filter(|f| f.dem_values().is_some()) {
            let fel = reference::fel(f);
            let flow = d8_flow(f.grid(), &fel, FEL_NODATA, &[f.spacing], false);
            let honored = d8_reference(f.width, f.height, &fel, FEL_NODATA, f.spacing, true);
            assert!(
                honored.direction == flow.direction,
                "transcription differs on {}",
                f.name
            );
            if f.category != Category::DontCross {
                continue;
            }
            let on_edge = |i: usize| {
                let (x, y) = (i % f.width, i / f.width);
                [x % t, y % t].iter().any(|&r| r == 0 || r + 1 == t)
            };
            let ties = tie_cells(f, &fel);
            // Cells with no lower neighbor: the flat passes route them.
            let w = f.width;
            let flat_cells = (0..f.cells())
                .filter(|&i| {
                    let (x, y) = (i % w, i / w);
                    on_edge(i)
                        && x > 0
                        && y > 0
                        && x + 1 < w
                        && y + 1 < f.height
                        && (y - 1..=y + 1)
                            .all(|ny| (x - 1..=x + 1).all(|nx| fel[ny * w + nx] >= fel[i]))
                })
                .count();
            println!(
                "{} at T={t}: {} tie cells ({} on tile edges), {} crossing cells, {} flat passes",
                f.name,
                ties.len(),
                ties.iter().filter(|&&i| on_edge(i)).count(),
                honored.crossing_cells.len(),
                honored.passes
            );
            assert!(
                ties.iter().filter(|&&i| on_edge(i)).count() >= 4,
                "{} at T={t}",
                f.name
            );
            assert!(
                honored.passes >= 1 && flat_cells > 0,
                "{} at T={t}: no flats on tile edges",
                f.name
            );
        }
    }
}

/// Splits a raster into tiles through a tile store (tiles in every tier,
/// in parallel) and a border table, and reassembles it.
fn through_tiles<T: taudem_rs::tiling::TileElement>(
    spec: &TileSpec,
    values: &[T],
    fill: T,
) -> Vec<T> {
    let cache = TileCache::new(CacheConfig::new(
        1 << 16,
        Some(std::env::temp_dir().join("taudem-rs-harness")),
    ));
    let store = cache.store::<T>("layer", StoreOptions::default());
    let borders = BorderTable::new(*spec, fill);
    let width = spec.raster_width();
    let ids: Vec<TileId> = spec.row_major().collect();
    ids.par_iter().for_each(|&id| {
        let r = spec.rect(id);
        let tile: Vec<T> = (r.y..r.bottom())
            .flat_map(|y| {
                values[y * width + r.x..y * width + r.right()]
                    .iter()
                    .copied()
            })
            .collect();
        borders.set_from_tile(id, &tile).unwrap();
        store.put_owned(id, tile).unwrap();
        match spec.index(id) % 3 {
            1 => store.evict(id, Tier::Compressed).unwrap(),
            2 => store.evict(id, Tier::Spilled).unwrap(),
            _ => {}
        }
    });
    let mut out = vec![fill; values.len()];
    let mut tile = Vec::new();
    for id in spec.row_major() {
        let r = spec.rect(id);
        store.get_into(id, &mut tile).unwrap();
        for (row, chunk) in tile.chunks_exact(r.width).enumerate() {
            let start = (r.y + row) * width + r.x;
            out[start..start + r.width].copy_from_slice(chunk);
        }
        // The halo ring from the border table is the raster around the tile.
        let ring = borders.ring(id, fill);
        if r.y > 0 {
            for (k, &v) in ring.north[1..=r.width].iter().enumerate() {
                assert!(v == values[(r.y - 1) * width + r.x + k]);
            }
        }
    }
    out
}

#[test]
fn harness_accepts_stages_routed_through_tiles() {
    let pit = stage(
        "pitremove via tiles",
        |f| vec![Layer::f32("fel", reference::fel(f))],
        |f, spec| {
            let mut z = through_tiles(spec, f.dem_values().unwrap(), 0.0f32);
            let mask = f.mask.as_ref().map(|m| through_tiles(spec, m, 0u8));
            pit_remove(f.grid(), &mut z, f.nodata, f.four_way, mask.as_deref());
            vec![Layer::f32("fel", z)]
        },
    );
    let d8 = stage(
        "d8flowdir via tiles",
        |f| {
            let (p, sd8) = reference::d8(f);
            vec![Layer::u8("p", p), Layer::f32("sd8", sd8)]
        },
        |f, spec| {
            let fel = through_tiles(spec, &reference::fel(f), FEL_NODATA);
            let flow = d8_flow(f.grid(), &fel, FEL_NODATA, &[f.spacing], true);
            vec![
                Layer::u8("p", flow.direction),
                Layer::f32("sd8", flow.slope.unwrap()),
            ]
        },
    );
    let area = stage(
        "aread8 via tiles",
        |f| vec![Layer::f32("ad8", reference::ad8(f))],
        |f, spec| {
            let p = through_tiles(spec, &reference::directions(f), DIRECTION_NODATA);
            vec![Layer::f32(
                "ad8",
                area_d8(f.grid(), &p, f.check_contamination),
            )]
        },
    )
    .applies_to(|_| true);
    for s in [&pit as &dyn TiledStage, &d8, &area] {
        assert_suite(s, &TILE_SIZES, &THREADS);
    }
}

/// Runs an in-memory kernel on each tile alone (with `radius` halo cells
/// read from the raster) and keeps the core: what a tiled stage that
/// ignored cross-tile dependencies would produce.
fn tile_local<T: Copy + Send + Sync>(
    spec: &TileSpec,
    values: &[T],
    fill: T,
    radius: usize,
    kernel: impl Fn(Grid, Vec<T>) -> Vec<T> + Sync,
) -> Vec<T> {
    let width = spec.raster_width();
    let tiles: Vec<(TileId, Vec<T>)> = spec
        .row_major()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|id| {
            let halo = spec.halo(id, radius);
            let w = halo.window;
            let data: Vec<T> = (w.y..w.bottom())
                .flat_map(|y| {
                    values[y * width + w.x..y * width + w.right()]
                        .iter()
                        .copied()
                })
                .collect();
            let mut buffer = Vec::new();
            halo.pad_into(&data, fill, &mut buffer).unwrap();
            let grid = Grid::new(halo.buffer_width(), halo.buffer_height()).unwrap();
            let result = kernel(grid, buffer);
            let mut core = Vec::new();
            halo.core_from_buffer(&result, &mut core).unwrap();
            (id, core)
        })
        .collect();
    let mut out = vec![fill; values.len()];
    for (id, core) in tiles {
        let r = spec.rect(id);
        for (row, chunk) in core.chunks_exact(r.width).enumerate() {
            let start = (r.y + row) * width + r.x;
            out[start..start + r.width].copy_from_slice(chunk);
        }
    }
    out
}

#[test]
fn harness_rejects_tile_local_stages_on_each_category() {
    let pit = stage(
        "tile-local pitremove",
        |f| vec![Layer::f32("fel", reference::fel(f))],
        |f, spec| {
            // Each tile floods alone: its edges become outlets.
            let filled = tile_local(spec, f.dem_values().unwrap(), f.nodata, 0, |grid, mut z| {
                pit_remove(grid, &mut z, f.nodata, f.four_way, None);
                z
            });
            vec![Layer::f32("fel", filled)]
        },
    );
    let d8 = stage(
        "tile-local d8flowdir",
        |f| vec![Layer::u8("p", reference::d8(f).0)],
        |f, spec| {
            let fel = reference::fel(f);
            let bits: Vec<u32> = fel.iter().map(|v| v.to_bits()).collect();
            let p = tile_local(spec, &bits, FEL_NODATA.to_bits(), 1, |grid, z| {
                let z: Vec<f32> = z.into_iter().map(f32::from_bits).collect();
                let flow = d8_flow(grid, &z, FEL_NODATA, &[f.spacing], false);
                flow.direction.into_iter().map(u32::from).collect()
            });
            vec![Layer::u8("p", p.into_iter().map(|v| v as u8).collect())]
        },
    );
    let area = stage(
        "tile-local aread8",
        |f| vec![Layer::f32("ad8", reference::ad8(f))],
        |f, spec| {
            let p: Vec<u32> = reference::directions(f)
                .into_iter()
                .map(u32::from)
                .collect();
            let bits = tile_local(spec, &p, u32::from(DIRECTION_NODATA), 0, |grid, p| {
                let p: Vec<u8> = p.into_iter().map(|v| v as u8).collect();
                area_d8(grid, &p, f.check_contamination)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect()
            });
            vec![Layer::f32(
                "ad8",
                bits.into_iter().map(f32::from_bits).collect(),
            )]
        },
    )
    .applies_to(|_| true);

    let cases: [(&dyn TiledStage, &[Category]); 3] = [
        (&pit, &[Category::CrossSeamPits, Category::NodataAtSeams]),
        (&d8, &[Category::DontCross, Category::MultiTileFlats]),
        (&area, &[Category::RiversAndQuirks]),
    ];
    for t in TILE_SIZES {
        let fixtures = suite(t);
        for (s, categories) in cases {
            for &category in categories {
                let caught: Vec<&str> = fixtures
                    .iter()
                    .filter(|f| f.category == category && s.applies(f))
                    .filter(|f| compare(s, f, t, 8).is_err())
                    .map(|f| f.name.as_str())
                    .collect();
                assert!(
                    !caught.is_empty(),
                    "{} passes every {category:?} fixture at T={t}",
                    s.name()
                );
            }
        }
    }
}

/// The large AreaD8 ramp's mainstem passes 2^24 cells, where in-memory f32
/// accumulation rounds: a tiled AreaD8 summing perimeter inflows exactly
/// (u64) would differ from it there.
#[test]
#[ignore = "17.6 M cells: cargo test --release --test adversarial -- --ignored"]
fn large_ramp_passes_the_f32_integer_limit() {
    for f in large_suite() {
        let area = reference::ad8(&f);
        let bottom = (f.height - 3) * f.width + 2;
        let exact = ((f.width - 4) * (f.height - 4)) as f64;
        println!(
            "{}: mainstem outlet {} (exact {exact})",
            f.name, area[bottom]
        );
        assert!(f64::from(area[bottom]) > 16_777_216.0);
        assert_ne!(
            f64::from(area[bottom]),
            exact,
            "f32 accumulation should have rounded"
        );
    }
}
