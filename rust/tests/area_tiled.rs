// SPDX-License-Identifier: GPL-3.0-only
//! Tiled AreaD8 (WP-S3) must reproduce the in-memory `area_d8` bit for bit:
//! on the §6.1 adversarial suite at T = 16, 17, 32 and 1 or 8 threads, on
//! random D8 grids with tiny tiles, through a spilling tile store, with the
//! f32 replay forced on every exit, and (release, `--ignored`) on the
//! 17.6 M-cell ramp whose mainstem passes 2^24.

mod support;

use std::sync::Mutex;

use support::fixtures::{Category, Fixture, Input, NODATA, THREADS, TILE_SIZES, large_suite};
use support::harness::{Layer, assert_suite, compare, reference, stage};
use taudem_rs::area_tiled::{
    AreaOptions, AreaScratch, AreaTiles, Arithmetic, GridDirections, StoredDirections,
    TiledAreaOptions, TiledAreaStats, area_d8_tiled,
};
use taudem_rs::tiling::{
    BorderTable, CacheConfig, Rect, StoreOptions, Tier, TileCache, TileId, TileSpec,
};
use taudem_rs::{DIRECTION_NODATA, SCALAR_NODATA, area_d8};

/// Runs tiled AreaD8 on a fixture's codes and gathers the tiles' areas.
fn run_tiled(
    f: &Fixture,
    spec: &TileSpec,
    area: AreaOptions,
    cache: Option<TileCache>,
) -> (Vec<f32>, TiledAreaStats) {
    let p = reference::directions(f);
    let source = GridDirections::new(f.width, f.height, &p).unwrap();
    let out = Mutex::new(vec![f32::NAN; f.cells()]);
    let width = f.width;
    let sink = |_id: TileId, rect: Rect, data: &[f32]| {
        let mut out = out.lock().unwrap();
        for row in 0..rect.height {
            let to = (rect.y + row) * width + rect.x;
            out[to..to + rect.width].copy_from_slice(&data[row * rect.width..][..rect.width]);
        }
        Ok(())
    };
    let options = TiledAreaOptions {
        area,
        workers: 0,
        cache,
    };
    let stats = area_d8_tiled(spec, &source, &options, &sink).expect("tiled aread8");
    (out.into_inner().unwrap(), stats)
}

fn options(f: &Fixture) -> AreaOptions {
    AreaOptions::new(f.check_contamination)
}

fn ad8_stage(
    name: &str,
    configure: fn(&Fixture) -> (AreaOptions, Option<TileCache>),
) -> impl support::harness::TiledStage {
    stage(
        name,
        |f| vec![Layer::f32("ad8", reference::ad8(f))],
        move |f, spec| {
            let (area, cache) = configure(f);
            vec![Layer::f32("ad8", run_tiled(f, spec, area, cache).0)]
        },
    )
    .applies_to(|_| true)
}

fn spilling_cache() -> TileCache {
    // Far below one tile: every tile is compressed and spilled at once.
    TileCache::new(CacheConfig::new(1024, Some(std::env::temp_dir())))
}

#[test]
fn tiled_area_matches_in_memory_on_the_adversarial_suite() {
    assert_suite(
        &ad8_stage("tiled aread8", |f| (options(f), None)),
        &TILE_SIZES,
        &THREADS,
    );
}

#[test]
fn tiled_area_matches_through_a_spilling_tile_store() {
    assert_suite(
        &ad8_stage("tiled aread8 via a spilling tile store", |f| {
            (options(f), Some(spilling_cache()))
        }),
        &TILE_SIZES,
        &THREADS,
    );
}

#[test]
fn tiled_area_matches_with_every_exit_replayed() {
    // Replaying every exit with an inflow exercises the f32 programs
    // (constants, slots, external inflows, river cycles in the tile graph)
    // on rasters far below 2^24 cells.
    for above in [0, 7] {
        let stage = stage(
            "tiled aread8 replaying exits",
            |f| vec![Layer::f32("ad8", reference::ad8(f))],
            move |f, spec| {
                let area = AreaOptions {
                    replay_above: above,
                    ..options(f)
                };
                vec![Layer::f32("ad8", run_tiled(f, spec, area, None).0)]
            },
        )
        .applies_to(|_| true);
        assert_suite(&stage, &TILE_SIZES, &THREADS);
    }
}

#[test]
fn the_meandering_river_cycles_through_the_tile_graph() {
    for t in TILE_SIZES {
        let fixtures = support::fixtures::suite(t);
        let river = fixtures
            .iter()
            .find(|f| f.name == "c5-meandering-river-tile-cycle")
            .unwrap();
        let spec = TileSpec::new(river.width, river.height, t, t).unwrap();
        let (_, stats) = run_tiled(river, &spec, options(river), None);
        assert!(stats.graph.edges > 0 && stats.graph.cycle_nodes == 0);
        // Exits chained through other tiles: some exit's path passes another
        // exit's tile at least twice (A -> B -> C -> A -> B).
        assert!(stats.graph.inflow_nodes > 4, "{:?}", stats.graph);
        let everything = AreaOptions {
            replay_above: 0,
            ..options(river)
        };
        let (_, stats) = run_tiled(river, &spec, everything, None);
        assert!(stats.graph.replay_nodes > 4 && stats.graph.replay_tiles > 4);
    }
}

/// A small deterministic generator.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn directions_fixture(name: String, width: usize, p: Vec<u8>, check: bool) -> Fixture {
    Fixture {
        name,
        category: Category::RiversAndQuirks,
        width,
        height: p.len() / width,
        input: Input::Directions(p),
        nodata: NODATA,
        mask: None,
        four_way: false,
        check_contamination: check,
        spacing: (1.0, 1.0),
    }
}

/// Random code grids: uniform codes (cycles, code-0 NW neighbors, nodata
/// and invalid codes everywhere), and drainage-like fields that mostly
/// follow one direction so that long paths cross many tiles.
fn random_fixtures(seed: u64) -> Vec<Fixture> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut out = Vec::new();
    for case in 0..6 {
        let w = 1 + rng.below(40) as usize;
        let h = 1 + rng.below(40) as usize;
        let mut p = vec![0u8; w * h];
        let main = 1 + rng.below(8) as u8;
        for (i, code) in p.iter_mut().enumerate() {
            let r = rng.below(100);
            *code = match case % 3 {
                0 => match r {
                    0..=4 => DIRECTION_NODATA,
                    5 => 254,
                    6..=10 => 0,
                    _ => 1 + rng.below(8) as u8,
                },
                1 => match r {
                    0 => DIRECTION_NODATA,
                    1 => 0,
                    2..=69 => main,
                    _ => 1 + rng.below(8) as u8,
                },
                // Flow toward the grid's left edge with jitter, nodata frame.
                _ => {
                    let (x, y) = (i % w, i / w);
                    if x == 0 || y == 0 || x + 1 == w || y + 1 == h {
                        DIRECTION_NODATA
                    } else {
                        [5, 4, 6, 5, 5, 3, 7][rng.below(7) as usize]
                    }
                }
            };
        }
        for check in [true, false] {
            out.push(directions_fixture(
                format!(
                    "random-{seed}-{case}-{w}x{h}{}",
                    if check { "" } else { "-nc" }
                ),
                w,
                p.clone(),
                check,
            ));
        }
    }
    out
}

#[test]
fn tiled_area_matches_on_random_grids_with_tiny_tiles() {
    let plain = ad8_stage("tiled aread8", |f| (options(f), None));
    let replayed = ad8_stage("tiled aread8 replaying everything", |f| {
        (
            AreaOptions {
                replay_above: 0,
                ..options(f)
            },
            None,
        )
    });
    let stored = ad8_stage("tiled aread8 via tiles", |f| {
        (options(f), Some(spilling_cache()))
    });
    let mut failures = Vec::new();
    for seed in 0..40 {
        for f in random_fixtures(seed) {
            for t in [1, 2, 3, 4, 5, 7, 16] {
                for threads in [1, 3] {
                    for s in [
                        &plain as &dyn support::harness::TiledStage,
                        &replayed,
                        &stored,
                    ] {
                        if let Err(report) = compare(s, &f, t, threads) {
                            failures.push(report);
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures[..failures.len().min(5)].join("\n")
    );
}

/// Below 2^24 cells TauDEM's f32 sums are exact, so exact counts agree.
fn exact_reference(f: &Fixture) -> Vec<f32> {
    reference::ad8(f)
}

#[test]
fn exact_arithmetic_agrees_below_two_to_the_24() {
    let stage = stage(
        "tiled aread8 exact",
        |f| vec![Layer::f32("ad8", exact_reference(f))],
        |f, spec| {
            let area = options(f).with_arithmetic(Arithmetic::Exact);
            vec![Layer::f32("ad8", run_tiled(f, spec, area, None).0)]
        },
    )
    .applies_to(|_| true);
    assert_suite(&stage, &TILE_SIZES, &THREADS);
}

#[test]
fn kernels_compose_for_a_fused_pipeline() {
    // WP-S4 hands D8's tiles over in a tile store and border table; the
    // kernels run on them directly, without any raster read.
    let fixtures = support::fixtures::suite(17);
    let f = fixtures
        .iter()
        .find(|f| f.name == "c5-corner-zero-blocks-diagonal-tile")
        .unwrap();
    let p = reference::directions(f);
    let spec = TileSpec::new(f.width, f.height, 17, 17).unwrap();
    let cache = TileCache::new(CacheConfig::new(64 << 10, None));
    let store = cache.store::<u8>(
        "dir",
        StoreOptions {
            prefilter: None,
            initial: Tier::Ram,
        },
    );
    let borders = BorderTable::new(spec, DIRECTION_NODATA);
    for id in spec.row_major() {
        let r = spec.rect(id);
        let tile: Vec<u8> = (r.y..r.bottom())
            .flat_map(|y| {
                p[y * f.width + r.x..y * f.width + r.right()]
                    .iter()
                    .copied()
            })
            .collect();
        borders.set_from_tile(id, &tile).unwrap();
        store.put_owned(id, tile).unwrap();
    }
    let source = StoredDirections::new(&store, &borders, false);
    let tiles = AreaTiles::new(spec, AreaOptions::new(f.check_contamination)).unwrap();
    let mut scratch = AreaScratch::new();
    let mut dir = Vec::new();
    use taudem_rs::area_tiled::DirectionSource;
    for id in spec.row_major() {
        source
            .read_halo(id, &spec.halo_window(id), &mut dir)
            .unwrap();
        tiles.pass1(id, &dir, &mut scratch).unwrap();
    }
    assert!(tiles.pass1(TileId::new(0, 0), &dir, &mut scratch).is_err());
    let solved = tiles.solve(&source, 1).unwrap();
    let expected = area_d8(f.grid(), &p, f.check_contamination);
    for id in spec.row_major() {
        source
            .read_halo(id, &spec.halo_window(id), &mut dir)
            .unwrap();
        let (area, _) = solved.pass2(id, &dir, &mut scratch).unwrap();
        let r = spec.rect(id);
        for (i, &a) in area.iter().enumerate() {
            let (x, y) = (r.x + i % r.width, r.y + i / r.width);
            assert_eq!(
                a.to_bits(),
                expected[y * f.width + x].to_bits(),
                "({x}, {y})"
            );
        }
    }
    assert!(expected.contains(&SCALAR_NODATA));
}

#[test]
fn one_tile_and_one_cell_rasters() {
    for (w, h) in [(1, 1), (1, 9), (9, 1), (3, 3)] {
        let p = vec![0u8; w * h];
        let f = directions_fixture(format!("zeros-{w}x{h}"), w, p, true);
        for t in [1, 2, 16] {
            compare(&ad8_stage("tiled aread8", |f| (options(f), None)), &f, t, 1).unwrap();
        }
    }
}

/// The 17.6 M-cell ramp: in memory its mainstem rounds past 2^24; tiled
/// AreaD8 replays the mainstem in f32 and matches bit for bit.
#[test]
#[ignore = "17.6 M cells: cargo test --release --test area_tiled -- --ignored"]
fn large_ramp_is_bit_exact_past_two_to_the_24() {
    let plain = ad8_stage("tiled aread8", |f| (options(f), None));
    let mut failures = Vec::new();
    for f in large_suite() {
        for t in TILE_SIZES {
            for threads in THREADS {
                let start = std::time::Instant::now();
                if let Err(report) = compare(&plain, &f, t, threads) {
                    failures.push(report);
                }
                println!(
                    "{} T={t} threads={threads}: {:.1} s",
                    f.name,
                    start.elapsed().as_secs_f64()
                );
            }
        }
        // The replay is what makes it exact: without it the outlet differs.
        let spec = TileSpec::new(f.width, f.height, 32, 32).unwrap();
        let (tiled, stats) = run_tiled(&f, &spec, options(&f), None);
        println!("{}: {:?}", f.name, stats.graph);
        assert!(stats.graph.replay_nodes > 0 && stats.graph.max_area > 1 << 24);
        let outlet = (f.height - 3) * f.width + 2;
        let exact = ((f.width - 4) * (f.height - 4)) as f32;
        assert_ne!(tiled[outlet], exact);
        let (exact_run, _) = run_tiled(
            &f,
            &spec,
            options(&f).with_arithmetic(Arithmetic::Exact),
            None,
        );
        assert_eq!(exact_run[outlet], exact);
        println!(
            "{}: outlet {} (TauDEM f32) vs {} (exact)",
            f.name, tiled[outlet], exact_run[outlet]
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
