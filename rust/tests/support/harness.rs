// SPDX-License-Identifier: GPL-3.0-only
//! The parity harness: a tiled stage must reproduce the in-memory stage bit
//! for bit on every fixture, tile size and thread count.
//!
//! A stage is a [`TiledStage`] (or closures through [`stage`]) producing
//! named output layers. WP-S1..S3 call, for example:
//!
//! ```ignore
//! let pit = stage("pitremove", |f| vec![Layer::f32("fel", in_memory_fill(f))],
//!                 |f, spec| vec![Layer::f32("fel", tiled_fill(f, spec))]);
//! assert_suite(&pit, &TILE_SIZES, &THREADS);
//! ```

use std::fmt::Write as _;

use taudem_rs::tiling::{TileId, TileSpec};
use taudem_rs::{FEL_NODATA, area_d8, d8_flow, pit_remove};

use super::fixtures::{Fixture, Input, suite};

/// One named output grid.
#[derive(Clone, Debug)]
pub enum Layer {
    F32(&'static str, Vec<f32>),
    U8(&'static str, Vec<u8>),
}

impl Layer {
    pub fn f32(name: &'static str, values: Vec<f32>) -> Self {
        Self::F32(name, values)
    }

    pub fn u8(name: &'static str, values: Vec<u8>) -> Self {
        Self::U8(name, values)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32(n, _) | Self::U8(n, _) => n,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::F32(_, v) => v.len(),
            Self::U8(_, v) => v.len(),
        }
    }

    /// Whether cell `i` is identical; floats compare by bits.
    fn same(&self, other: &Self, i: usize) -> bool {
        match (self, other) {
            (Self::F32(_, a), Self::F32(_, b)) => a[i].to_bits() == b[i].to_bits(),
            (Self::U8(_, a), Self::U8(_, b)) => a[i] == b[i],
            _ => false,
        }
    }

    fn show(&self, i: usize) -> String {
        match self {
            Self::F32(_, v) => format!("{:?}", v[i]),
            Self::U8(_, v) => v[i].to_string(),
        }
    }
}

/// A stage under test: its in-memory reference and its tiled version.
pub trait TiledStage: Sync {
    fn name(&self) -> &str;

    /// Whether the stage runs on this fixture (its input kind and options).
    fn applies(&self, fixture: &Fixture) -> bool {
        fixture.dem_values().is_some()
    }

    fn in_memory(&self, fixture: &Fixture) -> Vec<Layer>;

    /// Runs on the current Rayon pool, which the harness sizes.
    fn tiled(&self, fixture: &Fixture, spec: &TileSpec) -> Vec<Layer>;
}

/// A [`TiledStage`] from two closures; runs on DEM fixtures.
pub struct ClosureStage<M, T> {
    name: String,
    in_memory: M,
    tiled: T,
    applies: fn(&Fixture) -> bool,
}

pub fn stage<M, T>(name: &str, in_memory: M, tiled: T) -> ClosureStage<M, T>
where
    M: Fn(&Fixture) -> Vec<Layer> + Sync,
    T: Fn(&Fixture, &TileSpec) -> Vec<Layer> + Sync,
{
    ClosureStage {
        name: name.into(),
        in_memory,
        tiled,
        applies: |f| f.dem_values().is_some(),
    }
}

impl<M, T> ClosureStage<M, T> {
    /// Runs on the fixtures `applies` accepts instead of DEM fixtures only.
    pub fn applies_to(mut self, applies: fn(&Fixture) -> bool) -> Self {
        self.applies = applies;
        self
    }
}

impl<M, T> TiledStage for ClosureStage<M, T>
where
    M: Fn(&Fixture) -> Vec<Layer> + Sync,
    T: Fn(&Fixture, &TileSpec) -> Vec<Layer> + Sync,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn applies(&self, fixture: &Fixture) -> bool {
        (self.applies)(fixture)
    }

    fn in_memory(&self, fixture: &Fixture) -> Vec<Layer> {
        (self.in_memory)(fixture)
    }

    fn tiled(&self, fixture: &Fixture, spec: &TileSpec) -> Vec<Layer> {
        (self.tiled)(fixture, spec)
    }
}

/// Runs the tiled stage with square tiles of `tile_size` on a pool of
/// `threads` and compares it with the in-memory stage; `Err` describes the
/// differences: their count per layer, and the first cells with their
/// tile, tile-local position and distance to the nearest seam.
pub fn compare(
    stage: &dyn TiledStage,
    fixture: &Fixture,
    tile_size: usize,
    threads: usize,
) -> Result<(), String> {
    let spec =
        TileSpec::new(fixture.width, fixture.height, tile_size, tile_size).expect("tile spec");
    let expected = stage.in_memory(fixture);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool");
    let actual = pool.install(|| stage.tiled(fixture, &spec));
    let context = format!(
        "{} on {} ({}x{}), tiles {tile_size} ({}x{}), {threads} threads",
        stage.name(),
        fixture.name,
        fixture.width,
        fixture.height,
        spec.tiles_x(),
        spec.tiles_y()
    );
    let names = |layers: &[Layer]| layers.iter().map(Layer::name).collect::<Vec<_>>();
    if names(&expected) != names(&actual) {
        return Err(format!(
            "{context}: layers {:?}, expected {:?}",
            names(&actual),
            names(&expected)
        ));
    }
    let mut report = String::new();
    for (e, a) in expected.iter().zip(&actual) {
        if a.len() != e.len() || e.len() != fixture.cells() {
            writeln!(
                report,
                "{context}: layer {} has {} cells, expected {}",
                e.name(),
                a.len(),
                e.len()
            )
            .unwrap();
            continue;
        }
        let differing: Vec<usize> = (0..e.len()).filter(|&i| !e.same(a, i)).collect();
        if differing.is_empty() {
            continue;
        }
        let (tw, th) = spec.tile_size();
        let seam_distance = |x: usize, y: usize| {
            let dx = (x % tw).min(tw - 1 - x % tw);
            let dy = (y % th).min(th - 1 - y % th);
            dx.min(dy)
        };
        let near_seams = differing
            .iter()
            .filter(|&&i| seam_distance(i % fixture.width, i / fixture.width) == 0)
            .count();
        writeln!(
            report,
            "{context}: layer {}: {} of {} cells differ ({near_seams} on tile edges)",
            e.name(),
            differing.len(),
            e.len()
        )
        .unwrap();
        for &i in differing.iter().take(12) {
            let (x, y) = (i % fixture.width, i / fixture.width);
            let (TileId { tx, ty }, local) = spec.locate(x, y);
            let rect = spec.rect(TileId::new(tx, ty));
            writeln!(
                report,
                "  ({x}, {y}) tile ({tx}, {ty}) local ({}, {}) [index {local}], {} from a seam: expected {}, got {}",
                x - rect.x,
                y - rect.y,
                seam_distance(x, y),
                e.show(i),
                a.show(i)
            )
            .unwrap();
        }
    }
    if report.is_empty() {
        Ok(())
    } else {
        Err(report)
    }
}

/// Panics unless the tiled stage matches the in-memory stage on `fixture`.
pub fn assert_tiled_matches_in_memory(
    stage: &dyn TiledStage,
    fixture: &Fixture,
    tile_size: usize,
    threads: usize,
) {
    if let Err(report) = compare(stage, fixture, tile_size, threads) {
        panic!("tiled result differs from in-memory:\n{report}");
    }
}

/// Runs [`compare`] on every applicable fixture of [`suite`] for each tile
/// size and thread count, and panics with every failure.
pub fn assert_suite(stage: &dyn TiledStage, tile_sizes: &[usize], threads: &[usize]) {
    let mut failures = Vec::new();
    let mut runs = 0;
    for &t in tile_sizes {
        for fixture in suite(t).iter().filter(|f| stage.applies(f)) {
            for &n in threads {
                runs += 1;
                if let Err(report) = compare(stage, fixture, t, n) {
                    failures.push(report);
                }
            }
        }
    }
    assert!(runs > 0, "{} applies to no fixture", stage.name());
    assert!(
        failures.is_empty(),
        "{} of {runs} runs of {} differ:\n{}",
        failures.len(),
        stage.name(),
        failures.join("\n")
    );
}

/// The in-memory stages, for references and for tiled stages that reuse a
/// prefix of the pipeline (AreaD8 on a DEM fixture needs its directions).
pub mod reference {
    use super::*;

    /// PitRemove of a DEM fixture.
    pub fn fel(f: &Fixture) -> Vec<f32> {
        let mut z = f.dem_values().expect("a DEM fixture").to_vec();
        pit_remove(f.grid(), &mut z, f.nodata, f.four_way, f.mask.as_deref());
        z
    }

    /// D8 directions and slope of a DEM fixture's filled surface.
    pub fn d8(f: &Fixture) -> (Vec<u8>, Vec<f32>) {
        let flow = d8_flow(f.grid(), &fel(f), FEL_NODATA, &[f.spacing], true);
        (flow.direction, flow.slope.expect("slope"))
    }

    /// The D8 codes AreaD8 reads: the fixture's own, or D8 of its DEM.
    pub fn directions(f: &Fixture) -> Vec<u8> {
        match &f.input {
            Input::Directions(p) => p.clone(),
            Input::Dem(_) => d8(f).0,
        }
    }

    /// AreaD8 of a fixture.
    pub fn ad8(f: &Fixture) -> Vec<f32> {
        area_d8(f.grid(), &directions(f), f.check_contamination)
    }
}
