// SPDX-License-Identifier: GPL-3.0-only
//! `aread8 --mode tiled`: AreaD8 from a GeoTIFF of D8 codes to a GeoTIFF of
//! areas through [`taudem_rs::area_tiled`], within `--max-memory`.

use std::time::{Duration, Instant};

use taudem_rs::SCALAR_NODATA;
use taudem_rs::area_tiled::{
    AreaOptions, AreaPlan, DirectionSource, TiledAreaOptions, TiledAreaStats, area_d8_tiled, plan,
};
use taudem_rs::tiling::{self, ExecutionOptions, HaloWindow, TileCache, TileId, WriterStats};

use crate::io::{Raster, Result, TileReader, tile_writer};

pub struct Outcome {
    pub cells: u64,
    pub open: Duration,
    pub compute: Duration,
    pub write: Duration,
    pub plan: AreaPlan,
    pub stats: TiledAreaStats,
    pub writer: WriterStats,
}

/// D8 codes read tile by tile from a GeoTIFF, converted as
/// `Raster::directions` converts whole rasters.
struct GeoTiffDirections<'a>(TileReader<'a>);

impl DirectionSource for GeoTiffDirections<'_> {
    fn read_halo(
        &self,
        _id: TileId,
        halo: &HaloWindow,
        buffer: &mut Vec<u8>,
    ) -> tiling::Result<()> {
        self.0.codes_halo(halo, buffer)
    }
}

pub fn run(
    p_path: &str,
    ad8_path: &str,
    area: AreaOptions,
    execution: &ExecutionOptions,
) -> Result<Outcome> {
    let start = Instant::now();
    let mut raster = Raster::open(p_path)?;
    let (width, height) = raster.dimensions();
    let threads = tiling::workers::threads();
    // Caller-buffer reads of non-byte codes convert from a native window
    // (2 bytes per cell for TauDEM's Int16 directions).
    let plan = plan(
        width,
        height,
        execution,
        raster.block_size(),
        threads,
        &area,
        2,
    )?;
    let writer = tile_writer::<f32>(
        ad8_path,
        &raster.header,
        &plan.spec,
        f64::from(SCALAR_NODATA),
        2 * threads,
    )
    .map_err(|error| format!("writing {ad8_path}: {error}"))?;
    let cache = TileCache::new(execution.cache_config(plan.cache_bytes));
    let source = GeoTiffDirections(raster.tile_reader()?);
    let open = start.elapsed();

    let start = Instant::now();
    let sink = writer.sink();
    let stats = area_d8_tiled(
        &plan.spec,
        &source,
        &TiledAreaOptions {
            area,
            workers: plan.workers,
            cache: Some(cache),
        },
        &|_, rect, data| sink.submit_region(rect, data),
    )
    .map_err(|error| format!("aread8: {error}"))?;
    drop(sink);
    let compute = start.elapsed();
    let start = Instant::now();
    let writer = writer
        .finish()
        .map_err(|error| format!("writing {ad8_path}: {error}"))?;
    Ok(Outcome {
        cells: width as u64 * height as u64,
        open,
        compute,
        write: start.elapsed(),
        plan,
        stats,
        writer,
    })
}
