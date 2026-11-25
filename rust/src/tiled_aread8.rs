// SPDX-License-Identifier: GPL-3.0-only
//! `aread8 --mode tiled`: AreaD8 from a GeoTIFF of D8 codes to a GeoTIFF of
//! areas through [`taudem_rs::area_tiled`], within `--max-memory`.

use std::cell::RefCell;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use taudem_rs::area_tiled::{
    AreaOptions, AreaPlan, DirectionSource, TiledAreaOptions, TiledAreaStats, area_d8_tiled,
    in_memory_bytes, plan,
};
use taudem_rs::tiling::{
    self, AsyncTileWriter, ExecutionMode, ExecutionOptions, HaloWindow, TileCache, TileId,
    WriterStats,
};
use taudem_rs::{DIRECTION_NODATA, INVALID_DIRECTION, SCALAR_NODATA};
use terrio::raster::RasterData;
use terrio::{
    DataType, Error as TerrioError, LazyRaster, RasterDataset, RasterWriteOptions, WindowOptions,
    open_raster_lazy,
};

use crate::io::Result;

fn is_tiff(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff"))
}

/// Output blocks of every product, as the in-memory writer uses.
const BLOCK: u32 = 256;

/// Whether `aread8` runs tiled: always for `--mode tiled` or `--exact-area`,
/// and for `--mode auto` when the in-memory kernel would exceed the budget
/// or the raster exceeds its 32-bit cell index.
pub fn use_tiled(path: &str, execution: &ExecutionOptions, exact: bool) -> Result<bool> {
    match execution.mode {
        ExecutionMode::Tiled => Ok(true),
        ExecutionMode::InMemory if exact => {
            Err("--exact-area requires tiled execution (--mode tiled or auto)".into())
        }
        ExecutionMode::InMemory => Ok(false),
        ExecutionMode::Auto => {
            if exact {
                return Ok(true);
            }
            let raster =
                open_raster_lazy(path).map_err(|error| format!("opening {path}: {error}"))?;
            let cells = raster.width() as u64 * raster.height() as u64;
            Ok(cells > u64::from(u32::MAX)
                || in_memory_bytes(raster.width(), raster.height()) > execution.memory_budget())
        }
    }
}

pub struct Outcome {
    pub cells: u64,
    pub open: Duration,
    pub compute: Duration,
    pub write: Duration,
    pub plan: AreaPlan,
    pub stats: TiledAreaStats,
    pub writer: WriterStats,
}

thread_local! {
    static SAMPLES: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) };
}

/// D8 codes read tile by tile from a GeoTIFF, converted as `Raster::directions`
/// converts whole rasters.
struct GeoTiffDirections {
    path: String,
    raster: LazyRaster,
    nodata: f64,
    native_i16: bool,
    /// A second handle for owned window reads, when caller-buffer reads are
    /// unsupported (other sample types, some codecs).
    owned: Mutex<Option<LazyRaster>>,
}

fn code(z: f64, nodata: f64) -> u8 {
    if !z.is_finite() || z == nodata {
        DIRECTION_NODATA
    } else {
        match z.round() as i16 {
            k @ 0..=8 => k as u8,
            _ => INVALID_DIRECTION,
        }
    }
}

impl GeoTiffDirections {
    fn place(halo: &HaloWindow, buffer: &mut Vec<u8>, codes: impl Fn(usize) -> u8) {
        let window = halo.window;
        let bw = halo.buffer_width();
        buffer.clear();
        buffer.resize(halo.buffer_cells(), DIRECTION_NODATA);
        for row in 0..window.height {
            let dst = (row + halo.pad_top) * bw + halo.pad_left;
            for (col, cell) in buffer[dst..dst + window.width].iter_mut().enumerate() {
                *cell = codes(row * window.width + col);
            }
        }
    }

    fn read_owned(&self, halo: &HaloWindow, buffer: &mut Vec<u8>) -> tiling::Result<()> {
        let mut owned = self.owned.lock().unwrap_or_else(|p| p.into_inner());
        if owned.is_none() {
            *owned = Some(open_raster_lazy(&self.path)?);
        }
        let raster = owned.as_mut().expect("opened");
        let options = WindowOptions {
            bands: vec![1],
            ..Default::default()
        };
        let data: RasterData =
            raster.read_geotiff_samples(halo.window.terrio_window(), &options)?;
        drop(owned);
        let nodata = self.nodata;
        Self::place(halo, buffer, |i| code(data.get_f64(i), nodata));
        Ok(())
    }
}

impl DirectionSource for GeoTiffDirections {
    fn read_halo(
        &self,
        _id: TileId,
        halo: &HaloWindow,
        buffer: &mut Vec<u8>,
    ) -> tiling::Result<()> {
        if self.native_i16 {
            let read = SAMPLES.with(|samples| {
                let mut samples = samples.borrow_mut();
                samples.clear();
                samples.resize(halo.window.cells(), 0);
                match self.raster.read_geotiff_samples_into(
                    halo.window.terrio_window(),
                    1,
                    None,
                    &mut samples,
                ) {
                    Ok(()) => {
                        let nodata = self.nodata;
                        Self::place(halo, buffer, |i| {
                            let v = samples[i];
                            if f64::from(v) == nodata {
                                DIRECTION_NODATA
                            } else {
                                match v {
                                    0..=8 => v as u8,
                                    _ => INVALID_DIRECTION,
                                }
                            }
                        });
                        Ok(true)
                    }
                    Err(TerrioError::Unsupported(_)) => Ok(false),
                    Err(error) => Err(error),
                }
            })?;
            if read {
                return Ok(());
            }
        }
        self.read_owned(halo, buffer)
    }
}

/// The header of the area output, as the in-memory `aread8` writes it.
fn output_header(raster: &LazyRaster) -> RasterDataset {
    let mut header = raster.header();
    header.raster.bands = 1;
    header
        .raster
        .metadata
        .retain(|(key, _)| key == "AREA_OR_POINT");
    header.metadata_domains.clear();
    header.band_metadata.clear();
    header.color_table.clear();
    header.raster.data_type = DataType::F32;
    header.raster.data = RasterData::new_filled(DataType::F32, 0, 0.0);
    header.raster.nodata = f64::from(SCALAR_NODATA);
    header.has_nodata = true;
    header.band_nodata.clear();
    header
}

pub fn run(
    p_path: &str,
    ad8_path: &str,
    area: AreaOptions,
    execution: &ExecutionOptions,
) -> Result<Outcome> {
    let start = Instant::now();
    let raster = open_raster_lazy(p_path).map_err(|error| format!("opening {p_path}: {error}"))?;
    if raster.band_count() == 0 || raster.is_complex() {
        return Err(format!("{p_path} must have a real-valued raster band").into());
    }
    if !is_tiff(ad8_path) {
        return Err(
            format!("GeoTIFF output requires a .tif or .tiff extension: {ad8_path}").into(),
        );
    }
    let (width, height) = (raster.width(), raster.height());
    let threads = rayon::current_num_threads();
    let plan = plan(
        width,
        height,
        execution,
        raster.block_size(),
        threads,
        &area,
        2,
    )?;
    let header = output_header(&raster);
    let options = RasterWriteOptions {
        compression: Some("DEFLATE".into()),
        tile_size: Some(BLOCK),
        creation_options: vec![
            ("PREDICTOR".into(), "3".into()),
            ("ZLEVEL".into(), "1".into()),
        ],
        ..Default::default()
    };
    let writer = AsyncTileWriter::<f32>::create(ad8_path, &header, &options, 2 * threads)
        .map_err(|error| format!("writing {ad8_path}: {error}"))?;
    let source = GeoTiffDirections {
        path: p_path.to_owned(),
        nodata: raster.band_nodata(1).unwrap_or(-9999.0),
        native_i16: raster.band_data_type(1) == DataType::I16,
        raster,
        owned: Mutex::new(None),
    };
    let cache = TileCache::new(execution.cache_config(plan.cache_bytes));
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
