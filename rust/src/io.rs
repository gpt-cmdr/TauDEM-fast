// SPDX-License-Identifier: GPL-3.0-only

use std::{error::Error, path::Path, sync::RwLock};

use taudem_rs::tiling::{self, AsyncTileWriter, HaloWindow, TileSpec};
use taudem_rs::{DIRECTION_NODATA, Grid, INVALID_DIRECTION};
use terrio::raster::RasterData;
use terrio::{
    DataType, Error as TerrioError, GeoTiffWriter, LazyRaster, RasterDataset, RasterElement,
    RasterWriteOptions, TiffReadLimits, TiffSample, Window, WindowOptions, open_raster_lazy,
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const P_NODATA: i16 = i16::MIN;
const STRIPE_BYTES: usize = 8 * 1024 * 1024;
const TILE: usize = 256;

pub struct Raster {
    input: LazyRaster,
    pub header: RasterDataset,
}

impl Raster {
    pub fn open(path: &str) -> Result<Self> {
        let input = open_raster_lazy(path).map_err(|error| format!("opening {path}: {error}"))?;
        if input.band_count() == 0 || input.is_complex() {
            return Err(format!("{path} must have a real-valued raster band").into());
        }
        let mut header = input.header();
        header.raster.bands = 1;
        // Derived products inherit spatial metadata, but not source band
        // statistics, scale/offsets, or the input's palette.
        header
            .raster
            .metadata
            .retain(|(key, _)| key == "AREA_OR_POINT");
        header.metadata_domains.clear();
        header.band_metadata.clear();
        header.color_table.clear();
        Ok(Self { input, header })
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.input.width(), self.input.height())
    }

    /// The in-memory grid; errors above the 32-bit cell index limit.
    pub fn grid(&self) -> Result<Grid> {
        let (width, height) = self.dimensions();
        Grid::new(width, height).ok_or_else(|| "raster exceeds the 32-bit cell index limit".into())
    }

    pub fn block_size(&self) -> Option<(usize, usize)> {
        self.input.block_size()
    }

    /// Concurrent reads of tile halos.
    pub fn halo_reader(&mut self) -> tiling::Result<HaloReader<'_>> {
        HaloReader::new(&mut self.input)
    }

    pub fn nodata(&self) -> f32 {
        self.input.band_nodata(1).unwrap_or(-9999.0) as f32
    }

    pub fn spacing(&self) -> Result<Vec<(f64, f64)>> {
        let [_, dx, _, top, _, dy] = self.input.geo_transform();
        let (dx, dy) = (dx.abs(), dy.abs());
        if !dx.is_finite() || !dy.is_finite() || dx == 0.0 || dy == 0.0 {
            return Err("D8 requires nonzero, finite x and y cell sizes".into());
        }
        if !self.input.crs().is_geographic() {
            return Ok(vec![(dx, dy)]);
        }
        // TauDEM uses the WGS84 ellipsoid and computes distances at a
        // float32 latitude per row (tiffIO::geotoLength).
        const A: f64 = 6_378_137.0;
        const B: f64 = 6_356_752.314;
        #[allow(clippy::approx_constant)]
        const PI: f64 = 3.14159265359;
        let bottom = top - self.input.height() as f64 * dy - dy / 2.0;
        let dx = dx * PI / 180.0;
        let dlat = dy * PI / 180.0;
        let mut spacing = Vec::with_capacity(self.input.height());
        for row in 0..self.input.height() {
            let latitude = (bottom + (self.input.height() - row - 1) as f64 * dy) as f32;
            let latitude = f64::from(latitude) * PI / 180.0;
            let beta = (B / A * latitude.tan()).atan();
            let cos = beta.cos();
            let ratio = cos / latitude.cos();
            let dbeta = dlat * (B / A) * ratio * ratio;
            let dy = (((A * beta.sin()).powi(2) + (B * cos).powi(2)) * dbeta.powi(2)).sqrt();
            spacing.push((A * cos * dx, dy));
        }
        Ok(spacing)
    }

    pub fn check_mask(&self, mask: &Self) -> Result<()> {
        if self.dimensions() != mask.dimensions()
            || self.input.geo_transform() != mask.input.geo_transform()
            || self.header.raster.crs != mask.header.raster.crs
        {
            return Err("depression mask must share the DEM grid and CRS".into());
        }
        Ok(())
    }

    // Conversions and unsupported direct reads use native window buffers.
    // Keep windows block-aligned except when Terrio needs to read a large strip by rows.
    fn read_stripes(&mut self, mut append: impl FnMut(RasterData)) -> Result<()> {
        let width = self.input.width();
        let height = self.input.height();
        let (block_width, block) = self.input.block_size().unwrap_or((width, 1));
        let row_bytes = width * self.input.data_type().size_bytes();
        let target_rows = (STRIPE_BYTES / row_bytes / block).max(1) * block;
        let limit = TiffReadLimits::from_config()?.chunk_bytes;
        let strip_rows = limit / row_bytes / self.input.band_count();
        let options = WindowOptions {
            bands: vec![1],
            ..Default::default()
        };
        let mut y = 0;
        while y < height {
            let mut rows = (height - y).min(target_rows);
            if block_width == width && (1..block.min(height)).contains(&strip_rows) {
                rows = rows.min(strip_rows).min(block - y % block);
            }
            let window = Window::new(0, y, width, rows);
            append(self.input.read_geotiff_samples(window, &options)?);
            y += rows;
        }
        Ok(())
    }

    pub fn elevations(&mut self) -> Result<Vec<f32>> {
        let (width, height) = (self.input.width(), self.input.height());
        let cells = width * height;
        if self.input.band_data_type(1) == DataType::F32 {
            let mut elevations = vec![0.0f32; cells];
            match self.input.read_geotiff_samples_into(
                Window::new(0, 0, width, height),
                1,
                None,
                &mut elevations,
            ) {
                Ok(()) => return Ok(elevations),
                Err(TerrioError::Unsupported(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }

        let mut elevations = Vec::new();
        self.read_stripes(|stripe| match stripe {
            RasterData::F32(data) if data.len() == cells => elevations = data,
            RasterData::F32(data) => {
                elevations.reserve_exact(cells - elevations.len());
                elevations.extend(data);
            }
            other => {
                elevations.reserve_exact(cells - elevations.len());
                elevations.extend((0..other.len()).map(|i| other.get_f64(i) as f32));
            }
        })?;
        Ok(elevations)
    }

    pub fn directions(&mut self) -> Result<Vec<u8>> {
        let nodata = self.input.band_nodata(1).unwrap_or(-9999.0);
        let convert = |z: f64| {
            if !z.is_finite() || z == nodata {
                DIRECTION_NODATA
            } else {
                match z.round() as i16 {
                    k @ 0..=8 => k as u8,
                    _ => INVALID_DIRECTION,
                }
            }
        };
        let mut directions = Vec::with_capacity(self.input.width() * self.input.height());
        self.read_stripes(|stripe| match stripe {
            RasterData::I16(data) => {
                directions.extend(data.into_iter().map(|v| convert(f64::from(v))))
            }
            other => directions.extend((0..other.len()).map(|i| convert(other.get_f64(i)))),
        })?;
        Ok(directions)
    }

    pub fn mask(&mut self) -> Result<Vec<u8>> {
        let mut mask = Vec::with_capacity(self.input.width() * self.input.height());
        self.read_stripes(|stripe| {
            mask.extend((0..stripe.len()).map(|i| u8::from(stripe.get_f64(i).round() == 1.0)));
        })?;
        Ok(mask)
    }
}

pub enum Product<'a> {
    Float(&'a [f32], f32),
    /// Float rows produced on demand: `fill(first_row, rows)` writes whole rows.
    FloatRows(&'a (dyn Fn(usize, &mut [f32]) + Sync), f32),
    Direction(&'a [u8]),
}

pub fn write(path: &str, header: &RasterDataset, product: Product<'_>) -> Result<()> {
    if !is_tiff(path) {
        return Err(format!("GeoTIFF output requires a .tif or .tiff extension: {path}").into());
    }
    let (data_type, nodata, predictor) = match product {
        Product::Float(_, nd) | Product::FloatRows(_, nd) => (DataType::F32, f64::from(nd), "3"),
        Product::Direction(_) => (DataType::I16, f64::from(P_NODATA), "2"),
    };
    let header = output_header(header, data_type, nodata);
    let width = header.raster.cols;
    let rows = header.raster.rows;
    let options = RasterWriteOptions {
        compression: Some("DEFLATE".into()),
        tile_size: Some(TILE as u32),
        creation_options: vec![
            ("PREDICTOR".into(), predictor.into()),
            ("ZLEVEL".into(), "1".into()),
        ],
        ..Default::default()
    };
    let mut writer = GeoTiffWriter::create(path, &header, &options)?;
    match product {
        Product::Float(values, _) => {
            for stripe in values.chunks(width * TILE) {
                writer.write_rows_slice(stripe, stripe.len() / width, None)?;
            }
        }
        Product::FloatRows(fill, _) => {
            let mut stripe = vec![0.0f32; width * rows.min(TILE)];
            let mut producing = std::time::Duration::ZERO;
            for first in (0..rows).step_by(TILE) {
                let count = (rows - first).min(TILE);
                let stripe = &mut stripe[..count * width];
                let start = std::time::Instant::now();
                fill(first, stripe);
                producing += start.elapsed();
                writer.write_rows_slice(stripe, count, None)?;
            }
            taudem_rs::profile::count("write.rows_produce_us", producing.as_micros() as u64);
        }
        Product::Direction(values) => {
            // The kernels store byte directions. Convert to TauDEM's Int16
            // file representation one block row at a time.
            let mut stripe = Vec::with_capacity(width * rows.min(TILE));
            let mut converting = std::time::Duration::ZERO;
            for directions in values.chunks(width * TILE) {
                let start = std::time::Instant::now();
                stripe.clear();
                stripe.extend(directions.iter().map(|&code| {
                    if code == DIRECTION_NODATA {
                        P_NODATA
                    } else {
                        i16::from(code)
                    }
                }));
                converting += start.elapsed();
                writer.write_rows_slice(&stripe, directions.len() / width, None)?;
            }
            taudem_rs::profile::count("write.p.convert_us", converting.as_micros() as u64);
        }
    }
    writer.finish()?;
    Ok(())
}

fn is_tiff(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff"))
}

fn output_header(header: &RasterDataset, data_type: DataType, nodata: f64) -> RasterDataset {
    let mut header = header.clone();
    header.raster.data_type = data_type;
    header.raster.data = RasterData::new_filled(data_type, 0, 0.0);
    header.raster.nodata = nodata;
    header.has_nodata = true;
    header.band_nodata.clear();
    header
}

/// The output block side for tiles of `spec`: 256, or the largest smaller
/// power of two the tiles are a multiple of, so that tiles fill whole blocks.
pub fn block_size_for(spec: &TileSpec) -> usize {
    let (w, h) = spec.tile_size();
    [256, 128, 64, 32, 16]
        .into_iter()
        .find(|&block| {
            (spec.tiles_x() == 1 || w.is_multiple_of(block))
                && (spec.tiles_y() == 1 || h.is_multiple_of(block))
        })
        .unwrap_or(TILE)
}

/// A tiled GeoTIFF written in the background, compressed as [`write`]
/// compresses and with all-nodata blocks left sparse.
pub fn tile_writer<T: RasterElement + Default>(
    path: &str,
    header: &RasterDataset,
    spec: &TileSpec,
    nodata: f64,
    queue_depth: usize,
) -> tiling::Result<AsyncTileWriter<T>> {
    if !is_tiff(path) {
        return Err(tiling::Error::Invalid(format!(
            "GeoTIFF output requires a .tif or .tiff extension: {path}"
        )));
    }
    let predictor = if T::DATA_TYPE == DataType::F32 {
        "3"
    } else {
        "2"
    };
    let header = output_header(header, T::DATA_TYPE, nodata);
    let options = RasterWriteOptions {
        compression: Some("DEFLATE".into()),
        tile_size: Some(block_size_for(spec) as u32),
        creation_options: vec![
            ("PREDICTOR".into(), predictor.into()),
            ("ZLEVEL".into(), "1".into()),
            ("SPARSE_OK".into(), "YES".into()),
        ],
        ..Default::default()
    };
    AsyncTileWriter::create(path, &header, &options, queue_depth)
}

/// Shared caller-buffer reads of tile halos. Layouts that Terrio reads only
/// into owned windows take an exclusive lock.
pub struct HaloReader<'a> {
    input: RwLock<&'a mut LazyRaster>,
    dtype: DataType,
    strip_rows: Option<(usize, usize)>,
}

impl<'a> HaloReader<'a> {
    fn new(input: &'a mut LazyRaster) -> tiling::Result<Self> {
        let dtype = input.band_data_type(1);
        let (block, strip_height) = input.block_size().unwrap_or((0, 0));
        let row_bytes = input.width() * input.data_type().size_bytes() * input.band_count();
        let max_rows = TiffReadLimits::from_config()?.chunk_bytes / row_bytes.max(1);
        let strip_rows = (block == input.width() && max_rows < strip_height.min(input.height()))
            .then_some((max_rows.max(1), strip_height));
        Ok(Self {
            input: RwLock::new(input),
            dtype,
            strip_rows,
        })
    }

    /// Fills a halo buffer with the window's elevations as `f32`, each at
    /// its place in the buffer; cells outside the raster are left as they are.
    pub fn halo(&self, halo: &HaloWindow, buffer: &mut [f32]) -> tiling::Result<()> {
        let window = halo.window;
        let width = halo.buffer_width();
        // Read the window to the front, then move rows back to their place.
        self.samples(window, &mut buffer[..window.cells()])?;
        for row in (0..window.height).rev() {
            let src = row * window.width;
            let dst = (row + halo.pad_top) * width + halo.pad_left;
            buffer.copy_within(src..src + window.width, dst);
        }
        Ok(())
    }

    fn samples(&self, rect: tiling::Rect, output: &mut [f32]) -> tiling::Result<()> {
        if self.strip_rows.is_none() {
            let input = self.input.read().map_err(|_| lock_error())?;
            let window = rect.terrio_window();
            let direct = match self.dtype {
                DataType::F32 => input.read_geotiff_samples_into(window, 1, None, output),
                DataType::F64 => typed::<f64>(&input, window, output),
                DataType::I16 => typed::<i16>(&input, window, output),
                DataType::U16 => typed::<u16>(&input, window, output),
                DataType::I32 => typed::<i32>(&input, window, output),
                DataType::U32 => typed::<u32>(&input, window, output),
                DataType::U8 => typed::<u8>(&input, window, output),
                DataType::I8 => typed::<i8>(&input, window, output),
                _ => Err(TerrioError::Unsupported("owned window".into())),
            };
            match direct {
                Ok(()) => return Ok(()),
                Err(TerrioError::Unsupported(_)) => {}
                Err(error) => return Err(error.into()),
            }
        }
        let mut input = self.input.write().map_err(|_| lock_error())?;
        let options = WindowOptions {
            bands: vec![1],
            ..Default::default()
        };
        let mut y = rect.y;
        while y < rect.bottom() {
            let mut rows = rect.bottom() - y;
            if let Some((max_rows, strip_height)) = self.strip_rows {
                rows = rows.min(max_rows).min(strip_height - y % strip_height);
            }
            let data =
                input.read_geotiff_samples(Window::new(rect.x, y, rect.width, rows), &options)?;
            let offset = (y - rect.y) * rect.width;
            for (i, out) in output[offset..offset + rows * rect.width]
                .iter_mut()
                .enumerate()
            {
                *out = data.get_f64(i) as f32;
            }
            y += rows;
        }
        Ok(())
    }
}

/// Reads a window of native samples and converts them to `f32`.
fn typed<T: TiffSample + Copy + Default + Into<f64>>(
    input: &LazyRaster,
    window: Window,
    output: &mut [f32],
) -> terrio::Result<()> {
    let mut native = vec![T::default(); output.len()];
    input.read_geotiff_samples_into(window, 1, None, &mut native)?;
    for (out, z) in output.iter_mut().zip(native) {
        *out = z.into() as f32;
    }
    Ok(())
}

fn lock_error() -> tiling::Error {
    tiling::Error::Invalid("the GeoTIFF window decoder lock was poisoned".into())
}
