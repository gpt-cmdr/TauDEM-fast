// SPDX-License-Identifier: GPL-3.0-only

use std::{error::Error, path::Path};

use taudem_rs::{DIRECTION_NODATA, Grid, INVALID_DIRECTION};
use terrio::raster::RasterData;
use terrio::{
    DataType, GeoTiffWriter, LazyRaster, RasterDataset, RasterWriteOptions, Window, WindowOptions,
    open_raster_lazy,
};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub const P_NODATA: i16 = i16::MIN;
const STRIPE_BYTES: usize = 8 * 1024 * 1024;
const TILE: usize = 256;

pub struct Raster {
    input: LazyRaster,
    pub header: RasterDataset,
    pub grid: Grid,
}

impl Raster {
    pub fn open(path: &str) -> Result<Self> {
        let input = open_raster_lazy(path).map_err(|error| format!("opening {path}: {error}"))?;
        if input.band_count() == 0 || input.is_complex() {
            return Err(format!("{path} must have a real-valued raster band").into());
        }
        let grid = Grid::new(input.width(), input.height())
            .ok_or("raster exceeds the 32-bit cell index limit")?;
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
        Ok(Self {
            input,
            header,
            grid,
        })
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
        if self.grid != mask.grid
            || self.input.geo_transform() != mask.input.geo_transform()
            || self.header.raster.crs != mask.header.raster.crs
        {
            return Err("depression mask must share the DEM grid and CRS".into());
        }
        Ok(())
    }

    // Terrio returns owned, typed window buffers. Aligning reads to source
    // blocks bounds temporary pixels without decoding a block twice.
    fn read_stripes(&mut self, mut append: impl FnMut(RasterData)) -> Result<()> {
        let width = self.input.width();
        let height = self.input.height();
        let block = self.input.block_size().map_or(1, |(_, rows)| rows);
        let row_bytes = width * self.input.data_type().size_bytes();
        let target_rows = (STRIPE_BYTES / row_bytes / block).max(1) * block;
        let options = WindowOptions {
            bands: vec![1],
            ..Default::default()
        };
        for y in (0..height).step_by(target_rows) {
            let window = Window::new(0, y, width, (height - y).min(target_rows));
            append(self.input.read_geotiff_samples(window, &options)?);
        }
        Ok(())
    }

    pub fn elevations(&mut self) -> Result<Vec<f32>> {
        let cells = self.input.width() * self.input.height();
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
    Direction(&'a [u8]),
}

pub fn write(path: &str, header: &RasterDataset, product: Product<'_>) -> Result<()> {
    if !is_tiff(path) {
        return Err(format!("GeoTIFF output requires a .tif or .tiff extension: {path}").into());
    }
    let mut header = header.clone();
    let (data_type, nodata, predictor) = match product {
        Product::Float(_, nd) => (DataType::F32, f64::from(nd), "3"),
        Product::Direction(_) => (DataType::I16, f64::from(P_NODATA), "2"),
    };
    header.raster.data_type = data_type;
    header.raster.data = RasterData::new_filled(data_type, 0, 0.0);
    header.raster.nodata = nodata;
    header.has_nodata = true;
    header.band_nodata.clear();
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
    let mut stripe = RasterData::new_filled(data_type, 0, 0.0);
    for y in (0..rows).step_by(TILE) {
        let height = (rows - y).min(TILE);
        let range = y * width..(y + height) * width;
        match (&product, &mut stripe) {
            (Product::Float(values, _), RasterData::F32(target)) => {
                target.clear();
                target.extend_from_slice(&values[range]);
            }
            (Product::Direction(values), RasterData::I16(target)) => {
                target.clear();
                target.extend(values[range].iter().map(|&code| {
                    if code == DIRECTION_NODATA {
                        P_NODATA
                    } else {
                        i16::from(code)
                    }
                }));
            }
            _ => unreachable!(),
        }
        writer.write_rows(&stripe, height, None)?;
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
