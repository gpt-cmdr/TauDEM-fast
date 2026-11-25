// SPDX-License-Identifier: GPL-3.0-only
//! The background GeoTIFF writer: compute tiles of any size submitted from
//! many threads in any order read back exactly, through terrio here and
//! through GDAL with `tests/gdal_readback.py`.
//!
//! Set `TAUDEM_GDAL_CHECK_DIR` to keep each GeoTIFF with a `.bin` of its
//! expected pixels in that directory, then run
//! `python tests/gdal_readback.py <dir>` with GDAL's Python bindings.

use std::path::PathBuf;

use rayon::prelude::*;
use taudem_rs::tiling::{AsyncTileWriter, Rect, TileElement, TileId, TileSpec, WriterStats};
use terrio::raster::RasterData;
use terrio::{
    DataType, Raster, RasterConfig, RasterDataset, RasterElement, RasterWriteOptions, TiffSample,
    Window,
};

fn header(cols: usize, rows: usize, data_type: DataType, nodata: f64) -> RasterDataset {
    let raster = Raster::new(RasterConfig {
        cols,
        rows,
        bands: 1,
        x_min: 500_000.0,
        y_min: 4_000_000.0,
        cell_size: 10.0,
        nodata,
        data_type,
        ..Default::default()
    });
    let mut header = RasterDataset::new(raster, true);
    header.raster.data = RasterData::new_filled(data_type, 0, 0.0);
    header
}

fn output_dir() -> (PathBuf, bool) {
    match std::env::var_os("TAUDEM_GDAL_CHECK_DIR") {
        Some(dir) => (PathBuf::from(dir), true),
        None => (std::env::temp_dir().join("taudem-rs-async-writer"), false),
    }
}

struct Case<'a, T> {
    name: &'a str,
    cols: usize,
    rows: usize,
    block: u32,
    compute_tile: usize,
    data_type: DataType,
    nodata: f64,
    predictor: &'a str,
    sparse: bool,
    queue: usize,
    values: Vec<T>,
}

fn roundtrip<T>(case: Case<'_, T>) -> WriterStats
where
    T: RasterElement + TiffSample + TileElement + std::fmt::Debug,
{
    let (dir, keep) = output_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{}.tif", case.name));
    let path = path.to_str().unwrap();
    let mut creation_options = vec![
        ("PREDICTOR".to_string(), case.predictor.to_string()),
        ("ZLEVEL".to_string(), "1".to_string()),
    ];
    if case.sparse {
        creation_options.push(("SPARSE_OK".into(), "YES".into()));
    }
    let options = RasterWriteOptions {
        compression: Some("DEFLATE".into()),
        tile_size: Some(case.block),
        creation_options,
        ..Default::default()
    };
    let writer = AsyncTileWriter::<T>::create(
        path,
        &header(case.cols, case.rows, case.data_type, case.nodata),
        &options,
        case.queue,
    )
    .unwrap();
    let spec = TileSpec::new(case.cols, case.rows, case.compute_tile, case.compute_tile).unwrap();
    // Submit compute tiles in reverse Morton order from 8 threads.
    let mut ids: Vec<TileId> = spec.morton_order();
    ids.reverse();
    let sink = writer.sink();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        ids.par_iter().for_each(|&id| {
            let r = spec.rect(id);
            let tile: Vec<T> = (r.y..r.bottom())
                .flat_map(|y| {
                    case.values[y * case.cols + r.x..y * case.cols + r.right()]
                        .iter()
                        .copied()
                })
                .collect();
            sink.submit_region(Rect::new(r.x, r.y, r.width, r.height), &tile)
                .unwrap();
        })
    });
    drop(sink);
    let stats = writer.finish().unwrap();

    let raster = terrio::open_raster_lazy(path).unwrap();
    assert_eq!(
        raster.block_size(),
        Some((case.block as usize, case.block as usize))
    );
    assert_eq!((raster.width(), raster.height()), (case.cols, case.rows));
    let mut back = vec![T::default(); case.cols * case.rows];
    raster
        .read_geotiff_samples_into(Window::new(0, 0, case.cols, case.rows), 1, None, &mut back)
        .unwrap();
    let same = T::as_bytes(&back) == T::as_bytes(&case.values);
    assert!(same, "{}: pixels differ after reading back", case.name);
    if keep {
        std::fs::write(
            dir.join(format!("{}.bin", case.name)),
            T::as_bytes(&case.values),
        )
        .unwrap();
    }
    stats
}

#[test]
fn compute_tiles_of_any_size_read_back_exactly() {
    let (cols, rows) = (300, 211);
    let surface: Vec<f32> = (0..cols * rows)
        .map(|i| {
            let (x, y) = ((i % cols) as f32, (i / cols) as f32);
            ((300.0 + (x * 0.05).sin() * 9.0 + y * 0.1) * 32.0).round() / 32.0
        })
        .collect();
    for (name, tile) in [
        ("fel-aligned-128", 128),
        ("fel-unaligned-37", 37),
        ("fel-unaligned-17", 17),
        ("fel-one-tile", 512),
    ] {
        let stats = roundtrip(Case {
            name,
            cols,
            rows,
            block: 64,
            compute_tile: tile,
            data_type: DataType::F32,
            nodata: -3.0e38,
            predictor: "3",
            sparse: false,
            queue: 16,
            values: surface.clone(),
        });
        assert_eq!(stats.blocks, 5 * 4, "{name}");
        assert_eq!(stats.sparse_blocks, 0);
    }
}

#[test]
fn sparse_blocks_and_integer_types_read_back_exactly() {
    let (cols, rows) = (333, 190);
    // Directions with a nodata region covering whole 32 x 32 blocks.
    let p: Vec<i16> = (0..cols * rows)
        .map(|i| {
            let (x, y) = (i % cols, i / cols);
            if (64..192).contains(&x) && y < 128 {
                i16::MIN
            } else {
                ((x * 7 + y * 3) % 9) as i16
            }
        })
        .collect();
    let stats = roundtrip(Case {
        name: "p-sparse-i16",
        cols,
        rows,
        block: 32,
        compute_tile: 48,
        data_type: DataType::I16,
        nodata: f64::from(i16::MIN),
        predictor: "2",
        sparse: true,
        queue: 4,
        values: p,
    });
    assert_eq!(stats.blocks, 11 * 6);
    assert_eq!(stats.sparse_blocks, 4 * 4);
    let bytes: Vec<u8> = (0..cols * rows).map(|i| (i % 251) as u8).collect();
    roundtrip(Case {
        name: "bytes-u8",
        cols,
        rows,
        block: 16,
        compute_tile: 100,
        data_type: DataType::U8,
        nodata: 255.0,
        predictor: "1",
        sparse: false,
        queue: 1,
        values: bytes,
    });
    let area: Vec<f32> = (0..cols * rows)
        .map(|i| {
            if i % 97 == 0 {
                -1.0
            } else {
                (i % 1000) as f32 + 1.0
            }
        })
        .collect();
    let stats = roundtrip(Case {
        name: "ad8-backpressure",
        cols,
        rows,
        block: 16,
        compute_tile: 20,
        data_type: DataType::F32,
        nodata: -1.0,
        predictor: "3",
        sparse: true,
        queue: 1,
        values: area,
    });
    assert_eq!(stats.blocks, 21 * 12);
    assert!(stats.compress_seconds > 0.0 && stats.write_seconds > 0.0);
}

#[test]
fn incomplete_or_repeated_blocks_are_errors() {
    let dir = std::env::temp_dir().join("taudem-rs-async-writer");
    std::fs::create_dir_all(&dir).unwrap();
    let options = RasterWriteOptions {
        tile_size: Some(16),
        ..Default::default()
    };
    let path = dir.join("partial.tif");
    let writer = AsyncTileWriter::<f32>::create(
        path.to_str().unwrap(),
        &header(40, 40, DataType::F32, -1.0),
        &options,
        2,
    )
    .unwrap();
    writer
        .submit_region(Rect::new(0, 0, 10, 10), &[1.0; 100])
        .unwrap();
    assert!(writer.finish().is_err(), "a block was only partly written");

    let path = dir.join("repeated.tif");
    let writer = AsyncTileWriter::<f32>::create(
        path.to_str().unwrap(),
        &header(40, 40, DataType::F32, -1.0),
        &options,
        1,
    )
    .unwrap();
    writer.submit_block(0, 0, &[1.0; 256]).unwrap();
    // The writer thread refuses the second copy of block (0, 0); a later
    // submit or the finish reports it.
    let _ = writer.submit_block(0, 0, &[1.0; 256]);
    let later = writer
        .submit_block(1, 0, &[1.0; 256])
        .and_then(|()| writer.submit_block(2, 0, &[1.0; 128]));
    assert!(later.is_err() || writer.finish().is_err());
    assert!(writer_rejects_misfit_regions());
}

fn writer_rejects_misfit_regions() -> bool {
    let dir = std::env::temp_dir().join("taudem-rs-async-writer");
    let path = dir.join("misfit.tif");
    let options = RasterWriteOptions {
        tile_size: Some(16),
        ..Default::default()
    };
    let writer = AsyncTileWriter::<f32>::create(
        path.to_str().unwrap(),
        &header(40, 40, DataType::F32, -1.0),
        &options,
        2,
    )
    .unwrap();
    let outside = writer
        .submit_region(Rect::new(30, 30, 20, 5), &[0.0; 100])
        .is_err();
    let wrong_len = writer
        .submit_region(Rect::new(0, 0, 5, 5), &[0.0; 24])
        .is_err();
    // Pixels given twice are found once a block's count overflows.
    let twice = writer
        .submit_region(Rect::new(0, 0, 10, 16), &[0.0; 160])
        .is_ok()
        && writer
            .submit_region(Rect::new(0, 0, 10, 16), &[0.0; 160])
            .is_err();
    let untiled = AsyncTileWriter::<f32>::create(
        dir.join("strips.tif").to_str().unwrap(),
        &header(40, 40, DataType::F32, -1.0),
        &RasterWriteOptions::default(),
        2,
    )
    .is_err();
    outside && wrong_len && twice && untiled
}
