# TauDEM Rust stage one

Build with `cargo build --release --locked` from this directory. The executable
is `target/release/taudem-rs` (`.exe` on Windows). Set `RAYON_NUM_THREADS` to
limit the number of compute and GeoTIFF worker threads.

```text
taudem-rs pitremove -z dem.tif -fel fel.tif [-4] [-depmask mask.tif]
taudem-rs d8flowdir -fel fel.tif -p p.tif -sd8 sd8.tif
taudem-rs aread8 -p p.tif -ad8 ad8.tif [-nc]
taudem-rs pipeline -z dem.tif [-fel fel.tif] [-p p.tif] [-sd8 sd8.tif] [-ad8 ad8.tif]
                   [-4] [-depmask mask.tif] [-nc]
```

The pipeline passes arrays between stages and writes only the requested
products. With no output paths it still computes contributing area and prints
timings. Use `-4` or `-4way` for cardinal-only PitRemove connectivity. A depression mask
value of 1 fixes the DEM elevation at that cell. By default AreaD8 propagates
edge contamination to downstream cells. Use `-nc` to disable that check.

All inputs and outputs are GeoTIFF. The first band is used. Output nodata
values are `-3e38` for Float32 fel, `-32768` for Int16 p, and `-1` for Float32
sd8 and ad8. D8 direction is 1 east, increasing counterclockwise.
Slope is measured on the original filled surface before flat routing.
Output uses tiled Deflate with automatic BigTIFF selection. Float32 DEMs use
Terrio's caller-buffer read when the source and codec support it. For a strip
over Terrio's 256 MiB decoded-chunk limit, TauDEM requests smaller windows and
uses Terrio's row-based decoder when available. Compressed strips may be
decoded from the start for each window. For trusted inputs that still hit the
limit, set `TERRIO_TIFF_MAX_CHUNK_BYTES` to at least the largest decoded block.

The in-memory implementation supports up to 4,294,967,295 cells, subject
to available memory. The kernels are also available through the library.
AreaD8 outlet filtering (`-o`) and weight grids (`-wg`), D8FlowDir imposed
directions (`-sfdr`), and out-of-core processing are future work.

## Algorithm sources

- PitRemove implements improved Priority-Flood with a FIFO for flooded cells.
  See Barnes, Lehman & Mulla (2014), [doi:10.1016/j.cageo.2013.04.024](https://doi.org/10.1016/j.cageo.2013.04.024).
  Outlet and mask rules follow TauDEM's `src/flood.cpp`.
- D8FlowDir and AreaD8 follow TauDEM's `src/d8.cpp`, `src/aread8.cpp` and
  `src/commonLib.cpp` by David Tarboton and contributors. The flat resolver
  replaces repeated scans with breadth-first traversals of each gradient.
