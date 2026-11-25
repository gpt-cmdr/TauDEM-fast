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

Every subcommand also accepts `--profile-json PATH`. The JSON file has the
read, compute and write totals, nested phase spans (seconds, heap in use at
entry and exit, heap high-water mark inside the span and per cell), counters
such as flat-cell and pass counts, the sizes of the main working buffers, and
the heap and process (peak working set or VmHWM) high-water marks. The heap
figures come from a counting global allocator in the binary.

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

Every subcommand also accepts `--mode auto|in-memory|tiled`, `--max-memory
BYTES` (such as `4G` or `512M`; default the smaller of 65% of usable and 75%
of available memory), `--tile-size CELLS` and `--scratch-dir DIR` (default
the system temporary directory). Tiled, out-of-core execution is not
implemented yet: `--mode tiled` stops with an error, and `auto` and
`in-memory` run in memory as before. The library's `tiling` module holds its
infrastructure (tile geometry, a three-tier tile store under a memory
budget, tile border tables, and a background GeoTIFF tile writer); see
`docs/rust-port/STREAMING.md` section 3.

The in-memory implementation supports up to 4,294,967,295 cells, subject
to available memory. The kernels are also available through the library.
AreaD8 outlet filtering (`-o`) and weight grids (`-wg`), D8FlowDir imposed
directions (`-sfdr`), and out-of-core processing are future work.

## Memory and threads

Every stage runs on all `RAYON_NUM_THREADS` workers, and the results do not
depend on the thread count. The fused pipeline peaks at
about 9 to 10 bytes per cell, during flat routing: the filled grid (4),
directions (1), both flat increment fields packed in one 32-bit word per cell
(4), and the search frontiers. PitRemove holds the grid, 16-bit tile labels
(2) and small per-tile buffers; AreaD8 holds directions, a state byte and the
area grid (6). The CLI streams sd8 to disk in row blocks before routing flats
and then reuses the filled grid as the flats' artificial surface, so neither
needs its own grid. Flat increments use 16-bit halves. A pass whose
breadth-first distance inside a flat would exceed 65,533 cells repeats with
32-bit halves (8 bytes per cell).

## Algorithm sources

- PitRemove implements improved Priority-Flood with a FIFO for flooded cells.
  See Barnes, Lehman & Mulla (2014), [doi:10.1016/j.cageo.2013.04.024](https://doi.org/10.1016/j.cageo.2013.04.024).
  Outlet and mask rules follow TauDEM's `src/flood.cpp`. Large grids are
  flooded in parallel on tiles (512 cells square by default) whose perimeter
  cells carry labels, and the per-tile spill graphs are joined as in Barnes
  (2016), [doi:10.1016/j.cageo.2016.07.001](https://doi.org/10.1016/j.cageo.2016.07.001):
  each tile keeps a minimum spanning forest of its label adjacencies, a
  Kruskal pass over all tiles gives each label its spill elevation, and every
  cell ends at the higher of its tile fill and its label's spill elevation.
  Filling to the lowest spill elevation has one answer, so the tiled and
  whole-grid floods agree on every value. They can differ only in the sign of
  a filled zero, since -0.0 and 0.0 compare equal; when the DEM contains -0.0,
  a final pass assigns each filled zero the sign the whole-grid flood writes.
  `pit_remove_with_tiles` selects the tile side, or the whole-grid flood with 0.
- D8FlowDir and AreaD8 follow TauDEM's `src/d8.cpp`, `src/aread8.cpp` and
  `src/commonLib.cpp` by David Tarboton and contributors. The flat resolver
  computes TauDEM's Garbrecht–Martz increments as breadth-first distances
  instead of repeated scans, with level-synchronous parallel searches; the
  distances, and so the directions, do not depend on visit order. The
  crossing rule is checked in parallel and only crossing cells, and cells
  whose north or west neighbor then changes, are revisited in row-major order.
  AreaD8 walks downstream from headwater cells in parallel; the walk that
  completes a cell's last upstream neighbor sums it, adding upstream areas in
  TauDEM's fixed neighbor order, so the sums are identical for any schedule.

## Tests for tiled stages

`tests/support/` generates the tiling-boundary fixtures of
`docs/rust-port/STREAMING.md` section 6.1 for a tile size and provides
`assert_tiled_matches_in_memory`; include it with `mod support;`.
`cargo test --release --test adversarial -- --ignored` also runs the
17.6-million-cell AreaD8 ramp. To check the background writer's GeoTIFFs
with GDAL, run `TAUDEM_GDAL_CHECK_DIR=DIR cargo test --test async_writer`,
then `python tests/gdal_readback.py DIR` with GDAL's Python bindings.
`cargo run --release --example tile_codec_bench -- DEM.tif [TILE]` reports
the tile codec's compression ratios and speeds on a Float32 DEM.
