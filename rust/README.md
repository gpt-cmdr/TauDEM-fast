# TauDEM Rust stage one

Build with `cargo build --release --locked` from this directory. The executable
is `target/release/taudem-rs` (`.exe` on Windows). Set `RAYON_NUM_THREADS` to
limit the number of compute and GeoTIFF worker threads.

```text
taudem-rs pitremove -z dem.tif -fel fel.tif [-4] [-depmask mask.tif]
taudem-rs d8flowdir -fel fel.tif -p p.tif -sd8 sd8.tif
taudem-rs aread8 -p p.tif -ad8 ad8.tif [-nc] [--exact-area]
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
BYTES` (such as `4G` or `512M`; default the smaller of 65% of Terrio's
`usable_memory()` and 75% of available memory), `--tile-size CELLS` and
`--scratch-dir DIR` (default the system temporary directory). Every
subcommand runs out of core with `--mode tiled`, and its outputs are
identical to the in-memory ones cell for cell. `--mode auto`, the default,
runs tiled when the in-memory estimate exceeds the budget or the raster
passes the in-memory limit of 4,294,967,295 cells, and prints its choice.
The estimate is 48 MiB plus a measured peak per cell: PitRemove 6.5 bytes
(7.5 with `-depmask`), D8FlowDir 10, AreaD8 7.5 and the pipeline 10
(measured on Muncie, Spring Creek, the RAS 7.0 mosaic and ELC; see
`docs/rust-port/STREAMING.md` section 6.2). An explicit tile side can be
16 through 4096 if the working buffers and perimeters fit the budget.

The tiled pipeline is fused (`taudem_rs::pipeline_tiled`): the DEM is read
once, PitRemove hands each filled tile with its exact one-cell halo
straight to D8FlowDir, D8FlowDir's final directions go to AreaD8 through
the tile cache, and fel, sd8, p and ad8 are written by background tile
writers only when requested. All stages share one tile grid, one tile
cache and one budget. The planner divides `--max-memory` between the
workers' buffers, the tile perimeters and graphs, the writers, a process
reserve and the cache (at most half the budget), and halves the tile side
from 2048 until the workers fit. Tiles beyond the cache are compressed in
RAM and then spilled to `--scratch-dir`. On ELC (1.1 billion cells) the
fused pipeline takes 72 to 90 s with a 1.5 GiB peak working set at
`--max-memory 2G` (2.6 GiB at `4G`), against 68 to 96 s and 9.7 GiB in
memory, on 8 threads of a shared workstation.

Tiled PitRemove reads a halo around each tile, retains local fills and
watershed labels in the budgeted `TileStore`, and writes the result as a
sparse tiled GeoTIFF. The `pitremove_tiled::PitRemovePlan` library API takes a
window reader and returns a solved flood. `SolvedPitRemove::emit` supplies
filled cores and their exact one-cell halos directly to a concurrent
consumer. The library's `tiling` module holds the shared infrastructure
(tile geometry, a three-tier tile store under a memory budget, tile border
tables, a background GeoTIFF tile writer, and the worker threads); see
`docs/rust-port/STREAMING.md` section 3.

Tiled AreaD8 (`taudem_rs::area_tiled`) follows Barnes (2017),
[doi:10.1016/j.envsoft.2017.02.022](https://doi.org/10.1016/j.envsoft.2017.02.022):
each tile is accumulated alone, the tiles' perimeter exits form a graph
solved with exact integer counts, and a second pass injects each tile's
solved inflows. Its output equals in-memory `aread8` bit for bit, including
the `f32` rounding of TauDEM's sums once a mainstem passes 2^24 cells: the
few exits past that are recomputed in TauDEM's summation order from the
tiles that hold them. `--exact-area` (tiled only) instead writes exact
counts rounded once to `f32`, which differ from TauDEM above 2^24. The
module documentation gives the exactness argument. The direction tiles are
kept compressed in the tile store between passes and spill to
`--scratch-dir` beyond the budget.

Striped GeoTIFFs (whole rows per block, as Spring Creek's LZW strips) are
read for tiles from full-width bands, each decoded once per row of tiles;
the bands are charged to the budget.

The in-memory implementation supports up to 4,294,967,295 cells, subject
to available memory. The kernels are also available through the library.
AreaD8 outlet filtering (`-o`) and weight grids (`-wg`) and D8FlowDir
imposed directions (`-sfdr`) are future work.

## Memory and threads

Every stage runs on all `RAYON_NUM_THREADS` workers, and the results do not
depend on the thread count. The in-memory pipeline peaks at
about 9 to 10 bytes per cell, during flat routing: the filled grid (4),
directions (1), both flat increment fields packed in one 32-bit word per cell
(4), and the search frontiers. PitRemove holds the grid, 16-bit tile labels
(2) and small per-tile buffers; AreaD8 holds directions, a state byte and the
area grid (6). The CLI streams sd8 to disk in row blocks before routing flats
and then reuses the filled grid as the flats' artificial surface, so neither
needs its own grid. Flat increments use 16-bit halves. A pass whose
breadth-first distance inside a flat would exceed 65,533 cells repeats with
32-bit halves (8 bytes per cell).

Tiled runs stay within `--max-memory`. Their tile loops run on plain worker
threads (`tiling::workers`), one per Rayon thread, never as Rayon jobs, so a
GeoTIFF read inside a tile cannot start further tiles on the waiting thread.

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
- Tiled D8FlowDir (`d8_tiled`) routes each tile's flats that stay inside it
  in the tile, and the flats that reach a tile edge pass by pass across
  tiles: TauDEM's two increment fields are completed by label-correcting
  relaxation over tile perimeters, and every pass uses the raster-wide
  highest levels the in-memory resolver uses. The module documentation and
  `docs/rust-port/STREAMING.md` section 2.3 give the exactness argument.
- The fused tiled pipeline (`pipeline_tiled`) composes the three tiled
  stages on one tile grid; each stage receives exactly the inputs its
  standalone tiled form reads, so the composition is exact
  (`docs/rust-port/STREAMING.md` section 2.5).

## Tests for tiled stages

`tests/support/` generates the tiling-boundary fixtures of
`docs/rust-port/STREAMING.md` section 6.1 for a tile size and provides
`assert_tiled_matches_in_memory`; include it with `mod support;`.
`tests/pipeline_tiled.rs` checks the fused pipeline's four outputs on the
whole suite at T = 16, 17 and 32 and 1 and 8 threads, with a spilling cache
and in wavefront order.
`cargo test --release --test adversarial -- --ignored` also runs the
17.6-million-cell AreaD8 ramp, and `cargo test --release --test area_tiled
-- --ignored` checks tiled AreaD8 on it. To check the background writer's GeoTIFFs
with GDAL, run `TAUDEM_GDAL_CHECK_DIR=DIR cargo test --test async_writer`,
then `python tests/gdal_readback.py DIR` with GDAL's Python bindings.
`cargo run --release --example tile_codec_bench -- DEM.tif [TILE]` reports
the tile codec's compression ratios and speeds on a Float32 DEM.
