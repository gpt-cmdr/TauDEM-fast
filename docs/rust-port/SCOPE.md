# TauDEM Rust Rewrite — Architecture & Implementation Scope

**Target**: High-performance, pure-Rust rewrite of TauDEM (`G:\GH\TauDEM`, branch `rust-port` at upstream `dtarb/TauDEM` Develop `b994ed2`) optimized for $10^9$–$10^{10}$ cell LiDAR DEMs (e.g., HEC-RAS terrain GeoTIFFs), using our `terrio` fork (`H:\CLB-Repos\terrio-clb`, `main` at `90160a28`) for pure-Rust geospatial I/O and validated against `dtarb/TauDEM-Test-Data` (`master`) plus the local HEC-RAS terrain tiers in [`TERRAINS.md`](TERRAINS.md).

**Design goals**:
1. **10–50× faster wall time** on $10^8$–$4\times 10^9$ cell DEMs by replacing $O(N^{1.5})$ scans, serialized MPI file writes, and per-step halo IPC with $O(N \log N)$/$O(N)$ algorithms, shared-memory `rayon` parallelism, tiled out-of-core execution, and fused in-memory pipelines.
2. **Drop-in CLI parity** with TauDEM simple and flag-based syntax, backed by a testable parity contract distinguishing bit-exact outputs from documented algorithmic improvements.
3. **Zero C/MPI runtime dependencies** by default (single static binary via `terrio`), with optional `gdal` / `libzstd` features where legacy formats or max-ratio ZSTD encoding are needed.

---

## 1. TauDEM Tool Inventory & C++ Bottlenecks

### 1.1 C++ Architecture & Partitioning Bottlenecks

Inspection of `G:\GH\TauDEM\src` reveals five structural bottlenecks in C++ TauDEM that dominate large-terrain runtimes:

1. **1D Row-Strip Partitioning & Wasted Memory** (`src/linearpart.h:133-145`, `src/partition.h:94-125`): Every grid is split into contiguous horizontal strips (`ny = totaly / size`) allocated fully in RAM (`new datatype[nx * ny]`). Only `SHORT_TYPE` (`i16`), `LONG_TYPE` (`i32`), and `FLOAT_TYPE` (`f32`) exist, forcing D8 flow directions (9 states) into 2 B/cell (`i16`) instead of 1 B/cell (`u8`).
2. **Per-Iteration Halo IPC & $O(P)$ Ring Termination** (`src/linearpart.h:195-219, 254-309, 344-384`): Every `share()` allocates/frees buffers and calls `MPI_Buffer_attach`/`MPI_Bsend`/`MPI_Buffer_detach`; `ringTerm()` passes a sequential token around all $P$ ranks instead of using a collective reduction.
3. **Completely Serialized Raster Writes** (`src/tiffIO.cpp:263-428`): Rank 0 calls `GDALCreate`, writes its strip, and closes the file; each subsequent rank waits for an MPI token, reopens with `GA_Update`, writes its strip, and closes. Also note a latent bounds bug at `src/tiffIO.cpp:326` (`index == 6` for a 6-element extension table; `.tiff` is `index == 5`).
4. **Subtle Parity Rules Embedded in I/O** (`src/tiffIO.cpp:127-167, 434-445`; `src/tiffIO.h:96-98`; `src/linearpart.h:476`; `src/commonLib.h:81`):
   - Unset input nodata defaults silently to `-9999` (`tiffIO.cpp:164-167`).
   - `isNodata()` uses `abs((float)(val - noData)) < MINEPS` with `MINEPS = 1e-5f` (`linearpart.h:476`, `commonLib.h:81`).
   - Geographic CRS (`IsGeographic == 1`) computes per-row WGS84 ellipsoidal cell sizes `dxc[j]`, `dyc[j]` (`elipa = 6378137.0`, `elipb = 6356752.314`, `tiffIO.cpp:434-445`).
   - Output extensions map to GTiff LZW (`.tif`), HFA (`.img`), SAGA (`.sdat`), EHdr (`.bil`), ENVI (`.bin`), uncompressed GTiff (`.tiff`); unknown extensions rewrite to `.tif` (`tiffIO.cpp:268-270`). Vector I/O supports `.shp`, `.sqlite`, `.json`/`.geojson`, `.kml`, and `.gdb` point reads via `ReadOutlets.cpp:49-198` (case-sensitive `"id"` field, fallback 1-based index).
5. **Superlinear C++ Algorithms on Large Terrains**:
   - `flood.cpp:289-479` (Planchon-Darboux): alternating dual-`std::stack<long>` passes can require $O(\sqrt{N})$ sweeps across winding valleys/strip boundaries ($O(N^{1.5})$ worst case) plus ~16 B/unresolved cell in stack storage.
   - `d8.cpp:459-680` & `dinf.cpp:598-833` (Garbrecht-Martz `resolveflats`): scans every remaining flat cell in `que` on *every* wave step ($O(N \cdot D_{\text{flat}})$), allocates three `i16` grids (`elev2`, `dn`, `s`), and **overflows `i16`** on flats wider than 32,767 cells.
   - `linklib.h:165-187` (`FindLink`): singly-linked list linear search inside `streamnet`, yielding $O(L^2)$ link lookups on dense LiDAR stream networks.

### 1.2 Complete Tool Inventory (37 CMake Targets + 2 Non-CMake Tools = 39)

All 37 targets in `src/CMakeLists.txt:143-179` plus 2 non-CMake sources (`D8VDistToStrm`, `ogrtest`). **Hot-path tools** for large-terrain HEC-RAS / FIM workflows are in Tier A. Test-data coverage refers to `Input/testall.bat` / `Input/taudem-tests.sh` in `dtarb/TauDEM-Test-Data` (36/37 CMake targets covered; `twi` is the only CMake target absent).

#### Tier A — Hot-Path Large-Terrain Core (10 tools)

| Tool | Source (`src/`) | Algorithm | Inputs -> Outputs (dtypes) | C++ Complexity | MPI Pattern | Test-Data |
|---|---|---|---|---|---|---|
| **`pitremove`** | `flood.cpp`, `PitRemovemn.cpp` | Planchon-Darboux iterative 2-stack fill; `-4way`, `-depmask` | `z`(f32), opt `depmask`(i16) -> `fel`(f32, nd `-3e38`) | $O(N^{1.5})$ worst; $O(N)$ typical | `share` + `ringTerm` every 2 passes; 2× f32 + stacks | `Base`, `Geographic`, `NoEPSG`, `gridtypes`, `sinmapsi`, `RetLimFlow` |
| **`d8flowdir`** | `d8.cpp`, `D8FlowDirmn.cpp`, `Node.cpp` | Steepest 8-nb (`dontCross`) + Garbrecht-Martz `resolveflats` (`i16` grids); opt `-sfdr` | `fel`(f32), opt `sfdr`(i16) -> `p`(i16, nd `-32768`), `sd8`(f32, nd `-1`) | $O(N)$ steep; $O(N \cdot D_{\text{flat}})$ flats | `share` + `ringTerm` per flat wave; 3× i16 flat grids | `Base`, `Geographic`, `NoEPSG`, `gridtypes`, `FlowdirCond`, `gwunittest`, `fts` |
| **`dinfflowdir`** | `dinf.cpp`, `DinfFlowDirmn.cpp`, `Node.cpp` | Tarboton (1997) 8-facet `VSLOPE`/`SET2` (`f64`) + Garbrecht-Martz `resolveflats` | `fel`(f32) -> `ang`(f32, $[0,2\pi)$, nd `-FLT_MAX`), `slp`(f32, nd `-1`) | $O(N)$ steep; $O(N \cdot D_{\text{flat}})$ flats | Same as `d8flowdir` | `Base`, `Geographic`, `AreaDinf`, `Dinf*`, `sinmapsi`, `RetLimFlow` |
| **`aread8`** | `aread8.cpp`, `aread8mn.cpp` | Kahn topological sort (`initNeighborD8up`); opt `-o`, `-wg`, `-nc` | `p`(i16), opt `o`(vec), `wg`(f32) -> `ad8`(f32, nd `-1`) | $O(N)$ compute; boundary-limited MPI | `addBorders` + `share` + `ringTerm` per wave; `transferPack` for `-o` | `Base`, `AreaD8_data`, `Geographic`, `NoEPSG`, `gridtypes`, `GDAL_unset_nodata` |
| **`areadinf`** | `areadinf.cpp`, `areadinfmn.cpp` | Kahn topological sort on Dinf DAG (`prop` threshold `1e-5`); unweighted adds `dxc` (SCA) | `ang`(f32), opt `o`(vec), `wg`(f32) -> `sca`(f32, nd `-1`) | $O(N)$ | Same as `aread8` | `Base`, `AreaDinf`, `sinmapsi` |
| **`gridnet`** | `gridnet.cpp`, `gridnetmn.cpp` | Kahn D8 sort for longest/total path & Strahler order; opt `-mask`/`-thresh`, `-o` | `p`(i16), opt `mask`(f32), `o`(vec) -> `plen`(f32), `tlen`(f32), `gord`(i16) | $O(N)$ | `addBorders` + `share` + `ringTerm` | `Base`, `Gridnet` |
| **`threshold`** | `Threshold.cpp`, `Thresholdmn.cpp` | Per-pixel `ssa >= thresh` (opt `mask >= 0`) | `ssa`(f32), opt `mask`(f32) -> `src`(i16, 0/1, nd `-32768`) | $O(N)$ | Embarrassingly parallel; serialized write | `Base`, `Geographic`, `streamnet_data`, `MoveOutlets*`, `CatchHydroGeo` |
| **`streamnet`** | `streamnet.cpp`, `streamnetmn.cpp`, `linklib.h` | 3-pass Kahn/BFS: down->up lengths, up->down links/order/Shreve, down->up watersheds | `fel`(f32), `p`(i16), `ad8`(f32), `src`(i16), opt `o`(vec) -> `ord`(i16), `w`(i32), `tree.dat`, `coord.dat`, `net`(vec, 17 attrs) | $O(N + L^2)$ (`FindLink` list scan) | Heavy `sendLink`/`recvLink` + rank-0 vector/table write | `Base`, `Geographic`, `streamnet_data`, `LoganSampleSQLite`, `CatchOutlets` |
| **`moveoutletstostreams`** | `MoveOutletsToStrm.cpp` | Trace each outlet downstream along `p` up to `-md` cells until `src >= 1` | `p`(i16), `src`(i16), `o`(vec) -> `om`(vec + `Dist_moved`) | $O(N_{\text{read}} + K \cdot md)$ (loads full grids!) | Broadcasts points; ranks trace across strip handoffs | `Base`, `MoveOutlets2`, `MoveOutlets3`, `MovedOutletstoStream_data` |
| **`dinfdistdown`** | `DinfDistDown.cpp` | Down->up Dinf Kahn sort: `h`/`v`(HAND)/`p`/`s` × `ave`/`min`/`max`; opt `-wg`, `-nc` | `ang`(f32), `fel`(f32), `src`(i16), opt `wg`(f32) -> `dd`(f32, nd `-1`) | $O(N)$ | `addBorders` + `share` + `ringTerm` (down->up) | `Base`, `CatchHydroGeo`, `Inundepth` |

#### Tier B — Secondary D8 Graph Tools (8 CMake + 1 Non-CMake)

| Tool | Source (`src/`) | Algorithm | Inputs -> Outputs | Complexity & MPI Notes | Test-Data |
|---|---|---|---|---|---|
| `d8hdisttostrm` | `D8HDistToStrm.cpp` | Down->up D8 Kahn horizontal distance to `src >= thresh` | `p`(i16), `src`(i16) -> `dist`(f32) | $O(N)$; down->up `share`+`ringTerm` | `Base` |
| `d8flowpathextremeup` | `D8flowpathextremeup.cpp` | Up->down D8 Kahn max/min of `ssa` along upslope paths; opt `-o`, `-nc` | `p`(i16), `ssa`(f32), opt `o`(vec) -> `ssa`(f32) | $O(N)$; up->down `share`+`ringTerm` | `Base`, `D8flowextreme` |
| `slopeavedown` | `SlopeAveDown.cpp` | Trace downstream `dn` meters along `p`, compute drop/`dn` | `p`(i16), `fel`(f32) -> `slpd`(f32) | $O(N \cdot dn/\Delta x)$; MPI cross-strip trace | `Base` |
| `dropanalysis` | `DropAnalysis.cpp` | For each threshold in `[min,max]` (`num` steps), compute order-1 vs higher stream drop t-test | `fel`(f32), `p`(i16), `ad8`(f32), `ssa`(f32), `o`(vec) -> `drp.txt` | $O(T \cdot N)$; loops `T` thresholds serially in MPI | `Base` |
| `gagewatershed` | `gagewatershed.cpp` | Down->up D8 watershed labeling from outlets + optional downstream ID table (`-id`) | `p`(i16), `o`(vec) -> `gw`(i32), opt `id.txt` | $O(N)$; down->up `share`+`ringTerm` | `Base`, `GageWatershed`, `gwunittest`, `fts` |
| `connectdown` | `ConnectDown.cpp` | Find max `ad8` per watershed `w`, trace `d` cells downstream along `p` | `p`(i16), `ad8`(f32), `w`(i32) -> `o`(vec), `od`(vec) | $O(N)$; gathers watershed maxima to rank 0 | `ConnectDown` |
| `flowdircond` | `flowdircond.cpp` | Up->down D8 conditioning (monotone non-increasing DEM along `p`) | `p`(i16), `z`(f32) -> `zfdc`(f32) | $O(N)$; up->down Kahn `share`+`ringTerm` | `FlowdirCond` |
| `catchoutlets` | `CatchOutlets.cpp` | Walk `net` LineStrings + `p` to place outlets by `-mindist`/`-minarea` | `p`(i16), `net`(vec) -> `o`(vec) | $O(L + \text{len})$; **single-process only** in C++ | `CatchOutlets` |
| `d8vdisttostrm` *(non-CMake)* | `D8VDistToStrm.cpp` | Down->up D8 vertical drop to stream (D8 HAND); present in VS `.sln`, omitted from CMake | `p`(i16), `fel`(f32), `src`(i16) -> `vdist`(f32) | $O(N)$; down->up `share`+`ringTerm` | None |

#### Tier C — Secondary Dinf Graph Tools (8 CMake)

| Tool | Source (`src/`) | Algorithm | Inputs -> Outputs | Complexity & MPI Notes | Test-Data |
|---|---|---|---|---|---|
| `dinfdistup` | `DinfDistUp.cpp` | Up->down Dinf Kahn dist to ridge: `h`/`v`/`p`/`s` × `ave`/`min`/`max` | `ang`(f32), `fel`(f32), opt `slp`(f32) -> `du`(f32) | $O(N)$; up->down `share`+`ringTerm` | `Base` |
| `dinfavalanche` | `DinfAvalanche.cpp` | Up->down Dinf runout zone (`rz` max angle) & path distance (`dfs`) | `ang`(f32), `fel`(f32), `ass`(i16) -> `rz`(f32), `dfs`(f32) | $O(N)$; up->down `share`+`ringTerm` | `Base` |
| `dinfconclimaccum` | `DinfConcLimAccum.cpp` | Up->down Dinf concentration-limited accumulation (`cs`, `ctpt`, `q`, `dg`) | `ang`, `dg`, `dm`, `q`(f32), opt `o` -> `ctpt`, `q`(f32) | $O(N)$; up->down Kahn | `Base`, `DinfConcLimAccum` |
| `dinfdecayaccum` | `dinfdecayaccum.cpp` | Up->down Dinf accumulation with multiplicative decay `dm` | `ang`, `dm`(f32), opt `wg`, `o` -> `dsca`(f32) | $O(N)$; up->down Kahn | `Base` |
| `dinfrevaccum` | `DinfRevAccum.cpp` | Down->up Dinf reverse accumulation (`racc`) + max downslope (`dmax`) | `ang`, `wg`(f32) -> `racc`, `dmax`(f32) | $O(N)$; down->up Kahn | `Base` |
| `dinftranslimaccum` | `DinfTransLimAccum.cpp` | Up->down Dinf transport-limited supply/capacity (`tla`, `tdep`, opt `cs`) | `ang`, `tsup`, `tc`(f32), opt `cs`, `o` -> `tla`, `tdep`, opt `ctpt`(f32) | $O(N)$; up->down Kahn | `Base`, `DinfTransLimAcc` |
| `dinfupdependence` | `DinfUpDependence.cpp` | Down->up Dinf fractional contribution to target `dg` | `ang`(f32), `dg`(i16) -> `dep`(f32) | $O(N)$; down->up Kahn | `Base` |
| `retlimflow` | `RetlimFlow.cpp` | Up->down Dinf retention-limited flow (`q = max(0, in + r - c)`) | `ang`, `r`, `c`(f32) -> `q`(f32) | $O(N)$; up->down Kahn | `RetLimFlow` |

#### Tier D — Elementwise, Stencil, Table Reduction & Utility Tools (11 CMake + 1 Non-CMake)

| Tool | Source (`src/`) | Algorithm | Inputs -> Outputs | Complexity & MPI Notes | Test-Data |
|---|---|---|---|---|---|
| `peukerdouglas` | `PeukerDouglas.cpp` | 3×3 weighted smoothing + 2×2 upward-curved ridge detection | `fel`(f32) -> `ss`(i16, 0/1) | $O(N)$; 1-row halo (`share` once) | `Base`, `peukerDouglas` |
| `slopearea` | `SlopeArea.cpp` | Pixelwise $S^m A^n$ | `slp`, `sca`(f32) -> `sa`(f32) | $O(N)$; no halo | `Base` |
| `slopearearatio` | `SlopeAreaRatio.cpp` | Pixelwise $S / A$ (handles $A=0$) | `slp`, `sca`(f32) -> `sar`(f32) | $O(N)$; no halo | `Base` |
| `lengtharea` | `LengthArea.cpp` | Pixelwise `ad8 >= M * plen^y` -> 1/0 | `plen`, `ad8`(f32) -> `ss`(i16) | $O(N)$; no halo | `Base` |
| `twi` | `TWI.cpp` | Pixelwise $\ln(a / \tan\beta)$ | `slp`, `sca`(f32) -> `twi`(f32) | $O(N)$; no halo | **None** |
| `sinmapsi` | `SinmapSI.cpp` | Infinite-slope stability index `si` + wetness `sat` via region IDs & param CSV | `slp`, `sca`(f32), `reg`(i16/i32), `cal.csv` -> `si`, `sat`(f32) | $O(N)$; no halo | `sinmapsi` |
| `siregion` | `SIRegionToolmn.cpp` | **Non-MPI** GDAL rasterize/warp polygons -> region raster + default calibration CSV | `ref`(raster), `poly`(vec) -> `reg`(raster), `cal.csv` | $O(N)$; GDAL C API only | `sinmapsi` |
| `catchhydrogeo` | `CatchHydroGeo.cpp` | Per-catchment HAND stage-area-volume-bed-area-hydraulic-radius CSV table | `hand`, `slp`(f32), `catch`(i32), `net`(vec), `stage.txt` -> `hydrogeo.csv` | $O(N \cdot S_{\text{stages}})$; MPI `Allreduce` per catchment | `CatchHydroGeo` |
| `inundepth` | `InunDepth.cpp` | Pixelwise `max(0, stage[catch] - hand)` + opt depth/forecast CSV | `hand`(f32), `catch`(i32), `fc`(csv/nc) -> `depth`(f32) | $O(N)$; no halo | `Inundepth` |
| `editraster` | `EditRaster.cpp` | Apply `(x, y, val)` point edits from text file to raster | `in`(f32), `change.txt` -> `out`(f32) | $O(N)$ in C++; **O(edits) tiled** in Rust | `Editraster` |
| `setregion` | `SetRegion.cpp` | Mark cells outside `inreg` that flow into `inreg` (RWD preprocessing) | `p`(i16), `inreg`(i16) -> `outreg`(i16) | $O(N)$; 1-row halo | `SetRegion` |
| `ogrtest` *(non-CMake)* | `ogrtest.cpp` | OGR vector read/write smoke test (omit from Rust CLI; replace with unit tests) | — | — | None |

---

## 2. Rust Algorithm Choices & Parity Contract

### 2.1 `pitremove` — Priority-Flood (Barnes 2014) + Parallel Tile-Based (Barnes 2016)
- **In-memory ($N \le 1.5\times 10^9$ on 32 GB RAM)**: **Barnes, Lehman & Mulla (2014) Improved Priority-Flood** (`Algorithm 2`).
  - Uses 1× `f32` grid in-place + 1-bit `visited` bitset = **4.125 B/cell** (vs C++ 2× `f32` + stacks $\ge 16$ B/cell).
  - Separates spill cells (binary min-heap or radix-ordered `u32` IEEE-754 monotonic float key `f32_to_sortable_u32`, pushing only perimeter/non-pit cells) from flooded interior pit cells (plain `VecDeque<u64>`, $O(1)$ per cell with zero heap operations).
  - Supports `-4way` (4-connected cardinal neighbors) and `-depmask` (mask cells `== 1` seeded as boundary drains at their original DEM elevation, matching `flood.cpp:249-250`).
  - Nodata cells (`is_nodata` within `MINEPS = 1e-5f`) are marked visited immediately; valid cells adjacent to nodata or raster edges seed the priority queue (matching `flood.cpp:251-257`).
- **Out-of-core / Multi-core ($N > 10^9$ or multi-thread acceleration)**: **Barnes (2016) Parallel Tile-Based Priority-Flood**.
  - Partition raster into $T_w \times T_h$ super-tiles (e.g., $4096 \times 4096 = 16.7\text{M}$ cells, 64 MB/tile, aligned to 256×256 GeoTIFF blocks).
  - **Pass 1 (embarrassingly parallel via `rayon`)**: Run Barnes 2014 inside each tile independently, treating tile perimeter cells as tentative outlets; record each tile's perimeter-to-perimeter spill graph (compressed via union-find/MST on the tile's perimeter cells — $O(T_w + T_h)$ nodes per tile).
  - **Global Solve**: Run single-threaded/parallel Dijkstra on the stitched perimeter spill graph across all tiles (for $4\times 10^9$ cells with $4096^2$ tiles, ~240 tiles × ~16K perimeter cells $\approx 3.9\text{M}$ nodes — solves in <0.5 s in RAM).
  - **Pass 2 (parallel, streaming write)**: For each tile, re-run local raise from the solved perimeter spill elevations (or cache Pass-1 pit basins) and stream tiles directly to `GeoTiffWriter::write_tile`.
- **Parity**: **Bit-for-bit identical (`max_diff == 0.0`)** to Planchon-Darboux. Minimum-spill-elevation depression filling has a unique mathematical fixed point; every cell receives the exact `f32` elevation of the lowest saddle on a valid path to a boundary/mask drain.

### 2.2 `d8flowdir` & `dinfflowdir` — Exact Steep Slopes + Barnes (2014) $O(N)$ Flat Resolution
- **Steep-slope cells (exact parity)**:
  - `d8flowdir`: Port `d8.cpp:54-100, 153-177, 359-409` verbatim: 8-neighbor nodata propagation (any nodata/border neighbor makes center nodata unless `-sfdr` imposes a valid direction), cardinal (`k=1,3,5,7`) checked before diagonal (`k=2,4,6,8`) with `dontCross` anti-crossing check (`d8.cpp:54-100`), per-row WGS84 ellipsoidal `dxc[j]`, `dyc[j]` when geographic (`tiffIO.cpp:434-445`), and slope `sd8` in `f32` (`-1.0` nodata).
  - `dinfflowdir`: Port Tarboton (1997) 8-facet `VSLOPE`/`SET2` (`dinf.cpp:286-373`) in `f64` returning `ang` (`f32`, $[0,2\pi)$, nodata `-FLT_MAX`) and `slp` (`f32`, nodata `-1.0`).
- **Flat resolution — two modes**:
  - **Default fast mode (Barnes, Lehman & Mulla 2014, $O(N)$ BFS)**:
    1. Identify flat cells (`p == 0` / unresolved), label connected flat components (optional union-find for tile stitching), seed low-edge cells (flats bordering a lower cell) and high-edge cells (flats bordering a higher cell).
    2. Run **two linear $O(N_{\text{flat}})$ BFS passes** storing `u32` distances: `away_from_higher` (`h_dist`) and `toward_lower` (`l_dist`), combining into `flat_mask = 2 * l_dist + (max_h - h_dist)` (still `u32` — **never overflows** like C++ `i16` at 32,767).
    3. Compute D8 (`setPosDir`) or Dinf (`SET2` on `flat_mask`) over flat cells only. Runs in milliseconds even on million-cell floodplains/reservoirs where C++ Garbrecht-Martz takes hours.
  - **Exact-parity mode (`--compat-flats`)**: Queue-frontier BFS implementation of Garbrecht-Martz (`incfall` + `incrise` with `i32` accumulators and the exact tie-breaking + outer re-pass loop of `d8.cpp:307-316, 459-680` / `dinf.cpp:598-833`). Produces identical flat gradients to C++ TauDEM in $O(N_{\text{flat}})$ instead of $O(N \cdot D_{\text{flat}})$ (matching C++ everywhere C++ does not overflow `i16`).
- **Out-of-core flats**: Most LiDAR flats are small and tile-local; flats touching a tile border are resolved via a boundary-label graph across neighboring tiles (sparse multi-tile BFS only over flat pixels touching tile seams — typically <2% of cells).
- **Parity**:
  - Steep cells (typically 95–99.9% of LiDAR cells): **exact parity** (`p` identical; `sd8`, `ang`, `slp` within `1e-6` relative error).
  - Flat cells: **exact parity under `--compat-flats`**; under default Barnes 2014, slopes are identically `0.0`, drainage is guaranteed acyclic to the same flat outlet(s), and individual flow lines across wide flats may follow slightly smoother/differently centered paths (**acceptable difference**, documented in parity report).

### 2.3 `aread8`, `areadinf`, `gridnet`, `dinfdistdown` & Kahn DAG Tools
- **In-memory ($N \le 10^9$)**:
  - Store D8 as `u8` (1 B/cell) and in-degree as `u8` (1 B/cell; max 8 neighbors) — **2 B/cell** total graph state vs C++ `i16` dir + `i16` in-degree + `MPI` buffers.
  - **Parallel Kahn / Sub-Basin Partitioning**: Divide into $T_w \times T_h$ tiles (e.g., $512 \times 512$ in-RAM work units). Phase 1 runs Kahn topological sort in parallel across all tiles for cells with zero cross-tile dependencies; Phase 2 propagates boundary inflows along the coarse tile-edge DAG (or atomic `AtomicU8::fetch_sub` in-degree decrements on tile-border cells only, avoiding atomic contention on >99% interior cells).
  - For `aread8` and `areadinf`, accumulate internally in **`f64`** (avoiding C++ TauDEM's silent integer precision loss above $2^{24} = 16,777,216$ cells in `f32`, `aread8.cpp:173`) and cast to `f32` on write (or optionally write `f64` via `--out-f64`). Provide `--compat-f32` for exact C++ `f32` rounding in parity tests.
- **Out-of-core ($N > 10^9$, e.g., 4B-cell `Tickfaw`)**: **2D Tile-Graph Flow Accumulation (Barnes 2017 / Zhou-style 2-pass)**:
  - **Pass 1 (parallel over super-tiles)**: For each tile, treat incoming perimeter cells as external sources. Run local Kahn accumulation: (a) compute local interior accumulation `A_local`, and (b) trace/propagate unit impulse or exact linear dependency from each incoming perimeter cell to outgoing perimeter cell(s) (for D8, each perimeter entry exits at most 1 perimeter cell — $O(P_{\text{tile}})$ sparse map!; for Dinf, sparse CSR weights on the tile perimeter).
  - **Global Perimeter Solve**: Topologically sort the inter-tile perimeter graph (~$4\text{M}$ edges for 4B cells) in RAM to compute exact incoming accumulation at every tile perimeter cell.
  - **Pass 2 (parallel over super-tiles, streaming write)**: Add solved incoming perimeter values along local paths inside each tile and emit completed tiles via `GeoTiffWriter::write_tile`.
  - The identical 2-pass tile-graph pattern generalizes to **all** Tier A/B/C DAG tools: up->down (`aread8`, `areadinf`, `gridnet`, `dinfdistup`, `dinfdecayaccum`, `retlimflow`, `flowdircond`, `d8flowpathextremeup`) and down->up (`dinfdistdown` HAND, `d8hdisttostrm`, `d8vdisttostrm`, `gagewatershed`, `dinfrevaccum`, `dinfupdependence`). Note: nonlinear coupled tools (`dinfconclimaccum`, `dinftranslimaccum`, `dinfavalanche`) use iterative perimeter convergence if run out-of-core.
- **Parity**:
  - `aread8` (with identical `p`), `gridnet`, `d8hdisttostrm`, `gagewatershed`, `flowdircond`: **exact** on `<16.7M` cells; `f64` accumulation improves accuracy on $>16.7\text{M}$ cells (bit-exact under `--compat-f32`).
  - `areadinf`, `dinfdistdown` (HAND), other Dinf tools (with identical `ang`): **`rtol <= 1e-6`** (matches `compare_results.py`, accounting for FMA/order-of-addition differences in `f32`/`f64`).

### 2.4 `streamnet`, `moveoutletstostreams` & Elementwise/Reduction Tools
- **`streamnet`**:
  - Replace `linklib.h:165-187` singly-linked `FindLink` ($O(L^2)$) with `Vec<StreamLink>` indexed by `LinkId(u32)` ($O(1)$ lookup) and flat CSR coordinate arrays (`coords_x: Vec<f64>`, `coords_y: Vec<f64>`, `offsets: Vec<u32>`).
  - Single shared-memory 3-pass traversal eliminates all MPI `sendLink`/`recvLink` shuffling (`streamnet.cpp:976-1190`). For $4\times 10^9$ cells, stream cells (`src == 1`) are typically $<1\%$ of pixels ($\sim 10^7$ cells), so Pass 1 & 2 operate over a compact sparse stream-cell graph while Pass 3 (`wsGrid` watershed labeling) uses the same down->up tile-graph propagation as `gagewatershed`.
  - **Parity**: Exact `ord` raster, exact `w` watershed boundaries, exact link geometries, Strahler orders, Shreve magnitudes, lengths, and drop/area attributes. **Acceptable difference**: `LINKNO`/`WSNO` integer IDs are **graph-isomorphic** (not required to match a specific MPI rank count's numbering, since C++ TauDEM's own IDs already change with `-n`).
- **`moveoutletstostreams`**:
  - Open `p` and `src` via `terrio::LazyRaster` with a small LRU tile cache (e.g., 64 tiles = 16 MB). Trace each outlet point downstream up to `-md` cells reading only touched tiles ($O(K \cdot md)$ I/O instead of loading 20 GB rasters). **Exact parity**.
- **Elementwise / Stencil / Table Tools** (`threshold`, `peukerdouglas`, `slopearea`, `slopearearatio`, `lengtharea`, `twi`, `sinmapsi`, `inundepth`, `catchhydrogeo`, `setregion`, `editraster`):
  - Single-pass tiled/strip streaming (`rayon` over tiles, 1-cell halo for `peukerdouglas`/`setregion`). `catchhydrogeo` accumulates per-`(catchment_id, stage)` thread-local `HashMap`s and merges them in $O(C \cdot S)$. `editraster` updates only touched tiles when cloning GeoTIFFs. **Exact parity** (`rtol <= 1e-6` for float arithmetic).
- **`dropanalysis`**: Parallelize across the `num` candidate thresholds (or reuse a single stream-network sweep sorting links by `ssa` threshold in $O(L \log L)$ instead of $T$ full-grid scans!). **Exact parity** on `drp.txt`.

### 2.5 Summary Parity Matrix

| Tool Group | Parity Requirement | Acceptable Differences |
|---|---|---|
| `pitremove`, `threshold`, `peukerdouglas`, `moveoutletstostreams`, `editraster`, `setregion`, `flowdircond` | **Bit-exact** (`max_diff == 0`) | None |
| `d8flowdir`, `dinfflowdir` | **Exact** on steep cells & all cells under `--compat-flats`; `rtol <= 1e-6` on slopes/angles | Default Barnes 2014 flat mode may pick different acyclic paths across multi-cell flats; Rust fixes C++ `i16` overflow on flats >32,767 cells |
| `aread8`, `gridnet`, `d8hdisttostrm`, `d8flowpathextremeup`, `gagewatershed`, `connectdown` | **Exact** on `<16.7M` cells (given identical `p`); `--compat-f32` exact at all sizes | Default `f64` internal accumulation fixes C++ `f32` precision drift above $2^{24}$ cells |
| `areadinf`, `dinfdistdown` (HAND), `dinf*`, `retlimflow`, `slopearea*`, `twi`, `sinmapsi`, `inundepth`, `catchhydrogeo` | **`rtol <= 1e-6`**, exact nodata mask & geotransform | Minor floating-point summation-order noise (`<1e-6` rel) |
| `streamnet`, `catchoutlets`, `dropanalysis` | **Exact** topology, geometry, orders, areas, watersheds (`ord`, `w` partition, `drp.txt`) | `LINKNO`/`WSNO` integer permutation invariance (already MPI-rank-dependent in C++) |

---

## 3. Memory, I/O, Parallelism & CLI Architecture ($10^9$–$10^{10}$ Cells)

### 3.1 Compact Data Representation & Memory Budget

| Grid / State | C++ TauDEM | Rust | Bytes/Cell | $10^8$ cells | $10^9$ cells | $4\times 10^9$ (`Tickfaw`) | $10^{10}$ cells |
|---|---:|---:|---:|---:|---:|---:|---:|
| Elevation (`z`/`fel`) | `f32` | `f32` | 4.0 | 400 MB | 4.0 GB | 16.0 GB | 40.0 GB |
| D8 Flow Dir (`p`) | `i16` (2 B) | `u8` (0..8, 255=nd) | 1.0 | 100 MB | 1.0 GB | 4.0 GB | 10.0 GB |
| Dinf Angle (`ang`) | `f32` | `f32` | 4.0 | 400 MB | 4.0 GB | 16.0 GB | 40.0 GB |
| Accumulation (`ad8`/`sca`) | `f32` | `f64` work -> `f32` out | 8.0 / 4.0 | 800 MB | 8.0 GB | 32.0 GB | 80.0 GB |
| Visited bitset (`pitremove`) | `f32` + stack (~16 B) | `BitVec` (1 bit) | 0.125 | 12.5 MB | 125 MB | 500 MB | 1.25 GB |
| Kahn in-degree (`aread8`) | `i16` (2 B) | `u8` (0..8) | 1.0 | 100 MB | 1.0 GB | 4.0 GB | 10.0 GB |
| **`pitremove` peak (in-mem)** | ~20–24 B/cell | `f32` + bitset + heap | **~5.0** | **0.5 GB** | **5.0 GB** | *out-of-core* | *out-of-core* |
| **`d8flowdir` peak (in-mem)** | ~12–16 B/cell | `f32` + `u8` + sparse flats | **~5.5** | **0.55 GB** | **5.5 GB** | *out-of-core* | *out-of-core* |
| **`aread8` peak (in-mem)** | ~10 B/cell | `u8` + `u8` + `f64` (or `f32`) | **~6–10** | **0.6–1.0 GB** | **6–10 GB** | *out-of-core* | *out-of-core* |
| **Tiled out-of-core peak** ($4096^2$ tile, 8 threads) | N/A (OOMs) | $T_{\text{active}} \times 64\text{–}160\text{ MB}$ + perimeter | **bounded** | — | **~1.5 GB** | **~2–4 GB** | **~3–6 GB** |

### 3.2 Automatic Execution Tiering & Out-of-Core Tile Store
- At startup, query `terrio::usable_memory()` (extended to Windows/macOS in `T4`, §4), defaulting budget to `min(0.65 * usable_ram, --mem-limit)`.
- **Tier 1 (In-Memory Fast Path)**: Selected when peak working set fits within budget (covers all $10^7$–$10^9$ cell terrains on a 32–64 GB workstation). Full arrays allocated once; parallel tiled decode fills them directly via `read_window_f32_into` (`T1`) with zero intermediate copies.
- **Tier 2 (Tiled Out-of-Core Path)**: Selected automatically when `estimated_peak > budget` (e.g., $4\times 10^9$ `Tickfaw` on 32 GB RAM, or $1.5\times 10^{11}$ `AdamsCowBayou`).
  - Super-tile size defaults to $4096 \times 4096$ (aligned to multiples of source GeoTIFF block size, typically 256×256; configurable via `--tile-size`) with a **1-cell halo** fetched directly from `LazyRaster` (no duplicate halo buffers).
  - **Sparse-tile short-circuit**: Before decoding any tile, query `LazyRaster::is_block_sparse()` (`T1`). All-empty/nodata super-tiles skip decode, compute, and write (`SPARSE_OK=YES`, `T2`), making ultra-sparse mosaics like 150B-cell `AdamsCowBayou` tractable in minutes.
  - Intermediate pass state (when a 2-pass algorithm needs scratch tiles) is written to an optional fast local scratch directory (`--scratch-dir`, default OS temp) as raw block-aligned `memmap2` or LZ4-compressed tile files, never holding >`2 × threads` tiles in RAM.

### 3.3 Parallelism & Thread-Pool Hygiene
- Use a single global `rayon` thread pool (`--threads N`, default physical cores) shared across compute and I/O. Because `terrio::par` also uses `rayon` (`crates/terrio/src/par.rs`), running `terrio`'s per-block decode/encode inside a `rayon` tile worker avoids nested thread oversubscription (unlike `geotiff.rs:1159` today, which spawns raw OS threads — fixed in `T1`).
- Overlapping I/O and compute: bounded channel (`crossbeam-channel`, capacity `2 * threads`) feeds decoded tiles to compute workers and compressed tile buffers (`compress_tile`, `T2`) to a single dedicated writer thread flushing sequentially to disk — achieving full NVMe + multi-core saturation with minimal lock contention.

### 3.4 Fused Pipelines (Eliminating Intermediate GeoTIFFs)
A standard HEC-RAS watershed/HAND workflow chains 4–6 tools (`pitremove -> d8flowdir -> aread8 -> threshold -> streamnet` or `pitremove -> dinfflowdir -> dinfdistdown`). On a $1.28\times 10^9$-cell terrain, intermediate files force ~30–50 GB of redundant GeoTIFF compression, disk write, disk read, and decompression.
- Provide `taudem pipeline <recipe>` (and a composable Rust library API `taudem_core::Pipeline`):
  - **D8 Stream Network pipeline**: `pitremove -> d8flowdir -> aread8 -> threshold -> streamnet` — keeps `fel`, `p`, `ad8`, `src` in RAM (or scratch tile store) and writes only requested outputs (`--save-fel`, `--save-p`, `--save-ad8`, `--save-net`, etc.).
  - **HAND / FIM pipeline**: `pitremove -> dinfflowdir -> [d8->aread8->threshold | external src] -> dinfdistdown (v, ave) -> [catchhydrogeo | inundepth]`.
  - Expected savings: **40–60% total wall-time reduction** on $10^8$–$10^9$ cell runs by eliminating intermediate DEFLATE encode/decode stages.

### 3.5 Drop-In TauDEM CLI Compatibility
- Single multicall binary `taudem` dispatching by `argv[0]` basename (so thin symlinks/hardlinks/wrappers named `pitremove.exe`, `d8flowdir.exe`, `aread8.exe`, etc. work identically in existing batch/Python/QGIS/ArcGIS/HEC-RAS scripts) OR subcommand (`taudem pitremove ...`).
- Supports **both** TauDEM invocation styles:
  1. **Simple Usage** (`pitremove logan.tif`): derives default output names via TauDEM's exact `nameadd()` suffix rules (`commonLib.cpp:477-504`: `fel`, `p`, `sd8`, `ang`, `slp`, `ad8`, `sca`, `src`, `ord`, `w`, `net.shp`, `tree.dat`, `coord.dat`, `dd`, etc.).
  2. **Specific Flags**: exact parity with all C++ flags (`-z`, `-fel`, `-p`, `-sd8`, `-ang`, `-slp`, `-ad8`, `-sca`, `-o`, `-lyrname`, `-lyrno`, `-wg`, `-nc`, `-4way`, `-depmask`, `-sfdr`, `-m`, `-md`, `-thresh`, `-par`, etc.).
- **Additive Rust-only long flags** (never colliding with single-dash TauDEM flags): `--threads N`, `--mem-limit <bytes>`, `--tile-size N`, `--scratch-dir <path>`, `--compat-flats`, `--compat-f32`, `--out-f64`, `--compression <lzw|deflate|zstd|none>`, `--tiled [256]`, `--bigtiff <yes|no|if_needed|if_safer>`, `--profile-json <path>`.
- **Output GeoTIFF defaults**: Match TauDEM `.tif` = LZW (or configurable DEFLATE + predictor) and `BIGTIFF=IF_NEEDED`, with `--tiled 256` default for rasters $>10^7$ cells (or striped under `--compat-layout`) and GDAL-compatible nodata tags (`TIFFTAG_GDAL_NODATA`).

---

## 4. Concrete `terrio` Fork Changes (Upstreamable PRs)

### 4.1 Verified Already Present in `terrio` (Do Not Re-Propose)
Verified directly against `H:\CLB-Repos\terrio-clb` (`main` at `90160a28`):
- **Lazy tiled/windowed reads**: `LazyRaster` (`crates/terrio/src/cog.rs:174`), `open_raster_lazy` (`cog.rs:334`), `read_window` (`cog.rs:1071`), `read_window_f32`/`read_window_f64` (`float_read.rs:171, 180`).
- **Positioned thread-safe file I/O**: `FileCursor` (`geotiff.rs:892`) using `read_at`/`seek_read` on shared `Arc<File>` (no shared seek offset).
- **Streaming sequential GeoTIFF/COG writer**: `GeoTiffWriter::write_rows` (`drivers/gtiff_write.rs:2943`) with Rayon chunk compression (`par::map_range_coarse`, `gtiff_write.rs:2051`), TIFF predictors 2 & 3 (`apply_predictor`, `gtiff_write.rs:2112`), and pure-Rust codecs (`NONE`, `LZW`, `DEFLATE`, `PACKBITS`, `ZSTD` via `ruzstd` or `libzstd`, `LERC`, `JXL`, `JPEG`, `WEBP`).
- **Pure-Rust vector I/O**: Read/write `.shp`, `.geojson`/`.json`, `.kml`, `.gpkg`; read-only `.gdb` (`filegdb` feature, `formats.rs:423-456`).
- **Scope boundary**: `ops/dem.rs` implements `gdaldem` primitives (slope/aspect/hillshade/TRI/TPI/roughness). Hydrology algorithms (Priority-Flood, flow direction/accumulation, stream networks) belong in TauDEM, **not** in `terrio`.

### 4.2 Four Self-Contained Upstreamable `terrio` PRs

All four PRs follow `H:\CLB-Repos\terrio-clb\AGENTS.md`: generic beyond TauDEM, pure Rust default, lazy window reads preserved, REUSE/SPDX headers, GDAL-parity + I/O-count tests, and consistent exposure across Rust/CLI/Python/JS where applicable.

#### Feature T1 — Lock-Free Concurrent `&self` GeoTIFF Window Reads, Caller-Owned Buffers & Sparse Block Queries
- **Verified gap**:
  - `LazyRaster::read_window(&mut self)` (`cog.rs:1071`) requires `&mut self`; `LazyRaster::shared(&mut self)` (`cog.rs:343`) wraps in `Arc<Mutex<LazyRaster>>` (`cog.rs:275`), serializing concurrent tile reads across Rayon workers.
  - `decode_in_threads` (`geotiff.rs:1159-1190`) spawns fresh OS threads (`std::thread::Builder::new().spawn_scoped`), calls `reopen()` -> `Decoder::new()` (`cog.rs:1826`, re-parsing the TIFF IFD per worker per window), and copies every decoded chunk on the single calling thread (`place` loop `geotiff.rs:1185-1190`).
  - `TiffRaster::read_samples` (`cog.rs:1769`) allocates a fresh `RasterDataset` + `Vec<T>` on every call; there is no `read_window_into` into a caller's preallocated slab/tile buffer, and no public sparse-block query (`byte_count == 0`).
- **Rust API sketch**:
  ```rust
  impl LazyRaster {
      /// Thread-safe shared window read for immutable backends (local GeoTIFF/COG via FileCursor).
      pub fn read_window_shared(&self, window: RasterWindow, bands: Option<&[usize]>) -> Result<RasterDataset>;
      /// Zero-overhead typed read into a caller-provided slice (band-major, length = bands * w * h).
      pub fn read_window_f32_into(&self, window: RasterWindow, band: usize, dst: &mut [f32]) -> Result<()>;
      pub fn read_window_u8_into(&self, window: RasterWindow, band: usize, dst: &mut [u8]) -> Result<()>;
      pub fn read_window_i16_into(&self, window: RasterWindow, band: usize, dst: &mut [i16]) -> Result<()>;
      /// True when the underlying TIFF block (bx, by) has byte_count == 0 (GDAL SPARSE_OK).
      pub fn is_block_sparse(&self, band: usize, bx: usize, by: usize) -> Option<bool>;
  }
  ```
  Internally: cache parsed IFD block offset/byte-count tables immutably inside `TiffRaster` (`Arc<TiffIfdIndex>`); decode chunks with `terrio::par` (not `spawn_scoped`) directly from `FileCursor` without reopening/re-parsing IFD; allow parallel placement into disjoint output rows.
- **Rationale**: Eliminates mutex serialization, IFD re-parsing, and multi-GB transient allocations when reading 1,000+ tiles in parallel, and enables O(1) sparse-tile skipping on 150B-cell COGs (`AdamsCowBayou`).
- **Tests**: Multi-threaded Rayon concurrent read test asserting byte-identical output vs GDAL across striped, tiled, planar, and sparse GeoTIFFs; I/O-count regression test verifying `0` file re-opens and `0` bytes read for sparse blocks; parity tests for `f32`/`i16`/`u8` + nodata/scale/offset.
- **Public surfaces**: Rust (`terrio::LazyRaster`); Python (`terrio.rio` `out=` buffer parameter on `DatasetReader.read()` matching `rasterio` + `is_block_sparse` helper); JS/WASM (optional `TypedArray` destination buffer overload; single-threaded fallback). CLI unchanged.

#### Feature T2 — Random-Order & Parallel Tiled `GeoTiffWriter`, Real `SPARSE_OK=YES` & Zero-Copy Aligned `write_rows`
- **Verified gap**:
  - `GeoTiffWriter` (`drivers/gtiff_write.rs:2943-3371`) only accepts sequential full-width `write_rows`. Out-of-core tiled algorithms cannot emit `(tx, ty)` tiles out-of-order without buffering an entire raster-width band of rows (`42,810 × 256` or `480,509 × 256` cells).
  - `TiffRows::write_rows` (`gtiff_write.rs:3192-3255`) performs three full-slab copies/reallocations (`merge!`, `slab!`, `keep!`) even when `pending_rows == 0` and input is already block-aligned.
  - `SPARSE_OK=YES` is parsed and validated (`gtiff_write.rs:689, 722, 1014`) but **never applied** — all-nodata tiles are still compressed and written instead of recording `offset = 0, byte_count = 0`.
- **Rust API sketch**:
  ```rust
  pub struct CompressedTile { pub bx: usize, pub by: usize, pub plane: usize, pub data: Vec<u8> }

  impl GeoTiffWriter {
      /// Write one uncompressed tile at block coordinates (bx, by) in any order (TILED=YES required).
      pub fn write_tile(&mut self, bx: usize, by: usize, data: &[u8]) -> Result<()>;
      /// Pure-CPU compression of a tile on any worker thread, decoupled from file I/O.
      pub fn compress_tile(&self, bx: usize, by: usize, plane: usize, data: &[u8]) -> Result<Option<CompressedTile>>;
      /// Append an already-compressed tile to the file and record its IFD offset/byte_count.
      pub fn write_compressed_tile(&mut self, tile: CompressedTile) -> Result<()>;
  }
  ```
  In `write_tile` / `compress_tile` / `write_rows`: when `SPARSE_OK=YES` and all pixels in a tile equal the configured nodata (or `0` when nodata is unset, matching GDAL's `SPARSE_OK` rule), return `None` / record `offset = 0, byte_count = 0`. In `TiffRows::write_rows`, short-circuit directly to chunk compression without `pending` copies when `pending_rows == 0 && input_rows % block_rows == 0`.
- **Rationale**: Essential for streaming out-of-core $4\times 10^9+$ cell outputs from parallel workers without holding full-width strips in RAM, cuts peak write memory 50%+, and shrinks sparse watershed/HAND GeoTIFFs by 2–10×.
- **Tests**: GDAL-parity roundtrip tests writing tiles in shuffled order (including partial edge tiles, predictor 2/3, LZW/DEFLATE/ZSTD, BigTIFF); GDAL-parity `SPARSE_OK=YES` test asserting file size reduction and exact pixel/nodata readback via both `terrio` and GDAL (`scripts/conformance.py`); benchmark verifying zero extra buffer copies on aligned `write_rows`.
- **Public surfaces**: Rust (`GeoTiffWriter`); Python (`terrio.rio` `DatasetWriter.write(arr, window=...)` block-aligned tiled write + `SPARSE_OK=YES` creation option); CLI (`terrio translate -co SPARSE_OK=YES`, already accepted as option — now actually effective); JS/WASM (in-memory tiled buffer writer).

#### Feature T3 — Pure-Rust SIMD `zlib-rs` DEFLATE on Core `geotiff` Feature + Explicit `BigTiffPolicy`
- **Verified gap**:
  - Workspace `Cargo.toml:66` sets `flate2 = "1"` (pure-Rust `miniz_oxide`), while `parquet` (`Cargo.toml:47`) activates `flate2-zlib-rs`. A consumer building with `default-features = false, features = ["geotiff"]` (like `G:\GH\TauDEM\rust\Cargo.toml:10`) silently loses `zlib-rs` and falls back to 2–4× slower `miniz_oxide` on DEFLATE GeoTIFFs (the exact compression of almost all HEC-RAS terrains!).
  - `gtiff_write.rs:865-871` collapses `BIGTIFF=NO`, `IF_NEEDED`, and `IF_SAFER` all into `bigtiff = false`, while `TiffRows::create` (`gtiff_write.rs:3116-3117`) unconditionally upgrades to BigTIFF when `raw + raw / 4 > 4 GiB - 16 MiB` — violating `BIGTIFF=NO`, risking `usize` overflow on 32-bit targets before `as u64`, and failing at `check_size()` (`gtiff_write.rs:3288`) under `IF_NEEDED` when compressed data crosses 4 GiB.
- **Rust API sketch**:
  - In `H:\CLB-Repos\terrio-clb\Cargo.toml`: enable `flate2 = { version = "1", default-features = false, features = ["zlib-rs"] }` on native targets (pure Rust, zero C compiler, SIMD-accelerated; keep `rust_backend` on `wasm32` if needed).
  - Replace `bigtiff: bool` internally (keeping `GeoTiffOptions::bigtiff` deprecated/compat alias) with:
    ```rust
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum BigTiffPolicy { #[default] IfNeeded, IfSafer, Yes, No }
    ```
    Honor `No` (error cleanly if >4 GiB), `Yes` (always BigTIFF), `IfSafer` (BigTIFF if uncompressed `u64` estimate >4 GiB, matching GDAL), and `IfNeeded` (classic TIFF if uncompressed fits OR compressed fits; clean error or auto-upgrade if streaming exceeds 4 GiB). Use `u64` checked math for `raw`.
- **Rationale**: Immediate 2–4× read/write speedup on DEFLATE HEC-RAS terrains (`Muncie`, `RAS_7.0_Template`, `Amite`, `Tickfaw`) with zero C dependencies, plus GDAL-conformant BigTIFF behavior at the 4 GiB boundary.
- **Tests**: Unit + GDAL-parity tests for all four `BIGTIFF` values across `<4 GiB` and `>4 GiB` synthetic sparse/dense rasters; DEFLATE decode/encode throughput microbenchmark verifying `zlib-rs` active with `--no-default-features --features geotiff`; WASM build check.
- **Public surfaces**: Rust (`BigTiffPolicy`, `GeoTiffOptions`); CLI/Python/JS (`-co BIGTIFF=YES|NO|IF_NEEDED|IF_SAFER` already parsed — behavior now matches GDAL).

#### Feature T4 — Cross-Platform `usable_memory()` (Windows/macOS) + `LazyRaster::block_windows()` Iterator
- **Verified gap**:
  - `crates/terrio/src/dataset.rs:1894-1947`: `usable_memory()` only reads Linux `/proc/meminfo` and `/proc/self/limits`, returning `None` on Windows and macOS — even though `windows-sys` is already a dependency (`crates/terrio/Cargo.toml:187`).
  - `block_windows` exists in Python (`python/python/terrio/rio/io.py:564`) but not as a first-class Rust/JS iterator on `LazyRaster`.
- **Rust API sketch**:
  ```rust
  // dataset.rs: add Win32_System_SystemInformation feature to windows-sys for GlobalMemoryStatusEx,
  // and sysctl(HW_MEMSIZE) / vm_statistics64 on macOS.
  pub fn usable_memory() -> Option<u64>;

  impl LazyRaster {
      /// Yield ((bx, by), RasterWindow) for every native tile/strip block, optionally clipped to a region.
      pub fn block_windows(&self) -> impl Iterator<Item = ((usize, usize), RasterWindow)>;
      /// Yield aligned super-tile windows (multiples of native block size) with optional halo.
      pub fn tiled_windows(&self, tile_w: usize, tile_h: usize, halo: usize) -> impl Iterator<Item = TileWindow>;
  }
  ```
- **Rationale**: TauDEM's primary target workstations are Windows (HEC-RAS/MS-MPI) and Linux; automatic in-memory vs out-of-core tiering requires working `usable_memory()` on Windows. `block_windows()`/`tiled_windows()` eliminates boilerplate block-alignment math across TauDEM and all `terrio` users.
- **Tests**: Platform unit tests asserting `usable_memory().is_some()` and plausible bounds on Windows/Linux/macOS; `block_windows()` / `tiled_windows()` coverage tests verifying exact partition of non-multiple raster dimensions and clipped edge halos.
- **Public surfaces**: Rust (`terrio::usable_memory`, `LazyRaster::block_windows`, `LazyRaster::tiled_windows`); Python (`terrio.rio` delegates to Rust iterator); JS/WASM (`blockWindows()` method on `LazyRaster`).

### 4.3 Secondary Format Gaps (Handled in TauDEM, Not Core `terrio` PRs)
- **HFA (`.img`) write**: `terrio` reads HFA purely in Rust (`formats.rs:240`) but does not write it (`gridtypes` test case writes `.img` via `tiffIO.cpp:269`).
- **SpatiaLite (`.sqlite`) vector**: `terrio`'s `sqlite` feature covers GeoPackage (`.gpkg`) and MBTiles, not SpatiaLite `.sqlite` (`formats.rs:421-456`; exercised by `LoganSampleSQLite`).
- **Strategy**: Pure-Rust TauDEM supports `.tif`/`.tiff`/`.sdat`/`.bil`/`.bin` rasters and `.shp`/`.gpkg`/`.geojson`/`.kml`/`.gdb`(read) vectors natively; optional `--features gdal` (delegating through `terrio`'s `gdal` feature) or explicit warning/fallback handles `.img` write and `.sqlite` SpatiaLite when requested.

---

## 5. Rust Workspace Layout, `terrio` Dependency & Licensing

### 5.1 Cargo Workspace Layout (`G:\GH\TauDEM\rust/`)

Evolve the existing stub `G:\GH\TauDEM\rust\Cargo.toml` (`edition = "2024"`, `rust-version = "1.94"`) into a clean multi-crate workspace so implementation agents can work in parallel with fast incremental rebuilds:

```text
G:\GH\TauDEM\rust\
├── Cargo.toml                  # Workspace root (resolver = "3", shared deps, release LTO profile)
├── Cargo.lock
└── crates\
    ├── taudem-core\            # Grid<T>, BitGrid, TileSpec, TileStore (mem/mmap scratch),
    │                           # WGS84 dxc/dyc (tiffIO.cpp parity), MINEPS nodata, D8/Dinf
    │                           # offset tables, PriorityQueue/RadixHeap, TileGraph DAG solver
    ├── taudem-io\              # terrio adapter: LazyRaster tile loader, GeoTiffWriter sink,
    │                           # ReadOutlets/vector writer, nameadd() suffix rules, timing stats
    ├── taudem-d8\              # pitremove (Barnes 2014+2016), d8flowdir (steep + Barnes 2014
    │                           # & --compat-flats), aread8, gridnet, threshold, peukerdouglas,
    │                           # d8hdisttostrm, d8vdisttostrm, d8flowpathextremeup, slopeavedown,
    │                           # flowdircond, setregion
    ├── taudem-dinf\            # dinfflowdir, areadinf, dinfdistup, dinfavalanche,
    │                           # dinfconclimaccum, dinfdecayaccum, dinfrevaccum,
    │                           # dinftranslimaccum, dinfupdependence, retlimflow
    ├── taudem-net\             # streamnet (O(1) Vec<StreamLink> + CSR coords), moveoutletstostreams,
    │                           # gagewatershed, connectdown, catchoutlets, dropanalysis
    ├── taudem-fim\             # dinfdistdown (HAND), catchhydrogeo, inundepth, slopearea,
    │                           # slopearearatio, lengtharea, twi, sinmapsi, siregion, editraster
    ├── taudem-cli\             # Multicall `taudem` binary + tool shims + `pipeline` subcommand,
    │                           # TauDEM simple & flag CLI parser, JSON profiling output
    └── taudem-parity\          # Integration test & benchmark runner (TauDEM-Test-Data + synthetic
                                # adversarial + HEC-RAS terrain tiers vs C++ MPI baseline)
```

### 5.2 `terrio` Fork Dependency & Feature Matrix
In `G:\GH\TauDEM\rust\Cargo.toml`:
```toml
[workspace.dependencies]
rayon = "1.12"
crossbeam-channel = "0.5"
memmap2 = "0.9"
terrio = { git = "https://github.com/CLB-Engineering-Corporation/terrio", branch = "main", default-features = false, features = ["geotiff", "sqlite", "filegdb"] }
```
- **During co-development**: track `branch = "main"` (or a feature branch / `[patch."https://github.com/CLB-Engineering-Corporation/terrio"]` local path override to `H:\CLB-Repos\terrio-clb` in `.cargo/config.toml` ignored by git); pin `rev = "<sha>"` for release tags and reproducible benchmark runs.
- **Pass-through workspace features** on `taudem-cli` / `taudem-io`:
  - Default: pure Rust (`terrio` `geotiff` + `sqlite` for `.gpkg` + `filegdb` for `.gdb` outlets).
  - `lerc`: enables `terrio/lerc` (required for LERC-compressed RBFS COGs noted in `TERRAINS.md:23-24`).
  - `libzstd`: enables `terrio/libzstd` (native C ZSTD for max compression ratio/speed when writing ZSTD GeoTIFFs).
  - `gdal`: enables `terrio/gdal` fallback for legacy HFA `.img` writes and SpatiaLite `.sqlite`.

### 5.3 Licensing Compatibility
- **TauDEM**: `G:\GH\TauDEM\license.txt:1-15` (2014) places TauDEM under **GPL-3.0** (`GPLv3license.txt`), with a multi-licensing notice (`license.txt:16-20`) and a copyright-assignment CLA for contributions back to `dtarb/TauDEM` (`license.txt:29-32`). Older `.cpp` file headers (2010) say GPL-2.0, and the current stub `G:\GH\TauDEM\rust\Cargo.toml:6` has `license = "GPL-2.0-or-later"`.
- **`terrio`**: `H:\CLB-Repos\terrio-clb\crates\terrio\Cargo.toml:11` declares `license = "MIT AND MPL-2.0"`. Inspection confirms the `MPL-2.0` component is **solely** `pointcloud/filters/selfadjoint.rs` (ported from Eigen, gated behind the `pointcloud` feature — disabled in our build); all core raster/vector modules are **MIT**.
- **Compatibility verdict**:
  - **MIT** is compatible with both GPL-2.0 and GPL-3.0.
  - **MPL-2.0** is compatible with **GPL-3.0** (MPL-2.0 §3.3 Secondary Licenses) and **GPL-2.0-or-later** (upgraded to GPLv3), but **NOT** with GPL-2.0-only.
  - **Recommendation**: Set the Rust workspace `license = "GPL-3.0-or-later"` (matching `license.txt` and safely compatible with both `terrio`'s MIT core and any future `MPL-2.0` feature). Keep `taudem-core` clean of copied C++ source where algorithms are fresh Barnes 2014/2016/2017 implementations, while clearly attributing Tarboton's TauDEM algorithms/formulas where ported.

---

## 6. Benchmark & Validation Plan

### 6.1 Layer 1 — Upstream `dtarb/TauDEM-Test-Data` Parity Harness
- **Dataset**: `github.com/dtarb/TauDEM-Test-Data` (`master`, 1,963 files, 28 input case folders, 81 input rasters). All inputs are small (7.5K to 12.96M cells; largest probed TIFFs: `MoveOutlets3/p.tif` $3600\times 3600 = 12.96\text{M}$ i16 LZW; `CatchOutlets/fdr.tif` $2704\times 4104 = 11.10\text{M}$ u8; `CatchHydroGeo/slp.tif` & `Inundepth/hand.tif` & `GDAL_unset_nodata/weights.tif` $2628\times 1171 = 3.08\text{M}$; `sinmapsi/dmcal.tif` $1582\times 1918 = 3.03\text{M}$; `gwunittest/fdr.tif` $1442\times 1546 = 2.23\text{M}$; `Base/demDoublewgt.tif` $1453\times 1440 = 2.09\text{M}$ f64 tiled; `FlowdirCond/wcdem.tif` $1690\times 986 = 1.67\text{M}$; `Base/logan.tif` $985\times 1660 = 1.64\text{M}$ f32 striped; `Geographic/enogeo.tif` $715\times 844 = 603\text{K}$ geographic CRS; `Editraster` $9.6\text{K}$; `SetRegion` $7.5\text{K}$).
- **Harness (`taudem-parity`)**:
  - Uses existing `G:\GH\TauDEM\bench\inventory.py` to index local clone (`TAUDEM_TEST_DATA`) and replays every command in `Input/testall.bat` / `Input/taudem-tests.sh` against both C++ TauDEM and `taudem-cli`.
  - Enforces `compare_results.py` checks (dimensions, geotransform `rtol=1e-5`, exact nodata mask, pixel `max_rel_diff <= 1e-6`, `max_diff == 0` for integer/exact tools), plus vector topology comparison for `streamnet`/`moveoutletstostreams`/`connectdown`/`catchoutlets` (graph-isomorphic link matching ignoring `LINKNO` permutation) and CSV comparison for `catchhydrogeo`/`inundepth`/`dropanalysis`.
  - Runs in both `--compat-flats --compat-f32` mode (strict regression gate) and default fast mode (asserting steep-cell parity + flat-basin equivalence + zero cycles).

### 6.2 Layer 2 — Synthetic Adversarial Edge-Case Suite
Programmatically generated rasters in `taudem-parity` testing bugs/limits absent from `TauDEM-Test-Data`:
1. **Giant flat ($40,000 \times 2$ or $500 \times 500$ nested flat)**: verifies Rust `u32` flat labels succeed without overflow (exposes C++ `i16` overflow at >32,767).
2. **Spiral labyrinth & nested multi-spill pits**: stresses Priority-Flood worst case where Planchon-Darboux degrades to $O(N^{1.5})$.
3. **High-latitude Geographic CRS ($60^\circ$–$75^\circ$ N)**: verifies row-dependent WGS84 `dxc[j]`, `dyc[j]` ellipsoidal parity.
4. **Nodata edge cases**: unset nodata (must default to `-9999`), IEEE `NaN`, `-3.4028235e38`, values within `MINEPS = 1e-5f` of nodata, and all-nodata sparse tiles.
5. **Large-basin accumulation ($2\times 10^7$ synthetic inclined plane)**: verifies `f64` exact count ($20,000,000$) vs `--compat-f32` truncation above $2^{24}$.
6. **Constraint flags & topology**: `-4way`, `-depmask`, `-sfdr`, `-nc`, multi-inflow stream junctions ($k \ge 3$), non-tile-multiple dimensions ($1027 \times 773$), and `twi` (missing from upstream test data).

### 6.3 Layer 3 — Large HEC-RAS Terrain Benchmark Matrix
Copy candidate terrains from [`docs/rust-port/TERRAINS.md`](TERRAINS.md) to local NVMe SSD before timing to exclude SMB/network variance:

| Tier | Terrain (`TERRAINS.md`) | Dimensions (Cells) | File Size / Format | Benchmark Role |
|---|---|---|---:|---|
| **$10^7$** | `Terrain.muncie_clip.tif` | $7,892 \times 4,538$ ($3.6\times 10^7$) | 17.9 MB (DEFLATE 256²) | Fast CI/dev perf smoke test |
| **$10^7$** | `Spring Creek .../terrain.tif` | $10,323 \times 5,380$ ($5.6\times 10^7$) | 116 MB (LZW striped) | Real-world parity against existing `taudem_work/` outputs |
| **$10^8$** | `Terrain.dem_mosaic_5070_basin_extent.tif` | $11,630 \times 7,158$ ($8.3\times 10^7$) | 150 MB (DEFLATE 256²) | Medium in-memory scaling tier |
| **$10^9$** | `Terrain.cudem_ft.tif` & `2022LiDAR+CoNED.tif` | $27.7\text{K}\times 33.4\text{K}$ ($9.2\times 10^8$) / $36\text{K}\times 31.6\text{K}$ ($1.14\times 10^9$) | 517–621 MB (DEFLATE 256²) | Local-disk $10^9$-cell workstation target |
| **$10^9+$ I/O** | `Amite WA1` (`Terrain.DEM_10ft.tif` vs `DEM_10ft.tif`) | $25.5\text{K}\times 50.3\text{K}$ ($1.28\times 10^9$) / $25\text{K}^2$ ($1.25\times 10^9$) | 1.86 GB DEFLATE vs 5.02 GB raw | Isolates DEFLATE codec throughput (`T3`) vs pure NVMe I/O |
| **$4\times 10^9$** | `Tickfaw` (`Terrain_Repaired.WS1_Repaired3.tif`) | $42,810 \times 93,236$ ($3.99\times 10^9$) | 2.86 GB (DEFLATE 256²) | Out-of-core tiled engine target (C++ MPI typically OOMs/stalls) |
| **Stress** | `AdamsCowBayou_terrain_cog.tif` | $480,509 \times 316,295$ ($1.5\times 10^{11}$) | 5.46 GB (Int16 sparse COG) | Sparse-tile skip (`T1`/`T2`) + BigTIFF stress test |
| **Codec** | RBFS `SpringCreek` / `UpperSanMarcos` COGs | *(see `TERRAINS.md:23-24`)* | ZSTD/LERC COG | Validates `terrio` pure-Rust ZSTD/LERC decode where local GDAL fails |

### 6.4 Metrics & Baseline Protocol
- **Baseline**: Installed TauDEM at `C:\Program Files\TauDEM\TauDEM5Exe` invoked via `C:\Program Files\Microsoft MPI\Bin\mpiexec.exe -n {1, 4, 8, 16}`.
  - *Baseline gap note*: `C:\Program Files\TauDEM\TauDEM5Exe` contains **30 executables** (TauDEM 5.3.x); 7 newer `b994ed2` Develop tools (`CatchHydroGeo`, `CatchOutlets`, `EditRaster`, `InunDepth`, `RetLimFlow`, `SetRegion`, `SIRegion`) are absent and require a local CMake build from `G:\GH\TauDEM\src` for baseline timing/parity.
- **Recorded metrics** (emitted via `--profile-json` for Rust and Windows Job Object / `GetProcessMemoryInfo` wrapper for C++ MPI across all ranks):
  1. **Total wall-clock time** (median of 3 warm-cache runs + 1 cold-cache run on $10^9+$ tiers).
  2. **Peak RSS** (sum of peak working sets across all MPI ranks for C++ vs single-process peak RSS for Rust).
  3. **Phase split**: `read_ms` (file open + decode), `compute_ms` (algorithm + flat/perimeter solve), `write_ms` (encode + flush).
  4. **I/O volume**: bytes read from disk, bytes written to disk, sparse tiles skipped.
  5. **Thread scaling**: Rust `--threads {1, 4, 8, 16, N_cores}` vs C++ `mpiexec -n {1, 4, 8, 16}` on both standalone tools (`pitremove`, `d8flowdir`, `aread8`, `dinfflowdir`, `dinfdistdown`, `streamnet`) and full end-to-end watershed/HAND pipelines.

---

## 7. Phased Implementation Plan (Agent-Sized Work Packages)

Each work package (WP) is sized for a single implementation agent with clear inputs, outputs, and acceptance criteria.

```text
  [WP-T1] [WP-T2] [WP-T3] [WP-T4]   (terrio fork PRs — all 4 parallel)
     \       \       |       /
      +-------+------+------+------------------+
                     |                         |
  [WP0: Workspace + CLI + Parity Harness]      | (parallel with WP-T1..T4)
                     |                         |
              [WP1: pitremove]                 |
                     |                         |
              [WP2: d8flowdir] --------+       |
              /              \         |       |
      [WP3: aread8]    [WP5: dinfflowdir + areadinf]   (WP3 || WP5)
        /       \              |
  [WP4: thresh, [WP6:      [WP7: dinfdistdown (HAND),  (WP4 || WP6 || WP7)
   gridnet,      streamnet] catchhydrogeo, inundepth]
   moveoutlets,       \        |
   peuker]             +-------+---------------+
                               |
               [WP8: Tiled Out-of-Core (4e9+) & Fused Pipelines]  (needs WP-T1..T4 + WP1..WP7)
                               |
         [WP9a: D8 ext]  [WP9b: Dinf ext]  [WP9c: Elementwise/SINMAP]  (all 3 parallel)
```

| WP | Scope & Deliverables | Depends On | Parallel With | Acceptance Criteria |
|---|---|---|---|---|
| **WP-T1** | `terrio` PR: lock-free `&self` `read_window_shared`, `read_window_{f32,i16,u8}_into`, `is_block_sparse`, Rayon chunk decode w/ cached IFD | None | WP-T2..T4, WP0 | `cargo clippy -D warnings`, GDAL-parity + 0-reopen + sparse-skip tests green; Python/WASM checks pass |
| **WP-T2** | `terrio` PR: `GeoTiffWriter::write_tile`, `compress_tile`, `write_compressed_tile`, real `SPARSE_OK=YES`, zero-copy aligned `write_rows` | None | WP-T1, T3, T4, WP0 | Shuffled-tile & `SPARSE_OK=YES` GDAL conformance tests green; measurable reduction in write RAM/size |
| **WP-T3** | `terrio` PR: `zlib-rs` on core `geotiff` feature + `BigTiffPolicy` (`Yes`/`No`/`IfNeeded`/`IfSafer`) w/ `u64` math | None | WP-T1, T2, T4, WP0 | $\ge 2\times$ DEFLATE decode speedup w/ `--no-default-features --features geotiff`; all 4 `BIGTIFF` modes GDAL-verified |
| **WP-T4** | `terrio` PR: Windows (`GlobalMemoryStatusEx`) + macOS `usable_memory()`, `LazyRaster::block_windows()` & `tiled_windows()` | None | WP-T1..T3, WP0 | `usable_memory().is_some()` on Windows & Linux CI; edge/halo window partition tests green |
| **WP0** | Convert `rust/` into multi-crate workspace; `taudem-core` (`Grid`, `BitGrid`, `dxc`/`dyc`, `MINEPS` nodata); `taudem-io` (`terrio` wrapper, `nameadd`); `taudem-cli` multicall & `--profile-json`; `taudem-parity` harness + C++ MPI baseline script | None | WP-T1..T4 | `cargo test --workspace` green; parity harness runs C++ baseline on `Base/logan.tif` and `Muncie` and records baseline JSON |
| **WP1** | `pitremove` (Barnes 2014 Improved Priority-Flood, `-4way`, `-depmask`, in-memory) in `taudem-d8` | WP0 | — | Bit-exact (`max_diff == 0`) on all 6 `TauDEM-Test-Data` pitremove cases + synthetic spiral pits; $\ge 5\times$ faster than 8-rank C++ MPI on `RAS_7.0_Template` ($8.3\times 10^7$) |
| **WP2** | `d8flowdir` (steep `dontCross`, geographic `dxc`/`dyc`, `-sfdr`, Barnes 2014 $O(N)$ BFS flats + `--compat-flats` Garbrecht-Martz) | WP1 | — | Bit-exact on all `TauDEM-Test-Data` `d8flowdir` cases under `--compat-flats`; steep-cell exact + acyclic zero-slope flats under default Barnes 2014; passes 40K-wide synthetic flat without overflow |
| **WP3** | `aread8` (compact `u8` D8 + in-degree, parallel sub-basin Kahn, `-o` outlets, `-wg`, `-nc`, `f64` default + `--compat-f32`) | WP2 | WP5 | Exact parity on all `TauDEM-Test-Data` `aread8` cases (`--compat-f32`); exact count on $2\times 10^7$-cell synthetic ramp (default `f64`); $\ge 5\times$ faster than C++ MPI on $8.3\times 10^7$ |
| **WP4** | `threshold`, `gridnet`, `moveoutletstostreams` (lazy tile-cached trace), `peukerdouglas` | WP2, WP3 | WP5, WP6, WP7 | Exact parity on `Base`, `Gridnet`, `MoveOutlets2/3`, `MovedOutletstoStream_data`, `peukerDouglas`; `moveoutletstostreams` reads <1% of tiles on $10^9$ test raster |
| **WP5** | `dinfflowdir` (Tarboton 8-facet `f64` + Barnes/compat flats) + `areadinf` (Dinf Kahn SCA, `-o`, `-wg`, `-nc`) | WP1 | WP3, WP4 | `rtol <= 1e-6` parity across `Base`, `Geographic`, `AreaDinf`, `sinmapsi`; exact nodata masks |
| **WP6** | `streamnet` (`Vec<StreamLink>` $O(1)$ lookup, CSR coords, `ord`/`w`/`tree.dat`/`coord.dat`/vector `net` w/ 17 attrs) | WP3, WP4 | WP7 | Exact `ord`, `w` watershed geometry, link geometry/orders/magnitudes/areas on `Base`, `Geographic`, `streamnet_data`; linear scaling on $10^6$-link synthetic network |
| **WP7** | FIM/HAND suite: `dinfdistdown` (`h`/`v`/`p`/`s` × `ave`/`min`/`max`, `-wg`, `-nc`), `catchhydrogeo`, `inundepth` | WP5 | WP4, WP6 | `rtol <= 1e-6` parity on `Base`, `CatchHydroGeo`, `Inundepth`; end-to-end HAND parity vs `Spring Creek` (`5.6e7`) |
| **WP8** | Tiled Out-of-Core Engine (Barnes 2016 parallel tile `pitremove`, cross-tile flat stitching, 2-pass tile-graph `aread8`/`areadinf`/`dinfdistdown`/`streamnet` watersheds, sparse-tile skip) + `taudem pipeline` fused D8 & HAND pipelines | WP-T1..T4, WP1..WP7 | — | Completes full D8 + HAND pipelines on `Amite WA1` ($1.28\times 10^9$) and `Tickfaw` ($3.99\times 10^9$) within bounded RAM (<8 GB in out-of-core mode), sparse-skips `AdamsCowBayou`, and fused pipeline beats unfused wall time by $\ge 35\%$ |
| **WP9a** | Remaining D8 tools: `d8hdisttostrm`, `d8vdisttostrm`, `d8flowpathextremeup`, `slopeavedown`, `gagewatershed`, `connectdown`, `flowdircond`, `catchoutlets`, `dropanalysis` | WP3, WP6 | WP9b, WP9c | Parity green across `Base`, `D8flowextreme`, `GageWatershed`, `gwunittest`, `fts`, `ConnectDown`, `FlowdirCond`, `CatchOutlets` |
| **WP9b** | Remaining Dinf tools: `dinfdistup`, `dinfavalanche`, `dinfconclimaccum`, `dinfdecayaccum`, `dinfrevaccum`, `dinftranslimaccum`, `dinfupdependence`, `retlimflow` | WP5 | WP9a, WP9c | Parity green across `Base`, `DinfConcLimAccum`, `DinfTransLimAcc`, `RetLimFlow` |
| **WP9c** | Elementwise & SINMAP/RWD tools: `slopearea`, `slopearearatio`, `lengtharea`, `twi`, `sinmapsi`, `siregion`, `editraster`, `setregion` | WP0, WP2 | WP9a, WP9b | Parity green across `Base`, `sinmapsi`, `Editraster`, `SetRegion` + synthetic `twi` test |

---

## 8. Risks & Open Questions

### 8.1 Technical Risks & Mitigations
1. **Flat-resolution divergence cascading downstream**: Different tie-breaking inside wide flats alters `p`/`ang`, which shifts `aread8`/`streamnet` lines inside flat floodplains. *Mitigation*: Dual mode — `--compat-flats` guarantees exact TauDEM regression parity; default Barnes 2014 gives $O(N)$ speed and eliminates C++ `i16` overflow.
2. **Dinf tile-graph perimeter density in out-of-core mode**: Unlike D8 (at most 1 exit per tile-entry pixel), Dinf splits flow across 2 neighbors so a perimeter entry can fan out across many exit pixels. *Mitigation*: Use sparse CSR perimeter propagation pruned at TauDEM's existing `1e-5` weight cutoff (`commonLib.cpp:89`), falling back to iterative tile-halo sweeps if perimeter density exceeds threshold.
3. **Windows memory pressure & pagefile thrashing at $4\times 10^9$ cells**: Windows working-set trimming can silently page instead of OOMing, destroying throughput. *Mitigation*: Implement `GlobalMemoryStatusEx` in `T4`, default memory budget conservatively to `60–65%` of available physical RAM (`ullAvailPhys`), and enforce explicit out-of-core tiling above that threshold.
4. **Upstream `terrio` divergence**: Carrying long-lived fork patches on `CLB-Engineering-Corporation/terrio` risks merge conflicts with `opengeos/terrio`. *Mitigation*: Keep `T1`–`T4` small, generic, GDAL-tested, and submitted upstream to `opengeos/terrio` immediately after landing in the CLB fork.
5. **Non-deterministic `streamnet` link IDs in downstream scripts**: Scripts hardcoding specific `LINKNO` integers from C++ TauDEM may break if IDs shift. *Mitigation*: Sort headwater seed discovery deterministically in row-major order (matching 1-rank C++ TauDEM) so Rust `LINKNO`s are 100% deterministic across thread counts and match single-rank C++ runs.
6. **`siregion` polygon rasterization without GDAL**: `SIRegionToolmn.cpp` currently invokes GDAL `GDALRasterizeGeometries` and `GDALAutoCreateWarpedVRT`. *Mitigation*: Use `terrio`'s pure-Rust rasterize/warp operations (or gate `siregion` full warp behind `terrio/gdal` if any edge-case projection diverges).

### 8.2 Open Questions for the User
1. **Default flat resolution**: Should `d8flowdir`/`dinfflowdir` default to **Barnes (2014) $O(N)$** (fastest, no `i16` overflow, slight path variation inside wide flats) with `--compat-flats` for exact C++ TauDEM Garbrecht-Martz parity — or vice versa?
2. **`aread8` accumulation precision**: Should default `aread8` accumulate in **`f64`** (exact counts on $>16.7\text{M}$-cell basins, cast to `f32` GeoTIFF by default, with `--out-f64` option) and offer `--compat-f32` for C++ rounding parity?
3. **Default output GeoTIFF layout**: C++ TauDEM writes **striped LZW** `.tif` files. For large terrains, **tiled 256×256 DEFLATE + predictor** (with `SPARSE_OK=YES` and `BIGTIFF=IF_NEEDED`) is much faster to write and read in HEC-RAS/QGIS/GDAL. Should tiled DEFLATE be the default for rasters $>10^7$ cells (with `--compat-layout` for striped LZW), or always match C++ LZW?
4. **License identifier in `rust/Cargo.toml`**: The stub currently says `GPL-2.0-or-later`, while `G:\GH\TauDEM\license.txt` is **GPL-3.0**. Confirm switching the Rust workspace to `GPL-3.0-or-later` (which also maintains clean compatibility with `terrio`'s `MIT AND MPL-2.0`).
5. **Legacy format scope (`.img` write, `.sqlite` SpatiaLite)**: Are HFA `.img` output and SpatiaLite `.sqlite` streamnet outputs needed in pure-Rust builds, or is supporting them via `--features gdal` (and `.tif`/`.shp`/`.gpkg`/`.geojson` in pure Rust) sufficient?
6. **C++ MPI baseline for the 7 newer tools**: Confirm whether to build `G:\GH\TauDEM\src` locally with CMake/vcpkg to benchmark/validate the 7 Develop tools absent from `C:\Program Files\TauDEM\TauDEM5Exe` (`CatchHydroGeo`, `InunDepth`, `CatchOutlets`, `EditRaster`, `RetLimFlow`, `SetRegion`, `SIRegion`), or compare those 7 solely against `TauDEM-Test-Data` expected outputs.
