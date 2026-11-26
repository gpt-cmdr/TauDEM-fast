# TauDEM C++ vs taudem-rs on real LiDAR terrains — CLB-t3, November 2026

Measured 2026-11-25/26 UTC on CLB-t3. Every input is a real HEC-RAS / FEMA LiDAR terrain; no synthetic DEMs are used in the main results (a short synthetic appendix from an earlier, interrupted sweep is labelled as such).

## Machine and builds

- **CLB-t3**: Proxmox guest, Debian 12.12, kernel 7.0.14-23-pve. Intel Core i9-11900 @ 2.50 GHz, 6 cores visible to the guest (SMT off), governor `powersave`. 32 GiB RAM, 512 MiB swap. The 80% RAM budget used by the memory guard is 25.6 GiB.
- Inputs and outputs on local `/home/clb/scratch` (ZFS on SSD, lz4 dataset). Terrains were copied from the NAS before timing; copy time is not counted. Each configuration ran alone, with the input read once (prefetch) before each run.
- **taudem-rs** `rust-port` @ `fca2623` ("Accumulate AreaD8 in parallel"), `cargo build --release --locked`, Rust 1.99.0, binary sha256 `93df90ac…`. Used for every in-memory Rust row and the parity reference. The streaming section uses `main` @ `7ba87f4` (sha256 `d7740f48…`).
- **C++ TauDEM** 5.6.0-beta, the Linux build from the earlier CLB-t3 sweep at `/home/clb/scratch/taudem-bench-clb-t3-2026-11-25/build-cpp/src` (source `86b02be`, whose `src/` matches upstream Develop `b994ed2`). CMake, GCC 12.2.0 `-O3 -DNDEBUG`, MPICH/Hydra 4.0.2, GDAL 3.6.2. sha256 pitremove `011d7aff…`, d8flowdir `b113a06f…`, aread8 `cbb63d44…`.
- Harness: `real/realbench.py` in the scratch directory (wraps the same process-tree sampler as `bench/bench.py`; adds `--profile-json` per run, a 3 h chain cap for C++, and a RAM guard). Python 3.14.8, rasterio 1.5.2 (GDAL 3.12.2), psutil 7.2.2.

## Protocol

- Chain: PitRemove → D8FlowDir → AreaD8, writing fel, p, sd8 and ad8. Rust ran fused (`taudem-rs pipeline`) and as three separate tools, with `RAYON_NUM_THREADS` 1 and 6, 3 repeats each, `--profile-json` saved for every run. Wall is the median.
- C++ ran under `mpiexec.hydra -n {6,4,1}`, one run per rank count, with a 3 h cap on the whole chain.
- Wall includes process (and MPI) launch. Peak RSS is the highest per-step sum of every worker process's VmHWM (summed over ranks for C++). B/cell divides peak RSS by total cells (nodata included).
- Read, compute and write are the tools' own printed times summed over the chain (Rust: the profile's read/compute/write).
- "Background" is system CPU busy time not used by the benchmark process tree or the driver, averaged over the run, in cores. It was about 1 core during the Rust runs. Most of that is kernel work, chiefly ZFS compression and checksums for the output writes. About 0.3 core of it is a helper that the agent session launched (`owner_job_store.py`, roughly 2 CPU-seconds every ~7 s). During much of the C++ phase (Spring Creek onward), another T3 agent session on this host (`claude-owner-checked`, git) added about 1 core more, so C++ background was about 2 cores of 6. That penalises the 6-rank C++ runs most. For runs stopped at the cap, the figure is inflated because the killed ranks are reaped outside the driver, so their CPU shows up as background; ignore it there.

## Terrains

Flat cells are cells that D8FlowDir must route across flats after filling (taudem-rs counter `d8.flat_cells`); the fraction is of valid (non-nodata) cells.

| Terrain | File | W × H | Cells | Valid cells | Flat cells | Flat fraction | Cell size | Layout | Size |
|---|---|---|---|---|---|---|---|---|---|
| Muncie (HEC-RAS example) | `Terrain.muncie_clip.tif` | 7,892 × 4,538 | 35.8 M | 35.6 M | 25.9 M | 73.0% | 5.00 ft (US) | float32, DEFLATE, tiled 256×256 | 18 MB |
| Spring Creek, Springfield IL | `SpringCreek.terrain.tif` | 10,323 × 5,380 | 55.5 M | 31.7 M | 4.0 M | 12.7% | 3.00 m | float32, LZW, striped (1 row) | 116 MB |
| RAS 7.0 template mosaic | `Terrain.dem_mosaic_5070_basin_extent.tif` | 11,630 × 7,158 | 83.2 M | 73.3 M | 12.3 M | 16.8% | 3.00 m | float32, DEFLATE, tiled 256×256 | 150 MB |
| Lake Maurepas (FEMA eBFE) | `Terrain.LakeMaurepasDEM.tif` | 27,477 × 22,133 | 608.1 M | 373.4 M | 180.7 M | 48.4% | 9.84 ft (US) | float32, DEFLATE, tiled 256×256 | 758 MB |
| RBFS East CUDEM | `Terrain.cudem_ft.tif` | 27,684 × 33,398 | 924.6 M | 550.3 M | 385.7 M | 70.1% | 9.00 ft (US) | float32, DEFLATE, tiled 256×256 | 517 MB |
| ELC 2022 LiDAR + CoNED | `2022LiDAR+CoNED.2022LiDAR+CoNED.tif` | 35,999 × 31,635 | 1,138.8 M | 763.0 M | 602.8 M | 79.0% | 3.00 ft (US) | float32, DEFLATE, tiled 256×256 | 621 MB |
| Amite WA1 (FEMA eBFE) | `Terrain.DEM_10ft.tif` | 25,548 × 50,272 | 1,284.3 M | 976.5 M | 247.1 M | 25.3% | 10.00 ft (US) | float32, DEFLATE, tiled 256×256 | 1,857 MB |
| Tickfaw WS1 (FEMA eBFE) | `Terrain_Repaired.WS1_Repaired3.tif` | 42,810 × 93,236 | 3,991.4 M | 2,032.6 M | 872.8 M | 42.9% | 3.28 ft (US) | float32, DEFLATE, tiled 256×256 | 2,858 MB |

## End-to-end results

Best C++ is the fastest rank count that finished the chain. Best Rust is the fastest in-memory configuration (fused or separate, 1 or 6 threads). Speedup is C++ best wall / Rust best wall.

| Terrain | Cells | C++ best wall | C++ ranks | C++ peak RSS | Rust best wall | Rust config | Rust peak RSS | Speedup |
|---|---|---|---|---|---|---|---|---|
| Muncie (HEC-RAS example) | 36 M | 1,951.2 s | 4 | 1.23 GiB | 1.85 s | fused, 6 thr | 0.44 GiB | **1,054×** |
| Spring Creek, Springfield IL | 56 M | 279.7 s | 4 | 1.52 GiB | 2.53 s | separate, 6 thr | 0.46 GiB | **111×** |
| RAS 7.0 template mosaic | 83 M | 1,035.9 s | 4 | 2.45 GiB | 4.60 s | fused, 6 thr | 0.81 GiB | **225×** |
| Lake Maurepas (FEMA eBFE) | 608 M | > 10,798 s (all tried rank counts hit the 3 h cap in D8FlowDir) | 6/4/1 | 15.10 GiB (at cap) | 31.38 s | fused, 6 thr | 5.02 GiB | **> 344×** (lower bound) |
| RBFS East CUDEM | 925 M | > 10,798 s (all tried rank counts hit the 3 h cap in D8FlowDir) | 4 | 22.27 GiB (at cap) | 46.49 s | fused, 6 thr | 7.57 GiB | **> 232×** (lower bound) |
| ELC 2022 LiDAR + CoNED | 1,139 M | — (not run: exceeds RAM, ≈25–28 GiB predicted at the measured 23.5–26.7 B/cell vs a 25.6 GiB guard) | — | — | 66.94 s | fused, 6 thr | 9.10 GiB | — |
| Amite WA1 (FEMA eBFE) | 1,284 M | — (not run: exceeds RAM, ≈28–32 GiB predicted at 23.5–26.7 B/cell) | — | — | 73.35 s | fused, 6 thr | 11.14 GiB | — |
| Tickfaw WS1 (FEMA eBFE) | 3,991 M | — (not run: exceeds RAM, ≈87–99 GiB predicted) | — | — | in-memory not run (≈34 GiB at 9 B/cell); tiled: see streaming | — | — | — |

## Timing and memory by configuration

Wall: median [min, max] over the runs (C++ n = 1). Phase times are medians summed over the chain.

| Terrain | Impl | Mode | Workers | n | Wall s [min, max] | Read s | Compute s | Write s | Peak RSS GiB | B/cell | Output MiB | Background cores | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Muncie (HEC-RAS example) | Rust | fused | 1 | 3 | 7.76 [7.67, 8.81] | 0.10 | 5.36 | 2.30 | 0.40 | 11.9 | 86 | 0.98 | ok |
| Muncie (HEC-RAS example) | Rust | fused | 6 | 3 | 1.85 [1.77, 1.96] | 0.03 | 1.23 | 0.53 | 0.44 | 13.2 | 86 | 0.64 | ok |
| Muncie (HEC-RAS example) | Rust | separate | 1 | 3 | 8.28 [8.25, 8.34] | 0.43 | 5.46 | 2.27 | 0.40 | 11.9 | 86 | 0.99 | ok |
| Muncie (HEC-RAS example) | Rust | separate | 6 | 3 | 2.11 [2.10, 2.15] | 0.16 | 1.29 | 0.56 | 0.44 | 13.2 | 86 | 1.10 | ok |
| Muncie (HEC-RAS example) | C++ | separate | 1 | 1 | 6,031.34 [6,031.34, 6,031.34] | 1.09 | 6,026.82 | 2.99 | 1.06 | 31.9 | 102 | 1.60 | ok |
| Muncie (HEC-RAS example) | C++ | separate | 4 | 1 | 1,951.20 [1,951.20, 1,951.20] | 0.20 | 1,947.04 | 1.85 | 1.23 | 36.8 | 109 | 0.88 | ok |
| Muncie (HEC-RAS example) | C++ | separate | 6 | 1 | 2,033.36 [2,033.36, 2,033.36] | 0.23 | 2,029.00 | 1.70 | 1.34 | 40.3 | 109 | 1.03 | ok |
| Spring Creek, Springfield IL | Rust | fused | 1 | 3 | 9.18 [8.96, 9.33] | 0.52 | 4.93 | 3.66 | 0.45 | 8.8 | 198 | 1.08 | ok |
| Spring Creek, Springfield IL | Rust | fused | 6 | 3 | 2.74 [2.63, 2.90] | 0.11 | 1.40 | 1.17 | 0.47 | 9.1 | 198 | 0.93 | ok |
| Spring Creek, Springfield IL | Rust | separate | 1 | 3 | 9.64 [9.59, 9.71] | 1.07 | 4.79 | 3.68 | 0.45 | 8.6 | 198 | 1.09 | ok |
| Spring Creek, Springfield IL | Rust | separate | 6 | 3 | 2.53 [2.43, 2.57] | 0.29 | 1.17 | 0.88 | 0.46 | 8.9 | 198 | 1.00 | ok |
| Spring Creek, Springfield IL | C++ | separate | 1 | 1 | 623.26 [623.26, 623.26] | 2.42 | 614.95 | 5.24 | 1.40 | 27.1 | 255 | 2.22 | ok |
| Spring Creek, Springfield IL | C++ | separate | 4 | 1 | 279.71 [279.71, 279.71] | 0.58 | 272.90 | 3.57 | 1.52 | 29.4 | 264 | 2.05 | ok |
| Spring Creek, Springfield IL | C++ | separate | 6 | 1 | 392.34 [392.34, 392.34] | 0.67 | 382.10 | 5.33 | 1.63 | 31.5 | 265 | 1.98 | ok |
| RAS 7.0 template mosaic | Rust | fused | 1 | 3 | 18.47 [17.94, 18.48] | 0.82 | 10.89 | 6.46 | 0.78 | 10.0 | 323 | 1.25 | ok |
| RAS 7.0 template mosaic | Rust | fused | 6 | 3 | 4.60 [4.55, 5.02] | 0.19 | 2.49 | 1.89 | 0.81 | 10.5 | 323 | 0.90 | ok |
| RAS 7.0 template mosaic | Rust | separate | 1 | 3 | 18.75 [18.73, 19.16] | 1.61 | 10.53 | 6.45 | 0.77 | 10.0 | 323 | 1.00 | ok |
| RAS 7.0 template mosaic | Rust | separate | 6 | 3 | 4.73 [4.27, 4.88] | 0.47 | 2.46 | 1.68 | 0.79 | 10.2 | 323 | 0.72 | ok |
| RAS 7.0 template mosaic | C++ | separate | 1 | 1 | 2,615.88 [2,615.88, 2,615.88] | 3.66 | 2,602.84 | 8.48 | 2.05 | 26.4 | 325 | 2.24 | ok |
| RAS 7.0 template mosaic | C++ | separate | 4 | 1 | 1,035.92 [1,035.92, 1,035.92] | 0.68 | 1,023.57 | 6.95 | 2.45 | 31.6 | 338 | 2.13 | ok |
| RAS 7.0 template mosaic | C++ | separate | 6 | 1 | 1,335.26 [1,335.26, 1,335.26] | 0.90 | 1,319.50 | 8.12 | 2.63 | 33.9 | 339 | 1.96 | ok |
| Lake Maurepas (FEMA eBFE) | Rust | fused | 1 | 3 | 114.20 [113.90, 115.91] | 3.63 | 75.35 | 35.14 | 4.75 | 8.4 | 1,327 | 1.02 | ok |
| Lake Maurepas (FEMA eBFE) | Rust | fused | 6 | 3 | 31.38 [31.23, 31.47] | 0.75 | 20.23 | 10.33 | 5.02 | 8.9 | 1,327 | 0.96 | ok |
| Lake Maurepas (FEMA eBFE) | Rust | separate | 1 | 3 | 118.67 [118.09, 119.37] | 8.10 | 75.30 | 34.97 | 4.74 | 8.4 | 1,327 | 0.96 | ok |
| Lake Maurepas (FEMA eBFE) | Rust | separate | 6 | 3 | 32.67 [32.32, 33.66] | 2.47 | 19.08 | 10.74 | 4.92 | 8.7 | 1,327 | 0.95 | ok |
| Lake Maurepas (FEMA eBFE) | C++ | separate | 1 | 1 | 10,798.15 [10,798.15, 10,798.15] | 7.63 | 801.00 | 12.94 | 13.34 | 23.5 | 720 | 1.63 | timed out at 10802 s (d8flowdir) |
| Lake Maurepas (FEMA eBFE) | C++ | separate | 4 | 1 | 10,798.22 [10,798.22, 10,798.22] | 2.08 | 347.27 | 10.41 | 14.83 | 26.2 | 756 | 3.13 | timed out at 10802 s (d8flowdir) |
| Lake Maurepas (FEMA eBFE) | C++ | separate | 6 | 1 | 10,798.25 [10,798.25, 10,798.25] | 2.42 | 441.94 | 17.17 | 15.10 | 26.7 | 760 | 4.00 | timed out at 10802 s (d8flowdir) |
| RBFS East CUDEM | Rust | fused | 1 | 3 | 161.55 [160.27, 162.55] | 3.16 | 109.60 | 48.39 | 7.23 | 8.4 | 1,570 | 1.02 | ok |
| RBFS East CUDEM | Rust | fused | 6 | 3 | 46.49 [46.27, 47.61] | 0.76 | 31.27 | 14.39 | 7.57 | 8.8 | 1,570 | 0.96 | ok |
| RBFS East CUDEM | Rust | separate | 1 | 3 | 167.19 [166.10, 167.98] | 9.63 | 108.67 | 48.06 | 7.23 | 8.4 | 1,570 | 0.70 | ok |
| RBFS East CUDEM | Rust | separate | 6 | 3 | 49.41 [44.52, 51.48] | 3.39 | 31.28 | 14.36 | 7.47 | 8.7 | 1,570 | 0.87 | ok |
| RBFS East CUDEM | C++ | separate | 4 | 1 | 10,798.23 [10,798.23, 10,798.23] | 2.80 | 1,631.58 | 15.24 | 22.27 | 25.9 | 839 | 3.15 | timed out at 10802 s (d8flowdir) |
| ELC 2022 LiDAR + CoNED | Rust | fused | 1 | 3 | 221.22 [218.96, 221.48] | 4.24 | 159.29 | 57.34 | 8.84 | 8.3 | 1,769 | 1.07 | ok |
| ELC 2022 LiDAR + CoNED | Rust | fused | 6 | 3 | 66.94 [63.85, 67.20] | 0.99 | 48.06 | 16.64 | 9.10 | 8.6 | 1,769 | 1.13 | ok |
| ELC 2022 LiDAR + CoNED | Rust | separate | 1 | 3 | 229.64 [227.20, 229.74] | 11.87 | 159.62 | 56.50 | 8.83 | 8.3 | 1,769 | 1.05 | ok |
| ELC 2022 LiDAR + CoNED | Rust | separate | 6 | 3 | 67.88 [63.14, 69.37] | 4.26 | 46.53 | 16.30 | 9.03 | 8.5 | 1,769 | 0.99 | ok |
| Amite WA1 (FEMA eBFE) | Rust | fused | 1 | 3 | 285.20 [282.95, 287.24] | 10.51 | 182.91 | 90.88 | 10.59 | 8.9 | 4,396 | 0.82 | ok |
| Amite WA1 (FEMA eBFE) | Rust | fused | 6 | 3 | 73.35 [70.03, 73.93] | 2.84 | 43.32 | 25.90 | 11.14 | 9.3 | 4,396 | 0.98 | ok |
| Amite WA1 (FEMA eBFE) | Rust | separate | 1 | 3 | 300.23 [298.60, 302.37] | 22.02 | 185.78 | 92.01 | 10.56 | 8.8 | 4,396 | 1.17 | ok |
| Amite WA1 (FEMA eBFE) | Rust | separate | 6 | 3 | 81.17 [79.41, 82.19] | 7.02 | 44.31 | 28.03 | 11.00 | 9.2 | 4,396 | 1.19 | ok |

### C++ D8FlowDir

| Terrain | Ranks | PitRemove s | D8FlowDir s | of which flat resolution s | AreaD8 s | Status |
|---|---|---|---|---|---|---|
| Muncie (HEC-RAS example) | 6 | 3.5 | 2,026.6 | 2,024.4 | 3.2 | ok |
| Muncie (HEC-RAS example) | 4 | 3.2 | 1,945.2 | 1,943.0 | 2.8 | ok |
| Muncie (HEC-RAS example) | 1 | 6.8 | 6,016.9 | 6,011.7 | 7.6 | ok |
| Spring Creek, Springfield IL | 6 | 31.6 | 355.1 | 349.7 | 5.6 | ok |
| Spring Creek, Springfield IL | 4 | 21.3 | 254.2 | 250.3 | 4.3 | ok |
| Spring Creek, Springfield IL | 1 | 41.7 | 574.1 | 565.8 | 7.5 | ok |
| RAS 7.0 template mosaic | 6 | 51.0 | 1,275.9 | 1,266.5 | 8.3 | ok |
| RAS 7.0 template mosaic | 4 | 37.2 | 991.2 | 984.1 | 7.5 | ok |
| RAS 7.0 template mosaic | 1 | 86.8 | 2,512.8 | 2,497.5 | 16.3 | ok |
| Lake Maurepas (FEMA eBFE) | 6 | 472.8 | 10,325.4 | — | — | timed out at 10802 s (d8flowdir) |
| Lake Maurepas (FEMA eBFE) | 4 | 366.3 | 10,431.9 | — | — | timed out at 10802 s (d8flowdir) |
| Lake Maurepas (FEMA eBFE) | 1 | 822.6 | 9,975.5 | — | — | timed out at 10802 s (d8flowdir) |
| RBFS East CUDEM | 4 | 1,659.4 | 9,138.8 | — | — | timed out at 10802 s (d8flowdir) |

## Parity (C++ vs Rust)

`python bench/parity.py compare <C++ outputs> <Rust fused 6-thread outputs>` (default tolerance: atol 0, rtol 1e-6). Counts are exact-match cells / valid cells.

| Terrain | C++ ranks | fel | p | sd8 | ad8 |
|---|---|---|---|---|---|
| Muncie (HEC-RAS example) | 1 | EXACT 35,588,572/35,588,572, max diff 0, mask Δ 0 | EXACT 35,563,716/35,563,716, max diff 0, mask Δ 0 | EXACT 35,563,716/35,563,716, max diff 0, mask Δ 0 | EXACT 35,476,726/35,476,726, max diff 0, mask Δ 0 |
| Muncie (HEC-RAS example) | 4 | EXACT 35,588,572/35,588,572, max diff 0, mask Δ 0 | EXACT 35,563,716/35,563,716, max diff 0, mask Δ 0 | EXACT 35,563,716/35,563,716, max diff 0, mask Δ 0 | EXACT 35,476,726/35,476,726, max diff 0, mask Δ 0 |
| Muncie (HEC-RAS example) | 6 | EXACT 35,588,572/35,588,572, max diff 0, mask Δ 0 | EXACT 35,563,716/35,563,716, max diff 0, mask Δ 0 | EXACT 35,563,716/35,563,716, max diff 0, mask Δ 0 | EXACT 35,476,726/35,476,726, max diff 0, mask Δ 0 |
| Spring Creek, Springfield IL | 1 | EXACT 31,729,691/31,729,691, max diff 0, mask Δ 0 | EXACT 31,692,383/31,692,383, max diff 0, mask Δ 0 | EXACT 31,692,383/31,692,383, max diff 0, mask Δ 0 | EXACT 31,578,344/31,578,344, max diff 0, mask Δ 0 |
| Spring Creek, Springfield IL | 4 | EXACT 31,729,691/31,729,691, max diff 0, mask Δ 0 | EXACT 31,692,383/31,692,383, max diff 0, mask Δ 0 | EXACT 31,692,383/31,692,383, max diff 0, mask Δ 0 | EXACT 31,578,344/31,578,344, max diff 0, mask Δ 0 |
| Spring Creek, Springfield IL | 6 | EXACT 31,729,691/31,729,691, max diff 0, mask Δ 0 | EXACT 31,692,383/31,692,383, max diff 0, mask Δ 0 | EXACT 31,692,383/31,692,383, max diff 0, mask Δ 0 | EXACT 31,578,344/31,578,344, max diff 0, mask Δ 0 |
| RAS 7.0 template mosaic | 1 | EXACT 73,378,861/73,378,861, max diff 0, mask Δ 0 | EXACT 73,343,025/73,343,025, max diff 0, mask Δ 0 | EXACT 73,343,025/73,343,025, max diff 0, mask Δ 0 | EXACT 73,166,740/73,166,740, max diff 0, mask Δ 0 |
| RAS 7.0 template mosaic | 4 | EXACT 73,378,861/73,378,861, max diff 0, mask Δ 0 | EXACT 73,343,025/73,343,025, max diff 0, mask Δ 0 | EXACT 73,343,025/73,343,025, max diff 0, mask Δ 0 | EXACT 73,166,740/73,166,740, max diff 0, mask Δ 0 |
| RAS 7.0 template mosaic | 6 | EXACT 73,378,861/73,378,861, max diff 0, mask Δ 0 | EXACT 73,343,025/73,343,025, max diff 0, mask Δ 0 | EXACT 73,343,025/73,343,025, max diff 0, mask Δ 0 | EXACT 73,166,740/73,166,740, max diff 0, mask Δ 0 |

## Streaming (tiled) pipeline — `main` @ 7ba87f4

`taudem-rs pipeline --mode tiled --max-memory <cap> --scratch-dir <local scratch>`, 6 threads. Amite and ELC: 3 repeats per cap, plus 3 in-memory runs of the same 7ba87f4 binary (`--mode in-memory`) as a same-build control. Tickfaw: one run per cap. Spill is what the tile cache wrote to the scratch directory (`tiles.spill_write_bytes`); peak spilled is the most held on disk at once.

| Terrain | Cells | Mode | n | Wall s [min, max] | vs in-memory fca2623 | Peak RSS GiB | B/cell | Spill written GiB | Peak spilled GiB | Scratch peak GiB | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ELC 2022 LiDAR + CoNED | 1,139 M | in-memory fca2623 (reference) | 3 | 66.94 [63.85, 67.20] | 1.00× | 9.10 | 8.6 | — | — | — | ok |
| ELC 2022 LiDAR + CoNED | 1,139 M | in-memory 7ba87f4 | 3 | 67.21 [65.20, 68.47] | 1.00× | 9.08 | 8.6 | — | — | — | ok |
| ELC 2022 LiDAR + CoNED | 1,139 M | tiled, --max-memory 2G | 3 | 69.61 [69.20, 70.29] | 1.04× | 2.11 | 2.0 | 0.10 | 0.08 | 0.00 | ok |
| ELC 2022 LiDAR + CoNED | 1,139 M | tiled, --max-memory 4G | 3 | 76.11 [76.00, 76.45] | 1.14× | 4.69 | 4.4 | 0.00 | 0.00 | 0.00 | ok |
| Amite WA1 (FEMA eBFE) | 1,284 M | in-memory fca2623 (reference) | 3 | 73.35 [70.03, 73.93] | 1.00× | 11.14 | 9.3 | — | — | — | ok |
| Amite WA1 (FEMA eBFE) | 1,284 M | in-memory 7ba87f4 | 3 | 74.35 [73.92, 75.21] | 1.01× | 11.13 | 9.3 | — | — | — | ok |
| Amite WA1 (FEMA eBFE) | 1,284 M | tiled, --max-memory 2G | 3 | 92.68 [92.32, 92.69] | 1.26× | 2.03 | 1.7 | 1.17 | 0.65 | 0.00 | ok |
| Amite WA1 (FEMA eBFE) | 1,284 M | tiled, --max-memory 4G | 3 | 97.51 [95.97, 97.83] | 1.33× | 4.30 | 3.6 | 0.00 | 0.00 | 0.00 | ok |
| Tickfaw WS1 (FEMA eBFE) | 3,991 M | tiled, --max-memory 8G | 1 | 201.41 [201.41, 201.41] | — | 7.34 | 2.0 | 0.00 | 0.00 | 0.00 | ok |
| Tickfaw WS1 (FEMA eBFE) | 3,991 M | tiled, --max-memory 16G | 1 | 227.40 [227.40, 227.40] | — | 13.10 | 3.5 | 0.00 | 0.00 | 0.00 | ok |

Output identity (`bench/parity.py compare`, exact-match cells / valid cells):

| Terrain | Reference | Compared | fel | p | sd8 | ad8 |
|---|---|---|---|---|---|---|
| Amite WA1 (FEMA eBFE) | in-memory fca2623 | in-memory 7ba87f4 | EXACT 976,711,722/976,711,722, max diff 0, mask Δ 0 | EXACT 976,542,458/976,542,458, max diff 0, mask Δ 0 | EXACT 976,542,458/976,542,458, max diff 0, mask Δ 0 | EXACT 975,785,402/975,785,402, max diff 0, mask Δ 0 |
| Amite WA1 (FEMA eBFE) | in-memory fca2623 | tiled-2G | EXACT 976,711,722/976,711,722, max diff 0, mask Δ 0 | EXACT 976,542,458/976,542,458, max diff 0, mask Δ 0 | EXACT 976,542,458/976,542,458, max diff 0, mask Δ 0 | EXACT 975,785,402/975,785,402, max diff 0, mask Δ 0 |
| Amite WA1 (FEMA eBFE) | in-memory fca2623 | tiled-4G | EXACT 976,711,722/976,711,722, max diff 0, mask Δ 0 | EXACT 976,542,458/976,542,458, max diff 0, mask Δ 0 | EXACT 976,542,458/976,542,458, max diff 0, mask Δ 0 | EXACT 975,785,402/975,785,402, max diff 0, mask Δ 0 |
| ELC 2022 LiDAR + CoNED | in-memory fca2623 | in-memory 7ba87f4 | EXACT 763,160,413/763,160,413, max diff 0, mask Δ 0 | EXACT 763,023,769/763,023,769, max diff 0, mask Δ 0 | EXACT 763,023,769/763,023,769, max diff 0, mask Δ 0 | EXACT 762,283,265/762,283,265, max diff 0, mask Δ 0 |
| ELC 2022 LiDAR + CoNED | in-memory fca2623 | tiled-2G | EXACT 763,160,413/763,160,413, max diff 0, mask Δ 0 | EXACT 763,023,769/763,023,769, max diff 0, mask Δ 0 | EXACT 763,023,769/763,023,769, max diff 0, mask Δ 0 | EXACT 762,283,265/762,283,265, max diff 0, mask Δ 0 |
| ELC 2022 LiDAR + CoNED | in-memory fca2623 | tiled-4G | EXACT 763,160,413/763,160,413, max diff 0, mask Δ 0 | EXACT 763,023,769/763,023,769, max diff 0, mask Δ 0 | EXACT 763,023,769/763,023,769, max diff 0, mask Δ 0 | EXACT 762,283,265/762,283,265, max diff 0, mask Δ 0 |
| Tickfaw WS1 (FEMA eBFE) | tiled-8G | tiled-16G | EXACT 2,032,991,357/2,032,991,357, max diff 0, mask Δ 0 | EXACT 2,032,627,716/2,032,627,716, max diff 0, mask Δ 0 | EXACT 2,032,627,716/2,032,627,716, max diff 0, mask Δ 0 | EXACT 2,031,491,617/2,031,491,617, max diff 0, mask Δ 0 |

## Skipped or not run

| Item | Reason |
|---|---|
| C++ on ELC, Amite and Tickfaw | Exceeds RAM. Measured C++ peak is 23.5–26.7 B/cell (summed over ranks) at 6e8–9e8 cells, so these need ≈25–28, 28–32 and 87–99 GiB against a 25.6 GiB (80%) guard. |
| C++ on CUDEM at 1 and 6 ranks | The coordinator allowed one configuration. 4 ranks was chosen because it was the fastest rank count on every terrain C++ finished, and its predicted ≈22.5 GiB fit the guard. It peaked at 22.3 GiB and hit the 3 h cap in D8FlowDir. |
| taudem-rs in-memory on Tickfaw | Needs ≈34 GiB at ~9 B/cell. Run tiled instead (streaming section). |
| Spring River `Terrain (2).ep_extend.tif` | Not found. The Windows path (`G:\GH\ras-commander-docs\working\spring-river-source-local-20260718`) has no counterpart on the NAS mounts. A bounded `find` of `/mnt/h/CLB-Repos` (maxdepth 7) found no match; broader searches of `/mnt/h/CLB-Repos`, `/mnt/clb-active-projects` and `/mnt/clb-scratch` hit their 120 s timeouts. |
| Spring Creek and RAS 70 template from `H:\work\TauDEM-fast-terrains` | Spring Creek was copied and used. The RAS 70 template was byte-identical (`cmp`) to the copy already staged, so it was not copied again. |

## Observations

- **Results are identical.** Rust matches C++ TauDEM exactly on fel, p, sd8 and ad8, on every cell of every terrain where C++ finished: 9 comparisons, 0 differing cells, 0 nodata-mask mismatches. The tiled pipeline also matches the in-memory one exactly on ELC and Amite at both 2G and 4G caps. Tickfaw at 8G and 16G gives identical outputs.
- **C++ time is almost all D8FlowDir flat resolution**, more than 99% of the chain on every finished terrain. Its cost tracks the number of flat cells more than total size. Muncie (73% flat, 36 M cells) took 1,951 s at best, while Spring Creek (13% flat, 56 M cells) took 280 s. The speedup follows the same pattern: 1,054× on Muncie, 111× on Spring Creek, 225× on RAS 7.0. On the large real terrains (Lake Maurepas 48% flat, CUDEM 70% flat), C++ did not get through D8FlowDir in 3 h at any rank count tried. The speedups there are lower bounds (> 344× and > 232×).
- **C++ scaling stops at 4 ranks.** 4 ranks beat 6 on every finished terrain; 6 ranks is 4–40% slower. Part of that is the ~2 cores of background load from another session during the C++ phase, which only hurts when all six cores are in use. 1 → 4 ranks gives 2.2–3.1×.
- **Rust scales 3.3–4.2× from 1 to 6 threads.** Fused is 1–11% faster than separate tools at ≥ 6e8 cells (Amite 73.4 s vs 81.2 s), mostly because it skips the intermediate re-reads.
- **Rust in-memory memory use is 8.3–9.3 B/cell at ≥ 6e8 cells**, matching the ~9.1 B/cell target. It is 9–13 B/cell on the small terrains because of fixed overhead. C++ uses 23.5–40 B/cell summed over ranks, rising with rank count.
- **Streaming:** Tickfaw (4.0e9 cells, 2.0e9 valid) runs in 201 s in 7.3 GiB at `--max-memory 8G`, about 19.8 M cells/s, comparable to Amite in memory at 17.5 M cells/s. The tile cache never spilled to disk; compressed tiles fit in the budget. At 2G, ELC costs 1.04× in-memory wall (2.1 GiB peak) and Amite 1.26× (2.0 GiB peak, 1.2 GiB spilled).
- **A bigger memory cap made the tiled pipeline slower in every case:** ELC 4G 1.14× vs 2G 1.04×; Amite 4G 1.33× vs 2G 1.26×; Tickfaw 16G 227 s vs 8G 201 s. The repeat spread is tight (under 2%), so this is not noise and is worth a look in the tile-cache sizing.
- **The striped LZW input (Spring Creek, one-row strips) shows no read penalty** in the fused pipeline: 0.11 s at 6 threads for 56 M cells. Separate tools pay the read three times (0.29 s).
- **The background load (about 1 core, about 2 cores during the C++ phase) is not idle-machine noise.** It is ZFS write work plus agent-session helpers, and it affects the 6-worker numbers more than the 1-worker numbers.

## Earlier synthetic sweep (stopped)

A sweep started earlier on CLB-t3 (`resume-series.py`, older taudem-rs at `86b02be`, synthetic DEMs only) was still running when this work began. It was stopped on 2026-11-25 at 20:59 UTC. The publish waiter (PID 1690899) was stopped first so the driver's exit could not trigger a publish. Then the driver (1690898) and its running `taudem-rs pipeline` child (1730153, 100M flat-light, 1 thread, fused) were stopped with SIGTERM. The queued publish scripts were renamed `*.disabled-2026-11-25`; nothing was published. All completed result files were kept under `/home/clb/scratch/taudem-bench-clb-t3-2026-11-25/results`.

Completed before the stop: the whole 10M tier (both recipes, C++ 1/4/6/8 ranks, Rust 1/4/6/8 threads), Rust on 100M default, C++ 100M default at 4 ranks (4,131 s) and 6 ranks (3,556 s), and Rust 100M flat-light separate at 1 thread. **The 1-rank C++ 100M default run did not finish.** Its D8FlowDir process disappeared at ≥ 7,235 s of chain time, and no result was recorded.

### Appendix: synthetic 10M results (SYNTHETIC — older taudem-rs `86b02be`, not comparable to the main tables)

| Synthetic DEM (10,000,000 cells) | C++ best | Rust best | C++ / Rust | C++ peak MiB | Rust peak MiB |
|---|---|---|---|---|---|
| default recipe (flat fraction 0.15) | 104.32 s (4 ranks) | 1.16 s (6 threads, fused) | 89.7× | 490.0 | 306.4 |
| flat-light recipe (flat fraction 0.02) | 33.49 s (4 ranks) | 1.13 s (6 threads, fused) | 29.7× | 496.1 | 299.7 |

## Evidence

Raw rows (one JSON line per run, with every step's wall, VmHWM peaks, printed timings, background CPU and profile counters), the `--profile-json` files, per-run stdout/stderr, and parity outputs are kept on CLB-t3 under `/home/clb/scratch/taudem-bench-clb-t3-2026-11-25/real/` (`results/results.jsonl`, `results/profiles/`, `results/parity/`, `logs/`). The tables above are generated from them by `real/make_report.py`.
