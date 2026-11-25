# TauDEM parity and benchmark harness

These tools check that `taudem-rs`, the Rust rewrite in `../rust`, gives the same
results as C++ TauDEM, and measure both on large terrains. Only scripts and
configuration are in the repository. Test data, synthetic DEMs, scratch outputs and
results all live outside it.

| file | purpose |
|---|---|
| `parity.py` | Runs an implementation over the TauDEM-Test-Data cases and compares its outputs with `ReferenceResult`. `parity.py compare A B` compares any two rasters or directories. |
| `bench.py` | Times pitremove → d8flowdir → aread8 (as separate tools, plus the fused `taudem-rs pipeline`) and records peak memory, TauDEM phase timings and output sizes. `bench.py summarize` turns the results into markdown. |
| `make_dem.py` | Synthetic DEM generator for 1e6 to 1e9 cells. It writes a tiled, compressed GeoTIFF (BigTIFF when needed) that includes pits, broad flats, nodata holes and an irregular nodata edge. |
| `rastercompare.py` | Streaming cell-by-cell comparison, used by both harnesses. Also runs on its own. |
| `inventory.py` | Writes the test-data inventory in `TEST_DATA.md`. |
| `cases.json` | Parity cases, transcribed from `testall.bat`: steps, rank counts, outputs, tags. |
| `tools.json` | Maps each canonical tool to its C++ executable and its `taudem-rs` subcommand, says how a rank count becomes a Rust thread count, and holds the fused-pipeline argv. |
| `TEST_DATA.md` | What each test case contains (sizes, dtypes, nodata, CRS) and which tools it exercises. |

## Prerequisites

* Python 3.10 or newer, with `numpy`, `rasterio` (its wheels bundle GDAL) and `psutil`.
  Either run `uv run python ...` in this folder (uv reads `pyproject.toml`), or run
  `pip install numpy rasterio psutil`. Add `matplotlib` only if you want previews.
* The C++ TauDEM executables and an MPI launcher:
  * **Windows:** the TauDEM installer (`C:\Program Files\TauDEM\TauDEM5Exe`) and
    Microsoft MPI (`mpiexec` on PATH, or at `C:\Program Files\Microsoft MPI\Bin`).
  * **Linux:** build TauDEM with CMake (`cmake -S src -B build && cmake --build build -j`)
    and use OpenMPI or MPICH `mpirun`/`mpiexec`. Lowercase executable names
    (`pitremove`, `d8flowdir`, `aread8`) are found automatically.
* `taudem-rs`: `cargo build --release` in `../rust`. The binary is
  `rust/target/release/taudem-rs(.exe)`.
* The test data: `git clone --depth 1 https://github.com/dtarb/TauDEM-Test-Data <dir>`
  (about 1.8 GB).

## Configuration

Command-line arguments take precedence over environment variables. No script
contains a machine-specific path.

| variable | meaning |
|---|---|
| `TAUDEM_TEST_DATA` | TauDEM-Test-Data checkout (contains `Input/` and `ReferenceResult/`) |
| `TAUDEM_CPP_DIR` | directory holding the C++ executables (the default for `--impl cpp`) |
| `TAUDEM_RS_BIN` | path to the taudem-rs binary (the default for `--impl rs`) |
| `TAUDEM_BENCH_WORK` | scratch and results root. Defaults to `<system temp>/taudem-bench`. Put it on a local disk with room for about 4 x 4 bytes x cells (more than 16 GB at 1e9 cells). Do not use a cloud-synced drive. |
| `MPIEXEC` | MPI launcher, when it is not on PATH |

An implementation is given as `cpp[:<exe dir>]` or `rs[:<binary>]`. You can add
`=label` to tell two builds apart, for example `rs:old/taudem-rs.exe=rs-old`.

## Parity

```powershell
# Windows (PowerShell)
$env:TAUDEM_TEST_DATA = "D:\data\TauDEM-Test-Data"
$env:TAUDEM_BENCH_WORK = "D:\scratch\taudem-bench"
python parity.py run --impl "cpp:C:\Program Files\TauDEM\TauDEM5Exe"
python parity.py run --impl rs:..\rust\target\release\taudem-rs.exe --skip-tags ogr filegdb
python parity.py run --impl cpp --impl rs --cases Base,fts,Geographic
```

```bash
# Linux
export TAUDEM_TEST_DATA=~/data/TauDEM-Test-Data TAUDEM_BENCH_WORK=/scratch/taudem-bench
python parity.py run --impl cpp:$HOME/TauDEM/build/src --impl rs:../rust/target/release/taudem-rs
```

* Each case copies `Input/<case>` into `<work>/parity/<impl>/<case>` and runs the
  steps from `cases.json` with that step's rank count. `--np N` overrides every
  step's count.
* `--inputs reference` (the default) first overwrites a step's upstream inputs
  (`ref_inputs`) with the ReferenceResult copies. A d8flowdir bug then cannot turn
  into an aread8 failure. Use `--inputs chain` to run the real chain instead.
  Whatever each step produced is kept in `_produced/<step>/`.
* For every output, the harness checks dimensions, dtype, nodata value,
  geotransform and CRS (compared semantically, with a separate warning when only the
  WKT text differs). It also counts nodata-mask disagreements, exact-match cells,
  maximum and mean absolute difference, maximum relative difference, and the
  location of the worst cell.
* Statuses:
  * `EXACT`: identical.
  * `PASS`: within `--atol` (default 0) or `--rtol` (default 1e-6, as in upstream `compare_results.py`).
  * `CRSDIFF`: everything matches except the CRS definition.
  * `FAIL`, `MISSING`, `ERROR`, `NOREF`: the output is wrong, absent, failed to run, or has no reference to compare with.
  * `SKIP`: the tool is not implemented for this implementation (`tools.json` has `"rs": null`) or a tag filter excluded the step.
  * Use `--allow-crsdiff` so that `CRSDIFF` does not fail the run.
* Results are written to `<work>/results/parity-<stamp>.json` and `.md`. The exit
  code is 0 only when every output is clean.
* **Adding a tool to taudem-rs:** set its subcommand in `tools.json`. The cases in
  `cases.json` that use the tool then run for rs. To add a case, copy the matching
  line from `testall.bat` into `cases.json`.

## Synthetic DEMs

```bash
python make_dem.py --size 1e6 --out $TAUDEM_BENCH_WORK/dems/synth_1e6.tif
python make_dem.py --size 1e7 --out $TAUDEM_BENCH_WORK/dems/synth_1e7.tif   # about 10 s
python make_dem.py --size 1e8 --out $TAUDEM_BENCH_WORK/dems/synth_1e8.tif   # about 2 min, ~110 MB
python make_dem.py --size 1e9 --out $TAUDEM_BENCH_WORK/dems/synth_1e9.tif   # about 20 min, BigTIFF
```

* Presets are non-square (aspect 1.6): 1250x800, 4000x2500, 12500x8000 and
  40000x25000 cells. `--cols/--rows` sets any size.
* Defaults: 10 m cells, EPSG:26915, nodata -9999, DEFLATE with floating-point
  predictor (GDAL 2.1 in the TauDEM installer can read it), 512 x 512 tiles.
* The generator is deterministic for a given `--seed`. Generation streams in row
  strips, so memory stays at about 100 MB for any size. A `<out>.json` sidecar
  records the parameters and statistics (nodata %, lake-flat cells, z range).
* Features:
  * fBm terrain with ridged valleys and a regional tilt;
  * explicit pit bowls, 30 per million cells by default;
  * lake flats (`--flat-frac`, `--lake-wavelength`);
  * terraced, integer-metre-style flats (`--terrace-step`);
  * interior nodata holes (`--hole-frac`);
  * an irregular nodata edge, about 28% of the rectangle (`--no-edge` turns it off).
* **Flat size and C++ run time.** TauDEM's flat resolution (in d8flowdir) costs much
  more than linear in the size of each flat. By default, lakes and terraces have a
  fixed size in cells (256 and 512), so flat count grows with area rather than any
  single flat growing. Even so, flats dominate C++ run time. On the default 1e7 DEM
  (about 0.5 M lake-flat cells), C++ d8flowdir spends about 450 s (1 rank) or 190 s
  (4 ranks) in "Resolve Flat", against about 1 s for everything else. Plan C++ runs
  at 1e8 and above accordingly, or lower `--flat-frac` / `--terrace-step` for a
  flat-light variant.
* `--lake-wavelength 0` makes the lakes grow with the raster instead. This is a
  deliberate stress case: at 1e7 cells, C++ d8flowdir took 364 s with 1 rank.

## Benchmarks

```bash
python bench.py run --dem $TAUDEM_BENCH_WORK/dems/synth_1e7.tif \
    --impl cpp:"C:/Program Files/TauDEM/TauDEM5Exe" --impl rs:../rust/target/release/taudem-rs.exe \
    --ranks 1,2,4,8 --repeat 5 --label CLB04 --cross-check
python bench.py summarize $TAUDEM_BENCH_WORK/results/bench-*.jsonl --out summary.md
```

* `--modes separate,fused`:
  * `separate` runs `pitremove -z dem -fel` → `d8flowdir -fel -p -sd8` → `aread8 -p -ad8`
    (aread8 with its default edge-contamination checking).
  * `fused` makes a single `taudem-rs pipeline` call with the argv template from `tools.json`.
* Ranks are MPI processes for C++. For rs, they are threads: `tools.json`
  `rs.threads_env` (default `RAYON_NUM_THREADS`) and/or `rs.threads_flag`.
* Each invocation records:
  * `wall_s`, measured around the whole process, including the mpiexec launch.
  * The tool's own phase timings. `t_total` is TauDEM's "Total time", so wall
    minus `t_total` is launch overhead.
  * `peak_sum_bytes`: the sum, over worker processes, of each process's peak
    working set (Windows `peak_wset`) or `VmHWM` (Linux). mpiexec, smpd and conhost
    are excluded. Because the ranks peak at about the same time, this is a tight
    upper bound on aggregate memory. `peak_concurrent_bytes` is a sampled lower bound.
  * `out_bytes`, the output file sizes.
* Results go to `<work>/results/bench-<stamp>.{jsonl,csv,meta.json,md}`. Rows are
  appended as runs finish, so an interrupted run keeps what it measured. The
  meta file records the machine (CPU count, RAM, CPU load at start), the
  implementations and the arguments. The summary gives the median over repeats.
* `--cross-check` compares each implementation's outputs with the first
  implementation's outputs on the same DEM.
* **Cache state:**
  * Default: one untimed warm-up pass per (implementation, ranks, mode), so timed
    repeats are **warm**, with the DEM in the OS file cache.
  * Cold-cache numbers: `--cold-cmd` runs a shell command before every timed repeat.
    * Linux: `--cold-cmd "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches"`.
    * Windows: Sysinternals RAMMap, run elevated: `--cold-cmd "RAMMap64.exe -accepteula -Et"`.
      Without it, Windows cannot reliably drop the standby list.
    * Rows are tagged `cold`.
  * `--no-warmup` makes the first repeat `first`: cold for the DEM unless it was
    just written.
  * At 1e8 cells and above, the DEM (about 0.1 to 1 GB compressed) and the
    intermediate files (4 x 0.4 to 4 GB) can exceed the free standby memory, so
    "warm" stops meaning much for the intermediates. Report cold and warm separately.
* **Machine hygiene for formal runs:** use an idle machine (the meta file records
  load at start), AC power and a high-performance power plan, a local NVMe scratch
  disk, and no antivirus scanning the scratch folder.

### Running on another Windows machine (for example CLB04)

1. Install Python 3.10 or newer and uv, or pip-install `numpy rasterio psutil`.
   Install the TauDEM 5.3.x installer (it includes GDAL) and Microsoft MPI v10.
2. Get this `bench/` folder (clone the branch) and the test data clone. If it is
   built there, also get `rust/target/release/taudem-rs.exe`.
3. In PowerShell:
   ```powershell
   cd <repo>\bench
   $env:TAUDEM_TEST_DATA="C:\data\TauDEM-Test-Data"; $env:TAUDEM_BENCH_WORK="C:\scratch\taudem-bench"
   uv run python parity.py run --impl "cpp:C:\Program Files\TauDEM\TauDEM5Exe" --impl rs:..\rust\target\release\taudem-rs.exe
   foreach ($s in "1e7","1e8") { uv run python make_dem.py --size $s --out "$env:TAUDEM_BENCH_WORK\dems\synth_$s.tif" }
   uv run python bench.py run --dem "$env:TAUDEM_BENCH_WORK\dems\synth_1e7.tif" --dem "$env:TAUDEM_BENCH_WORK\dems\synth_1e8.tif" `
     --impl "cpp:C:\Program Files\TauDEM\TauDEM5Exe" --impl rs:..\rust\target\release\taudem-rs.exe `
     --ranks 1,2,4,8 --repeat 5 --label CLB04 --cross-check
   ```

### Running on Linux

Use the same commands with `mpirun` on PATH, or set `MPIEXEC`. Use
`--impl cpp:<build>/src` for the C++ build. If Open MPI refuses to run as root, run
as an unprivileged user, or set `OMPI_ALLOW_RUN_AS_ROOT=1
OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`. If the rank count exceeds the cores, set `MPIEXEC`
to a wrapper that adds `--oversubscribe`.

## C++ vs ReferenceResult (installed TauDEM 5.3.7, Windows, MS-MPI)

Run on 2026-11-25 on CLB-08 with `parity.py run --impl cpp`. 48 outputs were compared:
**44 EXACT**. The other 4 come from the bundled **GDAL 2.1.0**, which is older than
the GDAL that produced ReferenceResult. No cell value differs anywhere.

| case / output | result | cause |
|---|---|---|
| Geographic/enogeofel.tif | CRSDIFF | The output copies the input's `OGC:CRS83` (NAD83, lon/lat axis order). The reference has `EPSG:4269`. Values are exact. |
| GDAL_unset_nodata/ssa.tif | CRSDIFF | The output is `EPSG:4269`. The reference keeps p.tif's compound CRS (NAD83 + unknown VCS). Values are exact. |
| gridtypes/loganfelsd.sdat | FAIL (metadata) | GDAL 2.1's SAGA driver writes `NODATA_VALUE = -99999` in the .sgrd header but fills nodata cells with -3e38, so 27,364 nodata cells read back as valid. The reference header says -3e38. Valid cell values are exact. |
| AreaD8_data/loganad8_5.tif (`-o LoganOutlet.json -lyrname LoganOutlet`) | ERROR | GDAL 2.1 names a GeoJSON layer `OGRGeoJSON`, so the layer name `LoganOutlet` is not found and AreaD8 aborts. Newer GDAL names the layer after the file. |

What this means for taudem-rs: match ReferenceResult, which reflects modern GDAL
behaviour. Pass CRS through unchanged, compound CRSs included. Use the GeoJSON
layer name taken from the file name. Write a SAGA nodata value that matches the data.

## taudem-rs parity status (stage one, 2026-11-25)

`parity.py run --impl rs:../rust/target/release/taudem-rs.exe`: every output that
taudem-rs can produce is **EXACT** against ReferenceResult:
* Base: fel, 4way, depmask, depmask+4way, p/sd8 (filled and unfilled), ad8, ad8nf, int16 MED_01_01;
* fts: fel, p, sd8, ad8;
* Geographic: fel (including the CRS), p, sd8, ad8.

The other cases are `ERROR`, from features stage one does not have yet:
* positional basename arguments (`pitremove logan.tif`), so cases.json also has
  `explicit`-tagged versions of those steps;
* `-o` outlets;
* `-wg` weights;
* non-GeoTIFF input (VRT, HFA, AIG, BIL/BIN) and output (.img, .sdat, .bil, .bin);
* the TauDEM rule that an unknown or missing extension becomes `.tif`.

Use `--skip-tags positional ogr formats weights` to see only the algorithm parity.

## Smoke timings (CLB-08, 8 cores, about 50 to 97% loaded by other work, warm cache)

These numbers are indicative only. The machine was shared, and each result is
the median of 2 or 3 repeats. Wall time includes process and MPI launch.

| DEM | impl | ranks/threads | pitremove | d8flowdir | aread8 | total (separate) | fused pipeline | peak mem (largest step) |
|---|---|---|---|---|---|---|---|---|
| logan (1.6 M) | C++ 5.3.7 | 1 | 1.20 | 1.74 | 1.16 | 4.16 | - | 68 MiB |
| logan | C++ | 4 | 1.57 | 2.46 | 1.20 | 4.97 | - | 128 MiB |
| logan | rs | 1 | 0.51 | 0.50 | 0.37 | 1.31 | 0.84 | 60 MiB |
| logan | rs | 4 | 0.51 | 0.35 | 0.34 | 1.23 | 0.70 | 62 MiB |
| synth_1e7 (10 M, 28% nodata) | C++ | 1 | 2.89 | **454.6** | 2.95 | 460.5 | - | 331 MiB |
| synth_1e7 | C++ | 4 | 5.69 | **190.1** | 2.79 | 198.6 | - | 410 MiB |
| synth_1e7 | rs | 1 | 1.39 | 1.56 | 0.76 | 3.70 | 3.02 | 274 MiB |
| synth_1e7 | rs | 4 | 1.25 | 1.30 | 0.77 | 3.32 | 2.64 | 276 MiB |

On synth_1e7, taudem-rs outputs (separate and fused) are EXACT against C++ for fel,
p, sd8 and ad8, so its flat resolution reproduces TauDEM's directions cell for cell.
On logan, `--cross-check` gives EXACT for all four outputs at 1 and 4 threads, in
both modes.
