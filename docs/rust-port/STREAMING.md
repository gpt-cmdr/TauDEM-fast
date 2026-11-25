# `taudem-rs` Streaming & Out-of-Core Execution Design

**Scope**: Decision-ready architecture and work-package plan for streaming / out-of-core execution of `taudem-rs` (`pitremove`, `d8flowdir`, `aread8`, and fused `pipeline`) on `G:\GH\TauDEM` (branch `rust-port`), jointly optimizing **peak memory** and **wall time** while preserving bit-exact parity with TauDEM. Targets running Tickfaw ($3.99\times 10^9$ cells) on 32–64 GB workstations (and under explicit 4–16 GB budgets) and cutting $10^9$-cell peak RAM from 25 GiB to ~2 GiB while improving wall time ~3×.

---

## 1. Memory Model Today & Minimal In-Memory Floor

### 1.1 Per-Stage Buffer Inventory (Verified from `rust/src/`)

Current `taudem-rs` allocates full grids in RAM and rejects `cells > u32::MAX` (`4,294,967,295`, `rust/src/lib.rs:34` — barely covering Tickfaw's $3.991\times 10^9$ cells). C++ TauDEM (`src/linearpart.h:125-166`) does not stream either: it splits full grids into horizontal MPI strips allocated completely in RAM and exchanges row borders via `MPI_Bsend` (`src/linearpart.h:195-219`).

| Stage | Buffer | Code Location | Type | B/cell | Lifetime in `main.rs` (`pipeline -z -fel -p -sd8 -ad8`) |
|---|---|---|---|---:|---|
| Read | `elevations` | `rust/src/io.rs:127` | `Vec<f32>` | 4.000 | Read → dropped after `d8_flow` (`rust/src/main.rs:162`) |
| Read | `mask` (`-depmask`) | `rust/src/io.rs:178` | `Vec<u8>` | 1.000 | Read → dropped after `pit_remove` (`rust/src/main.rs:144`) |
| `pit_remove` | `visited` | `rust/src/flood.rs:20` | `Vec<u64>` | 0.125 | Local to `pit_remove` (1-bit bitset) |
| `pit_remove` | `heap` (`FloodQueue`) | `rust/src/flood.rs:21,106` | `[Vec<u64>; 33]` | ~0.1–1.2 | Local (8 B/queued frontier cell, `Vec` 2× doubling) |
| `pit_remove` | `pit` | `rust/src/flood.rs:22` | `VecDeque<u32>` | ~0.02–0.4 | Local (4 B/flooded FIFO cell) |
| `d8_flow` | `direction` | `rust/src/d8.rs:46` | `Vec<u8>` | 1.000 | `d8_flow` → end of `pipeline` (`rust/src/main.rs:191`) |
| `d8_flow` | `flats` | `rust/src/d8.rs:74,86` | `Vec<u32>` | `4f..8f` | Steep pass → end of `resolve_flats` ($f$ = flat fraction, 2× `Vec`) |
| `d8_flow` | `slope` (`-sd8`) | `rust/src/d8.rs:91-92` | `Vec<f32>` | 4.000 | **Allocated before `resolve_flats`** → dropped after write (`main.rs:177`) |
| `resolve_flats` | `fall` | `rust/src/d8.rs:157` | `Vec<u32>` | 4.000 | Dense full-grid buffer, entire `resolve_flats` |
| `resolve_flats` | `rise` | `rust/src/d8.rs:158` | `Vec<u32>` | 4.000 | Dense full-grid buffer, entire `resolve_flats` |
| `resolve_flats` | `frontier` | `rust/src/d8.rs:159` | `VecDeque<u32>` | ~0.05–0.6 | `incfall`/`incrise` BFS wavefront (4 B/queued, 2× doubling) |
| `resolve_flats` | `artificial` | `rust/src/d8.rs:156,277` | `Vec<f32>` | **0 or 4.0** | Allocated at `d8.rs:277` **iff** `resolve_flats` needs ≥2 iterations |
| Write (`-p`) | `stripe` | `rust/src/io.rs:226` | `Vec<i16>` | ~0.02 | `width × 256` conversion stripe (~18 MB on ELC) |
| `area_d8` | `state` | `rust/src/area.rs:14` | `Vec<u8>` | 1.000 | Local to `area_d8` |
| `area_d8` | `area` | `rust/src/area.rs:34` | `Vec<f32>` | 4.000 | `area_d8` → end of `pipeline` (`main.rs:189`) |

**Stage peaks in `main.rs` (`bench/tools.json:20` passes `-fel -p -sd8 -ad8`)**:
- **`pit_remove`**: `4.0 + 0.125 + heap/pit` ≈ **4.4–5.5 B/cell** (+1.0 with `-depmask`). Currently single-threaded (`flood.rs:26-76`).
- **`d8_flow` (`resolve_flats`) — Global Peak**: Fixed base `elevations` (4) + `direction` (1) + `slope` (4, sits dead across `resolve_flats`) + `fall` (4) + `rise` (4) = **17.00 B/cell**, plus `flats`, `frontier`, and conditional `artificial` (+4.0 B/cell).
- **`area_d8`**: `direction` (1) + `state` (1) + `area` (4) = **6.00 B/cell** (`elevations` and `slope` drop at `main.rs:162,177`; library `lib.rs:103-130` keeps them if retained, peaking at 14.0 B/cell in `area_d8`).

### 1.2 Why ELC Peaks at 23.6 B/cell (25.0 GiB) vs 17.4 B/cell on `1e8` Synthetic

Direct sampling of ELC (`C:\Temp\ELC_v2_validation\Terrain\2022LiDAR+CoNED.2022LiDAR+CoNED.tif`, $35,999\times 31,635 = 1.1388\times 10^9$ cells, 621 MB DEFLATE, 256×256 tiled, EPSG:6479) across 240M cells reveals:
- **Valid cells**: 62.35% (37.65% nodata); **elevations are quantized to `1/32 ft` (`0.03125`)** (only 112–1,138 distinct values per 8.2M-cell strip).
- **Pre-fill flat/pit fraction**: **38.84% of all cells (62.29% of valid cells)** already have no lower 8-neighbor before `pit_remove`; after pit filling, **~42–48% of all cells (~480–550M cells)** enter `resolve_flats`.
- **Exact +6.2 B/cell jump**:
  1. `17.00 B/cell` fixed `d8_flow` base. On `synth_1e8` (`bench/make_dem.py:202`, `flat_frac=0.15`, 28% nodata → $f \approx 0.08$, 1 flat pass), `flats` + `frontier` add ~0.40 B/cell → **17.4 B/cell (1.62 GiB)**.
  2. **`+4.00 B/cell` (`+4.24 GiB`) from `artificial`** (`d8.rs:277`): ELC's huge coastal flats leave secondary ties after pass 1, triggering iteration 2+.
  3. **`+1.89 B/cell` (`+2.00 GiB`) from `flats: Vec<u32>`** (`d8.rs:86`): ~480–530M flat indices double `Vec` capacity to $2^{29} = 536.87\text{M} \times 4\text{ B} = 2.15\text{ GB}$ (and would hit 4.29 GB above 536.87M).
  4. **`+0.47–0.71 B/cell` (`+0.5–0.75 GiB`)** from `frontier: VecDeque<u32>` (2× growth on wide coastal flat wavefronts) plus allocator/process overhead.
  - Sum: `17.00 + 4.00 + 1.89 + 0.71 = 23.60 B/cell = 25.03 GiB` (measured: **25 GiB / 23.6 B/cell**). Scaling linearly to Tickfaw ($3.991\times 10^9$ cells) requires **~88 GiB (~94 GB)**.

### 1.3 Minimal In-Memory Floor Without Streaming

Purely in-memory changes (the domain of the concurrent optimization agent) can cut peak RAM by **2.5–3.1×** while remaining 100% bit-exact:

1. **Stream `-sd8` directly after steep/`dontCross` and never hold full-grid `slope` (`-4.00 B/cell`)**: `resolve_flats` never reads `slope` (`d8.rs:91-112`). Compute and write `sd8` in 256-row stripes (`width × 256 × 4 B` ≈ 36 MB on ELC) via `write_rows_slice` before allocating flat buffers.
2. **Drop DEM or overwrite `elevations` in-place for `artificial` (`-4.00 B/cell`)**: `-fel` is already written at `main.rs:146` and `elevations` is dropped at `main.rs:162`. Passing `&mut [f32]` to `d8_flow` lets Iter 2+ overwrite `elevations` in-place (matching `src/d8.cpp:673`) and immediately reuse the same 4 B/cell allocation as `area: Vec<f32>` in `area_d8` (avoiding a 4.5 GB Windows page-fault/zero-fill cycle) before writing `-ad8` from that reused buffer.
3. **Eliminate `flats: Vec<u32>` (`-1.9` to `-3.8 B/cell` on ELC)**: Replace with a 1-bit `BitVec` (0.125 B/cell) or scan `direction == 0` directly (faster sequential scan when $f > 10\%$).
4. **Fuse `fall` + `rise` into ONE `surface` grid + 1-bit `has_rise` (`-3.875` to `-5.875 B/cell`)**:
   - Compute `incfall` into `surface` (`i32` = 4 B/cell, or `i16` = 2 B/cell with a tiny `HashMap<u32, i32>` for BFS depths $\ge 32,767$).
   - During `incrise` BFS, mark `has_rise` bitset (0.125 B/cell) and **subtract BFS wave depth `d` directly into `surface[i]`** (`surface[i] = fall[i] - d`). After `incrise`, add global `last_rise + 2` on the fly inside `surface(i)` (`d8.rs:244-251`) whenever `has_rise[i] == 1`. Eliminates the entire `rise` array bit-identically.
5. **Compact `area_d8`**: `direction` (1 B) + `state` (1 B) + reused `area` (4 B) = **6.0 B/cell** (or **5.0 B/cell** by packing 4-bit `in_degree`/flags into unused high bits of `direction`, since codes are `0..=8` and `255`).

**In-memory floor**: `elevations` (4.0) + `direction` (1.0) + `surface` (`2.0` `i16`+overflow or `4.0` `i32`) + `has_rise`/`flat_bits` (0.25) + `frontier` (~0.25) = **7.5 B/cell** (`i16`+overflow) to **9.5 B/cell** (`i32`). That yields **8.5–10.8 GiB** on ELC and **30–38 GiB** on Tickfaw — **still OOMing 32 GB workstations and missing the "few GB at 1e9" goal**. Streaming is strictly required.

---

## 2. Algorithm Options Per Stage & Exactness Analysis

### 2.1 Stage 1: Pit Filling (`pit_remove`)

| Option | Peak Memory | Compute & Parallelism | Input Passes | Bit-Exact? | Verdict |
|---|---|---|---|---|---|
| **Barnes et al. (2016) Parallel Tile Priority-Flood** | $O(P \cdot T^2 + \text{perim})$ (~0.4–1.4 GB active + ~170 MB perim on 4B) | $O(N \log T)$ across $P$ threads (parallel Pass 1 & 2; <0.2 s global Dijkstra) | **2** (Pass 1 + Pass 2) | **Yes (100%)** | **Adopt** |
| **RichDEM Tile-Cache (Evict / Retain / Compressed-RAM)** | Tunable: Evict = lowest; Compressed-RAM = ~0.8–1.4 B/cell | Same as Barnes 2016; skips Pass 2 on tiles with no raised perimeter | 1 (Retain) or 2 (Evict) | **Yes (100%)** | **Integrate via `TileStore`** |
| **Arge et al. (2003) TerraFlow (External EMST + Sort)** | $O(M)$ strict external memory | $O(\text{sort}(N))$ external merge-sorts; high constant factor | 6–12 full disk sort passes (~150–300 GB I/O for 4B) | Yes | **Reject** (10–20× slower; perimeter graph fits in 170 MB RAM) |
| **Strip-Streaming Planchon–Darboux** | $O(\text{strip})$ (~100 MB) | $O(N^{1.5})$ worst case; $\Omega(K)$ strip sweeps on winding valleys | 10–500+ full passes | Yes | **Reject** (pathological disk I/O) |

**Exactness proof & TauDEM semantics (`src/flood.cpp:243-271`, `rust/src/flood.rs:10-77`)**:
- **Uniqueness**: Fill-to-flat computes $Z_{\text{fel}}(v) = \min_{\pi: v \leadsto B} \max_{u \in \pi} Z(u)$ using only comparisons and exact `f32` copies (`flood.rs:68-73`, zero arithmetic). Any algorithm solving exact minimax paths to the TauDEM outlet set $B$ produces **bit-for-bit identical (`max_diff == 0`)** results.
- **Boundary, nodata, `-depmask`, and `-4` across tile seams**:
  1. **1-cell read halo**: Read each $T\times T$ tile with a 1-cell halo (`(T+2)×(T+2)`, clamped at raster borders with `nodata` padding). Any valid tile cell with a `step`-connected (`step=1` for 8-way, `step=2` for `-4`) nodata halo neighbor, on the global raster perimeter, or with `depmask == 1` (`flood.cpp:249-266`, `flood.rs:26-61`) is marked a **true outlet** (`label = OUTSIDE`).
  2. **Pass 1 compressed tile spill graph**: Allocate `label: Vec<u16>` (2 B/cell, serving as both `visited` and source basin ID for $T \le 8192$ since $4T - 4 + 2 \le 32,766 < 65,535$). Seed `OUTSIDE = 1` at all true outlets and distinct IDs `2..=P_T+1` at remaining valid perimeter cells. Run local Barnes 2014 (`FloodQueue` + `pit` FIFO). Whenever popped cell `i` touches visited neighbor `j` with `label[j] != label[i]`, emit undirected edge `(label[i], label[j], max(fel_local[i], fel_local[j]))`; compress edges via Union-Find/Kruskal to an MST of $\le P_T$ edges (~130–200 KB per $4096^2$ tile).
  3. **Global perimeter Dijkstra**: Stitch all tile MSTs with cross-seam edges between valid perimeter cells of adjacent tiles — horizontal, vertical, and (when `!four_way`) **diagonal across seams and 4-way tile corners** — with edge weight $\max(Z(u), Z(v))$. Run monotone radix Dijkstra from `OUTSIDE` to obtain exact global `Z_global[p]` for every perimeter cell `p`. Because Pass 1 pops every perimeter node `p` before any saddle incident to `p`, $Z(p) \le w(p, q)$ on every internal MST edge, so the compressed graph preserves exact global minimax paths (Hu 1961 / Barnes 2016).
  4. **Pass 2**: Re-read raw `z` (and optional `depmask`) for tile $T$, overwrite every non-outlet perimeter cell `p` with `Z_global[p]`, seed `OUTSIDE` at `Z` and perimeter cells at `Z_global[p]`, and run local Priority-Flood. Since `Z_global[p]` already incorporates all paths inside and outside $T$, Pass 2 produces exact global `fel` for every interior cell.
- **Evict vs Retain**: Evicting `z` and `fel_local` after Pass 1 costs **0 intermediate storage** because local DEFLATE GeoTIFFs sit in OS page cache (1.3 s to re-read ELC). Only retain if input is remote or slow uncompressed/striped.

### 2.2 Stage 2a: D8 Steep Pass & Slope (`sd8`)

- **Core 3×3 stencil** (`d8.rs:47-72`, `d8.cpp:359-409`): Any nodata or global-border 8-neighbor yields `DIRECTION_NODATA`; else `steepest` evaluates `ORDER = [1,3,5,7,2,4,6,8]` with per-row WGS84 `row_factors` (`io.rs:54-84`).
- **Critical exactness subtlety — `dontCross` (`d8.cpp:54-100,142`; `d8.rs:75-88,136-147`)**:
  - `crosses(grid, &direction, i, k, i)` depends **only** on already-finalized directions north (`i - W`: `k=2` checks `dirs[N]==8`, `k=4` checks `dirs[N]==6`) and west (`i - 1`: `k=4` checks `dirs[W]==2`, `k=6` checks `dirs[W]==8`). `k=8` never crosses in row-major order.
  - Because `calcSlope` (`d8.cpp:281-282`, `d8.rs:91-108`) reads post-`dontCross` `direction`, **both `p` and `sd8`** require cross-seam `dontCross` causality.
  - **Exact tiled solution**: Schedule Pass B tiles in **NW→SE anti-diagonal wavefront order** (`tx + ty = d`, parallelism $\min(M_x, M_y) \ge 8$ on ELC/Tickfaw). Each tile $(tx, ty)$ reads the finalized bottom row of `direction` from $(tx, ty-1)$ and right column from $(tx-1, ty)$ (just $2T$ bytes = 4–8 KB/tile), runs steep + row-major `dontCross`, and streams `-sd8` directly to `GeoTiffWriter`.
- **WP-S0 finding: `dontCross` does not fire.** In exact arithmetic, a cell and its north (or west) neighbor cannot both choose crossing diagonals in the steep pass. If `A = (x, y-1)` takes SE over E and `B = (x, y)` takes NE over E, with targets `p = (x+1, y-1)`, `q = (x+1, y)` and `r = D/dx > 1`, the two choices need `r p - q > (r-1) z_A` and `r q - p > (r-1) z_B`. Their sum gives `p + q > z_A + z_B`, which contradicts the positive drops `z_A > q` and `z_B > p`; the W/SW case and any row spacing are alike. The transcription in `rust/tests/support/d8_reference.rs` found **no** crossing event, in the steep pass or any flat pass, on the 6.1 suite, on 200,000 random small terrains (including `f32` near-ties and anisotropic cells), and on 3,000 smooth quantized terrains needing up to 3 flat passes. WP-S2 may therefore run the steep pass fully in parallel and keep the wavefront (`tiling::for_each_wavefront`) as a fallback for tiles where a crossing condition is detected, rather than as the default schedule.
- **Zero-read `fel` halo fusion**: In fused `pipeline`, the 1-cell outer `fel` halo of tile $(tx, ty)$ consists **exclusively of perimeter cells of its 8 adjacent tiles** — whose exact final `fel` elevations are **already in RAM** inside Global Solve 1's `Z_global`! In standalone `d8flowdir`, read `(T+2)×(T+2)` of `fel` directly from the input GeoTIFF instead.

### 2.3 Stage 2b: Flat Resolution Across Tiles (Hardest Exactness Problem)

**Exact dependency structure of TauDEM's Garbrecht–Martz (`src/d8.cpp:412-680`, `rust/src/d8.rs:149-284`)**:
1. **`incfall`** (`d8.rs:169-212`): Multi-source unweighted BFS over **equal-elevation** `dirs == 0` cells (skipping edges where `crosses` holds against all assigned `dirs`), seeded at `fall = 1` (adjacent to `dirs[j] in 1..=8 && elev[i] >= elev[j]`) and `fall = 2` (adjacent to `dirs[j] != 0 && elev[i] == elev[j]`, matching `d8.cpp:539-541`). Unreachable pits get `DIRECTION_NODATA` and `fall = global_last_fall.max(1) + 2` (`d8.rs:241`; absent after `pit_remove`).
2. **`incrise`** (`d8.rs:217-235`): Multi-source unweighted BFS over **all `dirs == 0` cells without an elevation-equality check** (`d8.cpp:616`, `d8.rs:230` — matters when `dontCross` leaves adjacent cells at different elevations flat), seeded at `rise = 1` where any 8-neighbor has `elev[j] > elev[i]`.
3. **Combined surface & iteration** (`d8.rs:244-283`): `surface(i) = fall[i].max(1) + (global_last_rise + 2 - rise[i])` when `rise[i] > 0`, else `fall[i].max(1)`. `setFlow2` picks steepest `surface(i) - surface(j)` over `rise[j] > 0` neighbors, or immediately breaks on the first `rise[j] == 0` neighbor with `elev[i] >= elev[j]`. Remaining flats replace `elevations` globally with `surface(i) as f32` (`artificial`) and repeat.

**Tile-Interior Decoupling Theorem (Proves Exactness of Option b)**:
Let $F$ be an 8-connected component of `dirs == 0` strictly interior to tile $T$ (touching no tile border and no unresolvable pit).
- Every 4-neighbor checked by `crosses(i, k)` for $i \in F$ is either in $F$ or already non-flat, so $F$ never interacts with another flat component via `crosses`.
- Replacing `global_last_rise` with $F$'s `local_last_rise` shifts `surface(u)` by a uniform constant $-C \le 0$ for every $u \in F$ with `rise[u] > 0`. In `setFlow2`, `surface(i) - surface(j)` cancels $C$ exactly. For any adjacent $v \notin F$, `surface(v) = 1` whereas `local_surface(u) >= 3 > 1` (any $u \in F$ with `rise[u] == 0` resolves on Iter 1 via `elev[u] >= elev[v]`). By induction on iterations $t = 1, 2, \dots$, every comparison `artificial[i] >= artificial[j]`, `==`, and `>` is **100% invariant to $C$** (up to `f32` exact-integer limit $2^{24}$).
- **Result**: Every tile-interior flat component resolves **bit-exactly** inside Pass B with zero global memory.

**Comparison of Cross-Tile Flat Resolution Options**:

| Option | Memory (ELC, $f \approx 45\%$) | Compute & I/O | Bit-Exact vs TauDEM? | Verdict |
|---|---|---|---|---|
| **(a) Global Sparse Flat Store** (all flat cells in RAM) | ~6–8 GiB on ELC (~500M flats × 12–16 B); **~20–28 GiB on Tickfaw** | Fast (1 RAM graph), 0 tile re-reads | **Yes (100%)** | **Reject as sole strategy** (breaks 4–16 GB budgets on coastal LiDAR) |
| **(b) Tile-Local Interior + Active-Tile Label-Correcting Relaxation for Border Flats** | **Bounded**: active $P \times T^2 \times 11\text{ B}$ + compressed `TileStore` (~1.0 GiB ELC; spills to NVMe at 4 GB) + 4–16 MB border strips | 0 extra work on interior-only tiles; $R \approx 2\text{–}6$ active-tile BFS rounds on border-flat tiles only | **Yes (100%)** — exact shortest-path fixed point + 8-byte global `last_rise`/`last_fall` reduction | **Adopt as default** |
| **(c) Barnes (2014) 1-Pass BFS (`2 * l_dist + away_from_high`)** | Same bounded model as (b), but **never needs Iter 2 / `artificial`** | ~30–50% faster flat pass (1 pass guaranteed, no `artificial`) | **No** — 2× weight on `toward_lower`, per-elevation `incrise`, lacks `fall=2` & `dontCross` rules (`d8.rs:175,183-196,230`); alters flat `p` & downstream `ad8` | **Optional `--fast-flats` mode only** (needs user approval) |

**Active-Tile Exact Garbrecht–Martz Protocol (Option b — Pass C)**:
1. In Pass B, after resolving all strictly interior flat components, tiles with any border-touching `dirs == 0` cell run initial local `incfall` and `incrise` from tile-interior seeds, publish their 4 border strips (`4T × 4 B = 32–64 KB/tile` in `BorderTable`), and store `fel` + `direction` + `fall` + `rise` (compressed ~1.2 B/cell) into `TileStore`. Tiles with **no** border-touching flats drop `fel` immediately.
2. **Active-tile `incfall` relaxation** (parallel over active tiles, prioritized by min border distance): A tile is active if any adjacent tile lowered a border cell's `fall` (matching equal elevation and `!crosses` across the seam, using `BorderTable<u8>` for neighbor `dirs` and `BorderTable<f32>` for seam `fel`). Load tile, push improved border cells (`d_nb + 1 < fall[local]`) into a bucketed queue, relax equal-elevation flat cells, update borders and write back. Converges to exact global shortest-path `fall`.
3. **Active-tile `incrise` relaxation**: Identical loop over all `dirs == 0` cells (no elevation-equality filter!), seeding border-adjacent higher-elevation cells from `BorderTable<f32>` on round 0. Reduce 8-byte global `last_rise` (and `last_fall` if any pits exist).
4. **Direction assignment & Iter 2+**: One parallel pass over border-flat tiles runs `setFlow2` using exact `surface(i)` and updates `direction`. If any cells remain `dirs == 0`, overwrite `fel` with `artificial[i] = surface(i) as f32` ONLY for tiles still containing unresolved flats and repeat Steps 2–4. Once converged, drop all `fel`/`fall`/`rise` tiles from `TileStore`.

**As implemented in WP-S2** (`rust/src/d8_tiled.rs`, `taudem_rs::d8_tiled`; CLI `d8flowdir --mode tiled|auto` in `rust/src/d8flowdir_tiled.rs`):

- **Pass B** (`TiledD8::process_tile`, tiles in any order, in parallel). One plain reader thread reads each tile's `fel` with a 1-cell halo and feeds the Rayon workers through a bounded channel: a worker that waited on Terrio's own read pool stole and nested further tiles (T-S2), which unbounded the tiles in flight. Per tile: unrestricted steep directions; crossing candidates (inside the tile now, on seams after the pass from `BorderTable<u8>`); `sd8` to the tile sink; 8-connected flat components labelled. Components touching no tile edge run every Garbrecht–Martz pass inside the tile with **tile-local** `last_fall`/`last_rise`; the others are *deferred*: the tile computes their first-pass `incfall`/`incrise` levels from the seeds and steps inside it, publishes perimeters, and stores `dir` (u8), `elev` (f32), `adj` (u16: equal-height step mask and unresolved-neighbor mask, in-tile steps only) and the levels (fall and rise packed 16+16 in one u32, unpacked when a tile's levels exceed 16 bits) in the `TileStore`. Tiles without deferred cells deliver final directions at once.
- **Pass C** (`TiledD8::finish`), one sweep per flat pass. (1) *Edge cells*: each active tile's unresolved perimeter cells get their steps, unresolved neighbors and seeds across the tile edge (drain, nodata-default and higher-neighbor) from `BorderTable`s of directions and the pass's surface; every crossing test there involves only perimeter and ring cells. (2) *Label-correcting rounds*. TauDEM's increments are breadth-first distances (`incfall` from the drains over equal-height steps, `incrise` from higher ground over every unresolved cell), so each pass's levels are multi-source shortest-hop distances, and the rounds propagate them across tiles through perimeter cells: a pending tile first checks from the perimeter tables alone whether a ring value + 1 or an edge seed lowers any perimeter level; only then is it loaded, lowered, searched again (a level-ordered BFS from the lowered cells) and stored, and if its perimeter changed its 8 neighbors are flagged for the next round. Fall and rise are independent searches and relax in the same rounds. (3) The pass's raster-wide `last_fall`/`last_rise` = max over Pass C tiles and the interior components' maxima of that pass (recorded in Pass B). (4) *Route + next init* in one visit: `setFlow2` with exact levels, the next pass's artificial surface (1 outside the pass's cells), new perimeters, and the next pass's in-tile levels; a tile with no unresolved cell left delivers its directions and drops its layers.
- **Exactness** (proof sketch; the module documentation has it in full). (i) Flat components evolve independently: every neighbor of a cell of component `C` is in `C` or was never unresolved, and from pass 2 the artificial surface is 1 outside the previous pass's cells, so `C`'s seeds, steps and routing read only `C` and fixed cells. (ii) Every pass resolves at least one cell of each component (a seeded cell drains; an unseeded one becomes nodata), so TauDEM's no-progress stop never fires and component pass counts are independent. (iii) `last_rise` shifts every rised cell of a component uniformly (the rise search covers the whole component, so it is rised everywhere or nowhere); the shift cancels in every level difference routing takes and in every `==`/`>=`/`>` of the next pass's surface while levels stay below 2^24 (`finish` returns an error otherwise), and against cells outside the component the surface is >= 3 > 1 either way. (iv) `last_fall` enters only through the level of cells no drain reaches; within a component that is wholly unreached it cancels, and drained cells never choose an unreached neighbor. It matters only for *mixed* components (drained and unreached cells side by side, which needs a crossing-blocked step or an underflowed slope). Pass B detects these and defers them; Pass C uses the exact raster-wide constants, so its artificial surfaces equal the in-memory ones. (v) The relaxation reaches the unique fixed point of the shortest-path equations, so levels — and every result — are identical for any tile size, round order and thread count. (vi) The steep pass equals TauDEM's row-major crossing rule when no candidate exists; a candidate makes `finish` report `D8Outcome::CrossingDetected`, and the driver reruns with `SteepSchedule::Wavefront` (`for_each_wavefront`, exact rule from final N/W rings). No terrain has triggered it.
- **Kernel interface for WP-S4.** `TiledD8::new(spec, nodata, spacing, &TileCache, SteepSchedule)`; `process_tile(id, fel_halo, Option<&mut Vec<f32>> sd8, finished: FnOnce(&[u8]))` takes the radius-1 halo buffer of `spec.halo_window(id)` (cells outside the raster are never read), so PitRemove Pass 2 can hand over each tile's filled buffer without a re-read; `finish(self, &dyn Fn(TileId, &[u8]))` runs Pass C and delivers the remaining tiles' directions (each tile exactly once over both calls; row-major cores, 255 = nodata). `d8_flow_tiled(spec, nodata, spacing, cache, schedule, read, sd8_sink, p_sink)` is the standalone driver; `tile_size_for_budget` and `WORK_BYTES_PER_CELL` size tiles and the cache; `D8TileStats` carries the counters (`record_profile` for `--profile-json`).
- **Deviations from the protocol above.** The steep pass is parallel with crossing detection and a wavefront restart (WP-S0 finding), not wavefront-scheduled. Interior components use tile-local constants (justified by (iii)–(iv)) and mixed ones are deferred; Pass C uses exact global constants rather than local ones. `incfall` and `incrise` relax together, routing of pass `t` is fused with initialisation of pass `t + 1`, and edge seeds are computed from the perimeter tables instead of a round-0 tile load. Rounds are chaotic Jacobi (tiles of a round run in parallel and read their neighbors' latest perimeters), not min-border-distance priority order. `fel` is stored for deferred tiles in Pass B rather than re-read.

- **Measured** (CLB-08, 8 logical CPUs; other agents' jobs were running, so wall times are ranges over repeated runs; filled DEM from the rust-port in-memory `pitremove` as input, outputs compared cell by cell and bitwise with in-memory `d8flowdir` on `rust-port`):

| Raster | Run | Tiles | Wall | Peak working set | Flat passes: relaxation rounds / tile loads |
|---|---|---|---:|---:|---|
| `1e8` synthetic (1.0e8 cells, 33.0M flat, 3 passes) | in-memory, 8 threads | — | 2.6 s | 898 MiB | — |
| | tiled `--max-memory 512M`, 8 threads | 187 × 768² | 4.5–5.3 s | 416 MiB (commit 436) | 5/244, 3/147, 2/1 |
| | tiled `2G`, 8 threads | 35 × 1792² | 3.6–4.5 s | 1,533 MiB | 3/52, 2/30, 2/1 |
| | tiled `512M`, 1 thread | 28 × 2048² | 14.7 s | 480 MiB | 3/40, 3/27, 1/0 |
| ELC (1.139e9 cells, 602.8M flat = 53%, 4 passes) | in-memory D8 phases of the rust-port pipeline, 8 threads | — | ≈ 39 s (steep 2.3, sd8 4.5, flats 30.4, p 2.1) | ≈ 10 GiB | — |
| | tiled `2G`, 8 threads | 378 × 1792² | 54–72 s | 1,803 MiB (commit 2,001) | 10/662, 3/180, 2/7, 1/0 |
| | tiled `4G`, 8 threads | 288 × 2048² | 54–78 s | 3,641 MiB (commit 3,883) | 8/498, 3/150, 2/10, 1/0 |
| | tiled `2G`, 3 threads | 288 × 2048² | 79.4 s | 1,902 MiB | 7/490, 3/151, 2/10, 1/0 |

  ELC flats at `T = 1792`: 79.7M flat cells (13%) are in components inside one tile; 523.1M are in 20,158 components reaching a tile edge, of which 10,411 (holding 99.97% of those cells) cross tiles; the largest has 16.3M cells and the widest spans 26 tiles (a 6 × 6 box). The per-pass highest levels (fall 10,616 / 436 / 13 / 1, rise 933 / 113 / 6 / 1) sum to the in-memory counters (11,066 and 1,053), and pass 4 routes the same 3 cells. At `2G` Pass B takes 24 s and Pass C 30 s (pass 1 20 s); about 39 GB of tile state goes through the LZ4 codec, which dominates Pass C. Relaxation rounds stay at 7–10 for the first pass because a round lowers a whole tile at once: levels cross several tiles per round, and the number of rounds follows the tile hops of the longest search path (11,066 cells over 1792-cell tiles), not its length.

### 2.4 Stage 3: Flow Accumulation (`area_d8`)

**Algorithm: Barnes (2017) 2-Pass Parallel Tile Flow Accumulation**:
- **Pass 1 (parallel over tiles)**: Load tile `direction` (`u8`, 1 B/cell — from `TileStore` in `pipeline`, or from GeoTIFF via `read_geotiff_samples_into::<i16>` in standalone `aread8`) + 1-cell `direction` halo from `BorderTable<u8>` (1.2 MB ELC, 4.0 MB Tickfaw — zero neighbor-tile loads!).
  - Initialize exact contamination (global border or any 8-neighbor `DIRECTION_NODATA`, `area.rs:25-30`) and preserve TauDEM's `k == 4 && other == 0` quirk (`area.rs:28`, `commonLib.cpp:267`, where an unresolved NW flat permanently blocks a cell even under `-nc`).
  - Run local chain accumulation (`area.rs:35-68`) without decrementing external halo inflows. For every **perimeter entry cell `p`** (receiving flow from an outside halo cell), trace its unique D8 downstream path inside the tile to its **perimeter exit cell `q`** (or interior termination), recording link `p → q` plus `q`'s local sum `A_local[q]` and state flags (`contaminated`, `blocked`). Evict `state` (1 B) and `area` (4–8 B) immediately.
- **Global Perimeter DAG Solve (<0.05 s in RAM)**:
  - Because D8 is non-divergent, every perimeter cell has **out-degree ≤ 1**. For Tickfaw ($253$ tiles of $4096^2$), total perimeter nodes $\le 4.14\text{M} \times 16\text{ B} = \mathbf{66\text{ MB}}$ RAM.
  - Topologically sort the stitched perimeter-cell graph (acyclic even when a meandering river creates cycles in the coarse tile-to-tile graph!), propagating `(area: u64, contaminated: bool, blocked: bool)`.
- **Pass 2 (parallel over tiles + streaming `-ad8` write)**:
  - Reload `direction`, inject solved incoming halo `(area, contaminated, blocked)` at perimeter entry cells, run exact local chain accumulation in **fixed `k = 1..8` neighbor order** (`area.rs:43-47`), and stream `-ad8` tiles to `GeoTiffWriter`.
- **Critical Bit-Exactness Subtlety — `f32` Non-Associativity Above $2^{24} = 16,777,216$ Cells**:
  - For all cells with area $\le 2^{24}$ (all `TauDEM-Test-Data` and ~99.9% of ELC/Tickfaw cells), integer addition in `f32` is exact and associative, so `u64` perimeter superposition cast to `f32` is **100% bit-exact** with `area.rs` / `aread8.cpp`.
  - Above $2^{24}$ (mainstem rivers on ELC/Tickfaw), `(16777216.0f32 + 1.0f32) == 16777216.0f32`, so linear superposition produces the mathematically exact count (`16777316`) whereas C++ TauDEM / `area.rs` loses `+1` increments.
  - **Exactness solution**: The `u64` perimeter solve identifies every perimeter link with `area_u64 > 2^24` (a few dozen 1D mainstem rivers, `<0.05%` of cells). Under `--compat-f32` (or default if Decision 1 selects strict C++ parity), replay only those 1D mainstem tile segments in topological segment order using `f32` arithmetic (<0.2 s, <5 MB RAM) to guarantee **100% bit-exact parity at all sizes**.

### 2.5 Fused Tile Pipeline (Passes & Intermediate Spilling)

```text
Pass A (parallel, DEM read #1):
  Read z (+1-cell halo) -> PitRemove Pass 1 (tile MST) -> Evict z, fel_local
  └─> Global Solve 1: Perimeter Dijkstra -> Z_global (~170 MB on 4B)

Pass B (NW->SE wavefront, DEM read #2):
  Read z (+1-cell halo) -> PitRemove Pass 2 (using Z_global)
    ├─> [-fel] TileEncoder -> async GeoTiffWriter (0 spill)
    ├─> D8 Steep + dontCross (exact fel halo = Z_global! N/W dir borders from wavefront)
    ├─> [-sd8] TileEncoder -> async GeoTiffWriter (0 full-grid slope!)
    ├─> Local Garbrecht-Martz on tile-interior flats (bit-exact)
    └─> Put direction -> TileStore (LZ4 ~0.4 B/cell);
        Put fel+fall+rise -> TileStore ONLY for tiles with border-touching flats!

Pass C (active-tile parallel, ONLY border-flat tiles):
  Iterative active-tile incfall + incrise -> setFlow2 (repeat if Iter 2+)
  └─> Update direction in TileStore; DROP all fel/fall/rise tiles!

Pass D (parallel):
  Get direction (+BorderTable<u8>) -> [-p] TileEncoder -> async GeoTiffWriter
    -> AreaD8 Pass 1 (perimeter link graph) -> Evict state, area
  └─> Global Solve 2: Perimeter DAG (~66 MB on 4B) [+ optional >2^24 f32 mainstem pass]

Pass E (parallel):
  Get direction (+solved halo inflows) -> AreaD8 Pass 2 (exact k=1..8)
    -> [-ad8] TileEncoder -> async GeoTiffWriter -> Drop direction tile
```

**Total I/O**: **2 reads** of input DEM GeoTIFF, **1 streaming write** per requested output, **0 bytes** ever stored for `slope` or `area`, **0 bytes** of `fel` stored on tiles without border-crossing flats, and `direction` compressed at ~0.4 B/cell.

---

## 3. Tiling Infrastructure, Shared Contracts & Composition

### 3.1 Tile Geometry (`TileSpec`) & Core Contracts (WP-S0 Interface)

Replace `Grid`'s `u32::MAX` cell limit (`rust/src/lib.rs:34`) with 64-bit raster dimensions (`width: usize, height: usize, cells: u64`) + **tile-local `u32` cell indices** ($T^2 \le 67\text{M} \ll 2^{32}$), supporting $>4.29\times 10^9$ rasters (`AdamsCowBayou`).

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TileId { pub tx: u32, pub ty: u32 }

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum TileLayer { Fel, Dir, Fall, Rise }

pub struct TileSpec {
    pub raster_w: usize, pub raster_h: usize,
    pub tile_w: usize,   pub tile_h: usize,   // aligned to native GeoTIFF block_size
    pub tiles_x: u32,    pub tiles_y: u32,
}
impl TileSpec {
    /// Core interior rect (x0, y0, w, h) clamped at right/bottom raster edges.
    pub fn rect(&self, id: TileId) -> (usize, usize, usize, usize);
    /// Window with 1-cell halo clamped to raster bounds + pad offsets (pad_l, pad_t).
    pub fn halo_window(&self, id: TileId) -> (terrio::Window, usize, usize);
    /// All tile IDs in Morton (Z-order) for locality, or anti-diagonals `tx + ty == d`.
    pub fn morton_order(&self) -> Vec<TileId>;
    pub fn wavefront_groups(&self) -> Vec<Vec<TileId>>;
}

/// 3-tier tile store: Tier 0 hot RAM -> Tier 1 LZ4-compressed RAM -> Tier 2 positioned disk spill.
pub struct TileStore { /* Mutex/RwLock sharded map + AtomicU64 ram_bytes + SpillFile */ }
impl TileStore {
    pub fn put_u8(&self, id: TileId, layer: TileLayer, data: &[u8]) -> Result<()>;
    pub fn put_f32(&self, id: TileId, layer: TileLayer, data: &[f32]) -> Result<()>;
    pub fn put_u32(&self, id: TileId, layer: TileLayer, data: &[u32]) -> Result<()>;
    pub fn get_into_u8(&self, id: TileId, layer: TileLayer, dst: &mut Vec<u8>) -> Result<()>;
    pub fn get_into_f32(&self, id: TileId, layer: TileLayer, dst: &mut Vec<f32>) -> Result<()>;
    pub fn get_into_u32(&self, id: TileId, layer: TileLayer, dst: &mut Vec<u32>) -> Result<()>;
    pub fn remove(&self, id: TileId, layer: TileLayer);
}

/// Permanently RAM-resident 1-cell border strips + 4 corners for every tile (4 MB on 4B `u8`).
pub struct BorderTable<T: Copy> { /* top/bottom [tile_w], left/right [tile_h], 4 corners per TileId */ }
```

**As implemented in WP-S0** (`rust/src/tiling/`, `taudem_rs::tiling`). The sketch above had untyped `put_u8`/`put_f32` methods keyed by a `TileLayer` enum; it is replaced by typed layers that share one cache, budget and scratch file:

```rust
pub struct TileId { pub tx: u32, pub ty: u32 }            // Ord is row-major (ty, tx)
pub struct Rect { pub x: usize, pub y: usize, pub width: usize, pub height: usize }
pub struct HaloWindow { pub core: Rect, pub window: Rect, pub radius: usize,
                        pub pad_left: usize, pub pad_top: usize, pub pad_right: usize, pub pad_bottom: usize }
impl HaloWindow {
    fn buffer_width/buffer_height/buffer_cells(&self) -> usize;
    fn index(&self, lx: usize, ly: usize) -> u32;          // core cell -> halo-buffer index
    fn pad_into<T: Copy>(&self, window_data: &[T], fill: T, buffer: &mut Vec<T>) -> Result<()>;
    fn core_from_buffer<T: Copy>(&self, buffer: &[T], core: &mut Vec<T>) -> Result<()>;
}
impl TileSpec {                                            // 64-bit raster dims; (T+2)^2 <= u32::MAX
    fn new(raster_w, raster_h, tile_w, tile_h) -> Result<Self>;               // exact (forced T=16/17)
    fn aligned(raster_w, raster_h, target, native_block: Option<(usize, usize)>) -> Result<Self>;
    fn cells(&self) -> u64;  fn tile_size(&self) -> (usize, usize);  fn tiles_x/tiles_y(&self) -> u32;
    fn rect(&self, id) -> Rect;  fn halo(&self, id, radius) -> HaloWindow;  fn halo_window(&self, id) -> HaloWindow;
    fn locate(&self, x, y) -> (TileId, u32);  fn local_index(&self, id, x, y) -> u32;
    fn global_coords(&self, id, local: u32) -> (usize, usize);  fn neighbor(&self, id, k: 1..=8) -> Option<TileId>;
    fn index(&self, id) -> usize;  fn id_at(&self, index) -> TileId;  fn is_aligned_to(&self, bw, bh) -> bool;
    fn row_major(&self) -> impl Iterator<Item = TileId>;  fn morton_order(&self) -> Vec<TileId>;
    fn wavefront_groups(&self) -> Vec<Vec<TileId>>;        // anti-diagonals tx + ty = d
}
/// Runs `visit` on the current Rayon pool, each tile after its W and N neighbors (dependency counting, no barriers).
pub fn for_each_wavefront<E: Send>(spec: &TileSpec, visit: impl Fn(TileId) -> Result<(), E> + Sync) -> Result<(), E>;

pub struct TileCache;                                      // budget, LRU order, scratch file; Clone shares it
impl TileCache {
    fn new(CacheConfig { budget_bytes: u64, scratch_dir: Option<PathBuf> }) -> Self;
    fn store<T: TileElement>(&self, name: &str, StoreOptions { prefilter: Option<Prefilter>, initial: Tier }) -> TileStore<T>;
    fn stats(&self) -> TileStats;  fn enforce_budget(&self) -> Result<()>;
}
impl<T: TileElement> TileStore<T> {                        // u8 i8 u16 i16 u32 i32 u64 i64 f32 f64; &self, thread-safe
    fn put(&self, id, data: &[T]) -> Result<()>;  fn put_owned(&self, id, data: Vec<T>) -> Result<()>;
    fn get_into(&self, id, dst: &mut Vec<T>) -> Result<()>;  fn try_get_into(..) -> Result<bool>;
    fn get(&self, id) -> Result<Option<Vec<T>>>;  fn take(&self, id) -> Result<Option<Vec<T>>>;  // take: no copy from Tier 0
    fn remove(&self, id) -> bool;  fn clear(&self);  fn contains/tier/ids/len/elements(..);
    fn pin(&self, id) -> Result<TilePin<'_, T>>;            // RAII; pinned tiles are never evicted
    fn evict(&self, id, to: Tier) -> Result<()>;            // explicit compress / spill
}
pub enum Tier { Ram, Compressed, Spilled }
pub enum Prefilter { None, Shuffle, Delta, DeltaShuffle }  // TileElement::DEFAULT_PREFILTER per type
pub fn encode_tile<T: TileElement>(data: &[T], filter: Prefilter, out: &mut Vec<u8>);
pub fn decode_tile<T: TileElement>(blob: &[u8], dst: &mut Vec<T>) -> Result<()>;   // verifies checksums

impl<T: Copy + Send + Sync> BorderTable<T> {               // per-tile RwLock; ~2(Tw+Th) values per tile
    fn new(spec: TileSpec, fill: T) -> Self;
    fn set_from_tile(&self, id, tile: &[T]) -> Result<()>;  fn set_from_halo(&self, id, &HaloWindow, buffer: &[T]) -> Result<()>;
    fn read(&self, id) -> RwLockReadGuard<'_, Border<T>>;   // top/bottom/left/right/corner(Corner)/get(lx, ly)
    fn update<R>(&self, id, f: impl FnOnce(&mut Border<T>) -> R) -> R;
    fn ring(&self, id, outside: T) -> Ring<T>;              // 1-cell ring from the 8 neighbors' borders
    fn fill_halo_ring(&self, id, &HaloWindow, buffer: &mut [T], outside: T) -> Result<()>;
    fn cell(&self, x, y) -> Option<T>;
}

impl<T: terrio::RasterElement + Default> AsyncTileWriter<T> {   // GeoTiffWriter owned by one background thread
    fn create(path, &RasterDataset, &RasterWriteOptions /* tile_size required */, queue_depth) -> Result<Self>;
    fn from_writer(GeoTiffWriter, queue_depth) -> Result<Self>;
    fn sink(&self) -> TileSink<T>;                          // Clone + Send + Sync handle for compute threads
    fn submit_block(&self, col, row, data: &[T]) -> Result<()>;    // compress on caller (TileEncoder), bounded send
    fn submit_region(&self, rect: Rect, data: &[T]) -> Result<()>; // any compute tile; partial blocks assembled
    fn finish(self) -> Result<WriterStats>;
}

pub struct ExecutionOptions { pub mode: ExecutionMode, pub max_memory: Option<u64>, pub tile_size: Option<usize>, pub scratch_dir: Option<PathBuf> }
// memory_budget(), scratch_dir(), tile_spec(w, h, native_block), cache_config(budget); default_memory_budget();
// tiling::memory::{usable_memory, available_memory, process_memory}
```

Test support for WP-S1..S3 lives in `rust/tests/support/` (`mod support;` from an integration test): `fixtures::suite(T)` and `large_suite()`; `harness::{TiledStage, stage(..), compare, assert_tiled_matches_in_memory(stage, fixture, tile_size, threads), assert_suite(stage, &TILE_SIZES, &THREADS)}`; `harness::reference::{fel, d8, directions, ad8}`; and `d8_reference`, a transcription of `d8.rs` that can disable and record `dontCross`.

### 3.2 Tile Size Selection, `TileStore` Tiers & Memory Budget (`--max-memory`)
- **Tile size (`--tile-size`, auto-aligned to native `block_size`, typically 256×256)**:
  - **$T = 2048$** ($4.19\text{M}$ cells, $8\times 8$ native blocks) when `budget <= 8 GiB` or `cells < 5e8`: active 8-thread RAM ~340 MB; 288 tiles on ELC, 966 on Tickfaw; `u8` dir (4 MB) + `u16` label (8 MB) fit in L3 cache.
  - **$T = 4096$** ($16.78\text{M}$ cells, $16\times 16$ blocks) when `budget >= 16 GiB && cells >= 5e8`: active 8-thread RAM ~1.35 GB; halves Tickfaw perimeter graphs to 66–170 MB and halves flat seams.
  - Striped inputs (`block_w == raster_w`): align `tile_h` to strip height (or horizontal super-tiles for tall strips).
- **Budget resolution (`--max-memory <bytes>`, `--scratch-dir <path>`, `--mode auto|memory|stream`)**:
  - Default budget: `min(0.65 * terrio::usable_memory(), 0.75 * avail_ram)` (fallback `4 GiB` if unavailable).
  - Reserve `R_work = threads * tile_scratch_bytes + perimeter_graph_bytes + writer_channel_bytes`; remaining `B_cache` bounds `TileStore`.
  - *As implemented*: `--mode auto|in-memory|tiled` (not `memory|stream`); `--max-memory` takes `4G`, `512M`, `1.5GiB` (binary units). The default is `min(0.65 * usable, 0.75 * available)`, where `tiling::memory::usable_memory()` is `terrio::usable_memory()` or, where that is `None` (Windows today), the total physical memory from `GlobalMemoryStatusEx` capped by the virtual address space; `available` is `ullAvailPhys` or `MemAvailable`. An explicit `--tile-size` is used exactly; the automatic size is aligned to native blocks.
  - *Measured* (`rust/examples/tile_codec_bench.rs`, Muncie LiDAR 7892 x 4538, 1024^2 tiles, one thread on a busy CLB-08; the clip's nodata flatters the ratios): `fel` f32 shuffle **0.353 B/cell** (11.3x; 2.0 GB/s compress, 2.7 GB/s decompress; raw LZ4 0.490); `p` u8 raw **0.464 B/cell**; `p` i16 shuffle 0.468 (delta 0.738, so 16-bit types default to shuffle, not delta); `ad8` f32 shuffle 1.49 (raw 1.63); BFS-distance u32 delta **0.018 B/cell**. Blobs are 1 MiB chunks compressed independently (bounded per-thread scratch) behind a 64-byte header with the element type, the tile key and xxHash64-style checksums of header and payload.
  - *Measured budget*: under a forced 64 MiB budget, 768 MiB of tiles from 8 threads (two rounds of put and verified get) peaked at 69.6 MiB of accounted tiles and 101 MiB of working-set growth, against a bound of budget + 8 x 4 tiles of working buffers = 128 MiB; 325 tiles were spilled (`rust/tests/tile_store_stress.rs`).
  - **Tier 1 (Compressed-RAM via `lz4_flex`)**: Pure Rust, >15 GB/s aggregate across 8 cores with dtype prefilters:
    - `Dir` (`u8`): raw LZ4 → **~0.35–0.45 B/cell** (ELC ~450 MB, Tickfaw ~1.6 GiB).
    - `Fel` (`f32`): 4-plane byte shuffle + LZ4 → **~0.7–1.2 B/cell**.
    - `Fall`/`Rise` (`u32`/`u16`): row-wise delta + LZ4 → **~0.25–0.5 B/cell** (adjacent BFS steps differ by $\{-1,0,1\}$).
  - **Tier 2 (Disk Spill)**: When Tier 1 exceeds `B_cache`, evict LRU **already-compressed** blobs via positioned I/O (`FileExt::seek_write`/`seek_read` on Windows, `write_at`/`read_at` on Unix) into a scratch file. **Never use mutable `mmap`** on Windows (dirty mapped pages inflate `peak_wset` and trigger pagefile thrashing).
- **Overlapped output writes & determinism**: Each requested output (`-fel`, `-sd8`, `-p`, `-ad8`) spawns one writer thread owning a `GeoTiffWriter` fed via `std::sync::mpsc::sync_channel(2 * threads)` (`TileEncoder` compresses 256×256 sub-tiles on Rayon workers; backpressure caps inflight bytes at <32 MB; `SPARSE_OK=YES` omits all-nodata blocks). All pixel values are **100% deterministic across thread counts**; optional `--deterministic-file` drains tiles in `(ty, tx)` order if byte-identical TIFF chunk ordering is desired.

### 3.3 Composition with Concurrent In-Memory Optimization
Factor every kernel into a pure tile-slice function (`tile_pit_pass1`, `tile_pit_pass2`, `tile_d8_steep`, `tile_resolve_flats_local`, `tile_flat_relax`, `tile_area_pass1`, `tile_area_pass2`) operating on `&mut [T]` buffers + 1-cell border slices. Optimized in-memory mode invokes these kernels on views of RAM arrays (1 input read, 0 `TileStore` overhead); streaming mode invokes the exact same kernels on worker-pooled `TileStore` buffers. Any single-tile SIMD, radix-queue, or single-buffer `incrise` optimization from the concurrent agent automatically accelerates streaming 1:1.

---

## 4. Concrete `terrio` Needs (Verified Against `taudem-integration`)

**Already present in `G:\GH\terrio-work\integration` (do NOT re-propose)**:
- Lock-free `&self` caller-buffer reads: `LazyRaster::read_geotiff_samples_into<T: TiffSample>` (`crates/terrio/src/cog/into.rs:131-162`) with cached `OnceLock<Arc<ChunkTables>>` (`cog.rs:2613`) and zero decoder re-opens.
- Zero-copy borrowed row writes: `GeoTiffWriter::write_rows_slice` (`drivers/gtiff_write.rs:2898,3215`).
- Any-order parallel tile writes + real `SPARSE_OK=YES`: `TileGrid`, `TileEncoder` (`Clone + Send + Sync`), `CompressedTile`, `write_tile`, `write_compressed_tile` (`gtiff_write.rs:2675-3012`).
- GDAL-exact `BigTiffPolicy` (`IfNeeded`/`IfSafer`/`Yes`/`No`) (`gtiff_write.rs:705-715,3128`).
- Fast pure-Rust DEFLATE: `fdeflate` decode (`tiff_codecs.rs:666`) + libdeflate-derived `deflate_compress.rs:13-27` encode.

**Remaining generic, upstreamable `terrio` gaps** (all adhere to `H:\CLB-Repos\terrio-clb\AGENTS.md`: pure Rust, generic beyond TauDEM, REUSE/SPDX headers, GDAL/I/O-count tests, consistent Rust/Py/JS surfaces):

| ID | Priority | Verified Gap & Source | Generic API / Change & Surfaces | Validation |
|---|---|---|---|---|
| **T-S1** | **High** | `terrio::usable_memory()` (`dataset.rs:2605-2658`) reads only Linux `/proc/meminfo` & `/proc/self/limits`, returning `None` on Windows/macOS. `windows-sys` is already in `crates/terrio/Cargo.toml:200`. | Implement Windows (`GlobalMemoryStatusEx` + Job Object limit via `Win32_System_SystemInformation` / `Win32_System_JobObjects`) and macOS (`sysctl HW_MEMSIZE`), matching GDAL `CPLGetUsablePhysicalRAM`. Rust API unchanged; no Py/JS break. | Platform unit tests on Windows/Linux/macOS asserting `Some(bytes)` in plausible range; Windows Job Object limit test. |
| **T-S2** | **High** | `cog/into.rs:294-298` calls `crate::par::install(...)` directly onto `terrio`'s **separate** Rayon pool (`par.rs:166`), bypassing `Pool::enter()`'s `OUTSIDE_WORK` guard (`par.rs:514-531`). 8 outer `taudem-rs` threads reading tiles concurrently spawn 16 competing threads and allocate fresh `Vec<f32>` + `raw: Vec<u8>` per 256×256 block (`tiff_codecs.rs:595,660`). | Route `cog/into.rs:298` through `Pool::run` (automatic serial decode on caller thread when called concurrently from outside, zero API break); reuse per-worker `raw` & decode buffers across blocks within a window read. | Multi-thread concurrent `read_geotiff_samples_into` test verifying exact pixels, no pool deadlock, and reduced allocation count / faster super-tile read throughput. |
| **T-S3** | **High** | `LazyRaster::tiff_block_range(&mut self, ...)` (`cog.rs:1308-1321,3004-3029`) requires `&mut self` even though `Level.tables` is `OnceLock<Arc<ChunkTables>>` (`cog.rs:2613`) and `decoder` is `Mutex<TiffDecoder>` (`cog/into.rs:172`). Prevents lock-free sparse-tile skipping. | Add `&self` `tiff_block_range_shared(&self, band, x, y, overview)` (or relax `tiff_block_range` to `&self` using `self.decoder.lock()` on first call, matching `direct_image()`). Expose consistently on Python `terrio.rio` if needed. | Concurrent `&self` sparse-query test asserting `0` bytes read for `byte_count == 0` blocks across threads. |
| **T-S4** | **Low (opt)** | `TileEncoder::compress` (`gtiff_write.rs:2746-2763`) wraps `Samples::whole(data, w, h, bands)`, forcing a 256×256 scratch copy per sub-tile even though `ChunkEncoder::encode` (`gtiff_write.rs:2073`) already accepts `(x0, y0)` into a wider `Samples`. | Add `TileEncoder::compress_subwindow<T>(&self, col, row, data: &[T], buf_w: usize, buf_h: usize, x0: usize, y0: usize)` forwarding to `ChunkEncoder::encode` with reusable LZW scratch. | Byte-identical roundtrip vs `compress` and GDAL (not a blocker: L2-resident 256 KiB copy adds <0.2 s on 4B cells). |

*Non-F32 DEMs*: `read_geotiff_samples_into` requires native `T::DATA_TYPE == band_data_type` (`cog/into.rs:145`); `taudem-rs` dispatches on `band_data_type(1)` to read `i16`/`f64` tiles and cast per tile with zero `terrio` changes.

---

## 5. Expected Outcomes Table

**Assumptions** (anchored to CLB-08: 8-core Intel i9-11900KF, 64 GB RAM, local NVMe ~3 GB/s, 8 Rayon threads, writing all 4 products `-fel -p -sd8 -ad8` as 256×256 DEFLATE `ZLEVEL=1` + `SPARSE_OK=YES`):
- **Measured baselines**: `1e8` synthetic = **1.62 GiB (17.4 B/cell) / 15 s**; `ELC` ($1.139\times 10^9$, 621 MB DEFLATE, 37.7% nodata, ~45% post-fill flats, ≥2 flat iters) = **25.0 GiB (23.6 B/cell) / 175 s** (1.3 s read, 12 s write, ~160 s mostly-1T compute); `Tickfaw` ($3.991\times 10^9$, 2.86 GB DEFLATE, 3.50× ELC cells) ≈ **~88 GiB (~94 GB, OOM on 64 GB)**.
- **Parallelism**: Moving `pit_remove`, `resolve_flats`, and `area_d8` from 1T to 8T via Barnes 2016/2017 + parallel tiled flats gives ~5–6× compute speedup (~160 s → ~32–42 s on ELC).
- **Streaming overhead**: +1 extra DEM read pass (1.3 s ELC, ~6 s Tickfaw); `lz4_flex` @ 16 GB/s aggregate (~1–4 s); Pass C border-flat relaxation 2–5 rounds (+3–15 s); NVMe spill of compressed tiles only when `TileStore` exceeds `--max-memory` (+4–8 s for ~12 GB compressed NVMe traffic on Tickfaw at 4 GiB).

| Terrain (Cells) | Current In-Memory (Measured) | (i) Optimized In-Memory (~8–9.5 B/cell) | (ii) Compressed-RAM Tiles (0 Disk Spill) | (iii) Disk-Spill Streaming (`--max-memory 4G`) | (iii) Disk-Spill Streaming (`--max-memory 16G`) |
|---|---|---|---|---|---|
| **`1e8` synthetic** ($1.00\times 10^8$) | **1.62 GiB** / **15 s** | **0.8–0.9 GiB** / **4.5–5.5 s** | **0.55 GiB** / **5.0–6.5 s** ($T=2048$) | **0.55 GiB** / **5.0–6.5 s** *(0 spill)* | **0.55 GiB** / **5.0–6.5 s** *(0 spill)* |
| **`ELC` coastal LiDAR** ($1.139\times 10^9$) | **25.0 GiB** / **175 s** | **8.5–10.5 GiB** / **48–57 s** | **1.8–2.4 GiB** / **54–66 s** ($T=2048$) | **1.8–2.4 GiB** / **54–66 s** *(0 spill)* | **2.6–3.2 GiB** / **52–63 s** ($T=4096$, *0 spill*) |
| **`Tickfaw`** ($3.991\times 10^9$) | **~88 GiB** *(OOM)* | **30–37 GiB** / **180–220 s** *(needs 64 GB)* | **5.8–7.2 GiB** / **200–250 s** ($T=4096$) | **≤ 3.9 GiB** / **215–275 s** ($T=2048$, ~1.8 GiB spill) | **5.8–7.2 GiB** / **200–250 s** ($T=4096$, *0 spill*) |

**Key takeaway**: Compressed-RAM tiling cuts ELC peak RAM **>10×** (25 GiB → ~2 GiB) while cutting wall time **~3×** (175 s → ~60 s), and runs Tickfaw within **4 GiB** in ~4 minutes (<12% overhead vs a 64 GB in-memory run).

---

## 6. Work Packages & Tiling-Boundary Adversarial Tests

Each work package (WP) is sized for **one implementation agent**.

```text
  [WP-T: terrio T-S1..T-S3]       [WP-S0: TileSpec, TileStore, BorderTable, AsyncWriter & Test Harness]
               \                       /           |            \
                \        +------------+            |             +------------+
                 \       |                         |                          |
                  v      v                         v                          v
             [WP-S1: Tiled PitRemove]   [WP-S2: Tiled D8 & Flats]   [WP-S3: Tiled AreaD8]   (S1 || S2 || S3)
                         \                         |                         /
                          +------------------------+------------------------+
                                                   v
                             [WP-S4: Fused Streaming Pipeline, Auto-Tier & Benchmarks]
```

| WP | Deliverables | Depends On | Parallel With | Acceptance Criteria |
|---|---|---|---|---|
| **WP-T** | `terrio` PRs on `taudem-integration`: **T-S1** (Win/macOS `usable_memory`), **T-S2** (fix `cog/into.rs` dual-pool oversubscription + scratch reuse), **T-S3** (`&self` `tiff_block_range_shared`), optional **T-S4** (`compress_subwindow`). | None | WP-S0 | `cargo fmt --check`, `clippy -D warnings`, GDAL conformance & concurrent-read tests green; `usable_memory().is_some()` on Windows. |
| **WP-S0** | `TileSpec` (64-bit dims, tile-local `u32`, block alignment), 3-tier `TileStore` (RAM / `lz4_flex` w/ shuffle+delta / positioned disk-spill), `BorderTable<T>`, `AsyncTileWriter` (`TileEncoder` + bounded channel), CLI flags (`--max-memory`, `--tile-size`, `--scratch-dir`, `--mode`), **adversarial boundary test suite** (§6.1). | None | WP-T | Unit tests for Tier 0/1/2 eviction & integrity; RSS stays below forced 64 MB budget under spill stress; all §6.1 synthetic DEM fixtures generated. |
| **WP-S1** | Barnes (2016) tiled `pit_remove` (standalone + library tile kernels): Pass 1 `u16`-label Priority-Flood + MST spill graph, global radix Dijkstra (with 8-way seam+corner edges), Pass 2 fill, `-4`, `-depmask`, sparse-tile skip. | WP-S0 *(soft: WP-T)* | WP-S2, WP-S3 | **Bit-exact** (`max_diff == 0`) vs in-memory `pit_remove` across `TauDEM-Test-Data`, §6.1 adversarial suite at `T=16/17/32`, `1e8`, and `ELC`; identical across `RAYON_NUM_THREADS=1,4,8`. |
| **WP-S2** | Tiled `d8_flow` (standalone + library tile kernels): steep pass + NW→SE wavefront `dontCross`, streamed `-sd8`, tile-local exact Garbrecht–Martz, active-tile border-flat `incfall`/`incrise` + multi-iteration `artificial` exactness (plus optional `--fast-flats` if approved). | WP-S0 | WP-S1, WP-S3 | **Bit-exact** (`p` and `sd8` exact, 0 differing cells) vs in-memory `d8_flow` on `TauDEM-Test-Data`, §6.1 suite (multi-tile secondary flats & `dontCross` seams), `1e8`, and `ELC`; thread-count invariant. |
| **WP-S3** | Barnes (2017) tiled `area_d8` (standalone + library tile kernels): Pass 1 local accum + perimeter link graph `(area, contaminated, blocked)` preserving `k==4 && other==0` (`area.rs:28`), global perimeter DAG solve, Pass 2 exact `k=1..8` accum, `>2^24` `f32` mainstem compat path. | WP-S0 | WP-S1, WP-S2 | **Bit-exact** vs in-memory `area_d8` (default & `-nc`) on `TauDEM-Test-Data`, §6.1 suite, `1e8`, and `ELC` (incl. `>2^24` mainstems under `--compat-f32`); thread-count invariant. |
| **WP-S4** | Fused Pass A–E `pipeline` (Pass B `Z_global` zero-read `fel` halo handoff, `-fel`/`-sd8`/`-p`/`-ad8` streamed writes), auto memory-budget dispatch via `usable_memory()`, end-to-end `bench.py` parity & scaling validation on `1e8`, `ELC`, and `Tickfaw`. | WP-T, WP-S1, WP-S2, WP-S3 | — | **Bit-exact** end-to-end `pipeline` vs in-memory on `TauDEM-Test-Data`, `1e8`, and `ELC` (at `--max-memory 4G` and `16G`); `ELC` peak < 3 GiB & wall < 75 s; `Tickfaw` completes cleanly at `4G` and `16G`. |

**Recommended order**: Launch **WP-T** and **WP-S0** in parallel → launch **WP-S1, WP-S2, WP-S3** in parallel once WP-S0 lands → integrate and benchmark in **WP-S4**.

### 6.1 Tiling-Boundary Adversarial Test Suite (Built in WP-S0, Gated in WP-S1..S4)

Run every synthetic fixture with forced tiny tile sizes (`T = 16, 17, 32` — including prime/non-multiple widths) at `1` and `8` threads and assert **0 cell differences** against the in-memory reference:
1. **Cross-seam & 4-corner pits**: Deep pit centered on a 4-tile corner whose sole spill saddle lies 3 tiles away along a winding 1-cell diagonal channel (both 8-way and `-4`); pit drained by an interior `-depmask` cell in a neighboring tile.
2. **Nodata at tile edges/corners**: Single nodata cell on an adjacent tile's seam or diagonal corner (must act as PitRemove outlet, D8 nodata mask, and AreaD8 contamination source across the seam).
3. **`dontCross` cascades across seams**: Diagonal-tie checkerboard crossing horizontal and vertical tile boundaries, asserting both `p` and `sd8` match single-threaded row-major scanline order bit-for-bit. *(WP-S0: the fixtures put hundreds of `ORDER` tie-break cells and flat cells on tile edges, but no terrain found makes the crossing rule fire; see 2.2.)*
4. **Multi-tile flats & multi-iteration secondary flats**: L-shaped and spiral flats spanning $3\times 3$ tiles; flat bordering nodata across a seam (`fall = 2` seed); adjacent flats of different elevation blocked by `dontCross` (exercises `incrise` cross-elevation BFS); wide plateau triggering Garbrecht–Martz Iter 2 & 3 (`artificial`).
5. **Cross-tile meandering rivers, coarse tile cycles, & `k==4` quirk**: River exiting tile A → B → C → back into A → B (verifying perimeter-cell DAG solves despite cycles in the tile graph); unresolved flat (`p == 0`) at a tile corner blocking its SE neighbor in the diagonally adjacent tile (`k == 4 && other == 0`, tested with and without `-nc`); synthetic $>1.7\times 10^7$-cell narrow ramp crossing tiles to test the $2^{24}$ `f32` precision boundary.
6. **Non-multiple & degenerate geometries**: `100 × 67` with `T = 32`, 1-tile rasters, 1-row/1-column strips, and all-nodata sparse tiles (`SPARSE_OK` skip on read & write).

*WP-S0 status*: `rust/tests/support/fixtures.rs` generates all six categories for any `T >= 16` (`suite(T)`: 28 fixtures, 25 DEMs and 3 D8-code grids for AreaD8, including `-4`, `-depmask` and `-nc` variants). `large_suite()` adds the 4201 x 4200 ramp whose in-memory mainstem outlet is 17,610,416 against an exact 17,610,612 (f32 rounding above 2^24). `rust/tests/adversarial.rs` checks each fixture's feature, checks that stages routed through `TileStore`/`BorderTable` pass the harness at T = 16/17/32 and 1/8 threads, and checks that tile-local stages fail it in every category.

*WP-S2 status*: `rust/tests/d8_tiled.rs` runs tiled D8FlowDir on the whole suite at T = 16/17/32 and 1/8 threads (directions and sd8 compared bitwise), on the unfilled DEMs (undrained flats, so the raster-wide `last_fall` matters), with a few-kilobyte cache (every tile compressed or spilled), with the wavefront schedule, on 720 random quantized terrains, and on a mixed-component fixture (a patch one subnormal step above a flat, whose slope underflows to 0) that fails when mixed components are routed with tile-local constants; a unit test forces the unpacked level storage. Disabling the cross-tile relaxation makes 90 of 150 suite runs differ.

---

## 7. Risks & Decisions for the User

### 7.1 Technical Risks & Mitigations
1. **Convergence rounds on continental-scale coastal flats in Pass C**: A single flat spanning 20+ tiles could take 20+ synchronous sweeps. *Mitigation*: Schedule active tiles in min-border-distance priority order (tile-level Dijkstra / Gauss-Seidel ordering) so waves sweep multiple tiles per pass; log Pass C round count in timings and verify on ELC & Tickfaw in WP-S2.
2. **`terrio` dual Rayon pool oversubscription**: Until **T-S2** lands, 8 `taudem-rs` threads calling `read_geotiff_samples_into` oversubscribe cores 2×. *Mitigation*: Land **T-S2** in WP-T before WP-S4 benchmarking (interim fallback: semaphore limiting concurrent tile reads).
3. **`f32` non-associativity above $2^{24}$ in `area_d8`**: Linear perimeter superposition diverges from C++ `f32` rounding on rivers $>16.78\text{M}$ cells. *Mitigation*: Exact `u64` perimeter solve + lightweight 1D mainstem `f32` replay when strict `f32` parity is enabled (Decision 1).
4. **Windows working-set bloat / allocator fragmentation**: Per-tile heap allocations inflate Windows `peak_wset`. *Mitigation*: Reuse fixed-size per-worker scratch buffers across tiles, avoid mutable `mmap`, and set default budget to `65%` of `usable_memory()`.
5. **Striped LZW inputs (e.g., `Spring Creek`)**: Large compressed strips exceeding `TiffReadLimits::chunk_bytes` require `&mut self` row reads (`io.rs:98-121`). *Mitigation*: Detect striped layout in `TileSpec`, choose strip-aligned horizontal super-tiles, and use a mutex-guarded reader fallback.

### 7.2 Decisions Needed from the User
1. **`area_d8` $>2^{24}$ accumulation semantics**: Default to **(A, Recommended)** bit-exact `f32` parity with C++ TauDEM & current `area.rs` (automatic 1D mainstem `f32` replay on the `<0.05%` of cells above $2^{24}$), or **(B)** exact `u64`/`f64` accumulation cast to `f32` on write (fixing C++'s $>2^{24}$ freeze, with `--compat-f32` for C++ parity)?
2. **Default flat resolution in streaming**: Confirm **(A, Recommended)** bit-exact TauDEM Garbrecht–Martz as default (preserving the current `EXACT` parity contract) with optional `--fast-flats` (Barnes 2014 1-pass, non-bit-exact), rather than making Barnes 2014 default.
3. **Output `SPARSE_OK=YES` & chunk order**: Confirm enabling `SPARSE_OK=YES` on tiled GeoTIFF outputs by default (smaller/faster on irregular basins like ELC) and allowing arrival-order TIFF chunks (pixel-bit-identical across threads) unless `--deterministic-file` is passed.
4. **New Rust crate dependency**: Approve adding `lz4_flex` (pure Rust, MIT, already in `terrio`'s workspace) to `rust/Cargo.toml` for `TileStore` compression (async writer channels use `std::sync::mpsc::sync_channel` with zero extra dependency).
5. **Default execution mode**: Confirm `--mode auto` (switching to streaming when `est_in_mem > --max-memory`, default `min(0.65 * usable_memory(), 0.75 * avail_ram)`) vs always running the tiled engine (which already beats current single-threaded in-memory wall time ~3× while bounding memory).
