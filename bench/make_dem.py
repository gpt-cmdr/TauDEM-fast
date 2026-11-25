#!/usr/bin/env python3
"""Synthetic large-DEM generator for TauDEM benchmarking.

Terrain is a sum of coordinate-hashed value-noise octaves (fractional Brownian motion
with Hurst exponent H) plus ridged-noise valleys and a regional tilt. Because every
cell is a pure function of its (row, col), the raster is generated in row strips and
streamed to disk: memory stays bounded at any size, including 1e9 cells.

Hydrologically awkward features, on purpose:
  * real pits         - fine-scale noise leaves many small depressions; additional
                        explicit bowls (radius 4..250 cells) are carved in;
  * broad flats       - lake surfaces: inside a low-frequency basin mask the terrain is
                        raised to a fixed level, giving large exactly-flat areas; terrace
                        zones quantize elevation to a step (integer-meter-DEM style flats);
  * nodata holes      - scattered blobs of nodata inside the domain;
  * irregular edge    - the valid domain is a noisy ellipse, so 20-30% of the rectangle is
                        nodata with a ragged boundary (like a clipped watershed DEM).

Output: tiled (512x512), DEFLATE-compressed float32 GeoTIFF, BigTIFF when needed, with a
projected CRS (default EPSG:26915, NAD83 / UTM 15N), plus a <out>.json sidecar with the
parameters and cell statistics.

  python make_dem.py --size 1e7 --out G:/data/synth_1e7.tif
  python make_dem.py --cols 40000 --rows 25000 --out synth_1e9.tif   # 1e9 cells
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

# Non-square presets (aspect 1.6) so row/column mix-ups show up.
PRESETS = {"1e6": (1250, 800), "1e7": (4000, 2500), "1e8": (12500, 8000), "1e9": (40000, 25000)}

M1 = np.uint64(0x9E3779B97F4A7C15)
M2 = np.uint64(0xC2B2AE3D27D4EB4F)
M3 = np.uint64(0x165667B19E3779F9)
S1 = np.uint64(0xBF58476D1CE4E5B9)
S2 = np.uint64(0x94D049BB133111EB)


def _hash01(ix: np.ndarray, iy: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic uniform [0,1) per integer lattice point (splitmix64 finalizer)."""
    with np.errstate(over="ignore"):
        h = ix.astype(np.uint64) * M1 ^ iy.astype(np.uint64) * M2 ^ np.uint64(salt & 0xFFFFFFFFFFFFFFFF) * M3
        h ^= h >> np.uint64(30)
        h *= S1
        h ^= h >> np.uint64(27)
        h *= S2
        h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def value_noise(xs: np.ndarray, ys: np.ndarray, wavelength: float, salt: int) -> np.ndarray:
    """Smooth value noise in [0,1] on the grid ys x xs (1-D pixel-centre coordinates)."""
    off = _hash01(np.array([salt]), np.array([7]), 99)[0] * 1e4  # decorrelate octaves
    fx = xs / wavelength + off
    fy = ys / wavelength + off * 0.7
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = fx - x0
    ty = fy - y0
    tx = tx * tx * tx * (tx * (tx * 6 - 15) + 10)  # quintic smoothstep
    ty = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
    lx = np.arange(x0.min(), x0.max() + 2)
    ly = np.arange(y0.min(), y0.max() + 2)
    lat = _hash01(lx[None, :], ly[:, None], salt).astype(np.float32)
    ix = (x0 - lx[0])[None, :]
    iy = (y0 - ly[0])[:, None]
    a = lat[iy, ix]
    b = lat[iy, ix + 1]
    c = lat[iy + 1, ix]
    d = lat[iy + 1, ix + 1]
    tx = tx[None, :].astype(np.float32)
    ty = ty[:, None].astype(np.float32)
    top = a + (b - a) * tx
    bot = c + (d - c) * tx
    return top + (bot - top) * ty


class Terrain:
    def __init__(self, cols: int, rows: int, cell: float, seed: int, relief: float, hurst: float,
                 min_wavelength: float, flat_frac: float, terrace_step: float, pit_density: float,
                 hole_frac: float, edge: bool, lake_wavelength: float = 256.0, terrace_wavelength: float = 512.0):
        self.cols, self.rows, self.cell, self.seed = cols, rows, cell, seed
        self.relief, self.hurst = relief, hurst
        self.L0 = max(cols, rows) / 2.0
        self.octaves = []
        L = self.L0
        while L >= min_wavelength:
            self.octaves.append(L)
            L /= 2.0
        self.flat_frac, self.terrace_step, self.edge = flat_frac, terrace_step, edge
        self.hole_frac = hole_frac
        # Feature scales for flats, fixed in cells by default so flat sizes (and TauDEM's
        # superlinear flat-resolution cost) do not grow with the raster: the count of flats
        # scales with area instead. <= 0 means "scale with the domain" (stress mode).
        self.lake_L = lake_wavelength if lake_wavelength > 0 else self.L0 / 2
        self.terrace_L = terrace_wavelength if terrace_wavelength > 0 else self.L0 / 3
        rng = np.random.default_rng(seed)
        n_bowls = int(pit_density * cols * rows / 1e6)
        r = np.exp(rng.uniform(math.log(4), math.log(250), n_bowls))
        self.bowls = np.column_stack([
            rng.uniform(0, cols, n_bowls), rng.uniform(0, rows, n_bowls), r,
            relief * rng.uniform(0.01, 0.08, n_bowls) * np.sqrt(r / 250)])  # cx, cy, radius(px), depth(m)
        self.bowls = self.bowls[np.argsort(self.bowls[:, 1])]
        self.lake_level = None
        self.lake_level = float(np.quantile(self._sample_for_levels(), flat_frac)) if flat_frac > 0 else None

    def _sample_for_levels(self) -> np.ndarray:
        step = max(1, int(math.sqrt(self.cols * self.rows / 250_000)))
        xs = np.arange(0, self.cols, step) + 0.5
        ys = np.arange(0, self.rows, step) + 0.5
        z = self._base(xs, ys)
        basin = value_noise(xs, ys, self.lake_L, self.seed + 501) > 0.5
        return z[basin] if basin.any() else z.ravel()

    def _base(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        z = np.zeros((ys.size, xs.size), np.float32)
        for k, L in enumerate(self.octaves):
            amp = self.relief * (L / self.L0) ** self.hurst
            z += np.float32(amp) * (value_noise(xs, ys, L, self.seed * 1000 + k) - np.float32(0.5))
        # Ridged valleys at mid wavelengths: carve narrow channels where noise ~ 0.5.
        for k, L in enumerate((self.L0 / 8, self.L0 / 32)):
            if L < 4:
                continue
            n = value_noise(xs, ys, L, self.seed * 1000 + 700 + k)
            v = np.float32(1) - np.abs(np.float32(2) * n - np.float32(1))
            z -= np.float32(0.25 * self.relief * (L / self.L0) ** self.hurst) * v ** 8
        # Regional tilt toward the lower-right corner so the domain drains somewhere.
        tilt = np.float32(self.relief * 0.6)
        z += tilt * (np.float32(1) - (xs[None, :] / self.cols + ys[:, None] / self.rows).astype(np.float32) / 2)
        return z

    def strip(self, y0: int, h: int, nodata: float) -> np.ndarray:
        xs = np.arange(self.cols) + 0.5
        ys = np.arange(y0, y0 + h) + 0.5
        z = self._base(xs, ys)
        # Explicit bowls (pits).
        lo = np.searchsorted(self.bowls[:, 1], y0 - 260)
        hi = np.searchsorted(self.bowls[:, 1], y0 + h + 260)
        for cx, cy, r, depth in self.bowls[lo:hi]:
            ya, yb = max(y0, int(cy - r)), min(y0 + h, int(cy + r) + 1)
            xa, xb = max(0, int(cx - r)), min(self.cols, int(cx + r) + 1)
            if ya >= yb or xa >= xb:
                continue
            yy = (np.arange(ya, yb) + 0.5 - cy)[:, None]
            xx = (np.arange(xa, xb) + 0.5 - cx)[None, :]
            q = 1 - (xx * xx + yy * yy) / (r * r)
            z[ya - y0:yb - y0, xa:xb] -= np.float32(depth) * np.clip(q, 0, None).astype(np.float32)
        # Broad flats: lake surfaces inside a low-frequency basin mask (noise > 0.5). Outside
        # the mask the fill surface drops away steeply, so lake shores join the terrain
        # continuously instead of as cliffs.
        if self.lake_level is not None:
            b = value_noise(xs, ys, self.lake_L, self.seed + 501)
            fill = np.float32(self.lake_level) + np.float32(1.5 * self.relief) * np.minimum(b - np.float32(0.5), 0)
            np.maximum(z, fill, out=z)
        # Terraced zones (integer-meter style flats).
        if self.terrace_step > 0:
            zone = value_noise(xs, ys, self.terrace_L, self.seed + 601) > 0.62
            zq = np.floor(z / np.float32(self.terrace_step)) * np.float32(self.terrace_step)
            np.copyto(z, zq, where=zone)
        # Nodata: irregular outer boundary + interior holes.
        nd = np.zeros(z.shape, bool)
        if self.edge:
            ex = ((xs - self.cols / 2) / (self.cols / 2))[None, :]
            ey = ((ys - self.rows / 2) / (self.rows / 2))[:, None]
            rr = np.sqrt(ex * ex + ey * ey).astype(np.float32)
            wob = value_noise(xs, ys, self.L0 / 4, self.seed + 801) + 0.5 * value_noise(xs, ys, self.L0 / 16, self.seed + 802)
            nd |= rr + np.float32(0.45) * (wob - np.float32(0.75)) > np.float32(0.98)
        if self.hole_frac > 0:
            hn = value_noise(xs, ys, max(8.0, min(64.0, self.L0 / 50)), self.seed + 901)
            hz = value_noise(xs, ys, self.L0 / 6, self.seed + 902)  # holes cluster in some regions
            nd |= (hn * hz) > np.float32(1 - 2.2 * self.hole_frac)
        z[nd] = np.float32(nodata)
        return z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=sorted(PRESETS), help="cell-count preset (cols x rows: " +
                    ", ".join(f"{k}={v[0]}x{v[1]}" for k, v in PRESETS.items()) + ")")
    ap.add_argument("--cols", type=int)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=float, default=10.0, help="cell size in CRS units (m)")
    ap.add_argument("--epsg", type=int, default=26915)
    ap.add_argument("--origin", type=float, nargs=2, default=(300000.0, 4600000.0), metavar=("X", "Y"),
                    help="upper-left corner")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relief", type=float, default=None, help="m; default scales with extent")
    ap.add_argument("--hurst", type=float, default=0.8)
    ap.add_argument("--min-wavelength", type=float, default=2.0, help="finest noise octave (cells)")
    ap.add_argument("--flat-frac", type=float, default=0.15, help="quantile of basin terrain flooded to a flat lake level (0 = off)")
    ap.add_argument("--lake-wavelength", type=float, default=256.0,
                    help="cells; size scale of lake flats (largest lakes ~ this wide). 0 = scale with the raster, "
                         "which makes single flats of 1e5-1e6+ cells: a deliberate stress case for flat resolution")
    ap.add_argument("--terrace-wavelength", type=float, default=512.0, help="cells; size scale of terraced zones (0 = scale with raster)")
    ap.add_argument("--terrace-step", type=float, default=2.0, help="m; 0 disables terraced zones")
    ap.add_argument("--pit-density", type=float, default=30.0, help="explicit bowls per million cells")
    ap.add_argument("--hole-frac", type=float, default=0.01, help="approx fraction of interior nodata holes")
    ap.add_argument("--no-edge", action="store_true", help="disable irregular nodata boundary")
    ap.add_argument("--nodata", type=float, default=-9999.0)
    ap.add_argument("--compress", default="DEFLATE", help="DEFLATE (default, readable by old GDAL), LZW, ZSTD, NONE")
    ap.add_argument("--predictor", type=int, default=3, help="TIFF predictor (3 = floating point, 1 = none)")
    ap.add_argument("--blocksize", type=int, default=512)
    ap.add_argument("--strip-rows", type=int, default=1024, help="rows generated per strip (memory knob)")
    a = ap.parse_args()
    if a.size:
        cols, rows = PRESETS[a.size]
    elif a.cols and a.rows:
        cols, rows = a.cols, a.rows
    else:
        ap.error("give --size or --cols and --rows")
    relief = a.relief if a.relief is not None else min(2500.0, 150.0 + 0.004 * max(cols, rows) * a.cell)
    t0 = time.perf_counter()
    T = Terrain(cols, rows, a.cell, a.seed, relief, a.hurst, a.min_wavelength, a.flat_frac,
                a.terrace_step, a.pit_density, a.hole_frac, not a.no_edge, a.lake_wavelength, a.terrace_wavelength)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    est = cols * rows * 4
    profile = dict(driver="GTiff", width=cols, height=rows, count=1, dtype="float32", nodata=a.nodata,
                   crs=f"EPSG:{a.epsg}", transform=from_origin(a.origin[0], a.origin[1], a.cell, a.cell),
                   tiled=True, blockxsize=a.blocksize, blockysize=a.blocksize,
                   BIGTIFF="YES" if est > 3.5e9 else "IF_SAFER")
    if a.compress.upper() != "NONE":
        profile.update(compress=a.compress.upper(), predictor=a.predictor)
    strip_rows = max(a.blocksize, (a.strip_rows // a.blocksize) * a.blocksize)
    n_nd = 0
    zmin, zmax = math.inf, -math.inf
    n_flat_lake = 0
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"), rasterio.open(out, "w", **profile) as ds:
        for y0 in range(0, rows, strip_rows):
            h = min(strip_rows, rows - y0)
            z = T.strip(y0, h, a.nodata)
            valid = z != np.float32(a.nodata)
            n_nd += int((~valid).sum())
            if valid.any():
                zmin = min(zmin, float(z[valid].min())); zmax = max(zmax, float(z[valid].max()))
            if T.lake_level is not None:
                n_flat_lake += int((z == np.float32(T.lake_level)).sum())
            ds.write(z, 1, window=Window(0, y0, cols, h))
            pct = 100 * (y0 + h) / rows
            print(f"\r{out.name}: {pct:5.1f}%  ({time.perf_counter() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    meta = {"path": str(out), "cols": cols, "rows": rows, "cells": cols * rows, "cell_size": a.cell,
            "epsg": a.epsg, "seed": a.seed, "relief_m": relief, "hurst": a.hurst, "octaves": len(T.octaves),
            "flat_frac": a.flat_frac, "lake_wavelength": T.lake_L, "terrace_wavelength": T.terrace_L, "lake_level": T.lake_level, "lake_flat_cells": n_flat_lake,
            "terrace_step": a.terrace_step, "bowls": int(len(T.bowls)), "hole_frac": a.hole_frac,
            "edge": not a.no_edge, "nodata": a.nodata, "nodata_cells": n_nd, "nodata_pct": round(100 * n_nd / (cols * rows), 2),
            "zmin": zmin, "zmax": zmax, "compress": a.compress, "predictor": a.predictor,
            "file_bytes": out.stat().st_size, "gen_seconds": round(time.perf_counter() - t0, 1)}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
