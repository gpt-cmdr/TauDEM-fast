"""Cell-by-cell raster comparison, streamed in row strips so 1e8+ cell rasters fit in memory.

    python rastercompare.py A.tif B.tif [--atol 0] [--rtol 1e-6] [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

STRIP_CELLS = 8_000_000  # cells per strip read from each raster


def _nodata_eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    # TauDEM writes float nodata as -FLT_MAX; compare at float32 precision.
    return np.float32(a) == np.float32(b) or a == b


def _mask(arr: np.ndarray, nodata) -> np.ndarray:
    """True where the cell is nodata."""
    m = np.zeros(arr.shape, dtype=bool)
    if np.issubdtype(arr.dtype, np.floating):
        m |= np.isnan(arr)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        if np.issubdtype(arr.dtype, np.floating):
            m |= arr == arr.dtype.type(nodata)
        else:
            info = np.iinfo(arr.dtype)
            if info.min <= nodata <= info.max:
                m |= arr == nodata
    return m


def compare(path_a: str | Path, path_b: str | Path, atol: float = 0.0, rtol: float = 1e-6,
            band: int = 1) -> dict:
    """Compare raster A (candidate) against B (reference).

    status: EXACT (identical metadata + values), PASS (within tolerance), CRSDIFF (everything
    matches except the CRS definition), FAIL.
    Value tolerance: a valid-in-both cell passes if |a-b| <= atol OR |a-b|/(|b|) <= rtol
    (rtol applies only to float rasters). Exact-match counts are always reported."""
    r: dict = {"a": str(path_a), "b": str(path_b), "problems": [], "crs_problems": [], "warnings": []}
    with rasterio.open(path_a) as A, rasterio.open(path_b) as B:
        r.update(width=A.width, height=A.height, dtype_a=A.dtypes[band - 1], dtype_b=B.dtypes[band - 1],
                 nodata_a=A.nodata, nodata_b=B.nodata,
                 crs_a=A.crs.to_string() if A.crs else None, crs_b=B.crs.to_string() if B.crs else None,
                 transform_a=list(A.transform)[:6], transform_b=list(B.transform)[:6],
                 driver_a=A.driver, driver_b=B.driver)
        if (A.width, A.height) != (B.width, B.height):
            r["problems"].append(f"shape {A.width}x{A.height} vs {B.width}x{B.height}")
            r["status"] = "FAIL"
            return r
        if A.dtypes[band - 1] != B.dtypes[band - 1]:
            r["problems"].append(f"dtype {A.dtypes[band-1]} vs {B.dtypes[band-1]}")
        if not _nodata_eq(A.nodata, B.nodata):
            r["problems"].append(f"nodata {A.nodata} vs {B.nodata}")
        if not np.allclose(list(A.transform)[:6], list(B.transform)[:6], rtol=1e-9, atol=1e-9):
            r["problems"].append("geotransform differs")
        if (A.crs is None) != (B.crs is None):
            r["crs_problems"].append(f"crs presence differs ({r['crs_a']} vs {r['crs_b']})")
        elif A.crs is not None and A.crs != B.crs:
            r["crs_problems"].append(f"crs differs ({r['crs_a'][:60]} vs {r['crs_b'][:60]})")
        elif A.crs is not None and A.crs.to_wkt() != B.crs.to_wkt():
            r["warnings"].append("crs equivalent but WKT text differs")
        r["crs_equal"] = not r["crs_problems"]

        total = A.width * A.height
        rows = max(1, STRIP_CELLS // max(1, A.width))
        valid_both = exact = within = mask_mismatch = nd_a = nd_b = 0
        max_abs = 0.0
        max_rel = 0.0
        max_loc = None
        sum_abs = 0.0
        hist_dirs = None
        for y0 in range(0, A.height, rows):
            h = min(rows, A.height - y0)
            w = Window(0, y0, A.width, h)
            a = A.read(band, window=w)
            b = B.read(band, window=w)
            ma = _mask(a, A.nodata)
            mb = _mask(b, B.nodata)
            nd_a += int(ma.sum()); nd_b += int(mb.sum())
            mask_mismatch += int((ma != mb).sum())
            v = ~ma & ~mb
            nv = int(v.sum())
            if nv == 0:
                continue
            valid_both += nv
            av = a[v].astype(np.float64)
            bv = b[v].astype(np.float64)
            d = np.abs(av - bv)
            exact += int((d == 0).sum())
            ok = d <= atol
            if np.issubdtype(b.dtype, np.floating) and rtol > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(bv != 0, d / np.abs(bv), np.where(d == 0, 0.0, np.inf))
                ok |= rel <= rtol
                mr = float(rel.max())
                max_rel = max(max_rel, mr)
            within += int(ok.sum())
            sum_abs += float(d.sum())
            i = int(d.argmax())
            if d[i] > max_abs:
                max_abs = float(d[i])
                ys, xs = np.nonzero(v)
                max_loc = {"row": int(ys[i]) + y0, "col": int(xs[i]), "a": float(av[i]), "b": float(bv[i])}
        r.update(total_cells=total, nodata_a_cells=nd_a, nodata_b_cells=nd_b, nodata_mask_mismatch=mask_mismatch,
                 valid_both=valid_both, exact_match=exact, differing=valid_both - exact,
                 beyond_tol=valid_both - within, max_abs_diff=max_abs, max_rel_diff=max_rel,
                 mean_abs_diff=(sum_abs / valid_both) if valid_both else 0.0, max_diff_at=max_loc,
                 atol=atol, rtol=rtol)
        if mask_mismatch:
            r["problems"].append(f"nodata mask differs in {mask_mismatch} cells")
        if valid_both - within:
            r["problems"].append(f"{valid_both - within} cells beyond tolerance (max abs {max_abs:g})")
    if r["problems"]:
        r["status"] = "FAIL"
    elif r["crs_problems"]:
        r["status"] = "CRSDIFF"  # values, mask, dtype, nodata, transform all fine; only the CRS differs
    elif r["differing"] == 0 and not r["warnings"]:
        r["status"] = "EXACT"
    else:
        r["status"] = "PASS"
    return r


def summary_line(r: dict) -> str:
    if "total_cells" not in r:
        return f"{r['status']}: " + "; ".join(r["problems"] + r["crs_problems"])
    s = (f"{r['status']}: exact {r['exact_match']:,}/{r['valid_both']:,} valid, max|d| {r['max_abs_diff']:.6g}, "
         f"mask mismatch {r['nodata_mask_mismatch']:,}")
    if r["problems"] or r["crs_problems"]:
        s += " | " + "; ".join(r["problems"] + r["crs_problems"])
    if r["warnings"]:
        s += " | warn: " + "; ".join(r["warnings"])
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = compare(a.a, a.b, a.atol, a.rtol)
    print(json.dumps(r, indent=2, default=str) if a.json else summary_line(r))
    return 0 if r["status"] in ("EXACT", "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
