#!/usr/bin/env python3
"""Inventory the TauDEM-Test-Data repo: raster sizes/dtypes/nodata per case folder,
and which TauDEM tools each folder's test scripts exercise. Emits markdown.

Usage: python inventory.py --data <TauDEM-Test-Data dir> [--out TEST_DATA.md]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import rasterio

RASTER_EXT = {".tif", ".tiff", ".img", ".bil", ".bin", ".sdat", ".vrt", ".asc", ".lg", ".ang", ".slp"}


def tools_by_folder(testall: Path) -> dict[str, list[str]]:
    """Parse testall.bat: track `cd X` and `mpiexec -n N Tool ...` lines."""
    out: dict[str, list[str]] = defaultdict(list)
    cur = "."
    for line in testall.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.lower().startswith("rem") or s.startswith("::"):
            continue
        m = re.match(r"cd\s+(\S+)", s, re.I)
        if m:
            target = m.group(1).replace("\\", "/")
            if target == "..":
                cur = "."
            elif target.startswith("../"):
                cur = target[3:]
            else:
                cur = target
            continue
        m = re.match(r"mpiexec\s+-n?p?\s+(\d+)\s+(\S+)", s, re.I)
        if m:
            tool = m.group(2).lower()
        else:
            m2 = re.match(r"(siregion)\b", s, re.I)
            if not m2:
                continue
            tool = m2.group(1).lower()
        if tool not in out[cur.lower()]:
            out[cur.lower()].append(tool)
    return out


def describe(path: Path) -> dict | None:
    try:
        with rasterio.open(path) as ds:
            crs = ds.crs
            epsg = crs.to_epsg() if crs else None
            return {
                "w": ds.width, "h": ds.height, "cells": ds.width * ds.height, "bands": ds.count,
                "dtype": ds.dtypes[0], "nodata": ds.nodata, "driver": ds.driver,
                "crs": f"EPSG:{epsg}" if epsg else ("custom" if crs else "none"),
                "geographic": bool(crs and crs.is_geographic),
                "compress": (ds.compression.value if ds.compression else "none"),
                "tiled": bool(ds.profile.get("tiled", False)),
            }
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("TAUDEM_TEST_DATA"), required=os.environ.get("TAUDEM_TEST_DATA") is None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.data)
    inp = root / "Input"
    tools = tools_by_folder(inp / "testall.bat")
    lines: list[str] = []
    allr = []
    for case in sorted(p for p in inp.iterdir() if p.is_dir()):
        rasters = []
        for f in sorted(case.rglob("*")):
            if f.is_dir() and f.suffix.lower() == ".gdb":
                continue
            if ".gdb" in f.parts[-2:-1][0].lower() if len(f.parts) > 1 else False:
                continue
            is_esri_grid = f.is_dir() and (f / "hdr.adf").exists()
            if not (is_esri_grid or (f.is_file() and f.suffix.lower() in RASTER_EXT) or (f.is_file() and f.suffix == "" and (f.parent / (f.name + ".hdr")).exists())):
                continue
            d = describe(f)
            if d:
                rel = f.relative_to(case).as_posix()
                rasters.append((rel, d))
                allr.append((case.name + "/" + rel, d))
        t = tools.get(case.name.lower(), [])
        lines.append(f"### {case.name}\n")
        lines.append(f"Tools exercised (testall.bat): {', '.join(t) if t else '(none in testall.bat)'}\n")
        if rasters:
            lines.append("| raster | cols x rows | cells | dtype | nodata | driver | CRS | compress/tiled |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for rel, d in rasters:
                lines.append(f"| {rel} | {d['w']} x {d['h']} | {d['cells']:,} | {d['dtype']} | {d['nodata']} | {d['driver']} | {d['crs']}{' (geo)' if d['geographic'] else ''} | {d['compress']}/{'tiled' if d['tiled'] else 'strip'} |")
        else:
            lines.append("(no rasters readable)")
        lines.append("")
    allr.sort(key=lambda x: -x[1]["cells"])
    top = ["| raster | cols x rows | cells | dtype |", "|---|---|---|---|"]
    seen = set()
    for name, d in allr:
        key = (d["w"], d["h"], Path(name).parts[0])
        if key in seen:
            continue
        seen.add(key)
        top.append(f"| {name} | {d['w']} x {d['h']} | {d['cells']:,} | {d['dtype']} |")
        if len(top) >= 12:
            break
    text = "## Largest rasters (one per case/size)\n\n" + "\n".join(top) + "\n\n## Per-case inventory\n\n" + "\n".join(lines)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
