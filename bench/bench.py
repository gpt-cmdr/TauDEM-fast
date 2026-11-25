#!/usr/bin/env python3
"""Benchmark pitremove -> d8flowdir -> aread8 for C++ TauDEM and taudem-rs.

Run:
  python bench.py run --dem synth_1e7.tif --impl cpp:"C:/Program Files/TauDEM/TauDEM5Exe" \
      --impl rs:path/to/taudem-rs --ranks 1,4 --repeat 3
Summarize one or more result files into markdown:
  python bench.py summarize <results>/bench-*.jsonl [--out summary.md]

Modes (--modes, default "separate,fused"):
  separate  three tool invocations, each timed (all implementations)
  fused     one 'taudem-rs pipeline' invocation (rs only; argv template in tools.json)

Per invocation we record wall time, peak memory over the process tree (Windows: peak
working set of every worker process via psutil; Linux: VmHWM per process), TauDEM's own
printed phase timings ("Read time", "Compute time", ...), and output file sizes. Rows are
appended to <results>/bench-<stamp>.jsonl and .csv as they finish, so an interrupted run
keeps what it measured. The summary reports the median over repeats.

Cache state: by default one untimed warm-up pass of each (impl, ranks) combination runs
first so every timed repeat sees the DEM in the OS file cache ("warm"). For cold-cache
numbers pass --cold-cmd, a shell command run before every timed repeat that evicts the
file cache (Linux: "sync; echo 3 | sudo tee /proc/sys/vm/drop_caches"; Windows: e.g.
"RAMMap64.exe -Et" from Sysinternals, elevated). Rows record cache = warm|cold|first.

Ranks: C++ runs under mpiexec -n R. For rs, R becomes a thread count via tools.json
(threads_flag / threads_env).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from taudem_bench_common import Impl, fmt_bytes, load_json, machine_info, run_measured, work_root

STEPS = [
    ("pitremove", ["-z", "{dem}", "-fel", "{fel}"], ["fel"]),
    ("d8flowdir", ["-fel", "{fel}", "-p", "{p}", "-sd8", "{sd8}"], ["p", "sd8"]),
    ("aread8", ["-p", "{p}", "-ad8", "{ad8}"], ["ad8"]),
]
OUTS = ("fel", "p", "sd8", "ad8")


def _fmt(argv: list[str], subst: dict) -> list[str]:
    return [a.format(**subst) for a in argv]


def _file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    n = path.stat().st_size
    for side in path.parent.glob(path.name + ".*"):
        n += side.stat().st_size
    return n


def run_cmd_shell(cmd: str) -> None:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"warning: cold-cmd exited {r.returncode}: {r.stderr.strip()[:300]}", file=sys.stderr)


class Recorder:
    FIELDS = ["stamp", "label", "dem", "dem_cells", "impl", "impl_kind", "mode", "step", "ranks", "repeat", "cache",
              "rc", "wall_s", "peak_sum_bytes", "peak_max_proc_bytes", "peak_concurrent_bytes", "nprocs",
              "out_bytes", "t_header_read", "t_read", "t_compute", "t_write", "t_total", "timings_json", "cmd"]

    def __init__(self, res_dir: Path, stamp: str):
        res_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = res_dir / f"bench-{stamp}.jsonl"
        self.csv = res_dir / f"bench-{stamp}.csv"
        with self.csv.open("w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)

    def add(self, row: dict) -> None:
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        with self.csv.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") if k != "timings_json" else json.dumps(row.get("timings", {}))
                                    for k in self.FIELDS])


def _timing_fields(t: dict) -> dict:
    """Normalize TauDEM's per-tool phase names into header/read/compute/write/total."""
    out = {"t_header_read": t.get("header_read"), "t_total": t.get("total")}
    out["t_read"] = sum(v for k, v in t.items() if k in ("data_read", "read")) or None
    out["t_compute"] = sum(v for k, v in t.items() if k.startswith("compute") or k.startswith("resolve")) or None
    out["t_write"] = sum(v for k, v in t.items() if k.startswith("write")) or None
    return out


def cmd_run(a) -> int:
    tools_cfg = load_json(a.tools_config)
    impls = [Impl.parse(s, tools_cfg, a.mpiexec) for s in a.impl]
    for im in impls:
        if not im.available():
            sys.exit(f"{im.label}: not found at {im.path}")
    ranks_list = [int(r) for r in str(a.ranks).split(",")]
    modes = [m.strip() for m in a.modes.split(",")]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    root = work_root(a.work)
    res_dir = Path(a.results) if a.results else root / "results"
    rec = Recorder(res_dir, stamp)
    meta = {"stamp": stamp, "label": a.label, "machine": machine_info(), "impls": [i.describe() for i in impls],
            "ranks": ranks_list, "repeat": a.repeat, "modes": modes, "cold_cmd": a.cold_cmd,
            "warmup": not a.no_warmup, "dems": a.dem}
    (res_dir / f"bench-{stamp}.meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"results -> {rec.jsonl}", flush=True)
    rs_pipe = tools_cfg.get("rs", {}).get("pipeline")
    for dem in a.dem:
        dem_p = Path(dem).resolve()
        try:
            import rasterio
            with rasterio.open(dem_p) as ds:
                cells = ds.width * ds.height
        except Exception:
            cells = None
        for im in impls:
            for ranks in ranks_list:
                for mode in modes:
                    if mode == "fused" and (im.kind != "rs" or not rs_pipe):
                        continue
                    run_dir = root / "bench" / dem_p.stem / im.label / f"r{ranks}" / mode
                    reps = ([0] if not a.no_warmup and not a.cold_cmd else []) + list(range(1, a.repeat + 1))
                    for rep in reps:
                        if run_dir.exists():
                            shutil.rmtree(run_dir)
                        run_dir.mkdir(parents=True)
                        if rep > 0 and a.cold_cmd:
                            run_cmd_shell(a.cold_cmd)
                        cache = "cold" if a.cold_cmd else ("warm" if not a.no_warmup or rep > 1 else "first")
                        subst = {"dem": str(dem_p), **{k: str(run_dir / f"{dem_p.stem}{k}.tif") for k in OUTS}}
                        if mode == "separate":
                            jobs = [(tool, _fmt(argv, subst), outs) for tool, argv, outs in STEPS]
                        else:
                            jobs = [("pipeline", _fmt(rs_pipe["argv"], subst), list(OUTS))]
                        total_wall = 0.0
                        ok = True
                        for tool, argv, outs in jobs:
                            if tool == "pipeline":
                                cmd = [str(im.path), rs_pipe.get("subcommand", "pipeline"), *argv]
                                env = dict(os.environ)
                                rcfg = tools_cfg.get("rs", {})
                                if rcfg.get("threads_flag"):
                                    cmd += [rcfg["threads_flag"], str(ranks)]
                                if rcfg.get("threads_env"):
                                    env[rcfg["threads_env"]] = str(ranks)
                            else:
                                if not im.supports(tool):
                                    print(f"  {im.label} lacks {tool}; skipping mode {mode}", file=sys.stderr)
                                    ok = False
                                    break
                                cmd, env = im.command(tool, argv, ranks)
                            res = run_measured(cmd, run_dir, env, timeout=a.timeout)
                            total_wall += res.wall_s
                            row = {"stamp": stamp, "label": a.label, "dem": dem_p.name, "dem_cells": cells,
                                   "impl": im.label, "impl_kind": im.kind, "mode": mode, "step": tool,
                                   "ranks": ranks, "repeat": rep, "cache": cache if rep > 0 else "warmup",
                                   "rc": res.returncode, "wall_s": round(res.wall_s, 4),
                                   "peak_sum_bytes": res.peak_sum_bytes, "peak_max_proc_bytes": res.peak_max_proc_bytes,
                                   "peak_concurrent_bytes": res.peak_concurrent_bytes, "nprocs": res.nprocs_seen,
                                   "out_bytes": sum(_file_bytes(Path(subst[o])) for o in outs),
                                   "timings": res.timings, **_timing_fields(res.timings), "cmd": " ".join(res.cmd)}
                            if res.returncode != 0:
                                row["error"] = (res.stderr or res.stdout)[-1500:]
                                ok = False
                            rec.add(row)
                            tag = "warmup" if rep == 0 else f"rep {rep}"
                            print(f"{dem_p.name} {im.label} r{ranks} {mode} {tool} {tag}: {res.wall_s:.2f}s "
                                  f"peak {fmt_bytes(res.peak_sum_bytes)} rc={res.returncode}", flush=True)
                            if res.returncode != 0:
                                print("   " + row["error"].strip().replace("\n", "\n   ")[-800:], file=sys.stderr)
                                break
                        if ok and len(jobs) > 1 and rep > 0:
                            rec.add({"stamp": stamp, "label": a.label, "dem": dem_p.name, "dem_cells": cells,
                                     "impl": im.label, "impl_kind": im.kind, "mode": mode, "step": "TOTAL",
                                     "ranks": ranks, "repeat": rep, "cache": cache, "rc": 0,
                                     "wall_s": round(total_wall, 4),
                                     "out_bytes": sum(_file_bytes(Path(subst[o])) for o in OUTS)})
                        if not ok:
                            break
    if a.cross_check and len(impls) > 1:
        cross_check(root, a, impls, ranks_list, modes, rec)
    md = summarize([rec.jsonl])
    mdp = res_dir / f"bench-{stamp}.md"
    mdp.write_text(md, encoding="utf-8")
    print(f"\n{md}\nsummary: {mdp}")
    return 0


def cross_check(root: Path, a, impls: list[Impl], ranks_list, modes, rec: Recorder) -> None:
    """Compare every impl's last outputs against the first impl's (same DEM)."""
    from rastercompare import compare, summary_line
    base = impls[0]
    out_path = rec.jsonl.with_suffix(".crosscheck.json")
    found = []
    for dem in a.dem:
        stem = Path(dem).stem
        bdir = root / "bench" / stem / base.label / f"r{ranks_list[0]}" / "separate"
        for im in impls[1:]:
            for ranks in ranks_list:
                for mode in modes:
                    d = root / "bench" / stem / im.label / f"r{ranks}" / mode
                    if not d.exists():
                        continue
                    for o in OUTS:
                        fa, fb = d / f"{stem}{o}.tif", bdir / f"{stem}{o}.tif"
                        if fa.exists() and fb.exists():
                            r = compare(fa, fb, 0.0, a.rtol)
                            found.append({"dem": stem, "impl": im.label, "ranks": ranks, "mode": mode, "output": o,
                                          "vs": base.label, "status": r["status"], "detail": summary_line(r)})
                            print(f"cross-check {stem} {im.label} r{ranks} {mode} {o} vs {base.label}: {summary_line(r)}")
    out_path.write_text(json.dumps(found, indent=2), encoding="utf-8")


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(paths: list[Path]) -> str:
    rows = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    timed = [r for r in rows if r.get("repeat", 0) > 0 and r.get("rc", 0) == 0]
    groups: dict = {}
    for r in timed:
        k = (r["dem"], r.get("dem_cells"), r["impl"], r["mode"], r["ranks"], r["step"], r.get("cache"))
        groups.setdefault(k, []).append(r)
    order = {"pitremove": 0, "d8flowdir": 1, "aread8": 2, "TOTAL": 3, "pipeline": 4}
    keys = sorted(groups, key=lambda k: (k[0], k[2], k[3], k[4], order.get(k[5], 9)))
    out = ["# TauDEM benchmark summary", "",
           f"Sources: {', '.join(Path(p).name for p in paths)}. Median over n timed repeats (warm-up excluded).",
           "'tool total s' is the tool's self-reported Total time; wall - tool total = process/MPI launch overhead.",
           "Peak mem = sum over worker processes of each one's peak working set / VmHWM (upper bound on the aggregate).", "",
           "| DEM | cells | impl | mode | ranks | step | cache | n | wall s (median) | min | max | peak mem | tool total s | read s | compute s | write s | output |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        g = groups[k]
        w = [r["wall_s"] for r in g]
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {} | {} | {} | {} | {} | {} |".format(
            k[0], f"{k[1]:,}" if k[1] else "-", k[2], k[3], k[4], k[5], k[6], len(g), statistics.median(w), min(w), max(w),
            fmt_bytes(_median([r.get("peak_sum_bytes") for r in g])), _f(_median([r.get("t_total") for r in g])),
            _f(_median([r.get("t_read") for r in g])), _f(_median([r.get("t_compute") for r in g])),
            _f(_median([r.get("t_write") for r in g])), fmt_bytes(_median([r.get("out_bytes") for r in g]))))
    failed = [r for r in rows if r.get("rc", 0) != 0]
    if failed:
        out += ["", "## Failed invocations", ""]
        for r in failed:
            err_lines = (r.get("error") or "").strip().splitlines()
            last = err_lines[-1][:200] if err_lines else ""
            out.append(f"- {r['dem']} {r['impl']} r{r['ranks']} {r['mode']} {r['step']} rep {r['repeat']}: rc={r['rc']} {last}")
    return "\n".join(out) + "\n"


def _f(x):
    return "-" if x is None else f"{x:.3f}"


def cmd_summarize(a) -> int:
    md = summarize([Path(p) for p in a.files])
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dem", action="append", required=True, help="input DEM (repeatable)")
    r.add_argument("--impl", action="append", required=True, help="cpp[:dir] | rs[:binary], optional =label")
    r.add_argument("--ranks", default="1,4", help="comma list of MPI ranks (cpp) / threads (rs)")
    r.add_argument("--repeat", type=int, default=3)
    r.add_argument("--modes", default="separate,fused")
    r.add_argument("--cold-cmd", help="shell command run before each timed repeat to drop the file cache")
    r.add_argument("--no-warmup", action="store_true", help="skip the untimed warm-up pass")
    r.add_argument("--cross-check", action="store_true", help="compare outputs of every impl against the first impl")
    r.add_argument("--rtol", type=float, default=1e-6)
    r.add_argument("--label", default="", help="free-text tag stored with every row (e.g. machine name)")
    r.add_argument("--timeout", type=float, default=6 * 3600)
    r.add_argument("--work", help="scratch root (or TAUDEM_BENCH_WORK)")
    r.add_argument("--results", help="results dir (default <work>/results)")
    r.add_argument("--mpiexec")
    r.add_argument("--tools-config", default="tools.json")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("summarize")
    s.add_argument("files", nargs="+")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_summarize)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
