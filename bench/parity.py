#!/usr/bin/env python3
"""Parity harness: run a TauDEM implementation over TauDEM-Test-Data cases and compare
its outputs with ReferenceResult (or compare two arbitrary rasters/directories).

Run cases:
  python parity.py run --impl cpp:"C:/Program Files/TauDEM/TauDEM5Exe" --data <TauDEM-Test-Data>
  python parity.py run --impl rs:<path/to/taudem-rs> --cases Base,fts --skip-tags ogr
  python parity.py run --impl cpp --impl rs --cases Base      # several implementations

Compare two directories (e.g. C++ vs Rust outputs of a benchmark run) or two files:
  python parity.py compare <dirA|fileA> <dirB|fileB> [--glob "*.tif"]

Implementation spec: cpp[:<exe dir>] | rs[:<binary>], optional '=label' suffix.
Defaults come from TAUDEM_CPP_DIR / TAUDEM_RS_BIN; data from TAUDEM_TEST_DATA;
scratch from TAUDEM_BENCH_WORK (default <tmp>/taudem-bench). Case definitions are
cases.json; the tool -> executable mapping is tools.json (both next to this file).

Statuses: EXACT, PASS (within atol/rtol), CRSDIFF (only the CRS definition differs), FAIL,
MISSING, ERROR, NOREF, SKIP. Exit status: 0 when every output is EXACT/PASS/SKIP
(plus CRSDIFF with --allow-crsdiff), 1 otherwise. Each produced output is kept under
<work>/parity/<impl>/<case>/_produced/<step>/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from taudem_bench_common import (Impl, copy_case, copy_with_sidecars, load_json, machine_info, run_measured,
                                 test_data_root, work_root)
from rastercompare import compare, summary_line


def _out_spec(o) -> tuple[str, float | None, float | None]:
    if isinstance(o, str):
        return o, None, None
    return o["name"], o.get("atol"), o.get("rtol")


def run_impl(impl: Impl, cases: dict, sel: list[str], data: Path, work: Path, args) -> list[dict]:
    rows: list[dict] = []
    for cname in sel:
        case = cases[cname]
        folder = case["folder"]
        src = data / "Input" / folder
        ref = data / "ReferenceResult" / folder
        wdir = work / impl.label / cname
        copy_case(src, wdir)
        for step in case["steps"]:
            tags = set(step.get("tags", []))
            base = {"impl": impl.label, "case": cname, "step": step["id"], "tool": step["tool"]}
            if args.tags and not tags & set(args.tags):
                continue
            if args.skip_tags and tags & set(args.skip_tags):
                rows.append({**base, "output": "-", "status": "SKIP", "detail": f"tag filter {sorted(tags & set(args.skip_tags))}"})
                continue
            if args.tools and step["tool"] not in args.tools:
                continue
            if not impl.supports(step["tool"]):
                rows.append({**base, "output": "-", "status": "SKIP", "detail": f"{impl.label} has no {step['tool']} (tools.json)"})
                continue
            if args.inputs == "reference":
                for f in step.get("ref_inputs", []):
                    if not copy_with_sidecars(ref, f, wdir):
                        rows.append({**base, "output": f, "status": "ERROR", "detail": "reference input missing"})
            ranks = args.np or step.get("np", 1)
            cmd, env = impl.command(step["tool"], step["argv"], ranks)
            res = run_measured(cmd, wdir, env, timeout=args.timeout)
            base.update(ranks=ranks, wall_s=round(res.wall_s, 3), rc=res.returncode, cmd=" ".join(res.cmd))
            if args.verbose:
                print(f"$ {' '.join(res.cmd)}\n{res.stdout[-1500:]}{res.stderr[-1500:]}", file=sys.stderr)
            for o in step["outputs"]:
                name, atol, rtol = _out_spec(o)
                cand = wdir / name
                refp = ref / name
                row = {**base, "output": name}
                if res.returncode != 0 and not cand.exists():
                    row.update(status="ERROR", detail=f"exit {res.returncode}: {(res.stderr or res.stdout).strip()[-300:]}")
                elif not cand.exists():
                    row.update(status="MISSING", detail="output not produced")
                elif not refp.exists():
                    row.update(status="NOREF", detail=f"no {refp}")
                else:
                    try:
                        c = compare(cand, refp, atol if atol is not None else args.atol,
                                    rtol if rtol is not None else args.rtol)
                    except Exception as e:  # unreadable output etc.
                        row.update(status="ERROR", detail=f"compare failed: {e}")
                    else:
                        row.update(status=c["status"], detail=summary_line(c), metrics={
                            k: c.get(k) for k in ("width", "height", "dtype_a", "dtype_b", "nodata_a", "nodata_b",
                                                  "crs_a", "crs_b", "crs_equal", "transform_a", "transform_b",
                                                  "total_cells", "valid_both", "exact_match", "differing",
                                                  "beyond_tol", "max_abs_diff", "max_rel_diff", "mean_abs_diff",
                                                  "nodata_mask_mismatch", "max_diff_at", "problems",
                                                  "crs_problems", "warnings")})
                    # Keep what this implementation produced: a later step may overwrite it
                    # with the reference copy (--inputs reference).
                    copy_with_sidecars(wdir, name, wdir / "_produced" / step["id"])
                if res.returncode != 0 and row["status"] in ("EXACT", "PASS"):
                    row["detail"] += f" (but tool exited {res.returncode})"
                    row["status"] = "FAIL"
                rows.append(row)
                print(f"[{impl.label}] {cname}/{step['id']} -> {name}: {row['status']}  {row.get('detail','')}", flush=True)
    return rows


def to_markdown(rows: list[dict], meta: dict) -> str:
    out = [f"# TauDEM parity report", "", f"- generated: {meta['generated']}",
           f"- data: `{meta['data']}`", f"- inputs mode: {meta['inputs']}",
           f"- tolerance: atol={meta['atol']} rtol={meta['rtol']}", ""]
    for im in meta["impls"]:
        out.append(f"- impl `{im['label']}`: {im['kind']} `{im['path']}`" + (f" (version {im['version']})" if im.get("version") else ""))
    out.append("")
    counts: dict = {}
    for r in rows:
        counts.setdefault(r["impl"], {}).setdefault(r["status"], 0)
        counts[r["impl"]][r["status"]] += 1
    for im, c in counts.items():
        out.append(f"**{im}**: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    out += ["", "| impl | case | step | output | status | ranks | wall s | detail |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        det = str(r.get("detail", "")).replace("|", "/").replace("\n", " ")
        out.append(f"| {r['impl']} | {r['case']} | {r['step']} | {r['output']} | {r['status']} | {r.get('ranks','')} | {r.get('wall_s','')} | {det} |")
    return "\n".join(out) + "\n"


def cmd_run(args) -> int:
    tools_cfg = load_json(args.tools_config)
    cases = load_json(args.cases_config)["cases"]
    data = test_data_root(args.data)
    work = work_root(args.work) / "parity"
    sel = list(cases) if not args.cases else [c.strip() for c in args.cases.split(",")]
    for c in sel:
        if c not in cases:
            sys.exit(f"unknown case {c!r}; known: {', '.join(cases)}")
    impls = [Impl.parse(s, tools_cfg, args.mpiexec) for s in (args.impl or ["cpp"])]
    rows: list[dict] = []
    for im in impls:
        if not im.available():
            print(f"[{im.label}] not found at {im.path}; skipping", file=sys.stderr)
            rows.append({"impl": im.label, "case": "-", "step": "-", "tool": "-", "output": "-", "status": "UNAVAILABLE", "detail": str(im.path)})
            continue
        rows += run_impl(im, cases, sel, data, work, args)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "data": str(data), "inputs": args.inputs,
            "atol": args.atol, "rtol": args.rtol, "impls": [i.describe() for i in impls], "machine": machine_info()}
    res_dir = Path(args.results) if args.results else work_root(args.work) / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    jpath = res_dir / f"parity-{stamp}.json"
    mpath = res_dir / f"parity-{stamp}.md"
    jpath.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, default=str), encoding="utf-8")
    mpath.write_text(to_markdown(rows, meta), encoding="utf-8")
    print(f"\nresults: {jpath}\nreport:  {mpath}")
    ok = {"EXACT", "PASS", "SKIP"} | ({"CRSDIFF"} if args.allow_crsdiff else set())
    bad = [r for r in rows if r["status"] not in ok]
    return 1 if bad else 0


def cmd_compare(args) -> int:
    a, b = Path(args.a), Path(args.b)
    pairs = []
    if a.is_dir():
        for f in sorted(a.glob(args.glob)):
            pairs.append((f, b / f.name))
    else:
        pairs.append((a, b))
    bad = 0
    for fa, fb in pairs:
        if not fb.exists():
            print(f"{fa.name}: MISSING in {b}")
            bad += 1
            continue
        r = compare(fa, fb, args.atol, args.rtol)
        print(f"{fa.name}: {summary_line(r)}")
        bad += r["status"] not in ("EXACT", "PASS", "CRSDIFF")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run cases and compare to ReferenceResult")
    r.add_argument("--impl", action="append", help="cpp[:dir] or rs[:binary], repeatable (default cpp)")
    r.add_argument("--data", help="TauDEM-Test-Data root (or TAUDEM_TEST_DATA)")
    r.add_argument("--cases", help="comma-separated case names from cases.json (default: all)")
    r.add_argument("--tools", nargs="*", help="only steps using these canonical tools")
    r.add_argument("--tags", nargs="*", help="only steps carrying one of these tags")
    r.add_argument("--skip-tags", nargs="*", help="skip steps carrying one of these tags (e.g. ogr filegdb)")
    r.add_argument("--inputs", choices=["reference", "chain"], default="reference")
    r.add_argument("--np", type=int, help="override every step's rank/thread count")
    r.add_argument("--atol", type=float, default=0.0)
    r.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for float outputs (upstream compare_results uses 1e-6)")
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--allow-crsdiff", action="store_true", help="do not fail the run on CRSDIFF (CRS-definition-only) results")
    r.add_argument("--work", help="scratch root (or TAUDEM_BENCH_WORK)")
    r.add_argument("--results", help="directory for JSON/markdown results (default <work>/results)")
    r.add_argument("--mpiexec")
    r.add_argument("--tools-config", default="tools.json")
    r.add_argument("--cases-config", default="cases.json")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare", help="compare two rasters or two directories of rasters")
    c.add_argument("a"); c.add_argument("b")
    c.add_argument("--glob", default="*.tif")
    c.add_argument("--atol", type=float, default=0.0)
    c.add_argument("--rtol", type=float, default=1e-6)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
