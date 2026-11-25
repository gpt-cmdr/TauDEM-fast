"""Shared helpers for the TauDEM parity and benchmark harness.

Portable: no machine-specific paths. Everything comes from CLI args, env vars, or
the JSON config files that live next to this module (tools.json, cases.json).

Environment variables (all optional; CLI args win):
  TAUDEM_TEST_DATA   path to a TauDEM-Test-Data checkout (has Input/ and ReferenceResult/)
  TAUDEM_CPP_DIR     directory holding the C++ TauDEM executables (PitRemove[.exe], ...)
  TAUDEM_RS_BIN      path to the taudem-rs binary
  TAUDEM_BENCH_WORK  scratch/output root (default: <system temp>/taudem-bench)
  MPIEXEC            mpiexec/mpirun to use for the C++ implementation
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"


def load_json(name: str | Path) -> dict:
    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = HERE / p
    return json.loads(p.read_text(encoding="utf-8"))


def work_root(cli: str | None = None) -> Path:
    root = Path(cli or os.environ.get("TAUDEM_BENCH_WORK") or (Path(tempfile.gettempdir()) / "taudem-bench"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_data_root(cli: str | None = None) -> Path:
    v = cli or os.environ.get("TAUDEM_TEST_DATA")
    if not v:
        sys.exit("TauDEM-Test-Data location unknown: pass --data or set TAUDEM_TEST_DATA")
    p = Path(v)
    if not (p / "Input").is_dir() or not (p / "ReferenceResult").is_dir():
        sys.exit(f"{p} does not look like a TauDEM-Test-Data checkout (needs Input/ and ReferenceResult/)")
    return p


# --------------------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------------------

def find_mpiexec(cli: str | None = None) -> str:
    cands = [cli, os.environ.get("MPIEXEC"), shutil.which("mpiexec"), shutil.which("mpirun")]
    if IS_WINDOWS:
        msmpi = os.environ.get("MSMPI_BIN")
        if msmpi:
            cands.append(str(Path(msmpi) / "mpiexec.exe"))
        cands.append(r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe")
    for c in cands:
        if c and Path(c).exists():
            return str(c)
        if c and shutil.which(c):
            return shutil.which(c)  # type: ignore[return-value]
    sys.exit("mpiexec not found: pass --mpiexec or set MPIEXEC")


@dataclass
class Impl:
    """One implementation under test.

    kind == "cpp": C++ TauDEM, one executable per tool, launched under mpiexec -n <ranks>.
    kind == "rs":  taudem-rs, one binary with subcommands; ranks become a thread count
                   when tools.json says how to pass it (flag and/or env var).
    """

    kind: str
    path: Path
    label: str
    tools_cfg: dict
    mpiexec: str | None = None
    extra_env: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, spec: str, tools_cfg: dict, mpiexec: str | None = None) -> "Impl":
        """spec: 'cpp', 'cpp:<exe dir>', 'rs', 'rs:<binary path>' (optionally '=label' suffix)."""
        label = None
        if "=" in spec:
            spec, label = spec.split("=", 1)
        kind, _, loc = spec.partition(":")
        if len(kind) == 1 and loc.startswith("\\") or len(kind) == 1 and loc.startswith("/"):
            sys.exit(f"implementation spec must start with cpp: or rs: (got {spec!r})")
        kind = kind.lower()
        if kind == "cpp":
            loc = loc or os.environ.get("TAUDEM_CPP_DIR", "")
            if not loc:
                sys.exit("C++ TauDEM dir unknown: use cpp:<dir> or set TAUDEM_CPP_DIR")
            return cls("cpp", Path(loc), label or "cpp", tools_cfg, find_mpiexec(mpiexec))
        if kind == "rs":
            loc = loc or os.environ.get("TAUDEM_RS_BIN", "")
            if not loc:
                sys.exit("taudem-rs binary unknown: use rs:<path> or set TAUDEM_RS_BIN")
            return cls("rs", Path(loc), label or "rs", tools_cfg)
        sys.exit(f"unknown implementation kind {kind!r} (expected cpp or rs)")

    def available(self) -> bool:
        return self.path.is_dir() if self.kind == "cpp" else self.path.is_file()

    def supports(self, tool: str) -> bool:
        return self._tool_name(tool) is not None

    def _tool_name(self, tool: str) -> str | None:
        t = self.tools_cfg["tools"].get(tool)
        if t is None:
            return None
        return t.get("cpp") if self.kind == "cpp" else t.get("rs")

    def _cpp_exe(self, name: str) -> str:
        for n in (name, name.lower()):
            for ext in ((".exe", "") if IS_WINDOWS else ("", ".exe")):
                p = self.path / (n + ext)
                if p.exists():
                    return str(p)
        raise FileNotFoundError(f"{name} not found in {self.path}")

    def command(self, tool: str, argv: list[str], ranks: int) -> tuple[list[str], dict]:
        """Return (argv list, env) for one tool invocation."""
        name = self._tool_name(tool)
        if name is None:
            raise NotImplementedError(f"{self.label} does not implement {tool}")
        env = dict(os.environ)
        env.update(self.extra_env)
        if self.kind == "cpp":
            return [self.mpiexec, "-n", str(ranks), self._cpp_exe(name), *argv], env  # type: ignore[list-item]
        rs = self.tools_cfg.get("rs", {})
        cmd = [str(self.path), name, *argv]
        if ranks and rs.get("threads_flag"):
            cmd += [rs["threads_flag"], str(ranks)]
        if ranks and rs.get("threads_env"):
            env[rs["threads_env"]] = str(ranks)
        return cmd, env

    def describe(self) -> dict:
        d = {"label": self.label, "kind": self.kind, "path": str(self.path)}
        if self.kind == "cpp":
            d["mpiexec"] = self.mpiexec
        try:
            if self.kind == "rs":
                out = subprocess.run([str(self.path), "--version"], capture_output=True, text=True, timeout=30)
                d["version"] = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None
        except Exception:
            pass
        return d


# --------------------------------------------------------------------------------------
# Measured process execution (wall time + peak memory over the whole process tree)
# --------------------------------------------------------------------------------------

_TIME_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/-]*?)\s+time:\s*([-+0-9.eE]+)\s*$", re.M)


def parse_taudem_timings(text: str) -> dict[str, float]:
    """TauDEM prints lines like 'Compute time: 0.34'. Return {'compute': 0.34, ...}.

    Repeated keys are summed (a tool printing the same phase twice)."""
    out: dict[str, float] = {}
    for k, v in _TIME_RE.findall(text):
        key = k.strip().lower().replace(" ", "_")
        try:
            out[key] = out.get(key, 0.0) + float(v)
        except ValueError:
            pass
    return out


def _linux_hwm(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


@dataclass
class RunResult:
    cmd: list[str]
    returncode: int
    wall_s: float
    stdout: str
    stderr: str
    peak_sum_bytes: int          # sum over worker processes of each one's peak working set / VmHWM
    peak_max_proc_bytes: int     # largest single-process peak
    peak_concurrent_bytes: int   # largest sampled concurrent sum of current RSS (lower bound)
    nprocs_seen: int
    timings: dict

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["stdout_tail"] = self.stdout[-2000:]
        d["stderr_tail"] = self.stderr[-2000:]
        del d["stdout"], d["stderr"]
        return d


# Launcher/infra processes whose memory we do not attribute to the tool.
_INFRA = {"mpiexec", "mpiexec.exe", "smpd", "smpd.exe", "conhost.exe", "hydra_pmi_proxy", "mpirun", "orterun", "prterun", "prted", "orted"}


def run_measured(cmd: list[str], cwd: Path, env: dict | None = None, timeout: float | None = None,
                 poll_s: float = 0.02) -> RunResult:
    """Run cmd, sampling the process tree for peak memory. Requires psutil."""
    import psutil

    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, errors="replace")
    out_chunks: list[str] = []
    err_chunks: list[str] = []
    th_o = threading.Thread(target=lambda: out_chunks.append(p.stdout.read()), daemon=True)  # type: ignore[union-attr]
    th_e = threading.Thread(target=lambda: err_chunks.append(p.stderr.read()), daemon=True)  # type: ignore[union-attr]
    th_o.start(); th_e.start()
    peaks: dict[tuple[int, str], int] = {}
    concurrent = 0
    try:
        root = psutil.Process(p.pid)
    except psutil.Error:
        root = None
    deadline = None if timeout is None else t0 + timeout
    while p.poll() is None:
        if root is not None:
            try:
                procs = [root] + root.children(recursive=True)
            except psutil.Error:
                procs = []
            cur = 0
            for c in procs:
                try:
                    name = c.name()
                    mi = c.memory_info()
                except psutil.Error:
                    continue
                if name.lower() in _INFRA:
                    continue
                pk = getattr(mi, "peak_wset", None)
                if pk is None:
                    pk = _linux_hwm(c.pid) or mi.rss
                key = (c.pid, name)
                peaks[key] = max(peaks.get(key, 0), pk)
                cur += mi.rss
            concurrent = max(concurrent, cur)
        if deadline and time.perf_counter() > deadline:
            for c in (root.children(recursive=True) if root else []):
                try:
                    c.kill()
                except psutil.Error:
                    pass
            p.kill()
            break
        time.sleep(poll_s)
    p.wait()
    wall = time.perf_counter() - t0
    th_o.join(5); th_e.join(5)
    stdout = "".join(out_chunks)
    stderr = "".join(err_chunks)
    if not peaks and not IS_WINDOWS:
        # Process finished between polls; fall back to rusage (max single child).
        import resource
        r = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
        peaks[(p.pid, "rusage")] = r
    return RunResult(cmd=[str(c) for c in cmd], returncode=p.returncode, wall_s=wall, stdout=stdout,
                     stderr=stderr, peak_sum_bytes=sum(peaks.values()),
                     peak_max_proc_bytes=max(peaks.values(), default=0), peak_concurrent_bytes=concurrent,
                     nprocs_seen=len(peaks), timings=parse_taudem_timings(stdout + "\n" + stderr))


def machine_info() -> dict:
    info = {"platform": platform.platform(), "python": sys.version.split()[0], "machine": platform.node(),
            "processor": platform.processor()}
    try:
        import psutil
        info["cpu_logical"] = psutil.cpu_count()
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["ram_bytes"] = psutil.virtual_memory().total
        info["cpu_load_pct_at_start"] = psutil.cpu_percent(interval=0.5)
    except Exception:
        pass
    return info


def copy_case(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_with_sidecars(src_dir: Path, name: str, dst_dir: Path) -> bool:
    """Copy a raster plus sidecars (foo.tif, foo.tif.aux.xml, foo.hdr, foo.prj, ...) or a dir."""
    src = src_dir / name
    if src.is_dir():
        dst = dst_dir / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True
    if not src.exists():
        return False
    stem = Path(name).stem
    for f in src_dir.iterdir():
        if f.is_file() and (f.name == name or f.name.startswith(name + ".") or f.stem == stem
                            and f.suffix.lower() in {".hdr", ".prj", ".aux", ".sgrd", ".rrd", ".ige"}):
            (dst_dir / f.name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_dir / f.name)
    return True


def fmt_bytes(n: float | int | None) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return str(n)
