"""Where counters come from.

GpgpuSimBackend  runs sim/harness on GPGPU-Sim (the A100_slice config by default), keeps every log
                 in results/autotune/logs/, and caches parsed counters in a JSON file keyed by
                 (design point, workload). The simulator is deterministic, so a cached result is
                 identical to a fresh run.
ReplayBackend    answers from that cache only and counts how many *distinct* simulations a search
                 asked for. This is how search strategies are compared fairly: every strategy sees
                 exactly the counters it would have seen live, and is charged one simulation per
                 new design point.

Both backends attach static compiler facts (StaticInfo: registers and spill bytes from ptxas for
sm_80). The simulator cannot see spills, because it executes PTX, so this is how the optimizer
knows that a simulated win from a register cap may be fake.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from .counters import Counters, parse_log
from .space import DesignPoint, Workload

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(REPO, "results", "autotune", "sim_cache.json")
DEFAULT_LOGS = os.path.join(REPO, "results", "autotune", "logs")
DEFAULT_CONFIG = os.path.join(REPO, "sim", "configs", "A100_slice")
DEFAULT_HARNESS = os.path.join(REPO, "sim", "harness")


# ---------------------------------------------------------------------------------- static facts
class StaticInfo:
    """Registers / spills per kernel from `ptxas -v` for sm_80 (the real A100 target)."""

    _ENTRY = re.compile(r"Compiling entry function '(\S+)' for '(sm_\d+)'")
    _SPILL = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
    _REGS = re.compile(r"Used (\d+) registers")

    def __init__(self, table: dict):
        self.table = table

    @classmethod
    def from_ptxas(cls, harness_src: str = os.path.join(REPO, "sim", "harness.cu"), arch: str = "sm_80",
                   cache: str = os.path.join(REPO, "results", "autotune", "static_sm80.json")) -> "StaticInfo":
        if os.path.exists(cache) and os.path.getmtime(cache) > max(
                os.path.getmtime(harness_src), *(os.path.getmtime(os.path.join(REPO, "csrc", "kernels", f))
                                                for f in os.listdir(os.path.join(REPO, "csrc", "kernels")))):
            return cls(json.load(open(cache)))
        out = tempfile.mktemp(suffix=".o")
        proc = subprocess.run(["nvcc", "-O3", "-std=c++17", f"-arch={arch}", "-Xptxas", "-v", "-c", "-o", out,
                               harness_src], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        table, cur = {}, None
        for line in proc.stdout.splitlines():
            m = cls._ENTRY.search(line)
            if m:
                cur = m.group(1)
                table[cur] = {"regs": 0, "spill_stores": 0, "spill_loads": 0, "stack": 0}
                continue
            if cur is None:
                continue
            m = cls._SPILL.search(line)
            if m:
                table[cur].update(stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
            m = cls._REGS.search(line)
            if m:
                table[cur]["regs"] = int(m.group(1))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(table, open(cache, "w"), indent=1)
        return cls(table)

    def for_kernel(self, mangled: str) -> dict:
        return self.table.get(mangled, {})


def _attach_static(c: Counters, static: StaticInfo | None):
    if static is None:
        return
    s = static.for_kernel(c.raw.get("dominant_kernel", ""))
    c.raw["hw_regs"] = s.get("regs", 0)
    c.raw["hw_spill_bytes"] = s.get("spill_stores", 0)  # bytes spilled per thread (stores)


# ---------------------------------------------------------------------------------- backends
class Backend:
    name = "base"

    def __init__(self):
        self.calls: list[str] = []      # distinct design points simulated, in order

    def evaluate(self, p: DesignPoint, wl: Workload) -> Counters:
        raise NotImplementedError

    @property
    def sims_used(self) -> int:
        return len(self.calls)


class _Cache:
    def __init__(self, path: str):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    @staticmethod
    def key(p: DesignPoint, wl: Workload) -> str:
        return f"{wl.key()}|{p.key()}"

    def get(self, p, wl):
        d = self.data.get(self.key(p, wl))
        return Counters.from_dict(d["counters"]) if d else None

    def put(self, p, wl, c: Counters, meta: dict):
        self.data[self.key(p, wl)] = {"point": p.key(), "workload": wl.key(), "counters": c.to_dict(), **meta}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        json.dump(self.data, open(tmp, "w"), indent=1, sort_keys=True)
        os.replace(tmp, self.path)


class GpgpuSimBackend(Backend):
    name = "gpgpusim"

    def __init__(self, cache: str = DEFAULT_CACHE, logs: str = DEFAULT_LOGS, config: str = DEFAULT_CONFIG,
                 harness: str = DEFAULT_HARNESS, static: StaticInfo | None = None):
        super().__init__()
        self.cache, self.logs, self.config, self.harness, self.static = _Cache(cache), logs, config, harness, static
        env = dict(os.environ)
        assert "gpgpu-sim" in env.get("LD_LIBRARY_PATH", ""), "source the GPGPU-Sim environment first (see sim/README.md)"
        self.env = env

    def log_path(self, p: DesignPoint, wl: Workload) -> str:
        return os.path.join(self.logs, f"{wl.key()}__{p.key().replace('/', '_')}.log")

    def run(self, p: DesignPoint, wl: Workload) -> Counters:
        work = tempfile.mkdtemp(prefix="sim_")
        for f in os.listdir(self.config):
            shutil.copy(os.path.join(self.config, f), work)
        t0 = time.time()
        proc = subprocess.run([self.harness, p.harness_spec(), str(wl.BH), str(wl.N), str(wl.causal)], cwd=work,
                              env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wall = time.time() - t0
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(self.logs, exist_ok=True)
        with open(self.log_path(p, wl), "w") as fh:
            fh.write(proc.stdout)
        c = parse_log(proc.stdout, p, wl)
        _attach_static(c, self.static)
        self.cache.put(p, wl, c, {"sim_wall_s": round(wall), "harness_spec": p.harness_spec(), "source": "simulated"})
        print(f"  [sim] {p.key():32} {c.status:5} cycles={c.raw.get('cycles', 0):>10,} ({wall:.0f}s)", flush=True)
        return c

    def evaluate(self, p: DesignPoint, wl: Workload) -> Counters:
        c = self.cache.get(p, wl)
        if c is None:
            c = self.run(p, wl)
        if p.key() not in self.calls:
            self.calls.append(p.key())
        return c

    def evaluate_many(self, pts: list[DesignPoint], wl: Workload, jobs: int = 2):
        todo = [p for p in pts if self.cache.get(p, wl) is None]
        with ThreadPoolExecutor(jobs) as pool:
            list(pool.map(lambda p: self.run(p, wl), todo))

    def import_log(self, p: DesignPoint, wl: Workload, log_path: str, note: str):
        """Add an existing simulator log (e.g. from the original v1-v4 study) to the cache."""
        log = open(log_path).read()
        c = parse_log(log, p, wl)
        _attach_static(c, self.static)
        os.makedirs(self.logs, exist_ok=True)
        shutil.copy(log_path, self.log_path(p, wl))
        self.cache.put(p, wl, c, {"harness_spec": p.harness_spec(), "source": note})


class ReplayBackend(Backend):
    """Deterministic replay from the simulation cache; raises if a point was never simulated."""
    name = "replay"

    def __init__(self, cache: str = DEFAULT_CACHE):
        super().__init__()
        self.cache = _Cache(cache)

    def has(self, p: DesignPoint, wl: Workload) -> bool:
        return self.cache.get(p, wl) is not None

    def evaluate(self, p: DesignPoint, wl: Workload) -> Counters:
        c = self.cache.get(p, wl)
        if c is None:
            raise KeyError(f"{p.key()} @ {wl.key()} not in the simulation cache; run it with GpgpuSimBackend")
        if p.key() not in self.calls:
            self.calls.append(p.key())
        return c

    def reset(self):
        self.calls = []
