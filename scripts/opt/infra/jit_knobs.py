#!/usr/bin/env python
"""
Phase-2 (infra): measure TorchScript runtime knobs that do NOT change the model.

For each config a SEPARATE copy of the artifact is loaded and first called with
that config's global flags set, so each copy gets its own GraphExecutor built
under its own settings.  Before every call the config's flags are set again,
then the copies are timed INTERLEAVED (round-robin), like bench_common.py.

MD-like inputs: every call gets fresh coordinates (base + N(0, sigma) noise,
pre-generated on the device), so the neighbour-list edge count changes from
call to call the way it does in NAMD.  That matters: the profiling executor
specialises on shapes, and a fixed-geometry benchmark hides re-specialisation.

Configs (the C++ equivalents are in brackets):
  default     torch 2.11 defaults: profiling executor, texpr fuser on, fusion strategy STATIC 2 + DYNAMIC 10
  noopt       torch._C._set_graph_executor_optimize(False)     [torch::jit::setGraphExecutorOptimize(false)]
  legacy      _jit_set_profiling_executor(False)+_jit_set_profiling_mode(False)   [getExecutorMode()=false, getProfilingMode()=false]
  notexpr     _jit_set_texpr_fuser_enabled(False)               [torch::jit::setTensorExprFuserEnabled(false)]
  dyn20       fusion strategy [("DYNAMIC", 20)]                  [torch::jit::setFusionStrategy]
  static20    fusion strategy [("STATIC", 20)]
  freeze      torch.jit.freeze(copy, preserved_attrs=["forward_batch"]) with default flags
  tf32        default + torch.backends.cuda.matmul.allow_tf32 = True   (CHANGES THE NUMBERS for fp32 matmuls)

Usage:
  python scripts/opt/infra/jit_knobs.py --model namd_benchmarks/models/schnet.pt --systems 30,300,900 \
      [--configs default,noopt,...] [--geom file.xyz] [--jitter 0.02] [--out json]
Takes the shared GPU lock (scripts/opt/.gpu_bench.lock).
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench_common as bc  # noqa: E402

DEFAULT_STRATEGY = [("STATIC", 2), ("DYNAMIC", 10)]


def apply_flags(cfg: str):
    torch._C._set_graph_executor_optimize(cfg != "noopt")
    legacy = cfg == "legacy"
    torch._C._jit_set_profiling_executor(not legacy)
    torch._C._jit_set_profiling_mode(not legacy)
    torch._C._jit_set_texpr_fuser_enabled(cfg != "notexpr")
    if cfg == "dyn20":
        torch._C._jit_set_fusion_strategy([("DYNAMIC", 20)])
    elif cfg == "static20":
        torch._C._jit_set_fusion_strategy([("STATIC", 20)])
    else:
        torch._C._jit_set_fusion_strategy(DEFAULT_STRATEGY)
    torch.backends.cuda.matmul.allow_tf32 = cfg == "tf32"
    torch.backends.cudnn.allow_tf32 = cfg == "tf32"


class Cfg:
    def __init__(self, name, path, dev):
        self.name = name
        apply_flags(name)
        self.r = bc.Runner(name, path, dev)
        if name == "freeze":
            keep = ["forward_batch"] if self.r.has_batch else []
            self.r.mod = torch.jit.freeze(self.r.mod, preserved_attrs=keep)
        self.ts = []

    def call(self, coords):
        apply_flags(self.name)
        self.r.coords = coords
        return self.r.call()


def geoms_from_args(args):
    out = []
    for s in [x for x in args.systems.split(",") if x.strip()]:
        out.append((f"water{s}", *bc.water_system(int(s))))
    for g in args.geom:
        out.append((Path(g).stem, *bc.load_geom(Path(g))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--systems", default="30,300,900")
    ap.add_argument("--geom", action="append", default=[])
    ap.add_argument("--configs", default="default,noopt,legacy,notexpr,dyn20,static20,freeze,tf32")
    ap.add_argument("--jitter", type=float, default=0.02, help="per-call coordinate noise sigma (A); 0 = fixed geometry")
    ap.add_argument("--pool", type=int, default=64, help="number of pre-generated jittered geometries")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    fh = open(bc.LOCK_PATH, "w")
    print("[knobs] waiting for GPU lock ...", flush=True)
    fcntl.flock(fh, fcntl.LOCK_EX)

    names = args.configs.split(",")
    results = []
    for gname, xyz, Z in geoms_from_args(args):
        cfgs = []
        for n in names:
            try:
                c = Cfg(n, Path(args.model), dev)
                c.r.prepare(xyz, Z, 1)
                cfgs.append(c)
            except Exception as ex:
                print(f"[knobs] {n}: load/prepare failed: {str(ex).splitlines()[0][:160]}")
        g = torch.Generator(device=dev).manual_seed(1234)
        base = xyz.to(dev)
        pool = [base + args.jitter * torch.randn(base.shape, generator=g, device=dev, dtype=base.dtype)
                for _ in range(args.pool)]
        # warm-up (each config compiles/profiles under its own flags) + parity on pool[0]
        ref = None
        rows = {}
        live = []
        for c in cfgs:
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                for i in range(args.warmup):
                    c.call(pool[i % len(pool)])
                torch.cuda.synchronize()
                warm = (time.perf_counter() - t0) * 1e3
                e, f = c.call(pool[0])
                torch.cuda.synchronize()
                e, f = e.clone(), f.clone()
                if ref is None:
                    ref = (e, f)
                rows[c.name] = dict(warmup_total_ms=round(warm, 1),
                                    peak_alloc_mib=round((torch.cuda.max_memory_allocated() - m0) / 2**20, 1),
                                    dE=float((e - ref[0]).abs().max()), dF=float((f - ref[1]).abs().max()))
                live.append(c)
            except Exception as ex:
                rows[c.name] = dict(error=str(ex).splitlines()[0][:200])
                print(f"[knobs] {c.name} {gname}: ERROR {rows[c.name]['error']}", flush=True)
        k = 0
        for _ in range(args.rounds):
            for c in live:
                for _ in range(args.iters):
                    x = pool[k % len(pool)]; k += 1
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    c.call(x)
                    torch.cuda.synchronize()
                    c.ts.append((time.perf_counter() - t) * 1e3)
        for c in live:
            ts = sorted(c.ts)
            rows[c.name].update(median_ms=round(statistics.median(ts), 3),
                                p10_ms=round(ts[len(ts) // 10], 3), p90_ms=round(ts[9 * len(ts) // 10], 3))
        b = rows.get(names[0], {}).get("median_ms")
        print(f"\n== {Path(args.model).name}  {gname} ({xyz.shape[0]} atoms)  jitter={args.jitter}")
        print(f"{'config':>10} {'median':>9} {'p10':>8} {'p90':>8} {'speedup':>7} {'warmup':>9} {'peakMiB':>8} {'dE':>9} {'dF':>9}")
        for n in names:
            r = rows.get(n)
            if r is None:
                continue
            if "median_ms" not in r:
                print(f"{n:>10}  ERROR {r.get('error', '')[:90]}")
                continue
            sp = b / r["median_ms"] if b else float("nan")
            print(f"{n:>10} {r['median_ms']:>9.3f} {r['p10_ms']:>8.3f} {r['p90_ms']:>8.3f} {sp:>6.2f}x "
                  f"{r['warmup_total_ms']:>9.1f} {r['peak_alloc_mib']:>8.1f} {r['dE']:>9.2e} {r['dF']:>9.2e}")
            results.append(dict(model=str(args.model), system=gname, n_atoms=int(xyz.shape[0]), config=n, **r))
        del cfgs, live
        torch.cuda.empty_cache()
    apply_flags("default")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(torch=torch.__version__, argv=sys.argv, results=results), indent=1))


if __name__ == "__main__":
    main()
