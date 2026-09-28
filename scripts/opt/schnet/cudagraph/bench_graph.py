#!/usr/bin/env python
"""
CUDA-graph benchmark for the SchNet fast artifact (schnet-nl agent).

bench_common.py can only call forward()/forward_batch(); the graph path is a
different calling convention (what a graph-capable NAMD shim would do), so it
is timed here, interleaved in ONE process with the plain forward() of the same
artifacts, using the same conventions as bench_common (fp64 coords resident on
the GPU, forces copied to pinned host memory, energy .cpu() = the sync point,
same water boxes, same --jitter pool semantics, GPU flock held throughout).

GraphRunner below is the reference for the shim logic:
    cap   = model.graph_capacity(coords)                 # eager, sizes the NL
    warm up model.graph_step(static_coords, Z, cap) x4 on a side stream
    capture it once -> (E, F, n_edges) static outputs
    every step: static_coords.copy_(coords); replay; copy E, F out;
                if n_edges > cap: re-capture with a new capacity and re-run.

  flock is taken inside; run as
  python scripts/opt/schnet/cudagraph/bench_graph.py --base models/opt/schnet_baseline.pt \
      --fast models/opt/schnet_fast.pt --systems 30,300,900,3000 --jitter 0.02 --out ...
"""
import argparse
import fcntl
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "scripts/opt"))
from bench_common import water_system, LOCK_PATH  # noqa: E402


class GraphRunner:
    def __init__(self, mod, dev):
        self.mod, self.dev = mod, dev
        self.graph = None
        self.recaptures = 0

    def prepare(self, xyz, Z):
        self.coords = xyz.to(self.dev)
        self.Z = Z.to(self.dev)
        self.static = self.coords.clone()
        self.h_forces = torch.empty(xyz.shape[0], 3, dtype=torch.float64).pin_memory()
        self.graph = None

    def _capture(self):
        self.graph = None
        torch.cuda.synchronize()
        self.cap = int(self.mod.graph_capacity(self.static))
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(4):
                self.mod.graph_step(self.static, self.Z, self.cap)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.out = self.mod.graph_step(self.static, self.Z, self.cap)
        self.graph = g
        self.recaptures += 1

    def call(self, src):
        self.static.copy_(src)
        if self.graph is None:
            self._capture()
        self.graph.replay()
        e, f, n = self.out
        self.h_forces.copy_(f, non_blocking=True)
        n_host = int(n)              # sync point (energy/count read back together)
        e_host = e.cpu()
        if n_host > self.cap:        # edges were dropped -> invalid, redo
            self._capture()
            self.graph.replay()
            e, f, n = self.out
            self.h_forces.copy_(f, non_blocking=True)
            e_host = e.cpu()
        return e_host, self.h_forces


class FwdRunner:
    def __init__(self, mod, dev):
        self.mod, self.dev = mod, dev
        self.pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev)
        self.pcq = torch.zeros(0, dtype=torch.float64, device=dev)
        self.cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)

    def prepare(self, xyz, Z):
        self.Z = Z.to(self.dev)
        self.h_forces = torch.empty(xyz.shape[0], 3, dtype=torch.float64).pin_memory()

    def call(self, src):
        c = src.detach().clone().requires_grad_(True)
        e, f = self.mod(c, self.Z, self.pcx, self.pcq, self.cell)[:2]
        self.h_forces.copy_(f.detach(), non_blocking=True)
        return e.detach().reshape(1).cpu(), self.h_forces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--fast", required=True)
    ap.add_argument("--systems", default="30,300,900,3000")
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    fh = open(LOCK_PATH, "w")
    print("[bench_graph] waiting for GPU lock", flush=True)
    fcntl.flock(fh, fcntl.LOCK_EX)
    base = torch.jit.load(a.base, map_location=dev).eval()
    fast = torch.jit.load(a.fast, map_location=dev).eval()
    runners = {"base": FwdRunner(base, dev), "fast": FwdRunner(fast, dev), "fast_graph": GraphRunner(fast, dev)}
    results = []
    for s in [int(x) for x in a.systems.split(",")]:
        xyz, Z = water_system(s)
        c0 = xyz.to(dev)
        if a.jitter > 0:
            g = torch.Generator(device=dev).manual_seed(1234)
            pool = [c0 + a.jitter * torch.randn(c0.shape, generator=g, device=dev, dtype=c0.dtype) for _ in range(64)]
        else:
            pool = [c0]
        rows, ref = {}, None
        for name, r in runners.items():
            r.prepare(xyz, Z)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            m0 = torch.cuda.memory_allocated(dev)
            for k in range(a.warmup):
                r.call(pool[k % len(pool)])
            torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated(dev) - m0) / 2**20
            # parity on the unjittered geometry
            e, f = r.call(c0); e = e.clone().reshape(1); f = f.clone()
            if ref is None:
                ref = (e, f)
            rows[name] = dict(peak_alloc_mib=round(peak, 1), dE_max_kcal=float((e - ref[0]).abs().max()),
                              dF_max_kcal_A=float((f - ref[1]).abs().max()), ts=[])
        k = 0
        for _ in range(a.rounds):
            for name, r in runners.items():
                for _ in range(a.iters):
                    src = pool[k % len(pool)]; k += 1
                    t = time.perf_counter(); r.call(src); torch.cuda.synchronize()
                    rows[name]["ts"].append((time.perf_counter() - t) * 1e3)
        for name, row in rows.items():
            ts = sorted(row.pop("ts"))
            row.update(median_ms=round(statistics.median(ts), 3), p10_ms=round(ts[len(ts) // 10], 3),
                       p90_ms=round(ts[9 * len(ts) // 10], 3), n=len(ts))
            if name == "fast_graph":
                row["captures"] = runners[name].recaptures
            results.append(dict(system=f"water{s}", n_atoms=s, walkers=1, model=name, **row))
            print(f"water{s:<5} {name:>10} median {row['median_ms']:7.3f} p10 {row['p10_ms']:7.3f} "
                  f"p90 {row['p90_ms']:7.3f} peak {row['peak_alloc_mib']:7.1f} MiB "
                  f"dE {row['dE_max_kcal']:.2e} dF {row['dF_max_kcal_A']:.2e}"
                  + (f" captures {row['captures']}" if 'captures' in row else ""), flush=True)
        runners["fast_graph"].graph = None
        torch.cuda.empty_cache()
    if a.out:
        meta = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(dev),
                    time=time.strftime("%Y-%m-%d %H:%M:%S"), argv=sys.argv)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(meta=meta, results=results), indent=1))
        print("[bench_graph] wrote", a.out)


if __name__ == "__main__":
    main()
