#!/usr/bin/env python
"""
Where does the SchNet wrapper spend its time?  (schnet-nl agent)

Run under the GPU lock:
  flock scripts/opt/.gpu_bench.lock python scripts/opt/schnet/profile_breakdown.py \
      --model models/opt/schnet_baseline.pt --systems 30,900,3000 --walkers 1,4

Prints, per system: wall time of the full call (like bench_common), the
standalone neighbour-list cost, and a torch.profiler table of the top CUDA
kernels / CPU ops so we can split NL / representation / backward / host syncs.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "opt"))
from bench_common import water_system  # noqa: E402
from src import edges  # noqa: E402


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--systems", default="30,900,3000")
    ap.add_argument("--walkers", default="1")
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--no-prof", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda:0")
    m = torch.jit.load(a.model, map_location=dev).eval()
    pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev)
    pcq = torch.zeros(0, dtype=torch.float64, device=dev)
    for s in [int(x) for x in a.systems.split(",")]:
        xyz, Z = water_system(s)
        for W in [int(w) for w in a.walkers.split(",")]:
            n = xyz.shape[0]
            if W == 1:
                c0 = xyz.to(dev); Zd = Z.to(dev)
                cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)
                call = lambda: m(c0.clone().requires_grad_(True), Zd, pcx, pcq, cell)
                nl = lambda: edges.build_edges(c0.float(), 5.0)
            else:
                g = torch.Generator().manual_seed(0)
                c0 = torch.cat([xyz + 0.01 * torch.randn(xyz.shape, generator=g, dtype=xyz.dtype) * (i > 0)
                                for i in range(W)]).to(dev)
                Zd = Z.repeat(W).to(dev)
                batch = torch.arange(W).repeat_interleave(n).to(dev)
                ptr = (torch.arange(W + 1) * n).to(dev)
                cells = torch.zeros(W, 3, 3, dtype=torch.float64, device=dev)
                call = lambda: m.forward_batch(c0.clone().requires_grad_(True), Zd, batch, ptr, pcx, pcq, cells)
                nl = lambda: edges.build_edges_batched(c0.float(), ptr, 5.0)
            t_call = timeit(call)
            t_nl = timeit(nl)
            E = nl()[0].shape[1]
            print(f"\n=== water{s} W={W}: full call {t_call:.3f} ms | eager NL {t_nl:.3f} ms | edges {E}")
            if a.no_prof:
                continue
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    call()
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.rows))


if __name__ == "__main__":
    main()
