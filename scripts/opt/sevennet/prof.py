"""Where a SevenNet artifact spends its time: torch.profiler over a few calls
made the way the NAMD shim makes them (bench_common.Runner), native OEQ lib only.

usage: flock scripts/opt/.gpu_bench.lock python scripts/opt/sevennet/prof.py ARTIFACT N [W] [--rows K]
Prints the top ops by self CPU and by self CUDA time, the op count per call, and
wall vs GPU-busy time per call.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from torch.profiler import ProfilerActivity, profile

import bench_common as bc

ap = argparse.ArgumentParser()
ap.add_argument("artifact")
ap.add_argument("n", type=int)
ap.add_argument("w", type=int, nargs="?", default=1)
ap.add_argument("--rows", type=int, default=25)
ap.add_argument("--calls", type=int, default=5)
a = ap.parse_args()

torch.ops.load_library(str(bc.REPO / "scripts/opt/nequip/oeq_native/liboeq_native.so"))
dev = torch.device("cuda:0")
bc.JITTER = 0.02
r = bc.Runner("x", a.artifact, dev)
xyz, Z = bc.water_system(a.n)
r.prepare(xyz, Z, a.w)
for _ in range(20):
    r.call()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    r.call()
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) / 20 * 1e3

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(a.calls):
        r.call()
    torch.cuda.synchronize()
ev = prof.key_averages()
n_ops = sum(e.count for e in ev if e.key.startswith("aten::")) / a.calls
gpu = sum(e.self_device_time_total for e in ev) / a.calls / 1e3
print(f"{a.artifact}  N={a.n} W={a.w}: wall {wall:.2f} ms/call, GPU busy {gpu:.2f} ms/call, "
      f"{n_ops:.0f} aten ops/call")
print(ev.table(sort_by="self_cpu_time_total", row_limit=a.rows, max_name_column_width=60))
print(ev.table(sort_by="self_device_time_total", row_limit=a.rows, max_name_column_width=60))
