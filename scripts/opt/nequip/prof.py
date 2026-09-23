"""torch.profiler breakdown of a wrapped NequIP artifact called like the NAMD shim.
usage: python prof.py MODEL.pt N [--lib SO] [--walkers W] [--rows 25]
Prints wall ms/call, GPU busy ms/call, and top CUDA kernels / CPU ops. Hold the GPU lock."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import bench_common as bc

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("n", type=int)
ap.add_argument("--lib", action="append", default=[])
ap.add_argument("--walkers", type=int, default=1)
ap.add_argument("--rows", type=int, default=25)
ap.add_argument("--iters", type=int, default=10)
a = ap.parse_args()
for so in a.lib:
    torch.ops.load_library(so)
bc.JITTER = 0.02
dev = torch.device("cuda:0")
xyz, Z = bc.water_system(a.n)
r = bc.Runner("m", a.model, dev); r.prepare(xyz, Z, a.walkers)
for _ in range(12): r.call()
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(a.iters): r.call()
torch.cuda.synchronize(); wall = (time.perf_counter() - t) / a.iters * 1e3
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(a.iters): r.call()
    torch.cuda.synchronize()
ev = p.key_averages()
gpu = sum(e.self_device_time_total for e in ev) / a.iters / 1e3
nk = sum(e.count for e in ev if e.self_device_time_total > 0 and e.device_type.name == "CUDA") / a.iters
print(f"N={a.n} W={a.walkers}: wall {wall:.2f} ms/call, GPU busy {gpu:.2f} ms/call, ~{nk:.0f} kernels/call")
print(ev.table(sort_by="self_cuda_time_total", row_limit=a.rows, max_name_column_width=70))
print(ev.table(sort_by="self_cpu_time_total", row_limit=15, max_name_column_width=70))
