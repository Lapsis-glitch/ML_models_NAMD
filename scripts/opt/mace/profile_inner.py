"""torch.profiler breakdown of the (eager) wrapper around a compiled inner.
usage: flock scripts/opt/.gpu_bench.lock python profile_inner.py N inner.pt [rows]"""
import sys, torch
sys.path.insert(0, "scripts/opt"); sys.path.insert(0, ".")
import cuequivariance_torch  # noqa
from bench_common import water_system
from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper
from torch.profiler import profile, ProfilerActivity
n = int(sys.argv[1]); xyz, Z = water_system(n); dev = "cuda"
xyz, Z = xyz.to(dev), Z.to(dev)
pc = torch.zeros(0, 3, dtype=torch.float64, device=dev); pq = torch.zeros(0, dtype=torch.float64, device=dev)
cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)
w = MACE_TS_Wrapper(sys.argv[2], device=dev)
def step():
    c = xyz.clone().requires_grad_(True)
    e, f, q, v = w(c, Z, pc, pq, cell); f.cpu()
for _ in range(8): step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(5): step()
    torch.cuda.synchronize()
rows = int(sys.argv[3]) if len(sys.argv) > 3 else 25
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=rows, max_name_column_width=60))
