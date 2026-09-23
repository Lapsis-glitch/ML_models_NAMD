"""torch.profiler view of a wrapped artifact (called like the NAMD shim does).
usage: flock scripts/opt/.gpu_bench.lock python scripts/opt/mace/prof.py MODEL.pt N [W] [--rows 25] [--sort cuda|cpu]"""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import torch
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("n", type=int); ap.add_argument("w", type=int, nargs="?", default=1)
ap.add_argument("--rows", type=int, default=25); ap.add_argument("--sort", default="cuda")
ap.add_argument("--cueq", action="store_true")
ap.add_argument("--native", action="store_true", help="load libcue_ops + native uniform_1d op instead of cuEq python")
a = ap.parse_args()
if a.cueq:
    import cuequivariance_torch  # noqa
if a.native:
    torch.ops.load_library("/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/cuequivariance_ops/lib/libcue_ops.so")
    torch.ops.load_library(os.path.join(HERE, "cueq_native", "libcueq_uniform1d_native.so"))
from bench_common import water_system
dev = torch.device("cuda")
xyz, Z = water_system(a.n)
m = torch.jit.load(a.model, map_location=dev)
pc = torch.zeros(0, 3, dtype=torch.float64, device=dev); pq = torch.zeros(0, dtype=torch.float64, device=dev)
X, ZZ = xyz.to(dev), Z.to(dev)
g = torch.Generator(device=dev).manual_seed(0)


def call():
    c = (X + 0.02 * torch.randn(X.shape, generator=g, device=dev, dtype=X.dtype)).requires_grad_(True)
    out = m(c, ZZ, pc, pq, torch.zeros(1, 3, 3, dtype=torch.float64, device=dev))
    return out[0].cpu(), out[1].cpu()


for _ in range(8):
    call()
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(10):
    call()
torch.cuda.synchronize()
print(f"wall {(time.perf_counter() - t) / 10 * 1e3:.2f} ms/call")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
key = "self_cuda_time_total" if a.sort == "cuda" else "self_cpu_time_total"
print(p.key_averages().table(sort_by=key, row_limit=a.rows, max_name_column_width=60))
