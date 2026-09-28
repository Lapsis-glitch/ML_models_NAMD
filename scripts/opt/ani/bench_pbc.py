"""Interleaved timing of wrapped ANI artifacts on the PERIODIC path (bench_common only sends a zero cell).
Water box of N atoms in a cubic cell (extent + 3 A, same as geoms.py), coords jittered per call, called like
the shim (fp64 coords requires_grad, int64 Z, empty pcs, [1,3,3] cell), output copied to host. Holds the GPU lock.
usage: python bench_pbc.py --lib SO N name=model.pt [...]"""
import argparse, fcntl, os, statistics, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import torch
ap = argparse.ArgumentParser(); ap.add_argument("--lib", "--extra-lib", dest="lib", action="append", default=[]); ap.add_argument("n", type=int)
ap.add_argument("--iters", type=int, default=20); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("models", nargs="+"); a = ap.parse_args()
for so in a.lib: torch.ops.load_library(so)
import bench_common as bc
fh = open(os.path.join(HERE, "..", ".gpu_bench.lock"), "w"); fcntl.flock(fh, fcntl.LOCK_EX)
dev = torch.device("cuda")
xyz, Z = bc.water_system(a.n)
L = float((xyz.max(0).values - xyz.min(0).values).max()) + 3.0
cell = (torch.eye(3, dtype=torch.float64) * L).to(dev)[None]
g = torch.Generator(device=dev).manual_seed(1)
pool = [xyz.to(dev) + 0.02 * torch.randn(xyz.shape, generator=g, device=dev, dtype=torch.float64) for _ in range(32)]
Zd = Z.to(dev); pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev); pcq = torch.zeros(0, dtype=torch.float64, device=dev)
mods = [(s.split("=")[0], torch.jit.load(s.split("=")[1], map_location=dev)) for s in a.models]
times = {n: [] for n, _ in mods}; k = [0]
def call(m):
    c = pool[k[0] % 32].clone().requires_grad_(True); k[0] += 1
    E, F, q, V = m(c, Zd, pcx, pcq, cell); return E.cpu(), F.cpu(), V.cpu()
for n, m in mods:
    for _ in range(8): call(m)
for r in range(a.rounds):
    for n, m in mods:
        for _ in range(a.iters):
            torch.cuda.synchronize(); t = time.perf_counter(); call(m); times[n].append((time.perf_counter() - t) * 1e3)
ref = None
for n, m in mods:
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    c = pool[0].clone().requires_grad_(True); E, F, _, V = m(c, Zd, pcx, pcq, cell)
    if ref is None: ref = (E, F, V)
    print(f"water{a.n} PBC L={L:.1f}  {n:>14}  median {statistics.median(times[n]):8.3f} ms  "
          f"dE {float((E-ref[0]).abs().max()):.2e}  dF {float((F-ref[1]).abs().max()):.2e}  dV {float((V-ref[2]).abs().max()):.2e}"
          f"  peak(1 call) {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
