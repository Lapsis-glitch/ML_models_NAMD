"""Parity of cuEq artifacts run through the NATIVE uniform_1d op library vs a stock (e3nn) baseline.

Runs in a fresh process that never imports cuequivariance_torch / cuequivariance_ops_torch
(asserted), loading only libcue_ops.so + libcueq_uniform1d_native.so, like the NAMD shim.
usage: python parity_native.py BASE.pt ART.pt [ART2.pt ...] [--n 30,300] [--walkers 1,4]
"""
import argparse, os, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # scripts/opt for bench_common.water_system
SP = "/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages"
ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("arts", nargs="+")
ap.add_argument("--n", default="30,300"); ap.add_argument("--walkers", default="1,4")
ap.add_argument("--jitter", type=float, default=0.02)
a = ap.parse_args()
torch.ops.load_library(f"{SP}/cuequivariance_ops/lib/libcue_ops.so")
torch.ops.load_library(os.path.join(HERE, "libcueq_uniform1d_native.so"))
from bench_common import water_system
dev = "cuda"
base = torch.jit.load(a.base, map_location=dev)
arts = {p: torch.jit.load(p, map_location=dev) for p in a.arts}
assert not any(m.startswith("cuequivariance") for m in sys.modules), "cuEq python got imported!"
empty = torch.zeros(0, 3, dtype=torch.float64, device=dev), torch.zeros(0, dtype=torch.float64, device=dev)
cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)


def run(m, xyz, Z, W):
    x = xyz.clone().requires_grad_(True)
    if W == 1:
        try:
            out = m.forward(x, Z, empty[0], empty[1], cell)
        except Exception:
            out = m.forward(x, Z, empty[0], empty[1])
    else:
        n = xyz.shape[0] // W
        batch = torch.arange(W, device=dev).repeat_interleave(n)
        ptr = torch.arange(W + 1, device=dev) * n
        try:
            out = m.forward_batch(x, Z, batch, ptr, empty[0], empty[1], cell.expand(W, 3, 3).contiguous())
        except Exception:
            out = m.forward_batch(x, Z, batch, ptr, empty[0], empty[1])
    return out[0].detach().double(), out[1].detach().double()


ok = True
for n in [int(v) for v in a.n.split(",")]:
    for W in [int(v) for v in a.walkers.split(",")]:
        xyz0, Z0 = water_system(n)
        g = torch.Generator().manual_seed(7)
        xs = [xyz0 + a.jitter * torch.randn(xyz0.shape, generator=g, dtype=xyz0.dtype) for _ in range(W)]
        xyz = torch.cat(xs).to(dev); Z = Z0.repeat(W).to(dev)
        for _ in range(3):  # profiling executor: 3rd call runs the optimised graph
            Eb, Fb = run(base, xyz, Z, W)
        for p, m in arts.items():
            for _ in range(3):
                E, F = run(m, xyz, Z, W)
            dE = (E - Eb).abs().max().item(); dF = (F - Fb).abs().max().item()
            good = dF < 1e-8 and dE < 1e-6
            ok &= good
            print(f"N={n:5d} W={W} {os.path.basename(p):28s} E={Eb.flatten()[0].item():.6f} dE={dE:.2e} dF={dF:.2e} "
                  f"{'OK' if good else 'FAIL'}")
print("cuEq python modules imported:", [m for m in sys.modules if m.startswith("cuequivariance")])
print("ALL OK" if ok else "PARITY FAILURES")
