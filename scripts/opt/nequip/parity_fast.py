"""FastNequIP parity (native op lib only, openequivariance python never imported): nequip_fast_oeq.pt vs
nequip_oeq.pt (same OEQ kernels, stock graph) and vs nequip_baseline.pt, incl. PERIODIC cells + virials.
Floor = nequip_oeq.pt run twice (atomic-scatter nondeterminism)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import bench_common as bc
torch.ops.load_library(str(bc.REPO / "scripts/opt/nequip/oeq_native/liboeq_native.so"))
dev = torch.device("cuda:0")
L = {k: torch.jit.load(str(bc.REPO / f"models/opt/nequip_{k}.pt"), map_location=dev) for k in ("baseline", "oeq", "fast_oeq")}
assert not any(m.startswith("openequivariance") for m in sys.modules)
pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev); pcq = torch.zeros(0, dtype=torch.float64, device=dev)
g = torch.Generator(device=dev).manual_seed(7)

def run(m, xyz, Z, W, cell):
    n = xyz.shape[0]
    c = torch.cat([xyz + 0.01 * k for k in range(W)]).detach().requires_grad_(True)
    if W == 1:
        return m(c, Z, pcx, pcq, cell.reshape(1, 3, 3))
    batch = torch.arange(W, device=dev).repeat_interleave(n); ptr = torch.arange(W + 1, device=dev) * n
    return m.forward_batch(c, Z.repeat(W), batch, ptr, pcx, pcq, cell.reshape(1, 3, 3).expand(W, 3, 3).contiguous())

def cmp(a, b):
    return [(x - y).abs().max().item() for x, y in ((a[0], b[0]), (a[1], b[1]), (a[3], b[3]))]

worst = 0.0
for n, W, periodic in ((30, 1, False), (300, 1, False), (300, 4, False), (300, 1, True), (300, 2, True), (900, 1, True)):
    xyz, Z = bc.water_system(n); Z = Z.to(dev)
    # periodic test box: droplet extent + 3 A, so cross-boundary edges exist (3-6 A) without overlaps
    Lbox = float((xyz.max(0).values - xyz.min(0).values).max()) + 3.0
    cell = torch.eye(3, dtype=torch.float64, device=dev) * (Lbox if periodic else 0.0)
    for rep in range(3):
        x = (xyz.to(dev) + 0.02 * torch.randn(xyz.shape, generator=g, device=dev, dtype=torch.float64))
        if periodic:
            x = x - x.min(0).values
        o = {k: run(m, x, Z, W, cell) for k, m in L.items()}
        o2 = run(L["oeq"], x, Z, W, cell)
        fl = cmp(o["oeq"], o2); fo = cmp(o["fast_oeq"], o["oeq"]); fb = cmp(o["fast_oeq"], o["baseline"]); ob = cmp(o["oeq"], o["baseline"])
        if rep == 0:
            print(f"   (max|F| {o['oeq'][1].abs().max().item():.1f} kcal/mol/A, max|virial| {o['oeq'][3].abs().max().item():.1f} kcal/mol)")
            print(f"N={n:4d} W={W} {'PBC' if periodic else 'open'}  [dE, dF, dVirial] "
                  f"fast-oeq {fo[0]:.1e} {fo[1]:.1e} {fo[2]:.1e} | oeq-oeq(floor) {fl[0]:.1e} {fl[1]:.1e} {fl[2]:.1e} | "
                  f"fast-base {fb[0]:.1e} {fb[1]:.1e} {fb[2]:.1e} | oeq-base {ob[0]:.1e} {ob[1]:.1e} {ob[2]:.1e}")
        worst = max(worst, fo[1])
print("max |dF| fast vs oeq over all cells:", f"{worst:.2e}", "(compare with the oeq-oeq floor column)")
