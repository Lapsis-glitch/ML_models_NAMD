"""SevenNet parity in a fresh process that loads ONLY the native OEQ op library
(openequivariance Python is never imported), open and PERIODIC cells, W = 1 and
batched, energies / forces / virials.

usage: python scripts/opt/sevennet/parity.py [label ...]   (default: every models/opt/sevennet_*.pt
       artifact except the baselines; `*_d3` labels are compared with sevennet_baseline_d3.pt,
       the rest with sevennet_baseline.pt)

Floor = the reference run twice on the same input (fp32 atomic-scatter
nondeterminism).  A cell passes when dF <= max(10 x floor, 1e-3 kcal/mol/A) and
dE <= 1e-4 kcal/mol per atom.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch

import bench_common as bc

torch.ops.load_library(str(bc.REPO / "scripts/opt/nequip/oeq_native/liboeq_native.so"))
dev = torch.device("cuda:0")
M = bc.REPO / "models/opt"
labels = sys.argv[1:] or sorted(
    p.stem[len("sevennet_"):] for p in M.glob("sevennet_*.pt")
    if not p.stem.startswith(("sevennet_inner", "sevennet_baseline")))


def ref_of(k):
    return "baseline_d3" if k.endswith("_d3") else "baseline"


refs = sorted({ref_of(k) for k in labels})
L = {k: torch.jit.load(str(M / f"sevennet_{k}.pt"), map_location=dev) for k in [*refs, *labels]}
assert not any(m.startswith("openequivariance") for m in sys.modules), "openequivariance imported!"
pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev)
pcq = torch.zeros(0, dtype=torch.float64, device=dev)
g = torch.Generator(device=dev).manual_seed(7)


def run(m, xyz, Z, W, cell):
    n = xyz.shape[0]
    c = torch.cat([xyz + 0.01 * k for k in range(W)]).detach().requires_grad_(True)
    if W == 1:
        return m(c, Z, pcx, pcq, cell.reshape(1, 3, 3))
    batch = torch.arange(W, device=dev).repeat_interleave(n)
    ptr = torch.arange(W + 1, device=dev) * n
    return m.forward_batch(c, Z.repeat(W), batch, ptr, pcx, pcq,
                           cell.reshape(1, 3, 3).expand(W, 3, 3).contiguous())


def diff(a, b):
    return [(x - y).abs().max().item() for x, y in ((a[0], b[0]), (a[1], b[1]), (a[3], b[3]))]


ok = True
cells = ((30, 1, False), (300, 1, False), (300, 4, False), (3000, 1, False),
         (300, 1, True), (300, 2, True), (900, 1, True))
for n, W, periodic in cells:
    xyz, Z = bc.water_system(n)
    Z = Z.to(dev)
    # periodic test box: droplet extent + 3 A, so cross-boundary edges exist
    Lbox = float((xyz.max(0).values - xyz.min(0).values).max()) + 3.0
    cell = torch.eye(3, dtype=torch.float64, device=dev) * (Lbox if periodic else 0.0)
    worst = {k: [0.0, 0.0, 0.0] for k in L}
    for rep in range(3):
        x = xyz.to(dev) + 0.02 * torch.randn(xyz.shape, generator=g, device=dev, dtype=torch.float64)
        if periodic:
            x = x - x.min(0).values
        ref = {r: run(L[r], x, Z, W, cell) for r in refs}
        for k, m in L.items():
            d = diff(run(m, x, Z, W, cell), ref[ref_of(k)])  # for a reference this is the floor
            worst[k] = [max(a, b) for a, b in zip(worst[k], d)]
    tag = f"N={n:4d} W={W} {'PBC ' if periodic else 'open'}"
    r0 = ref[refs[0]]
    print(f"{tag}  max|F| {r0[1].abs().max().item():.0f}  max|V| {r0[3].abs().max().item():.0f}  floor "
          + "  ".join(f"[{r}] dE {worst[r][0]:.1e} dF {worst[r][1]:.1e} dV {worst[r][2]:.1e}" for r in refs))
    for k in labels:
        de, df, dv = worst[k]
        fl = worst[ref_of(k)]
        good = df <= max(10 * fl[1], 1e-3) and de <= 1e-4 * n * W
        ok &= good
        print(f"    {k:14s} dE {de:.1e} kcal/mol  dF {df:.1e} kcal/mol/A  dV {dv:.1e} kcal/mol"
              f"  {'OK' if good else 'FAIL'}")
assert not any(m.startswith("openequivariance") for m in sys.modules)
print("ALL OK" if ok else "PARITY FAIL")
sys.exit(0 if ok else 1)
