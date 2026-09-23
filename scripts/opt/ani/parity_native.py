"""Parity of wrapped ANI artifacts in a FRESH process that never imports torchani (cuAEV comes only from
the native lib), called like the NAMD shim (fp64 coords requires_grad on cuda, int64 Z, empty pcs, cell).
Reports max|dE| (kcal/mol), max|dF| (kcal/mol/A), max|dVirial| vs the FIRST model and vs the fp64
torchani reference (ref_fp64.py).
usage: python parity_native.py --lib SO --ref REF.pt name=model.pt [name=model.pt ...]"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
ap = argparse.ArgumentParser()
ap.add_argument("--lib", action="append", default=[])
ap.add_argument("--ref", required=True)
ap.add_argument("models", nargs="+")
a = ap.parse_args()
for so in a.lib:
    torch.ops.load_library(so)
import geoms
dev = torch.device("cuda")
mods = [(s.split("=", 1)[0], torch.jit.load(s.split("=", 1)[1], map_location=dev)) for s in a.models]
ref = torch.load(a.ref)
assert not any(k.startswith("torchani") for k in sys.modules), "torchani must not be imported here"
pc_x = torch.zeros(0, 3, dtype=torch.float64, device=dev); pc_q = torch.zeros(0, dtype=torch.float64, device=dev)


def run(mod, xs, Z, cell):
    W = len(xs)
    if W == 1:
        c = xs[0].to(dev).clone().requires_grad_(True)
        E, F, _, V = mod(c, Z.to(dev), pc_x, pc_q, cell.to(dev)[None])
        return E.reshape(1).cpu(), F.cpu(), V.reshape(1, 3, 3).cpu()
    n = xs[0].shape[0]
    c = torch.cat(xs, 0).to(dev).clone().requires_grad_(True)
    ZZ = Z.repeat(W).to(dev)
    batch = torch.arange(W, device=dev).repeat_interleave(n)
    ptr = torch.arange(W + 1, device=dev) * n
    E, F, _, V = mod.forward_batch(c, ZZ, batch, ptr, pc_x, pc_q, cell.to(dev)[None].repeat(W, 1, 1))
    return E.reshape(W).cpu(), F.cpu(), V.reshape(W, 3, 3).cpu()


print(f"{'case':>12} {'model':>16} {'dE vs '+mods[0][0]:>14} {'dF vs '+mods[0][0]:>14} {'dV vs '+mods[0][0]:>14}"
      f" {'dE vs fp64':>11} {'dF vs fp64':>11} {'dV vs fp64':>11}")
worst = {}
for (n, W, pbc) in geoms.CASES:
    acc = {}
    for g in range(geoms.N_GEOM):
        xs, Z, cell = geoms.case_inputs(n, W, pbc, g)
        rE, rF, rV = ref.get(geoms.key(n, W, pbc, g), (None, None, None))
        outs = [(name, run(m, xs, Z, cell)) for name, m in mods]
        E0, F0, V0 = outs[0][1]
        for name, (E, F, V) in outs:
            d = acc.setdefault(name, [0.0] * 6)
            vals = [(E - E0).abs().max(), (F - F0).abs().max(), (V - V0).abs().max()]
            vals += [(E - rE).abs().max(), (F - rF).abs().max(), (V - rV).abs().max()] if rE is not None \
                else [float("nan")] * 3
            for i, v in enumerate(vals):
                v = float(v.detach()) if torch.is_tensor(v) else float(v)
                d[i] = v if v != v else max(d[i], v)
    tag = f"{n}x{W}{' pbc' if pbc else ''}"
    for name, d in acc.items():
        print(f"{tag:>12} {name:>16} {d[0]:14.2e} {d[1]:14.2e} {d[2]:14.2e} {d[3]:11.2e} {d[4]:11.2e} {d[5]:11.2e}")
        worst[name] = [max(x, y) if y == y else x for x, y in zip(worst.get(name, [0] * 6), d)]
print("worst over all cases:")
for name, d in worst.items():
    print(f"{name:>16}: dE {d[0]:.2e} dF {d[1]:.2e} dV {d[2]:.2e} | vs fp64: dE {d[3]:.2e} dF {d[4]:.2e} dV {d[5]:.2e}")
