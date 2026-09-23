"""Build + verify a scripted FastMACE inner (see fast_mace.py).

usage: python scripts/opt/mace/build_fast.py {e3nn|cueq|cueqf} OUT [--dtype float64|float32]
           [--no-half] [--no-species-skip] [--no-plain-linear] [--cache-species] [--check N]

Checks (GPU): eager FastMACE vs the stock rebuilt model, then the scripted
artifact vs the stock model, on water N (default 300) with a random jitter.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
import torch
from e3nn.util import jit as ej
from rebuild import rebuild
from fast_mace import FastMACE
from bench_common import water_system
from src.edges import build_edges

STATE = os.path.join(HERE, "..", "..", "..", "models", "opt", "mace_off23_medium_state.pt")

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["e3nn", "cueq", "cueqf"])
ap.add_argument("out")
ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
ap.add_argument("--no-half", action="store_true")
ap.add_argument("--no-species-skip", action="store_true")
ap.add_argument("--no-plain-linear", action="store_true")
ap.add_argument("--cache-species", action="store_true")
ap.add_argument("--check", type=int, default=300)
a = ap.parse_args()
dt = getattr(torch, a.dtype)
dev = "cuda"

if a.kind == "e3nn":
    m = rebuild(STATE, dt, device=dev)
else:
    import cuequivariance_torch  # noqa: F401
    from cueq_fusion import make_scriptable
    m = rebuild(STATE, dt, cueq=True, device=dev, conv_fusion=(a.kind == "cueqf"))
    make_scriptable(m)
fm = FastMACE(m, half_radial=not a.no_half, species_skip=not a.no_species_skip,
              plain_linear=not a.no_plain_linear, cache_species=a.cache_species).eval()


def data_for(n, jit_seed=0):
    xyz, Z = water_system(n)
    g = torch.Generator().manual_seed(jit_seed)
    xyz = (xyz + 0.02 * torch.randn(xyz.shape, generator=g, dtype=xyz.dtype)).to(dev)
    Z = Z.to(dev)
    ei, _, _ = build_edges(xyz.float(), float(m.r_max))
    E = ei.size(1)
    attrs = (Z[:, None] == m.atomic_numbers.to(dev)[None]).to(dt)
    return {"positions": xyz.to(dt).clone(), "node_attrs": attrs, "edge_index": ei,
            "shifts": torch.zeros(E, 3, dtype=dt, device=dev),
            "unit_shifts": torch.zeros(E, 3, dtype=dt, device=dev),
            "cell": torch.zeros(3, 3, dtype=dt, device=dev),
            "batch": torch.zeros(n, dtype=torch.long, device=dev),
            "ptr": torch.tensor([0, n], device=dev)}


def cmp(tag, f, ref_fn, n):
    d = data_for(n)
    r = ref_fn(dict(d), compute_force=True)
    o = f(dict(d), compute_force=True)
    dE = (o["energy"] - r["energy"]).abs().max().item()
    dF = (o["forces"] - r["forces"]).abs().max().item()
    print(f"[{tag}] N={n} E={r['energy'].item():.8f} eV  dE={dE:.2e} eV  dF={dF:.2e} eV/A  "
          f"|F|max={r['forces'].abs().max().item():.2f}")
    return dE, dF


tol_F = 1e-10 if dt == torch.float64 else 2e-4
dE, dF = cmp("eager", fm, m, a.check)
assert dF < tol_F, "eager FastMACE differs from stock model"
s = torch.jit.script(ej.compile(fm))
for n in (30, a.check):
    for _ in range(3):  # profiling executor: 3rd call runs the optimised graph
        dE, dF = cmp("scripted", s, m, n)
    assert dF < tol_F, "scripted FastMACE differs from stock model"
# move to CPU before saving so torch.jit.load(map_location=...) works anywhere
s = s.to("cpu")
s.save(a.out)
print("saved", a.out)
