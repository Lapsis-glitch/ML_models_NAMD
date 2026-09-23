"""Build a scripted FastANI inner (scripts/opt/ani/fast_ani.py) for ANI-2x from torchani 2.7.9's pretrained
weights, check the weights equal the deployed inner's, and save it. Wrap afterwards with wrap.py.

usage: python build_fast.py OUT.pt [--variant precise|upstream] [--acc32] [--cache-species]
                            [--pbc-cuaev] [--ref models/compiled_ani2x.pt]
--pbc-cuaev sets the stock `.ani` submodule (the wrapper's periodic path) to strategy 'cuaev'
(torchani neighbour list + cuAEV half-nbrlist kernels) instead of 'pyaev'.
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--variant", default="precise")
ap.add_argument("--acc32", action="store_true")
ap.add_argument("--cache-species", action="store_true")
ap.add_argument("--pbc-cuaev", action="store_true")
ap.add_argument("--group-max-atoms", type=int, default=0)
ap.add_argument("--ref", default=os.path.join(HERE, "..", "..", "..", "models", "compiled_ani2x.pt"))
a = ap.parse_args()
import ani_env
ani_env.setup(a.variant)
import torch, torchani
from fast_ani import FastANI

m = torchani.models.ANI2x(periodic_table_index=False).eval()
for p in m.parameters():
    p.requires_grad_(False)
if a.pbc_cuaev:
    m.set_strategy("cuaev")
# same weights as the deployed inner?
ref = torch.jit.load(a.ref, map_location="cpu").ani.state_dict()
mine = m.state_dict()
common = [k for k in ref if k in mine]
bad = [k for k in common if not torch.equal(ref[k].float(), mine[k].float())]
print(f"weights vs {os.path.basename(a.ref)}: {len(common)} common tensors, {len(bad)} differ, "
      f"{len(set(ref) ^ set(mine))} keys only on one side")
assert not bad and len(common) > 100, bad[:5]
fast = FastANI(m, acc64=not a.acc32, cache_species=a.cache_species,
               group_max_atoms=a.group_max_atoms).eval()
s = torch.jit.script(fast)
s.save(a.out)
print("saved", a.out, "| acc64", not a.acc32, "| cache_species", a.cache_species, "| group_max_atoms", a.group_max_atoms, "| pbc strategy",
      m.aev_computer.strategy, "| cuAEV lib", ani_env.lib_path(a.variant))
