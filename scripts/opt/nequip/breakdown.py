"""Per-module synced forward timing + whole forward/backward split for the EAGER NequIP-OAM-L
(OEQ modifier applied, python OEQ imported -> analysis only, not a NAMD path).
usage: python breakdown.py N [--modifier enable_OpenEquivariance] [--tf32]"""
import argparse, glob, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import bench_common as bc
from src.edges import build_edges

ap = argparse.ArgumentParser(); ap.add_argument("n", type=int)
ap.add_argument("--modifier", action="append", default=None)
a = ap.parse_args()
mods = a.modifier if a.modifier is not None else ["enable_OpenEquivariance"]
from nequip.model.saved_models.load_utils import load_saved_model
from nequip.model.modify_utils import modify
from nequip.model.utils import _EAGER_MODEL_KEY
z = glob.glob(os.path.expanduser('~/.nequip/model_cache/*.nequip.zip'))[0]
m = load_saved_model(z, _EAGER_MODEL_KEY, "sole_model")
m = modify(m, [{"modifier": x} for x in mods]).cuda().eval()
dev = torch.device("cuda")
xyz, Z = bc.water_system(a.n)
pos = xyz.to(dev).float()
ei, _, _ = build_edges(pos, 6.0)
E = ei.size(1); N = pos.size(0)
keys = ei[0] * N + ei[1]; rk = ei[1] * N + ei[0]
sk, si = torch.sort(keys); perm = si[torch.searchsorted(sk, rk)]
tmap = {1: 0, 8: 7}
at = torch.tensor([tmap[int(z)] for z in Z], device=dev)
def data():
    return {"pos": pos.clone().requires_grad_(False), "edge_index": ei, "atom_types": at,
            "edge_cell_shift": torch.zeros(E, 3, device=dev), "edge_transpose_perm": perm,
            "cell": torch.zeros(1, 3, 3, device=dev), "batch": torch.zeros(N, dtype=torch.long, device=dev),
            "num_atoms": torch.tensor([N], device=dev)}
print(f"N={N} E={E} ({E/N:.1f}/atom)")
times = {}
def pre(name):
    def f(mod, inp):
        torch.cuda.synchronize(); times.setdefault(name, []).append(-time.perf_counter())
    return f
def post(name):
    def f(mod, inp, out):
        torch.cuda.synchronize(); times[name][-1] += time.perf_counter()
    return f
for name, sub in m.named_modules():
    depth = name.count(".")
    if name and (depth <= 4 or name.endswith(("tp_scatter", "edge_mlp", "sc", "linear_1", "linear_2"))):
        sub.register_forward_pre_hook(pre(name)); sub.register_forward_hook(post(name))
for _ in range(3): m(data())
times.clear()
for _ in range(5): m(data())
rows = sorted(((sum(v) / len(v) * 1e3, k) for k, v in times.items() if not k.endswith("func") and k.count(".") >= 2), reverse=True)
for t, k in rows[:40]:
    print(f"{t:9.3f} ms  {k}")
