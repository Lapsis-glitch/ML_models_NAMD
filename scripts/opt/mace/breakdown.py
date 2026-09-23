"""Per-submodule GPU time breakdown of MACE-OFF (eager, synced hooks; fwd only
per module, backward as one block).  usage: breakdown.py N {e3nn|cueqf} [dtype]"""
import sys, os, time, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "scripts/opt"); sys.path.insert(0, ".")
import torch
from rebuild import rebuild
from bench_common import water_system
from src.edges import build_edges
n = int(sys.argv[1]); kind = sys.argv[2]; dt = getattr(torch, sys.argv[3] if len(sys.argv) > 3 else "float64")
dev = "cuda"
xyz, Z = water_system(n); xyz, Z = xyz.to(dev), Z.to(dev)
m = rebuild("models/opt/mace_off23_medium_state.pt", dtype=dt, device=dev, cueq=(kind != "e3nn"),
            conv_fusion=(kind == "cueqf"))
acc = collections.defaultdict(float)
def pre(name):
    def f(mod, inp):
        torch.cuda.synchronize(); mod._t = time.perf_counter()
    return f
def post(name):
    def f(mod, inp, out):
        torch.cuda.synchronize(); acc[name] += time.perf_counter() - mod._t
    return f
for name, mod in m.named_modules():
    if name.count(".") <= 2 and name and not name.endswith("interactions") and not name.endswith("products") and not name.endswith("readouts"):
        mod.register_forward_pre_hook(pre(name)); mod.register_forward_hook(post(name))
ei, _, _ = build_edges(xyz.float(), 5.0); E = ei.size(1)
def data():
    return {"positions": xyz.to(dt).clone(), "node_attrs": (Z[:, None] == m.atomic_numbers.to(dev)[None]).to(dt),
            "edge_index": ei, "shifts": torch.zeros(E, 3, dtype=dt, device=dev),
            "unit_shifts": torch.zeros(E, 3, dtype=dt, device=dev), "cell": torch.zeros(3, 3, dtype=dt, device=dev),
            "batch": torch.zeros(n, dtype=torch.long, device=dev), "ptr": torch.tensor([0, n], device=dev)}
K = 5
for it in range(K + 3):
    if it == 3: acc.clear(); tb = 0.0; tt = 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    o = m(data(), compute_force=True)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    if it >= 3: tt += t1 - t0
print(f"N={n} E={E} kind={kind} dtype={dt}  total fwd+bwd {tt/K*1e3:.2f} ms/step (hooks sync => upper bound)")
tot = sum(v for k, v in acc.items() if k.count(".") == 0)
for k, v in sorted(acc.items(), key=lambda kv: -kv[1])[:30]:
    print(f"  {k:55s} {v/K*1e3:8.3f} ms")
