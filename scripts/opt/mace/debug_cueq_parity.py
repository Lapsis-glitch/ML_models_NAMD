"""Eager (unscripted) parity: e3nn MACE-OFF vs cuEq-converted, raw MACE input dict.
usage: python debug_cueq_parity.py [reduced_cg 0|1]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, "scripts/opt"); sys.path.insert(0, ".")
import torch
from rebuild import rebuild
from bench_common import water_system
from src.edges import build_edges
dev = "cuda"
xyz, Z = water_system(30); xyz, Z = xyz.to(dev), Z.to(dev)
def data(m):
    an = m.atomic_numbers.to(dev)
    ei, _, _ = build_edges(xyz.float(), 5.0)
    E = ei.size(1)
    return {"positions": xyz.clone(), "node_attrs": (Z[:, None] == an[None]).double(),
            "edge_index": ei, "shifts": torch.zeros(E, 3, dtype=torch.float64, device=dev),
            "unit_shifts": torch.zeros(E, 3, dtype=torch.float64, device=dev),
            "cell": torch.zeros(3, 3, dtype=torch.float64, device=dev),
            "batch": torch.zeros(30, dtype=torch.long, device=dev),
            "ptr": torch.tensor([0, 30], device=dev)}
m = rebuild("models/opt/mace_off23_medium_state.pt", device=dev)
o = m(data(m), compute_force=True)
print("e3nn E", o["energy"].item())
mr = rebuild("models/opt/mace_off23_medium_state.pt", device=dev, rebase=True)
orr = mr(data(mr), compute_force=True)
print("e3nn rebased-U dE", (orr["energy"] - o["energy"]).abs().item(), "dF", (orr["forces"] - o["forces"]).abs().max().item())
for fus in (False, True):
    t = rebuild("models/opt/mace_off23_medium_state.pt", cueq=True, device=dev, conv_fusion=fus)
    ot = t(data(t), compute_force=True)
    print("cueq fusion", fus, "E", ot["energy"].item(), "dE", (ot["energy"] - o["energy"]).abs().item(),
          "dF", (ot["forces"] - o["forces"]).abs().max().item())
    # per-block check of node embedding / first interaction
