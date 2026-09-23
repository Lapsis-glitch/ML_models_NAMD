"""Compile MACE-OFF23-medium inner models in the allegro env.

  e3nn : rebuilt e3nn model (mace 0.3.15 / e3nn 0.6 codegen), same weights
  cueq : same model with cuEquivariance kernels (layout ir_mul), no conv fusion
  cueqf: cueq + fused gather/TP/scatter conv kernel
  (both cueq variants get the TorchScript fixes in cueq_fusion.py)

usage: python scripts/opt/mace/compile_inner.py {e3nn|cueq|cueqf} OUT [--dtype float64|float32]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from e3nn.util import jit as ej
from rebuild import rebuild

STATE = "models/opt/mace_off23_medium_state.pt"

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["e3nn", "cueq", "cueqf"])
ap.add_argument("out")
ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
ap.add_argument("--state", default=STATE)
a = ap.parse_args()
dt = getattr(torch, a.dtype)
if a.kind == "e3nn":
    m = rebuild(a.state, dt)
else:
    # cuEq modules are built on CUDA (kernel selection); weights are moved
    # back to CPU before saving so torch.jit.load(map_location=...) works.
    from cueq_fusion import make_scriptable
    m = rebuild(a.state, dt, cueq=True, device="cuda", conv_fusion=(a.kind == "cueqf"))
    make_scriptable(m)
s = ej.compile(m)
s.save(a.out)
print("saved", a.out, a.kind, a.dtype)
