"""Wrap a compiled MACE inner with src/wrappers/wrap_compiled_mace.MACE_TS_Wrapper
and export it (torch.jit.script) for NAMD.

usage: python scripts/opt/mace/wrap.py INNER OUT [--cueq] [wrapper kwargs as key=value ...]

--cueq imports cuequivariance_torch first (needed to torch.jit.load a cuEq inner).
Extra key=value pairs are passed to MACE_TS_Wrapper (ints/floats/bools parsed).
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("inner")
ap.add_argument("out")
ap.add_argument("--cueq", action="store_true")
ap.add_argument("kw", nargs="*")
a = ap.parse_args()
if a.cueq:
    import cuequivariance_torch  # noqa: F401  registers the cuEq torch ops


def _val(v):
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


kw = {k: _val(v) for k, v in (x.split("=", 1) for x in a.kw)}
from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper
from src.export import export_wrapped
w = MACE_TS_Wrapper(a.inner, device="cpu", **kw).eval()
export_wrapped(w, a.out, model_type=f"MACE {kw}")
