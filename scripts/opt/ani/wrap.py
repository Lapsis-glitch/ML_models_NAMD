"""Wrap a compiled TorchANI inner with src/wrappers/wrap_torchani.TorchANI_Wrapper and export it
(torch.jit.script) for NAMD.

usage: python scripts/opt/ani/wrap.py INNER OUT [--lib SO ...] [key=value ...]

--lib loads a custom-op library (e.g. the native cuAEV lib) before torch.jit.load of the inner.
Extra key=value pairs are passed to the wrapper constructor (opt-in flags).
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.environ.setdefault("TORCHANI_NO_WARN_EXTENSIONS", "1")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("inner")
ap.add_argument("out")
ap.add_argument("--lib", action="append", default=[])
ap.add_argument("--device", default="cpu")
ap.add_argument("kw", nargs="*")
a = ap.parse_args()
for so in a.lib:
    torch.ops.load_library(so)


def _val(v):
    if v.startswith("["):
        return [int(x) for x in v.strip("[]").split(",")]
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
from src.wrappers.wrap_torchani import TorchANI_Wrapper
from src.export import export_wrapped
w = TorchANI_Wrapper(a.inner, device=a.device, **kw).eval()
export_wrapped(w, a.out, model_type=f"TorchANI {kw}")
