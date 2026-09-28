"""Wrap a compiled NequIP inner with src/wrappers/wrap_compiled_nequip.NequIP_Allegro_Wrapper
and export it (torch.jit.script) for NAMD.

usage: python scripts/opt/nequip/wrap.py INNER OUT [--lib SO ...] [--py-oeq] [key=value ...]

--lib loads a custom-op library (e.g. the native OEQ op lib) before torch.jit.load of the inner.
--py-oeq imports openequivariance instead (python-registered autograd; benchmark only).
--import MOD imports a python module first (e.g. cuequivariance_torch; benchmark only).
Extra key=value pairs are passed to the wrapper constructor.
"""
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import torch

ap = argparse.ArgumentParser()
ap.add_argument("inner")
ap.add_argument("out")
ap.add_argument("--lib", action="append", default=[])
ap.add_argument("--py-oeq", action="store_true")
ap.add_argument("--import", dest="imports", action="append", default=[])
ap.add_argument("--device", default="cpu")
ap.add_argument("kw", nargs="*")
a = ap.parse_args()
if a.lib and (a.py_oeq or "openequivariance" in a.imports):
    raise SystemExit("do not load liboeq_native.so and import openequivariance in one process "
                     "(duplicate TORCH_LIBRARY(libtorch_tp_jit))")
for so in a.lib:
    torch.ops.load_library(so)
if a.py_oeq:
    import openequivariance  # noqa: F401
import importlib
for _m in a.imports:
    importlib.import_module(_m)


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
from src.wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper
from src.export import export_wrapped
w = NequIP_Allegro_Wrapper(a.inner, device=a.device, **kw).eval()
export_wrapped(w, a.out, model_type=f"NequIP {kw}")
