"""Import helper: make torchani see the NATIVE cuAEV lib (scripts/opt/ani/cuaev_native/libcuaev_native_*.so)
instead of its own (not shipped) cuaev.so, so AEVComputer exports its cuAEV methods when scripted.

    import ani_env; ani_env.setup(variant="precise")   # BEFORE importing torchani
    import torchani

It loads the .so with torch.ops.load_library and pre-registers a stub `torchani.csrc` module with
CUAEV_IS_INSTALLED=True (MNP / cell_list stay False). Build-time / benchmark helper only.
"""
import os, sys, types
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def lib_path(variant: str = "precise") -> str:
    return os.path.join(HERE, "cuaev_native", f"libcuaev_native_{variant}.so")


def setup(variant: str = "precise", cuaev: bool = True) -> None:
    os.environ.setdefault("TORCHANI_NO_WARN_EXTENSIONS", "1")
    if "torchani" in sys.modules:
        raise RuntimeError("ani_env.setup() must run before `import torchani`")
    if cuaev:
        torch.ops.load_library(lib_path(variant))
    stub = types.ModuleType("torchani.csrc")
    stub.CUAEV_IS_INSTALLED = bool(cuaev)
    stub.MNP_IS_INSTALLED = False
    stub.CLIST_IS_INSTALLED = False
    stub.__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED", "CLIST_IS_INSTALLED"]
    sys.modules["torchani.csrc"] = stub
