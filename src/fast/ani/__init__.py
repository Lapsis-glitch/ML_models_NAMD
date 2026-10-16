"""Optimised TorchANI inner: cuAEV kernels + fused per-element ensemble (``fast.py``).

Works for torchani's pretrained ANI-1x, ANI-1ccx and ANI-2x.

    python -m src.fast.ani OUT [--variant ani2x] [--acc32] [--no-cache-species] ...

exposes the ablation switches; ``python -m src.compile_torchani --fast`` is the
recommended build.
"""
import argparse
import os
import sys
import types

import torch

from .. import report_parity, water_box

HARTREE_TO_KCAL = 627.5094740631


def setup_cuaev(lib: str) -> None:
    """Make torchani use the native cuAEV library instead of its own (usually
    not installed) extension, so the scripted model calls ``torch.ops.cuaev``.
    Must run before ``import torchani``."""
    os.environ.setdefault("TORCHANI_NO_WARN_EXTENSIONS", "1")
    if "torchani" in sys.modules:
        raise RuntimeError("setup_cuaev() must run before `import torchani`")
    torch.ops.load_library(lib)
    stub = types.ModuleType("torchani.csrc")
    stub.CUAEV_IS_INSTALLED = True
    stub.MNP_IS_INSTALLED = False
    stub.CLIST_IS_INSTALLED = False
    stub.__all__ = ["CUAEV_IS_INSTALLED", "MNP_IS_INSTALLED", "CLIST_IS_INSTALLED"]
    sys.modules["torchani.csrc"] = stub


def _check(fast, stock, species_count, n=300):
    """FastANI vs the stock model evaluated in float64, on water with the
    species cycled through every element of the model."""
    dev = torch.device("cuda")
    species = (torch.arange(n) % species_count)[None].to(dev)
    xyz = water_box(n)[None]
    ref = stock.double().to(dev)
    c64 = xyz.to(dev).requires_grad_(True)
    e_ref = ref((species, c64)).energies
    f_ref = -torch.autograd.grad(e_ref.sum(), c64)[0]
    c32 = xyz.to(dev).float().requires_grad_(True)
    for _ in range(3):  # profiling executor: the 3rd call runs the optimised graph
        e = fast(species, c32)[1]
    f = -torch.autograd.grad(e.sum(), c32)[0]
    k = HARTREE_TO_KCAL
    err = report_parity("scripted FastANI vs fp64", n, e_ref.detach() * k, e.detach().double() * k,
                        f_ref * k, f.double() * k, "kcal/mol")
    dE = ((e.double() - e_ref) * k).abs().max().item()
    if err >= 1e-4 or dE >= 0.05:
        raise RuntimeError(f"FastANI differs from the stock model (dE {dE:.1e} kcal/mol, relative dF {err:.1e})")


def build_fast_ani(variant: str, out: str, cuaev_lib: str, acc64: bool = True,
                   cache_species: bool = True, group_max_atoms: int = 1024,
                   pbc_cuaev: bool = False, check: bool = True) -> list:
    """Build the optimised ANI inner ``out`` for a pretrained ``variant``.

    Returns the element list (atomic numbers in the model's species order).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("the optimised ANI build needs a CUDA GPU")
    setup_cuaev(cuaev_lib)
    import torchani
    from ...compile_torchani import _VARIANTS
    from .fast import FastANI

    cls_name, element_list = _VARIANTS[variant]
    m = getattr(torchani.models, cls_name)(periodic_table_index=False).eval()
    stock = getattr(torchani.models, cls_name)(periodic_table_index=False).eval()
    stock.set_strategy("pyaev")
    for p in list(m.parameters()) + list(stock.parameters()):
        p.requires_grad_(False)
    if pbc_cuaev:
        m.set_strategy("cuaev")
    fast = FastANI(m, acc64=acc64, cache_species=cache_species,
                   group_max_atoms=group_max_atoms).eval()
    s = torch.jit.script(fast)
    if check:
        _check(s.to("cuda"), stock, len(element_list))
    s.to("cpu").save(out)
    print(f"[TorchANI] optimised {cls_name} inner  ->  {out}")
    print(f"  elements (Z order): {element_list}")
    return element_list


def main(argv=None):
    from ..native import ensure
    ap = argparse.ArgumentParser(description="Build an optimised ANI inner (ablation switches).")
    ap.add_argument("out")
    ap.add_argument("--variant", default="ani2x", choices=["ani1x", "ani1ccx", "ani2x"])
    ap.add_argument("--lib", default=None, help="cuAEV library (default: build/use src/fast/native/cuaev)")
    ap.add_argument("--acc32", action="store_true", help="stock fp32 energy sum")
    ap.add_argument("--no-cache-species", action="store_true")
    ap.add_argument("--group-max-atoms", type=int, default=1024)
    ap.add_argument("--pbc-cuaev", action="store_true")
    ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args(argv)
    lib = a.lib or ensure("cuaev")[0]
    build_fast_ani(a.variant, a.out, lib, acc64=not a.acc32, cache_species=not a.no_cache_species,
                   group_max_atoms=a.group_max_atoms, pbc_cuaev=a.pbc_cuaev, check=not a.no_check)
