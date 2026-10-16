"""Optimised NequIP inner: OpenEquivariance kernels + FastNequIP rewrites (``fast.py``).

    python -m src.fast.nequip PACKAGE OUT [--kernels-only] [--no-half] [--no-species-sc]

exposes the ablation switches; ``python -m src.compile_nequip --fast`` is the
recommended build.
"""
import argparse
import copy

import torch

from .. import report_parity, water_box

EV_TO_KCAL = 23.0605


def _check(model, ref, sizes):
    """Eager parity of the rewritten model against the stock OEQ model, on
    water with the atom types cycled through every species of the model."""
    from ...edges import build_edges
    dev = next(model.parameters()).device
    n_types = len(model.metadata["type_names"].split())
    r_max = float(model.metadata.get("r_max", 6.0))
    for n in sizes:
        pos = water_box(n).to(dev).float()
        ei = build_edges(pos, r_max)[0]
        E = ei.size(1)
        d = {"pos": pos, "edge_index": ei, "atom_types": torch.arange(n, device=dev) % n_types,
             "edge_cell_shift": torch.zeros(E, 3, dtype=torch.float32, device=dev),
             "cell": torch.zeros(1, 3, 3, dtype=torch.float32, device=dev),
             "batch": torch.zeros(n, dtype=torch.long, device=dev),
             "num_atoms": torch.tensor([n], device=dev)}
        r, o = ref(dict(d)), model(dict(d))
        err = report_parity("eager FastNequIP", n, r["total_energy"] * EV_TO_KCAL,
                            o["total_energy"] * EV_TO_KCAL, r["forces"] * EV_TO_KCAL,
                            o["forces"] * EV_TO_KCAL, "kcal/mol")
        if err >= 1e-4:  # fp32 model
            raise RuntimeError(f"FastNequIP differs from the stock OEQ model (relative dF {err:.1e})")


def build_fast_nequip(package: str, out: str, half_edges: bool = True, species_sc: bool = True,
                      rewrites: bool = True, check=(30, 300)) -> str:
    """Build the optimised NequIP inner ``out`` from ``package``.

    Returns the native library set it needs at run time (``"oeq"``).
    """
    from nequip.utils.compile import conditional_torchscript_mode
    from nequip.utils.global_state import set_global_state
    from nequip.scripts._workflow_utils import set_workflow_state
    from nequip.model.saved_models.load_utils import load_saved_model
    from nequip.model.modify_utils import modify
    from nequip.model.inference_models.torchscript import save_torchscript_model
    from nequip.model.utils import _EAGER_MODEL_KEY
    from ...compile_nequip import metadata
    from .fast import make_fast

    if not torch.cuda.is_available():
        raise RuntimeError("the optimised NequIP build needs a CUDA GPU")
    dev = torch.device("cuda")
    set_workflow_state("compile")
    set_global_state(allow_tf32=False)
    try:
        with conditional_torchscript_mode(True):
            model = load_saved_model(package, _EAGER_MODEL_KEY, "sole_model")
            model = modify(model, [{"modifier": "enable_OpenEquivariance"}]).to(dev)
            if rewrites:
                ref = copy.deepcopy(model)
                try:
                    make_fast(model, half_edges=half_edges, species_sc=species_sc)
                except AssertionError as exc:
                    print(f"[NequIP] FastNequIP rewrites don't fit this model ({exc}); "
                          "building with the OpenEquivariance kernels only.")
                    model = ref
                else:
                    _check(model, ref, check)
                del ref
            save_torchscript_model(model, metadata(model), out, dev)
    finally:
        set_workflow_state(None)
    print(f"[NequIP] optimised inner  ->  {out}")
    return "oeq"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build an optimised NequIP inner (ablation switches).")
    ap.add_argument("package", help="*.nequip.zip package or nequip.net: model id")
    ap.add_argument("out")
    ap.add_argument("--kernels-only", action="store_true", help="OpenEquivariance without the rewrites")
    ap.add_argument("--no-half", action="store_true")
    ap.add_argument("--no-species-sc", action="store_true")
    ap.add_argument("--check", type=int, nargs="*", default=[30, 300])
    a = ap.parse_args(argv)
    build_fast_nequip(a.package, a.out, half_edges=not a.no_half, species_sc=not a.no_species_sc,
                      rewrites=not a.kernels_only, check=a.check)
