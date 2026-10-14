"""
TorchScript-compile a **NequIP or Allegro** package for
:class:`src.wrappers.wrap_compiled_nequip.NequIP_Allegro_Wrapper`.

Usage::

    python -m src.compile_nequip --package nequip.net:mir-group/NequIP-OAM-L:0.1 \
        --out models/compiled_nequip_oam_l.nequip.pth
    python -m src.compile_nequip --package my_model.nequip.zip --fast \
        --out models/my_nequip_fast.nequip.pth

``--package`` takes a ``*.nequip.zip`` from ``nequip-package build`` or a
``nequip.net:`` model id.  This does what ``nequip-compile --mode torchscript``
does, without its refusal to run on torch >= 2.10.

``--fast`` (NequIP, GPU): OpenEquivariance tensor-product kernels plus the exact
FastNequIP rewrites (``src/fast/nequip``), checked against the stock model.  If
the rewrites don't fit the model it builds with the kernels only.  Needs the
native ``oeq`` library at run time (``src.build`` handles it).
"""

import argparse

import torch


def compile_nequip(package: str, out_path: str, device: str = "cuda",
                   modifiers=(), tf32: bool = False) -> None:
    from nequip.utils.compile import conditional_torchscript_mode
    from nequip.utils.global_state import set_global_state
    from nequip.scripts._workflow_utils import set_workflow_state
    from nequip.model.saved_models.load_utils import load_saved_model
    from nequip.model.modify_utils import modify
    from nequip.model.inference_models.torchscript import save_torchscript_model
    from nequip.model.utils import _EAGER_MODEL_KEY

    set_workflow_state("compile")
    set_global_state(allow_tf32=tf32)
    try:
        with conditional_torchscript_mode(True):
            model = load_saved_model(package, _EAGER_MODEL_KEY, "sole_model")
            model = modify(model, [{"modifier": m} for m in modifiers])
            save_torchscript_model(model, metadata(model), out_path, torch.device(device))
    finally:
        set_workflow_state(None)
    print(f"[NequIP] Compiled {package}  ->  {out_path}")


def metadata(model) -> dict:
    """The metadata ``nequip-compile`` stores with a TorchScript model."""
    from nequip.utils.global_state import get_latest_global_state
    md = model.metadata.copy()
    md.update(get_latest_global_state(only_metadata_related=True))
    return {k: str(int(v)) if isinstance(v, bool) else v for k, v in md.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--package", required=True,
                        help="*.nequip.zip package or nequip.net: model id")
    parser.add_argument("--out", required=True, help="Output TorchScript .nequip.pth")
    parser.add_argument("--device", default="cuda",
                        help="Device NAMD will run the model on (default: cuda)")
    parser.add_argument("--fast", action="store_true",
                        help="OpenEquivariance kernels + FastNequIP rewrites (GPU)")
    # benchmark variants only (e.g. enable_CuEquivariance)
    parser.add_argument("--modifiers", nargs="*", default=[], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.fast:
        from .fast.native import ensure_cuda_lib_path
        ensure_cuda_lib_path("src.compile_nequip")
        from .fast.nequip import build_fast_nequip
        build_fast_nequip(args.package, args.out)
    else:
        compile_nequip(args.package, args.out, device=args.device, modifiers=args.modifiers)


if __name__ == "__main__":
    main()
