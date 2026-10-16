"""
TorchScript-compile a **pretrained built-in TorchANI model**
(ANI-1x, ANI-1ccx, or ANI-2x) into a deployable ``.pt`` file that the
existing :class:`src.wrappers.wrap_torchani.TorchANI_Wrapper` accepts.

Usage::

    python -m src.compile_torchani --variant ani2x \
        --out models/compiled_ani2x.pt

    python -m src.compile_torchani --variant ani2x --fast \
        --out models/ani2x_fast.pt

After this, wrap for NAMD with ``python -m src.cli --model-type torchani``,
passing the printed element order to ``--elements``.

``--fast`` (GPU): cuAEV kernels plus a fused per-element ensemble
(``src/fast/ani``), checked against the stock model in float64.  Builds and
needs the native ``cuaev`` library (``src.build`` handles it).
"""

import argparse
from pathlib import Path

import torch

from .training.train_torchani import _TorchANIExportWrapper


_VARIANTS = {
    "ani1x":   ("ANI1x",   [1, 6, 7, 8]),
    "ani1ccx": ("ANI1ccx", [1, 6, 7, 8]),
    "ani2x":   ("ANI2x",   [1, 6, 7, 8, 16, 9, 17]),  # H C N O S F Cl
}


def compile_torchani(variant: str, out_path: str, device: str = "cpu") -> list:
    import torchani

    cls_name, element_list = _VARIANTS[variant]
    builder = getattr(torchani.models, cls_name)
    model = builder(periodic_table_index=False).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    export_model = _TorchANIExportWrapper(model).eval()

    try:
        scripted = torch.jit.script(export_model)
    except Exception as e:
        print(f"[TorchANI] jit.script failed ({e}); falling back to jit.trace")
        species = torch.zeros((1, 1), dtype=torch.long, device=device)
        coords = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)
        scripted = torch.jit.trace(export_model, (species, coords))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(out_path)

    print(f"[TorchANI] Compiled {cls_name}  ->  {out_path}")
    print(f"  elements (Z order): {element_list}")
    print(f"  --elements arg for src.cli: {','.join(str(z) for z in element_list)}")
    return element_list


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--variant", required=True, choices=list(_VARIANTS),
                        help="Pretrained TorchANI variant")
    parser.add_argument("--out", required=True,
                        help="Output compiled TorchScript .pt file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fast", action="store_true",
                        help="cuAEV kernels + fused ensemble (GPU)")
    args = parser.parse_args(argv)
    if args.fast:
        from .fast.native import ensure, ensure_cuda_lib_path
        ensure_cuda_lib_path("src.compile_torchani")
        from .fast.ani import build_fast_ani
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        build_fast_ani(args.variant, args.out, ensure("cuaev")[0])
    else:
        compile_torchani(args.variant, args.out, args.device)


if __name__ == "__main__":
    main()