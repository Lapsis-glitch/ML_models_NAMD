"""
TorchScript-compile a **pretrained built-in TorchANI model**
(ANI-1x, ANI-1ccx, or ANI-2x) into a deployable ``.pt`` file that the
existing :class:`src.wrappers.wrap_torchani.TorchANI_Wrapper` accepts.

Usage::

    python -m src.compile_torchani --variant ani2x \
        --out models/compiled_ani2x.pt

After this, wrap for NAMD with::

    python -m src.cli --model-type torchani \
        --compiled models/compiled_ani2x.pt \
        --elements 1,6,7,8,16,17 \
        --out models/trpcage_ani2x_qmmm.pt
"""

import argparse
from pathlib import Path

import torch

from .training.train_torchani import _TorchANIExportWrapper


_VARIANTS = {
    "ani1x":   ("ANI1x",   [1, 6, 7, 8]),
    "ani1ccx": ("ANI1ccx", [1, 6, 7, 8]),
    "ani2x":   ("ANI2x",   [1, 6, 7, 8, 16, 17]),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=list(_VARIANTS),
                        help="Pretrained TorchANI variant")
    parser.add_argument("--out", required=True,
                        help="Output compiled TorchScript .pt file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    compile_torchani(args.variant, args.out, args.device)


if __name__ == "__main__":
    main()