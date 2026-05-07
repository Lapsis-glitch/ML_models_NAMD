"""
TorchScript-compile a raw MACE foundation-model checkpoint
(e.g. ``MACE-OFF23_medium.model``) into a deployable ``.pt`` file
that the existing :class:`MACE_TS_Wrapper` accepts.

Usage::

    python -m src.compile_mace_off \
        --weights /home/rat/.cache/mace/MACE-OFF23_medium.model \
        --out compiled_mace_off23_medium.pt

    python -m src.compile_mace_off \
        --weights /home/rat/.cache/mace/MACE-OFF23_medium.model \
        --out compiled_mace_off23_medium_fp32.pt \
        --dtype float32
"""

import argparse
import os
from pathlib import Path

import torch
from e3nn.util import jit as e3nn_jit  # type: ignore[import-not-found]


_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def compile_mace_off(
    weights_path: str,
    out_path: str,
    device: str = "cpu",
    dtype: str = "float64",
) -> None:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    if dtype not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'; choose from {sorted(_DTYPE_MAP)}")

    target_dtype = _DTYPE_MAP[dtype]

    model = torch.load(weights_path, map_location=device, weights_only=False)
    model = model.to(dtype=target_dtype, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # e3nn's jit.compile handles the codegen submodules that plain
    # torch.jit.script chokes on (e.g. activation functions).
    scripted = e3nn_jit.compile(model)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(out_path)

    print(f"[MACE-OFF] Compiled {weights_path}  ->  {out_path}")
    print(f"  dtype:          {dtype}")
    print(f"  atomic_numbers: {list(model.atomic_numbers.tolist())}")
    print(f"  r_max:          {float(model.r_max)} Å")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True,
                        help="Path to raw MACE-OFF .model file")
    parser.add_argument("--out", required=True,
                        help="Output compiled TorchScript .pt file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=sorted(_DTYPE_MAP),
                        help="Floating-point dtype for the compiled artifact")
    args = parser.parse_args(argv)
    compile_mace_off(args.weights, args.out, args.device, args.dtype)


if __name__ == "__main__":
    main()