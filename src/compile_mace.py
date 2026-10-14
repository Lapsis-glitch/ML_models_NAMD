"""
TorchScript-compile a MACE checkpoint (e.g. ``MACE-OFF23_medium.model``) for
:class:`src.wrappers.wrap_compiled_mace.MACE_TS_Wrapper`.

Usage::

    python -m src.compile_mace --weights MACE-OFF23_medium.model \
        --out models/compiled_mace_off23_medium.pt
    python -m src.compile_mace --weights MACE-OFF23_medium.model --fast \
        --mace-python /path/to/old-e3nn-env/bin/python \
        --out models/mace_off23_medium_fast.pt

The reference compile runs in an env that can unpickle the ``.model``: one
with the mace/e3nn versions that saved it (MACE-OFF23 needs ``e3nn 0.4.4``).

``--fast`` (GPU, env with a current mace and cuEquivariance): rebuilds the model with cuEquivariance
kernels plus the exact FastMACE rewrites (``src/fast/mace``) and checks it
against the stock model.  When this env cannot unpickle the ``.model`` (older
e3nn), ``--mace-python`` names an interpreter that can, used only to dump the
weights.  Needs the native ``cueq`` libraries at run time (``src.build``
handles it).
"""

import argparse
import os
from pathlib import Path

import torch


_DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def compile_mace(
    weights_path: str,
    out_path: str,
    device: str = "cpu",
    dtype: str = "float64",
) -> None:
    from e3nn.util import jit as e3nn_jit  # type: ignore[import-not-found]

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

    print(f"[MACE] Compiled {weights_path}  ->  {out_path}")
    print(f"  dtype:          {dtype}")
    print(f"  atomic_numbers: {list(model.atomic_numbers.tolist())}")
    print(f"  r_max:          {float(model.r_max)} Å")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--weights", required=True,
                        help="Path to the MACE .model file")
    parser.add_argument("--out", required=True,
                        help="Output compiled TorchScript .pt file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64", choices=sorted(_DTYPE_MAP),
                        help="Floating-point dtype for the compiled artifact")
    parser.add_argument("--fast", action="store_true",
                        help="cuEquivariance kernels + FastMACE rewrites (GPU)")
    parser.add_argument("--mace-python", default=os.environ.get("MACE_PY"),
                        help="[--fast] Python that can unpickle the .model "
                             "(default: $MACE_PY, else this one)")
    args = parser.parse_args(argv)
    if args.fast:
        from .fast.native import ensure_cuda_lib_path
        ensure_cuda_lib_path("src.compile_mace")
        from .fast.mace import build_fast_mace
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        build_fast_mace(args.weights, args.out, dtype=args.dtype,
                        mace_python=args.mace_python)
    else:
        compile_mace(args.weights, args.out, args.device, args.dtype)


if __name__ == "__main__":
    main()
