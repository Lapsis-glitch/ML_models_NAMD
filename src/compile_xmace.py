"""
TorchScript-compile an **X-MACE** model (``AutoencoderExcitedMACE``) for
:class:`src.wrappers.wrap_xmace.XMACE_TS_Wrapper`.

Usage (``x_mace`` env, with the patched X-MACE source installed)::

    python -m src.compile_xmace --weights fulvene.model \
        --out models/fulvene_compiled.pt

Upstream X-MACE cannot be TorchScript-compiled as published; the ``x_mace``
env carries the patched source (see the README).  Models saved with
``nac_indices=None`` / ``soc_indices=None`` are fixed up here.
"""

import argparse
import os
from copy import deepcopy
from pathlib import Path

import torch


def compile_xmace(weights_path: str, out_path: str) -> None:
    from e3nn.util import jit

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    m = torch.load(weights_path, map_location="cpu", weights_only=False).eval()
    if getattr(m, "nac_indices", 0) is None:
        m.nac_indices = 0
    if getattr(m, "soc_indices", 0) is None:
        m.soc_indices = 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(jit.compile(deepcopy(m)), out_path)
    print(f"[X-MACE] Compiled {weights_path}  ->  {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--weights", required=True, help="X-MACE .model file")
    parser.add_argument("--out", required=True, help="Output TorchScript .pt file")
    args = parser.parse_args(argv)
    compile_xmace(args.weights, args.out)


if __name__ == "__main__":
    main()
