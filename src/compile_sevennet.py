"""
Deploy a **SevenNet** checkpoint (pretrained or your own) to the TorchScript
file :class:`src.wrappers.wrap_sevennet.SevenNet_Wrapper` accepts.

This is SevenNet's own LAMMPS serial deployment (``sevenn get_model``), so the
inner model is the same file ``pair_e3gnn`` would load.

Usage::

    python -m src.compile_sevennet --checkpoint 7net-0 \
        --out models/compiled_sevennet_0.pt
    python -m src.compile_sevennet --checkpoint 7net-mf-ompa --modal mpa \
        --out models/compiled_sevennet_mf_ompa_mpa.pt
    python -m src.compile_sevennet --checkpoint my_run/checkpoint_best.pth \
        --out models/compiled_sevennet_mine.pt
    python -m src.compile_sevennet --checkpoint 7net-0 --fast \
        --out models/compiled_sevennet_0_fast.pt

After this, wrap for NAMD with::

    python -m src.cli --model-type sevennet \
        --compiled models/compiled_sevennet_0.pt --out mlff_model.pt

Multi-fidelity checkpoints (``7net-mf-ompa``, ``7net-omni``, ...) need
``--modal``; the error SevenNet raises without it lists the choices.  D3
dispersion is a separate CUDA kernel in SevenNet and is not part of the
deployed model (``src.cli --d3`` adds it).

Optimised deployments (same weights; GPU and ``openequivariance`` needed at
build time):

``--oeq``   SevenNet's own OpenEquivariance deployment (``use_oeq``): fused
            tensor-product convolution kernels.
``--fast``  ``--oeq`` plus the exact FastSevenNet rewrites of
            ``src/sevennet_fast.py`` (fewer, larger ops; the model is
            host-bound without them).  If a rewrite doesn't fit the model's
            structure, it falls back to ``--oeq`` and says so.

Both need the native op library ``scripts/opt/nequip/oeq_native/liboeq_native.so``
at run time: in NAMD via ``NAMD_MLFF_EXTRA_LIBS``; ``src.cli`` loads it by
itself when it wraps such a deployment.  See scripts/opt/sevennet/REPORT.md.
"""

import argparse
from pathlib import Path


def compile_sevennet(checkpoint: str, out_path: str, modal: str = None,
                     oeq: bool = False, fast: bool = False) -> str:
    """Deploy ``checkpoint`` to ``out_path``.  Returns what was built:
    ``"stock"``, ``"oeq"`` or ``"fast"``."""
    import sevenn._keys as KEY
    from sevenn.scripts.deploy import deploy
    from sevenn.util import load_checkpoint

    # deploy() writes the type map in dict-key order and pair_e3gnn takes a
    # symbol's position as its index, so the two must agree.
    type_map = load_checkpoint(checkpoint).config[KEY.TYPE_MAP]
    if list(type_map.values()) != list(range(len(type_map))):
        raise RuntimeError(
            "Checkpoint type map is not in index order; the deployed "
            "chemical_symbols_to_index would be wrong.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    kind = "fast" if fast else ("oeq" if oeq else "stock")
    if fast:
        from .sevennet_fast import RewriteNotApplicable, rewriting_deploy
        try:
            with rewriting_deploy():
                deploy(checkpoint, out_path, modal=modal, use_oeq=True)
        except RewriteNotApplicable as exc:
            print(f"[SevenNet] FastSevenNet rewrites don't fit this model ({exc}); "
                  "deploying with the OpenEquivariance kernels only.")
            kind = "oeq"
            deploy(checkpoint, out_path, modal=modal, use_oeq=True)
    else:
        deploy(checkpoint, out_path, modal=modal, use_oeq=oeq)

    from .wrappers.wrap_sevennet import read_sevennet_metadata
    type_map, r_max, dtype = read_sevennet_metadata(out_path)
    how = {"stock": "", "oeq": " with OpenEquivariance",
           "fast": " with OpenEquivariance + FastSevenNet"}[kind]
    print(f"[SevenNet] Deployed {checkpoint}"
          f"{f' (modal {modal})' if modal else ''}{how}  ->  {out_path}")
    print(f"  elements: {len(type_map)}   cutoff: {r_max} Å   dtype: {dtype}")
    if kind != "stock":
        print("  needs scripts/opt/nequip/oeq_native/liboeq_native.so at run time "
              "(NAMD_MLFF_EXTRA_LIBS)")
    return kind


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint path or pretrained name "
                             "(7net-0, 7net-l3i5, 7net-omat, 7net-mf-ompa, ...)")
    parser.add_argument("--modal", default=None,
                        help="Fidelity/task for multi-fidelity checkpoints")
    opt = parser.add_mutually_exclusive_group()
    opt.add_argument("--oeq", action="store_true",
                     help="Deploy with OpenEquivariance tensor-product kernels (GPU)")
    opt.add_argument("--fast", action="store_true",
                     help="--oeq plus the exact FastSevenNet rewrites (GPU; recommended)")
    parser.add_argument("--out", required=True, help="Output TorchScript file")
    args = parser.parse_args(argv)
    compile_sevennet(args.checkpoint, args.out, modal=args.modal,
                     oeq=args.oeq, fast=args.fast)


if __name__ == "__main__":
    main()
