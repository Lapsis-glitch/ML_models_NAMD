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

After this, wrap for NAMD with::

    python -m src.cli --model-type sevennet \
        --compiled models/compiled_sevennet_0.pt --out mlff_model.pt

Multi-fidelity checkpoints (``7net-mf-ompa``, ``7net-omni``, ...) need
``--modal``; the error SevenNet raises without it lists the choices.  D3
dispersion is a separate CUDA kernel in SevenNet and is not part of the
deployed model.
"""

import argparse
from pathlib import Path


def compile_sevennet(checkpoint: str, out_path: str, modal: str = None) -> None:
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
    deploy(checkpoint, out_path, modal=modal)

    from .wrappers.wrap_sevennet import read_sevennet_metadata
    type_map, r_max, dtype = read_sevennet_metadata(out_path)
    print(f"[SevenNet] Deployed {checkpoint}"
          f"{f' (modal {modal})' if modal else ''}  ->  {out_path}")
    print(f"  elements: {len(type_map)}   cutoff: {r_max} Å   dtype: {dtype}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True,
                        help="Checkpoint path or pretrained name "
                             "(7net-0, 7net-l3i5, 7net-omat, 7net-mf-ompa, ...)")
    parser.add_argument("--modal", default=None,
                        help="Fidelity/task for multi-fidelity checkpoints")
    parser.add_argument("--out", required=True, help="Output TorchScript file")
    args = parser.parse_args(argv)
    compile_sevennet(args.checkpoint, args.out, modal=args.modal)


if __name__ == "__main__":
    main()
