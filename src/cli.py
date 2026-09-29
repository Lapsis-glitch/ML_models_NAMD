"""
Unified CLI for exporting wrapped MLIP models for NAMD.

Usage examples::

    python -m src.cli --model-type mace    --compiled model.pt --out mlff.pt
    python -m src.cli --model-type nequip  --compiled model.pth --out mlff.pt
    python -m src.cli --model-type allegro --compiled model.pth --out mlff.pt
    python -m src.cli --model-type schnet  --compiled model.pt --r-max 5.0 --out mlff.pt
    python -m src.cli --model-type torchani --compiled model.pt --out mlff.pt --elements 1,6,7,8
    python -m src.cli --model-type sevennet --compiled deployed_serial.pt --out mlff.pt
    python -m src.cli --model-type sevennet --compiled deployed_serial.pt --d3 --out mlff.pt

Optimised inner models (scripts/opt/) use custom ops; pass the same native
libraries NAMD will load via NAMD_MLFF_EXTRA_LIBS::

    python -m src.cli --model-type mace --compiled mace_inner_fast.pt \
        --extra-libs "$NAMD_MLFF_EXTRA_LIBS" --out mlff.pt
"""

import argparse

import torch


from .export import export_wrapped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a wrapped MLIP TorchScript model for NAMD",
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=["mace", "nequip", "allegro", "schnet", "torchani", "xmace",
                 "sevennet"],
        help="Type of ML potential to wrap",
    )
    parser.add_argument(
        "--compiled",
        required=True,
        help="Path to the compiled / deployed model file",
    )
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file (default: mlff_model.pt)")
    parser.add_argument("--device", default="cpu",
                        help="Device to load onto (default: cpu)")
    parser.add_argument("--extra-libs", default="",
                        help="Colon-separated native op libraries to load before "
                             "the model (same format as NAMD_MLFF_EXTRA_LIBS); "
                             "needed for optimised inner models")

    # SchNetPack-specific (also used as fallback for new NequIP-framework
    # models that don't expose r_max as a module attribute, e.g. NequIP-OAM-L).
    parser.add_argument("--r-max", type=float, default=0.0,
                        help="[schnet] Cutoff radius in Å (required); "
                             "[nequip] optional fallback when the deployed "
                             "model has no r_max attribute (OAM-L: 6.0)")
    parser.add_argument("--energy-key", default="energy",
                        help="[schnet] Output dict key for energy")
    parser.add_argument("--forces-key", default="forces",
                        help="[schnet] Output dict key for forces")
    parser.add_argument("--fast", action="store_true",
                        help="[schnet] Opt-in fast energy route for "
                             "non-periodic inputs (optimised build)")
    parser.add_argument("--graph-max-atoms", type=int, default=2048,
                        help="[schnet --fast] Largest system a graph-capable "
                             "shim should capture")
    parser.add_argument("--no-half-filter", action="store_true",
                        help="[schnet --fast] Run the filter network per "
                             "directed edge (stock layout)")
    parser.add_argument("--half-min-atoms", type=int, default=1500,
                        help="[schnet --fast] Total atoms from which the "
                             "half-list filter is used")
    parser.add_argument("--nl-cell-min-pairs", type=int, default=16_000_000,
                        help="[schnet --fast] B*n^2 above which the cell-list "
                             "neighbour list is used")

    # TorchANI-specific
    parser.add_argument("--elements", default="1,6,7,8,16,9,17",
                        help="[torchani] Comma-separated atomic numbers "
                             "in species order (default: ANI-2x)")
    parser.add_argument("--lean", action="store_true",
                        help="[torchani] Opt-in low-overhead path, "
                             "bit-identical outputs (optimised build)")

    # X-MACE-specific
    parser.add_argument("--state", type=int, default=0,
                        help="[xmace] Electronic state index to expose "
                             "(default: 0 = ground state)")

    # D3 dispersion, any model type
    parser.add_argument("--d3", action="store_true",
                        help="Add D3(BJ) dispersion (SevenNet's D3, parameters "
                             "read from the installed sevenn package)")
    parser.add_argument("--d3-functional", default="pbe",
                        help="[--d3] Functional for the BJ parameters "
                             "(default: pbe, what SevenNet uses)")
    parser.add_argument("--d3-cutoff", type=float, default=9000.0,
                        help="[--d3] Dispersion cutoff on r^2 in bohr^2, as "
                             "SevenNet takes it (default 9000 = 50.2 A)")
    parser.add_argument("--d3-cn-cutoff", type=float, default=1600.0,
                        help="[--d3] Coordination-number cutoff on r^2 in "
                             "bohr^2 (default 1600 = 21.2 A)")

    args = parser.parse_args(argv)

    for lib in filter(None, args.extra_libs.split(":")):
        torch.ops.load_library(lib)

    model_type = args.model_type

    if model_type == "mace":
        from .wrappers.wrap_compiled_mace import MACE_TS_Wrapper
        wrapper = MACE_TS_Wrapper(args.compiled, device=args.device).eval()

    elif model_type in ("nequip", "allegro"):
        from .wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper
        r_max_override = args.r_max if args.r_max > 0.0 else None
        wrapper = NequIP_Allegro_Wrapper(
            args.compiled, device=args.device, r_max=r_max_override,
        ).eval()

    elif model_type == "schnet":
        if args.r_max <= 0.0:
            parser.error("--r-max is required for --model-type schnet")
        from .wrappers.wrap_schnetpack import SchNetPack_Wrapper
        wrapper = SchNetPack_Wrapper(
            model_path=args.compiled,
            r_max=args.r_max,
            device=args.device,
            energy_key=args.energy_key,
            forces_key=args.forces_key,
            fast=args.fast,
            graph_max_atoms=args.graph_max_atoms,
            nl_cell_min_pairs=args.nl_cell_min_pairs,
            half_filter=not args.no_half_filter,
            half_min_atoms=args.half_min_atoms,
        ).eval()

    elif model_type == "torchani":
        from .wrappers.wrap_torchani import TorchANI_Wrapper
        elem_list = [int(x) for x in args.elements.split(",")]
        wrapper = TorchANI_Wrapper(
            model_path=args.compiled,
            device=args.device,
            element_list=elem_list,
            lean=args.lean,
        ).eval()

    elif model_type == "xmace":
        from .wrappers.wrap_xmace import XMACE_TS_Wrapper
        wrapper = XMACE_TS_Wrapper(
            args.compiled, state_idx=args.state, device=args.device,
        ).eval()

    elif model_type == "sevennet":
        from .wrappers.wrap_sevennet import SevenNet_Wrapper
        wrapper = SevenNet_Wrapper(args.compiled, device=args.device).eval()

    else:
        parser.error(f"Unknown model type: {model_type}")
        return  # unreachable, parser.error raises

    if args.d3:
        from .d3 import D3_Wrapper
        wrapper = D3_Wrapper(
            wrapper, functional=args.d3_functional,
            cutoff=args.d3_cutoff, cn_cutoff=args.d3_cn_cutoff,
        ).to(args.device).eval()

    label = model_type.upper()
    if model_type == "nequip":
        label = "NequIP"
    elif model_type == "allegro":
        label = "Allegro"
    elif model_type == "schnet":
        label = "SchNetPack"
    elif model_type == "torchani":
        label = "TorchANI"
    elif model_type == "xmace":
        label = "X-MACE"
    elif model_type == "sevennet":
        label = "SevenNet"

    if args.d3:
        label += f"+D3({args.d3_functional})"

    export_wrapped(wrapper, args.out, model_type=label)


if __name__ == "__main__":
    main()

