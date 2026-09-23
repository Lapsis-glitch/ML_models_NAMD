"""
Unified CLI for exporting wrapped MLIP models for NAMD.

Usage examples::

    python -m src.cli --model-type mace    --compiled model.pt --out mlff.pt
    python -m src.cli --model-type nequip  --compiled model.pth --out mlff.pt
    python -m src.cli --model-type allegro --compiled model.pth --out mlff.pt
    python -m src.cli --model-type schnet  --compiled model.pt --r-max 5.0 --out mlff.pt
    python -m src.cli --model-type torchani --compiled model.pt --out mlff.pt --elements 1,6,7,8
"""

import argparse


from .export import export_wrapped


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Export a wrapped MLIP TorchScript model for NAMD",
    )

    parser.add_argument(
        "--model-type",
        required=True,
        choices=["mace", "nequip", "allegro", "schnet", "torchani", "xmace"],
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

    # TorchANI-specific
    parser.add_argument("--elements", default="1,6,7,8,16,9,17",
                        help="[torchani] Comma-separated atomic numbers "
                             "in species order (default: ANI-2x)")

    # X-MACE-specific
    parser.add_argument("--state", type=int, default=0,
                        help="[xmace] Electronic state index to expose "
                             "(default: 0 = ground state)")

    args = parser.parse_args(argv)

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
        ).eval()

    elif model_type == "torchani":
        from .wrappers.wrap_torchani import TorchANI_Wrapper
        elem_list = [int(x) for x in args.elements.split(",")]
        wrapper = TorchANI_Wrapper(
            model_path=args.compiled,
            device=args.device,
            element_list=elem_list,
        ).eval()

    elif model_type == "xmace":
        from .wrappers.wrap_xmace import XMACE_TS_Wrapper
        wrapper = XMACE_TS_Wrapper(
            args.compiled, state_idx=args.state, device=args.device,
        ).eval()

    else:
        parser.error(f"Unknown model type: {model_type}")
        return  # unreachable, parser.error raises

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

    export_wrapped(wrapper, args.out, model_type=label)


if __name__ == "__main__":
    main()

