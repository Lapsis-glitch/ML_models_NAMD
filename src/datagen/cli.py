"""
CLI for QM data generation.

Usage::

    python -m src.datagen.cli \\
        --input geometries.xyz \\
        --output dft_data.xyz \\
        --method "B3LYP def2-SVP" \\
        --n-workers 4 \\
        --orca-nprocs 2

    # With random-displacement sampling from a single geometry:
    python -m src.datagen.cli \\
        --input equilibrium.xyz \\
        --output dft_data.xyz \\
        --method "wB97X-D3 def2-TZVP" \\
        --sample random --n-samples 100 --amplitude 0.15
"""

import argparse
import logging
import sys

from .orca_generator import OrcaDataGenerator


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate QM training data with ORCA",
    )

    # I/O
    parser.add_argument("--input", required=True,
                        help="Input XYZ file with geometries")
    parser.add_argument("--output", required=True,
                        help="Output extended XYZ with energy/forces")

    # ORCA settings
    parser.add_argument("--method", default="B3LYP def2-SVP",
                        help="ORCA method line (default: 'B3LYP def2-SVP')")
    parser.add_argument("--orca-command", default=None,
                        help="Path to ORCA binary (default: $ASE_ORCA_COMMAND or 'orca')")
    parser.add_argument("--orca-nprocs", type=int, default=1,
                        help="MPI processes per ORCA calculation (default: 1)")
    parser.add_argument("--extra-blocks", default="",
                        help="Additional ORCA input blocks")

    # Parallelism
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel frame calculations (default: 1)")

    # Sampling
    parser.add_argument("--sample", choices=["none", "random"],
                        default="none",
                        help="Sampling method applied to input geometries")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="[random] Number of displaced structures per input frame")
    parser.add_argument("--amplitude", type=float, default=0.1,
                        help="[random] Displacement amplitude in Å (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load input frames
    from ase.io import read as ase_read
    from ase import Atoms

    frames = ase_read(args.input, index=":")
    if isinstance(frames, Atoms):
        frames = [frames]

    # Optional sampling
    if args.sample == "random":
        from .sampling import random_displacements

        all_displaced = []
        for f in frames:
            displaced = random_displacements(
                f, n_samples=args.n_samples,
                amplitude=args.amplitude, seed=args.seed,
            )
            all_displaced.extend(displaced)
        print(f"Generated {len(all_displaced)} displaced structures "
              f"from {len(frames)} input frame(s)")
        frames = all_displaced

    # Run calculations
    gen = OrcaDataGenerator(
        method=args.method,
        orca_command=args.orca_command,
        orca_nprocs=args.orca_nprocs,
        extra_blocks=args.extra_blocks,
    )

    n_success = gen.run(args.input, args.output, n_workers=args.n_workers) \
        if args.sample == "none" else _run_frames_and_write(
            gen, frames, args.output, args.n_workers,
        )

    print(f"\nDone. {n_success} frames written to {args.output}")
    if n_success == 0:
        sys.exit(1)


def _run_frames_and_write(gen, frames, output_path, n_workers):
    """Run frames through the generator and write output."""
    from pathlib import Path
    from ase.io import write as ase_write

    results = gen.run_frames(frames, n_workers=n_workers)
    if results:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ase_write(output_path, results, format="extxyz")
        print(f"Wrote {len(results)} frames to {output_path}")
    return len(results)


if __name__ == "__main__":
    main()

