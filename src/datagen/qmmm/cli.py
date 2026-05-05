"""
CLI for ORCA QM/MM data generation.

Usage::

    python -m src.datagen.qmmm.cli \\
        --input trajectory.xyz \\
        --output qmmm_data.xyz \\
        --method "B3LYP def2-SVP" \\
        --n-qm-atoms 6 \\
        --orcaff-file system.ORCAFF.prms \\
        --n-workers 8 --orca-nprocs 2

    # With topology conversion:
    python -m src.datagen.qmmm.cli \\
        --input trajectory.xyz \\
        --output qmmm_data.xyz \\
        --method "B3LYP def2-SVP" \\
        --n-qm-atoms 6 \\
        --amber-prmtop system.prmtop \\
        --amber-inpcrd system.rst7 \\
        --n-workers 8

    # With explicit QM atom indices:
    python -m src.datagen.qmmm.cli \\
        --input trajectory.xyz \\
        --output qmmm_data.xyz \\
        --method "B3LYP def2-SVP" \\
        --qm-indices 0,1,2,3,4,5 \\
        --orcaff-file system.ORCAFF.prms \\
        --n-workers 4
"""

from __future__ import annotations

import argparse
import logging
import sys

from .generator import QMMMDataGenerator, convert_amber_topology


def _parse_qm_indices(s: str) -> list[int]:
    """
    Parse a QM-indices string.

    Supports:
      * Comma-separated:  ``"0,1,2,3,4,5"``
      * Range notation:   ``"0-5"``  (inclusive)
      * Mixed:            ``"0-5,10,12-14"``
    """
    indices: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate QM/MM training data with ORCA.\n\n"
            "Runs electrostatic-embedding QM/MM single-point calculations "
            "for each frame in the input trajectory.  Output is an extended "
            "XYZ file with QM-region energy, forces, and charges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────
    io_grp = parser.add_argument_group("input / output")
    io_grp.add_argument(
        "--input", required=True,
        help="Input XYZ with full-system geometries (QM + MM atoms).",
    )
    io_grp.add_argument(
        "--output", required=True,
        help="Output extended XYZ with QM-region results.",
    )

    # ── QM region ────────────────────────────────────────────────
    qm_grp = parser.add_argument_group("QM region")
    qm_excl = qm_grp.add_mutually_exclusive_group(required=True)
    qm_excl.add_argument(
        "--n-qm-atoms", type=int, default=None,
        help="Number of QM atoms (first N atoms in each frame).",
    )
    qm_excl.add_argument(
        "--qm-indices", type=str, default=None,
        help=(
            "Explicit 0-based QM atom indices.  "
            "Comma-separated or range: '0,1,2,3' or '0-5' or '0-5,10'."
        ),
    )

    # ── ORCA settings ────────────────────────────────────────────
    orca_grp = parser.add_argument_group("ORCA settings")
    orca_grp.add_argument(
        "--method", default="B3LYP def2-SVP",
        help="ORCA simple-input line (default: 'B3LYP def2-SVP').",
    )
    orca_grp.add_argument(
        "--orca-command", default=None,
        help="Path to ORCA binary (default: $ASE_ORCA_COMMAND or 'orca').",
    )
    orca_grp.add_argument(
        "--orca-nprocs", type=int, default=1,
        help="MPI processes per ORCA calculation (default: 1).",
    )
    orca_grp.add_argument(
        "--extra-blocks", default="",
        help="Additional ORCA input blocks (e.g. '%%scf MaxIter 300 end').",
    )
    orca_grp.add_argument(
        "--charge-total", type=int, default=0,
        help="Total system charge (default: 0).",
    )
    orca_grp.add_argument(
        "--charge-qm", type=int, default=0,
        help="QM region charge (default: 0).",
    )
    orca_grp.add_argument(
        "--mult-qm", type=int, default=1,
        help="QM region spin multiplicity (default: 1).",
    )

    # ── Force field / topology ───────────────────────────────────
    ff_grp = parser.add_argument_group("force field / Amber topology")
    ff_grp.add_argument(
        "--orcaff-file", default=None,
        help=(
            "Pre-converted ORCA FF file (.ORCAFF.prms).  "
            "If not given, --amber-prmtop and --amber-inpcrd are required."
        ),
    )
    ff_grp.add_argument(
        "--amber-prmtop", default=None,
        help="Amber .prmtop topology file.",
    )
    ff_grp.add_argument(
        "--amber-inpcrd", default=None,
        help=(
            "Amber .inpcrd / .rst7 coordinate file "
            "(required for topology conversion)."
        ),
    )
    ff_grp.add_argument(
        "--orca-mm-command", default="orca_mm",
        help="Name or path of the orca_mm binary (default: 'orca_mm').",
    )
    ff_grp.add_argument(
        "--ff-output-dir", default="./orcaff",
        help="Directory for converted FF files (default: ./orcaff).",
    )

    # ── Parallelism ──────────────────────────────────────────────
    par_grp = parser.add_argument_group("parallelism")
    par_grp.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel frame calculations (default: 1).",
    )

    # ── Misc ─────────────────────────────────────────────────────
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── Resolve the ORCA FF file ─────────────────────────────────
    orcaff_file = args.orcaff_file
    if orcaff_file is None:
        # Need to convert from Amber topology
        if args.amber_prmtop is None or args.amber_inpcrd is None:
            parser.error(
                "Either --orcaff-file must be given, or both "
                "--amber-prmtop and --amber-inpcrd for automatic "
                "topology conversion."
            )
        print("Converting Amber topology to ORCA force field …")
        orcaff_file = convert_amber_topology(
            prmtop_path=args.amber_prmtop,
            inpcrd_path=args.amber_inpcrd,
            output_dir=args.ff_output_dir,
            orca_mm_command=args.orca_mm_command,
        )
        print(f"  → {orcaff_file}\n")

    # ── Resolve QM indices ───────────────────────────────────────
    qm_indices = None
    n_qm_atoms = None
    if args.qm_indices is not None:
        qm_indices = _parse_qm_indices(args.qm_indices)
    else:
        n_qm_atoms = args.n_qm_atoms

    # ── Build generator and run ──────────────────────────────────
    gen = QMMMDataGenerator(
        method=args.method,
        qm_indices=qm_indices,
        n_qm_atoms=n_qm_atoms,
        orcaff_file=orcaff_file,
        amber_prmtop=args.amber_prmtop,
        charge_total=args.charge_total,
        charge_qm=args.charge_qm,
        mult_qm=args.mult_qm,
        orca_command=args.orca_command,
        orca_nprocs=args.orca_nprocs,
        extra_blocks=args.extra_blocks,
    )

    n_success = gen.run(
        input_path=args.input,
        output_path=args.output,
        n_workers=args.n_workers,
    )

    print(f"\nDone.  {n_success} frames written to {args.output}")
    if n_success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

