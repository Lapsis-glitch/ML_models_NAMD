"""
Shared data preparation: extended XYZ → per-framework format.

Reads an extended XYZ file (with energy/forces in info/arrays fields),
splits into train / validation / test sets, and writes outputs in every
format the supported frameworks expect:

    ├── xyz/           train.xyz, val.xyz, test.xyz   (MACE, NequIP, Allegro)
    ├── schnetpack/    train.db, val.db, test.db      (SchNetPack ASE DB)
    └── torchani/      data.h5                        (TorchANI HDF5)

Usage::

    python -m src.training.prepare_data \\
        --xyz data.xyz \\
        --output-dir ./prepared \\
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \\
        --seed 42
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
except ImportError:
    raise ImportError(
        "ASE is required for data preparation: pip install ase"
    )


# -------------------------------------------------------------------
#  Energy / forces extraction helpers
# -------------------------------------------------------------------

_ENERGY_KEYS = ("energy", "Energy", "REF_energy", "dft_energy")
_FORCES_KEYS = ("forces", "REF_forces")


def get_energy(atoms: Atoms):
    """
    Extract energy from an Atoms object.

    Checks ``info`` dict first (common keys), then falls back to
    ``atoms.calc.results["energy"]`` (ASE SinglePointCalculator).
    Returns ``None`` if not found.
    """
    for key in _ENERGY_KEYS:
        if key in atoms.info:
            return float(atoms.info[key])
    # Fallback: calculator results
    if atoms.calc is not None:
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    return None


def get_forces(atoms: Atoms):
    """
    Extract forces from an Atoms object.

    Checks ``arrays`` dict first, then ``atoms.calc.results["forces"]``.
    Returns ``None`` if not found.
    """
    for key in _FORCES_KEYS:
        if key in atoms.arrays:
            return np.array(atoms.arrays[key], dtype=np.float64)
    if atoms.calc is not None:
        try:
            return np.array(atoms.get_forces(), dtype=np.float64)
        except Exception:
            pass
    return None


# -------------------------------------------------------------------
#  Splitting
# -------------------------------------------------------------------

def split_frames(
    frames: List[Atoms],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Atoms], List[Atoms], List[Atoms]]:
    """
    Randomly shuffle *frames* and split into train / val / test lists.

    The ratios are normalised so they always sum to 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    # test gets the remainder

    rng = random.Random(seed)
    indices = list(range(len(frames)))
    rng.shuffle(indices)

    n = len(frames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = [frames[i] for i in indices[:n_train]]
    val = [frames[i] for i in indices[n_train : n_train + n_val]]
    test = [frames[i] for i in indices[n_train + n_val :]]
    return train, val, test


# -------------------------------------------------------------------
#  Writers: XYZ  (MACE, NequIP, Allegro)
# -------------------------------------------------------------------

def write_xyz_splits(
    train: List[Atoms],
    val: List[Atoms],
    test: List[Atoms],
    output_dir: Path,
) -> None:
    """Write three extended-XYZ files."""
    xyz_dir = output_dir / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)

    for name, frames in [("train", train), ("val", val), ("test", test)]:
        path = xyz_dir / f"{name}.xyz"
        ase_write(str(path), frames, format="extxyz")
        print(f"  {path}  ({len(frames)} frames)")


# -------------------------------------------------------------------
#  Writer: SchNetPack ASE DB
# -------------------------------------------------------------------

def write_schnetpack_db(
    train: List[Atoms],
    val: List[Atoms],
    test: List[Atoms],
    output_dir: Path,
) -> None:
    """
    Write three ASE DB files that SchNetPack can consume directly.

    Each ``Atoms`` object must carry ``info['energy']`` (scalar) and
    ``arrays['forces']`` ([N,3]) – the standard extended-XYZ convention.
    These are stored as properties ``energy`` and ``forces`` in the DB.

    Uses the raw ``ase.db`` context-manager API to avoid a
    SchNetPack 2.2 / ASE ≥ 3.28 incompatibility where
    ``ASEAtomsData`` accesses the DB metadata outside an active
    connection.
    """
    try:
        import schnetpack  # noqa: F401
    except ImportError:
        print("  [skip] schnetpack not installed – skipping .db output")
        return

    from ase.db import connect as ase_db_connect

    db_dir = output_dir / "schnetpack"
    db_dir.mkdir(parents=True, exist_ok=True)

    for name, frames in [("train", train), ("val", val), ("test", test)]:
        db_path = db_dir / f"{name}.db"
        if db_path.exists():
            db_path.unlink()

        with ase_db_connect(str(db_path)) as db:
            # SchNetPack metadata (must be written inside the connection)
            db.metadata = {
                "_property_unit_dict": {"energy": "eV", "forces": "eV/Ang"},
                "_distance_unit": "Ang",
                "atomrefs": {},
            }

            for atoms in frames:
                data = {}
                e = get_energy(atoms)
                if e is not None:
                    data["energy"] = np.array([e], dtype=np.float64)
                f = get_forces(atoms)
                if f is not None:
                    data["forces"] = f
                db.write(atoms, data=data)

        print(f"  {db_path}  ({len(frames)} frames)")


# -------------------------------------------------------------------
#  Writer: TorchANI HDF5
# -------------------------------------------------------------------

def write_torchani_h5(
    train: List[Atoms],
    val: List[Atoms],
    test: List[Atoms],
    output_dir: Path,
) -> None:
    """
    Write an HDF5 file with top-level groups ``train``, ``val``, ``test``.

    Frames are bucketed by species composition (tuple of atomic numbers
    in system order) so each bucket has a uniform ``[n_frames, N, …]``
    shape.  Buckets become subgroups named ``bucket_<idx>``:

        train/bucket_0/species      [n_frames, N]     int64
        train/bucket_0/coordinates  [n_frames, N, 3]  float64  (Å)
        train/bucket_0/energies     [n_frames]        float64  (Hartree)
        train/bucket_0/forces       [n_frames, N, 3]  float64  (Hartree/Å)

    Mirrors TorchANI's native layout.  Energies/forces are stored in
    Hartree (TorchANI's native unit); the XYZ's eV values are converted
    automatically.
    """
    try:
        import h5py
    except ImportError:
        print("  [skip] h5py not installed – skipping .h5 output")
        return

    from ..constants import HARTREE_TO_EV  # 1 Ha = 27.21 eV → eV / factor = Ha

    h5_dir = output_dir / "torchani"
    h5_dir.mkdir(parents=True, exist_ok=True)
    h5_path = h5_dir / "data.h5"

    with h5py.File(str(h5_path), "w") as f:
        for name, frames in [("train", train), ("val", val), ("test", test)]:
            if len(frames) == 0:
                continue
            grp = f.create_group(name)

            # Bucket frames by species composition (tuple of Z in order).
            buckets: dict = {}
            for atoms in frames:
                key = tuple(int(z) for z in atoms.get_atomic_numbers())
                buckets.setdefault(key, []).append(atoms)

            for bucket_idx, (species_key, bucket_frames) in enumerate(buckets.items()):
                species_arr = np.tile(
                    np.asarray(species_key, dtype=np.int64),
                    (len(bucket_frames), 1),
                )
                coords_list = []
                energy_list = []
                forces_list = []
                for atoms in bucket_frames:
                    coords_list.append(atoms.get_positions())

                    e_ev = get_energy(atoms)
                    energy_list.append(
                        e_ev / HARTREE_TO_EV if e_ev is not None else 0.0
                    )

                    frc = get_forces(atoms)
                    if frc is not None:
                        forces_list.append(frc / HARTREE_TO_EV)
                    else:
                        forces_list.append(np.zeros_like(coords_list[-1]))

                sub = grp.create_group(f"bucket_{bucket_idx}")
                sub.create_dataset("species", data=species_arr)
                sub.create_dataset(
                    "coordinates",
                    data=np.asarray(coords_list, dtype=np.float64),
                )
                sub.create_dataset(
                    "energies",
                    data=np.asarray(energy_list, dtype=np.float64),
                )
                sub.create_dataset(
                    "forces",
                    data=np.asarray(forces_list, dtype=np.float64),
                )

            print(
                f"  {h5_path}:{name}  "
                f"({len(frames)} frames in {len(buckets)} composition bucket(s))"
            )


# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare training data from extended XYZ for all supported MLIP frameworks",
    )
    parser.add_argument("--xyz", required=True,
                        help="Path to extended XYZ file")
    parser.add_argument("--output-dir", default="./prepared_data",
                        help="Output directory (default: ./prepared_data)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.xyz} ...")
    frames = ase_read(args.xyz, index=":")
    print(f"  {len(frames)} frames loaded")

    # Quick sanity check
    sample = frames[0]
    has_energy = get_energy(sample) is not None
    has_forces = get_forces(sample) is not None
    if not has_energy:
        print("  WARNING: no energy found – training may fail")
    if not has_forces:
        print("  WARNING: no forces found – training may fail")

    print(f"Splitting ({args.train_ratio}/{args.val_ratio}/{args.test_ratio}, seed={args.seed}) ...")
    train, val, test = split_frames(
        frames,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  train={len(train)}, val={len(val)}, test={len(test)}")

    print("\nWriting XYZ splits (MACE / NequIP / Allegro) ...")
    write_xyz_splits(train, val, test, output_dir)

    print("\nWriting SchNetPack ASE DB splits ...")
    write_schnetpack_db(train, val, test, output_dir)

    print("\nWriting TorchANI HDF5 ...")
    write_torchani_h5(train, val, test, output_dir)

    print(f"\nDone. All outputs in: {output_dir}/")


if __name__ == "__main__":
    main()

