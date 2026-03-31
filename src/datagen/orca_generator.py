"""
ORCA-based QM data generator.

Runs DFT (or other QM) single-point calculations via ASE's ORCA
calculator and writes the results as extended XYZ files with ``energy``
(eV) in the info dict and ``forces`` (eV/Å) in the arrays dict —
exactly the format expected by :mod:`src.training.prepare_data`.

ORCA 6's Python Interface (OPI) is used through ASE, which handles
input generation, execution, and output parsing transparently.

Usage (Python API)::

    gen = OrcaDataGenerator(method="B3LYP def2-SVP", orca_nprocs=4)
    gen.run("input_geometries.xyz", "output_with_dft.xyz", n_workers=2)

Usage (CLI)::

    python -m src.datagen.cli \\
        --input geoms.xyz --output dft_data.xyz \\
        --method "B3LYP def2-SVP" --n-workers 4
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from ase import Atoms
from ase.io import read as ase_read, write as ase_write

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#  Single-frame worker (runs in a subprocess)
# -------------------------------------------------------------------

def _make_orca_calculator(
    orca_command: str,
    orca_nprocs: int,
    method: str,
    extra_blocks: str,
    directory: str,
):
    """
    Create an ASE ORCA calculator, supporting both the legacy API
    (ASE < 3.23) and the new ``OrcaProfile``-based API (ASE ≥ 3.23).
    """
    from ase.calculators.orca import ORCA

    orcasimpleinput = method
    orcablocks = f"%pal nprocs {orca_nprocs} end\n"
    if extra_blocks:
        orcablocks += extra_blocks + "\n"

    # ASE ≥ 3.23 uses OrcaProfile; older versions use command=/label=.
    try:
        from ase.calculators.orca import OrcaProfile
        profile = OrcaProfile(command=orca_command)
        calc = ORCA(
            profile=profile,
            directory=directory,
            orcasimpleinput=orcasimpleinput,
            orcablocks=orcablocks,
        )
    except ImportError:
        # Legacy ASE (< 3.23)
        calc = ORCA(
            label=os.path.join(directory, "orca"),
            command=f"{orca_command} PREFIX.inp > PREFIX.out 2>&1",
            orcasimpleinput=orcasimpleinput,
            orcablocks=orcablocks,
        )
    return calc


def _run_single_point(
    atoms_dict: dict,
    method: str,
    orca_command: str,
    orca_nprocs: int,
    extra_blocks: str,
    scratch_root: str,
    frame_idx: int,
) -> Optional[dict]:
    """
    Run one ORCA single-point calculation.

    This function is called in a worker process.  It reconstructs an
    ``Atoms`` object from a serialisable dict, attaches an ORCA
    calculator, computes energy + forces, and returns the results as
    a plain dict (picklable across process boundaries).

    Returns ``None`` if the calculation fails.
    """
    from ase import Atoms as _Atoms

    try:
        # Reconstruct Atoms
        atoms = _Atoms(
            symbols=atoms_dict["symbols"],
            positions=atoms_dict["positions"],
        )

        # Per-worker scratch directory (avoids ORCA file conflicts)
        scratch = os.path.join(scratch_root, f"frame_{frame_idx}")
        os.makedirs(scratch, exist_ok=True)

        calc = _make_orca_calculator(
            orca_command=orca_command,
            orca_nprocs=orca_nprocs,
            method=method,
            extra_blocks=extra_blocks,
            directory=scratch,
        )
        atoms.calc = calc

        energy = atoms.get_potential_energy()   # eV
        forces = atoms.get_forces()             # eV/Å

        return {
            "frame_idx": frame_idx,
            "symbols": atoms_dict["symbols"],
            "positions": atoms_dict["positions"],
            "energy": float(energy),
            "forces": forces.tolist(),
        }

    except Exception as exc:
        logger.warning(f"Frame {frame_idx} failed: {exc}")
        return None

    finally:
        # Clean up scratch
        scratch = os.path.join(scratch_root, f"frame_{frame_idx}")
        if os.path.isdir(scratch):
            shutil.rmtree(scratch, ignore_errors=True)


# -------------------------------------------------------------------
#  Atoms ↔ dict serialisation (for multiprocessing)
# -------------------------------------------------------------------

def _atoms_to_dict(atoms: Atoms) -> dict:
    """Serialise an Atoms object to a picklable dict."""
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
    }


def _dict_to_atoms(d: dict) -> Atoms:
    """Reconstruct an Atoms object from a dict, with energy/forces."""
    import numpy as np
    atoms = Atoms(symbols=d["symbols"], positions=d["positions"])
    atoms.info["energy"] = d["energy"]
    atoms.arrays["forces"] = np.array(d["forces"], dtype=np.float64)
    return atoms


# -------------------------------------------------------------------
#  Main generator class
# -------------------------------------------------------------------

class OrcaDataGenerator:
    """
    Generate QM training data by running ORCA single-point calculations.

    Args:
        method:       ORCA simple-input line (e.g. ``"B3LYP def2-SVP EnGrad"``).
                      ``EnGrad`` is appended automatically if not present.
        orca_command: Path to the ORCA binary.  Defaults to
                      ``$ASE_ORCA_COMMAND`` or ``"orca"``.
        orca_nprocs:  Number of MPI processes per ORCA calculation.
        extra_blocks: Additional ORCA input blocks (e.g. ``"%scf MaxIter 300 end"``).
    """

    def __init__(
        self,
        method: str = "B3LYP def2-SVP",
        orca_command: Optional[str] = None,
        orca_nprocs: int = 1,
        extra_blocks: str = "",
    ):
        # Ensure gradients are requested
        if "engrad" not in method.lower():
            method = method + " EnGrad"

        self.method = method
        self.orca_nprocs = orca_nprocs
        self.extra_blocks = extra_blocks

        # Resolve the ORCA binary to an absolute path so subprocess
        # workers always find it, even if PATH differs.
        cmd = orca_command or os.environ.get("ASE_ORCA_COMMAND", "")
        if not cmd:
            cmd = shutil.which("orca") or "orca"
        elif not os.path.isabs(cmd):
            resolved = shutil.which(cmd)
            if resolved:
                cmd = resolved
        self.orca_command = cmd

    def run_frames(
        self,
        frames: List[Atoms],
        n_workers: int = 1,
    ) -> List[Atoms]:
        """
        Run single-point calculations on a list of Atoms objects.

        Args:
            frames:    Input geometries.
            n_workers: Number of parallel frame calculations.
                       Set to 1 for serial execution (use
                       ``orca_nprocs`` for ORCA-internal parallelism).

        Returns:
            List of Atoms with ``info["energy"]`` (eV) and
            ``arrays["forces"]`` (eV/Å) populated.  Failed frames
            are silently dropped.
        """
        scratch_root = tempfile.mkdtemp(prefix="orca_datagen_")
        atoms_dicts = [_atoms_to_dict(f) for f in frames]
        results: List[Optional[dict]] = [None] * len(frames)

        try:
            if n_workers <= 1:
                # Serial
                for i, ad in enumerate(atoms_dicts):
                    res = _run_single_point(
                        ad, self.method, self.orca_command,
                        self.orca_nprocs, self.extra_blocks,
                        scratch_root, i,
                    )
                    results[i] = res
                    if res is not None:
                        logger.info(f"Frame {i}: E = {res['energy']:.6f} eV")
                    else:
                        logger.warning(f"Frame {i}: FAILED")
            else:
                # Parallel
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {}
                    for i, ad in enumerate(atoms_dicts):
                        fut = pool.submit(
                            _run_single_point,
                            ad, self.method, self.orca_command,
                            self.orca_nprocs, self.extra_blocks,
                            scratch_root, i,
                        )
                        futures[fut] = i

                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            res = fut.result()
                            results[idx] = res
                            if res is not None:
                                logger.info(f"Frame {idx}: E = {res['energy']:.6f} eV")
                        except Exception as exc:
                            logger.warning(f"Frame {idx}: worker exception: {exc}")

        finally:
            shutil.rmtree(scratch_root, ignore_errors=True)

        # Convert successful results to Atoms
        successful = []
        for r in results:
            if r is not None:
                successful.append(_dict_to_atoms(r))

        n_fail = len(frames) - len(successful)
        if n_fail > 0:
            logger.warning(f"{n_fail}/{len(frames)} frames failed")

        return successful

    def run(
        self,
        input_path: str,
        output_path: str,
        n_workers: int = 1,
    ) -> int:
        """
        Read geometries from *input_path*, run ORCA, write results to
        *output_path* as extended XYZ.

        Args:
            input_path:  Path to an XYZ file with input geometries.
            output_path: Path to write the output extended XYZ.
            n_workers:   Number of parallel frame calculations.

        Returns:
            Number of successfully computed frames.
        """
        frames = ase_read(input_path, index=":")
        if isinstance(frames, Atoms):
            frames = [frames]

        print(f"Loaded {len(frames)} frames from {input_path}")
        print(f"Method: {self.method}")
        print(f"ORCA: {self.orca_command} (nprocs={self.orca_nprocs})")
        print(f"Workers: {n_workers}")
        print()

        results = self.run_frames(frames, n_workers=n_workers)

        if results:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            ase_write(output_path, results, format="extxyz")
            print(f"\nWrote {len(results)} frames to {output_path}")
        else:
            print("\nNo frames succeeded — nothing written.")

        return len(results)

