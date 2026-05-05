"""
ORCA QM/MM data generator for explicit-solvent systems.

Runs ORCA QM/MM single-point + gradient calculations with electrostatic
embedding.  The QM region is treated with a chosen DFT method; the MM
region is described by force-field parameters converted from an Amber
topology using the ``orca_mm`` utility that ships with ORCA.

The output is an extended XYZ file containing **only the QM atoms**,
annotated with:

* ``info["energy"]``   – QM/MM total energy (eV)
* ``arrays["forces"]`` – forces on QM atoms (eV/Å)
* ``arrays["charges"]``– Mulliken charges on QM atoms (e)  *(if available)*
* ``info["pc_N"]``     – number of MM point charges
* ``info["pc_charges"]``   – MM charges (e), flat list
* ``info["pc_positions"]`` – MM positions (Å), flat list (x₁ y₁ z₁ x₂ …)

This format is directly consumable by :mod:`src.training.prepare_data`.

Usage (Python API)::

    gen = QMMMDataGenerator(
        method="B3LYP def2-SVP",
        n_qm_atoms=6,
        orcaff_file="system.ORCAFF.prms",
        amber_prmtop="system.prmtop",   # optional – for MM charge metadata
    )
    gen.run("trajectory.xyz", "qmmm_data.xyz", n_workers=8)

Usage (CLI)::

    python -m src.datagen.qmmm.cli \\
        --input trajectory.xyz --output qmmm_data.xyz \\
        --method "B3LYP def2-SVP" \\
        --n-qm-atoms 6 --orcaff-file system.ORCAFF.prms \\
        --amber-prmtop system.prmtop \\
        --n-workers 8 --orca-nprocs 2
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

logger = logging.getLogger(__name__)

# ===================================================================
#  Physical constants
# ===================================================================

HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANG: float = 0.529177210903
# gradient  Eₕ/a₀  →  eV/Å
GRAD_AU_TO_EV_ANG: float = HARTREE_TO_EV / BOHR_TO_ANG   # ≈ 51.422

# Amber stores charges multiplied by 18.2223
AMBER_CHARGE_FACTOR: float = 18.2223


# ===================================================================
#  Amber topology helpers
# ===================================================================

def parse_amber_charges(prmtop_path: str) -> np.ndarray:
    """
    Extract atomic partial charges from an Amber ``.prmtop`` file.

    Amber stores charges in internal units (q × 18.2223).  This
    function returns the charges in elementary-charge units (e).

    Args:
        prmtop_path: Path to the ``.prmtop`` file.

    Returns:
        1-D float64 array of shape ``[N_atoms]``.
    """
    charges: list[float] = []
    in_charge_section = False

    with open(prmtop_path) as fh:
        for line in fh:
            # The CHARGE section is terminated by the next %FLAG.  Any
            # line starting with '%' between the flag and the data
            # (%FORMAT, optional %COMMENT lines) should be skipped.
            if line.startswith("%FLAG"):
                if in_charge_section:
                    break
                in_charge_section = line.startswith("%FLAG CHARGE")
                continue
            if not in_charge_section:
                continue
            if line.startswith("%"):
                # Skip %FORMAT / %COMMENT inside the section
                continue
            charges.extend(float(tok) for tok in line.split())

    if not charges:
        raise ValueError(
            f"Could not parse CHARGE section from {prmtop_path}"
        )
    return np.array(charges, dtype=np.float64) / AMBER_CHARGE_FACTOR


def convert_amber_topology(
    prmtop_path: str,
    inpcrd_path: str,
    output_dir: str,
    orca_mm_command: str = "orca_mm",
) -> str:
    """
    Convert an Amber topology to ORCA force-field format.

    Runs::

        orca_mm --amber2orca <prmtop> <inpcrd>

    to produce an ``.ORCAFF.prms`` file that ORCA's ``%QMMM`` block
    can reference via ``ORCAFFFilename``.

    Args:
        prmtop_path:    Path to the Amber ``.prmtop`` file.
        inpcrd_path:    Path to the Amber ``.inpcrd`` / ``.rst7`` file.
        output_dir:     Directory for the converted output.
        orca_mm_command: Name or path of the ``orca_mm`` binary.

    Returns:
        Absolute path to the generated ``.ORCAFF.prms`` file.
    """
    prmtop_path = os.path.abspath(prmtop_path)
    inpcrd_path = os.path.abspath(inpcrd_path)
    os.makedirs(output_dir, exist_ok=True)

    # orca_mm writes output next to the input files, so we copy them
    # into the output directory first.
    prmtop_copy = os.path.join(output_dir, os.path.basename(prmtop_path))
    inpcrd_copy = os.path.join(output_dir, os.path.basename(inpcrd_path))

    if os.path.abspath(prmtop_path) != os.path.abspath(prmtop_copy):
        shutil.copy2(prmtop_path, prmtop_copy)
    if os.path.abspath(inpcrd_path) != os.path.abspath(inpcrd_copy):
        shutil.copy2(inpcrd_path, inpcrd_copy)

    cmd = [orca_mm_command, "--amber2orca", prmtop_copy, inpcrd_copy]
    logger.info(f"Converting topology: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=output_dir,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"orca_mm topology conversion failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )

    prms_files = list(Path(output_dir).glob("*.ORCAFF.prms"))
    if not prms_files:
        raise FileNotFoundError(
            f"No .ORCAFF.prms file found in {output_dir} after conversion.\n"
            f"orca_mm stdout: {result.stdout[:500]}"
        )

    path = str(prms_files[0].resolve())
    logger.info(f"ORCA FF written to: {path}")
    return path


# ===================================================================
#  ORCA input generation
# ===================================================================

def _format_qm_atoms(qm_indices: Sequence[int]) -> str:
    """
    Format QM atom indices for the ORCA ``%QMMM`` block.

    Uses compact range notation ``{start:end}`` when the indices form a
    contiguous block, otherwise lists them individually.
    """
    indices = sorted(qm_indices)
    if indices == list(range(indices[0], indices[-1] + 1)):
        return f"{{{indices[0]}:{indices[-1]}}}"
    return "{" + " ".join(str(i) for i in indices) + "}"


def _write_xyz_file(
    path: str,
    symbols: Sequence[str],
    positions: np.ndarray,
) -> None:
    """Write a plain XYZ file (no extended-XYZ metadata)."""
    n = len(symbols)
    with open(path, "w") as fh:
        fh.write(f"{n}\n\n")
        for sym, (x, y, z) in zip(symbols, positions):
            fh.write(f"{sym:4s} {x:20.12f} {y:20.12f} {z:20.12f}\n")


def _build_orca_input(
    xyz_filename: str,
    method: str,
    orca_nprocs: int,
    qm_indices: Sequence[int],
    orcaff_basename: str,
    charge_total: int,
    charge_qm: int,
    mult_qm: int,
    extra_blocks: str,
) -> str:
    """Return the full text of an ORCA QM/MM input file."""
    qm_atoms_str = _format_qm_atoms(qm_indices)

    lines = [
        f"! {method}",
        f"%pal nprocs {orca_nprocs} end",
    ]

    if extra_blocks:
        lines.append(extra_blocks)

    lines.extend([
        "%QMMM",
        f"    QMAtoms {qm_atoms_str} end",
        f"    Charge_Total {charge_total}",
        f"    Charge_QM {charge_qm}",
        f'    ORCAFFFilename "{orcaff_basename}"',
        "end",
        "",
        f"* xyzfile {charge_qm} {mult_qm} {xyz_filename}",
        "",
    ])

    return "\n".join(lines)


# ===================================================================
#  ORCA output parsing
# ===================================================================

def _parse_engrad(engrad_path: str) -> tuple[float, np.ndarray]:
    """
    Parse an ORCA ``.engrad`` file.

    Returns:
        ``(energy_eV, gradients_eV_per_Ang)`` where *gradients* has
        shape ``[N_atoms, 3]``.
    """
    with open(engrad_path) as fh:
        raw_lines = fh.readlines()

    # Keep only data lines (skip comments starting with '#')
    data: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            data.append(stripped)

    # Layout:  n_atoms, energy(Eh), grad_x1, grad_y1, grad_z1, …
    n_atoms = int(data[0])
    energy_hartree = float(data[1])
    grad_flat = [float(data[2 + j]) for j in range(3 * n_atoms)]
    gradients = np.array(grad_flat, dtype=np.float64).reshape(n_atoms, 3)

    return (
        energy_hartree * HARTREE_TO_EV,
        gradients * GRAD_AU_TO_EV_ANG,
    )


def _parse_final_energy(output_path: str) -> Optional[float]:
    """
    Parse ``FINAL SINGLE POINT ENERGY`` from ORCA output (Hartree).

    Returns the energy in **eV**, or ``None`` if not found.
    """
    energy_hartree: Optional[float] = None
    with open(output_path) as fh:
        for line in fh:
            if "FINAL SINGLE POINT ENERGY" in line:
                match = re.search(
                    r"[-+]?\d+(?:\.\d+)?(?:[eEdD][-+]?\d+)?", line,
                )
                if match:
                    token = match.group().replace("D", "E").replace("d", "e")
                    energy_hartree = float(token)
    if energy_hartree is not None:
        return energy_hartree * HARTREE_TO_EV
    return None


def _parse_mulliken_charges(
    output_path: str,
    qm_indices: Sequence[int],
) -> Optional[np.ndarray]:
    """
    Parse Mulliken atomic charges for the QM atoms from the ORCA output.

    ORCA prints the Mulliken block only for QM atoms in a QM/MM run.
    We take the **last** such block in the file (post-SCF).

    Returns:
        1-D float64 array of length ``len(qm_indices)``, or ``None``.
    """
    charges: list[float] = []
    in_mulliken = False

    with open(output_path) as fh:
        for line in fh:
            if "MULLIKEN ATOMIC CHARGES" in line:
                in_mulliken = True
                charges = []           # restart – keep only the last block
                continue
            if in_mulliken:
                stripped = line.strip()
                if not stripped or "Sum of" in stripped:
                    in_mulliken = False
                    continue
                # Typical line:  "   0 O :   -0.541234"
                parts = stripped.split(":")
                if len(parts) == 2:
                    try:
                        charges.append(float(parts[1].strip()))
                    except ValueError:
                        continue

    n_qm = len(qm_indices)
    if len(charges) >= n_qm:
        return np.array(charges[:n_qm], dtype=np.float64)
    return None


# ===================================================================
#  Worker function (executed in subprocess pool)
# ===================================================================

def _run_qmmm_frame(
    atoms_dict: dict,
    method: str,
    orca_command: str,
    orca_nprocs: int,
    qm_indices: list[int],
    orcaff_file: str,
    charge_total: int,
    charge_qm: int,
    mult_qm: int,
    extra_blocks: str,
    scratch_root: str,
    frame_idx: int,
) -> Optional[dict]:
    """
    Run one ORCA QM/MM single-point + gradient calculation.

    This function is designed to be called inside a worker process.
    It communicates only through picklable dicts.

    Returns ``None`` if the calculation fails.
    """
    scratch = os.path.join(scratch_root, f"frame_{frame_idx}")
    prefix = "orca_qmmm"

    try:
        os.makedirs(scratch, exist_ok=True)

        symbols: list[str] = atoms_dict["symbols"]
        positions = np.array(atoms_dict["positions"])
        n_total = len(symbols)

        # ---- write system.xyz ----
        xyz_path = os.path.join(scratch, "system.xyz")
        _write_xyz_file(xyz_path, symbols, positions)

        # ---- copy ORCAFF file into scratch ----
        orcaff_basename = os.path.basename(orcaff_file)
        shutil.copy2(orcaff_file, os.path.join(scratch, orcaff_basename))

        # ---- write ORCA input ----
        inp_text = _build_orca_input(
            xyz_filename="system.xyz",
            method=method,
            orca_nprocs=orca_nprocs,
            qm_indices=qm_indices,
            orcaff_basename=orcaff_basename,
            charge_total=charge_total,
            charge_qm=charge_qm,
            mult_qm=mult_qm,
            extra_blocks=extra_blocks,
        )
        inp_path = os.path.join(scratch, f"{prefix}.inp")
        with open(inp_path, "w") as fh:
            fh.write(inp_text)

        # ---- run ORCA ----
        result = subprocess.run(
            [orca_command, f"{prefix}.inp"],
            cwd=scratch,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(
                f"Frame {frame_idx}: ORCA exited with code "
                f"{result.returncode}\nstderr (last 500 chars):\n"
                f"{result.stderr[-500:]}"
            )
            return None

        # ---- parse results ----
        engrad_path = os.path.join(scratch, f"{prefix}.engrad")
        output_path = os.path.join(scratch, f"{prefix}.out")

        if not os.path.exists(engrad_path):
            logger.warning(f"Frame {frame_idx}: .engrad file not found")
            return None

        energy_ev, gradients_ev_ang = _parse_engrad(engrad_path)

        # Try to get the more-specific FINAL SINGLE POINT ENERGY from
        # the output file (this is the most reliable number).
        if os.path.exists(output_path):
            fsp_energy = _parse_final_energy(output_path)
            if fsp_energy is not None:
                energy_ev = fsp_energy

        # Forces on QM atoms (F = −∇E).  ORCA's .engrad layout depends on
        # the run type: for QM/MM it contains QM atoms only (in the order
        # they appear in the system XYZ); for pure QM it contains all
        # atoms.  Dispatch on the row count so both non-contiguous and
        # contiguous QM regions work.
        n_qm = len(qm_indices)
        n_grad_rows = gradients_ev_ang.shape[0]
        if n_grad_rows == n_qm:
            qm_forces = -gradients_ev_ang
        elif n_grad_rows == n_total:
            qm_forces = -gradients_ev_ang[qm_indices]
        else:
            logger.warning(
                f"Frame {frame_idx}: .engrad has {n_grad_rows} atoms, "
                f"expected {n_qm} (QM only) or {n_total} (full system)"
            )
            return None

        # Mulliken charges (optional)
        charges = None
        if os.path.exists(output_path):
            charges = _parse_mulliken_charges(output_path, qm_indices)

        # ---- assemble result dict ----
        qm_set = set(qm_indices)
        mm_indices = [i for i in range(n_total) if i not in qm_set]

        out: dict = {
            "frame_idx": frame_idx,
            "qm_symbols": [symbols[i] for i in qm_indices],
            "qm_positions": positions[qm_indices].tolist(),
            "energy": float(energy_ev),
            "forces": qm_forces.tolist(),
            "mm_symbols": [symbols[i] for i in mm_indices],
            "mm_positions": positions[mm_indices].tolist(),
        }

        if charges is not None:
            out["charges"] = charges.tolist()

        return out

    except Exception as exc:
        logger.warning(f"Frame {frame_idx} failed: {exc}")
        return None

    finally:
        if os.path.isdir(scratch):
            shutil.rmtree(scratch, ignore_errors=True)


# ===================================================================
#  Serialisation helpers
# ===================================================================

def _atoms_to_dict(atoms: Atoms) -> dict:
    """Serialise a full-system Atoms object to a picklable dict."""
    if bool(atoms.pbc.any()) or np.any(atoms.cell.array != 0.0):
        logger.warning(
            "Frame has PBC/cell set; the QM/MM driver runs non-periodic "
            "ORCA calculations and drops this information."
        )
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
    }


def _result_to_atoms(
    d: dict,
    mm_charges: Optional[np.ndarray],
    qm_indices: Sequence[int],
    n_total: int,
) -> Atoms:
    """
    Convert a worker result dict to an Atoms object (QM region only)
    with energy, forces, charges, and point-charge metadata.
    """
    atoms = Atoms(
        symbols=d["qm_symbols"],
        positions=d["qm_positions"],
    )
    atoms.info["energy"] = d["energy"]
    atoms.arrays["forces"] = np.array(d["forces"], dtype=np.float64)

    if "charges" in d:
        atoms.arrays["charges"] = np.array(d["charges"], dtype=np.float64)

    # Store MM / point-charge metadata so the training pipeline can
    # reconstruct the electrostatic environment.
    mm_pos = d.get("mm_positions", [])
    if mm_pos:
        mm_pos_arr = np.array(mm_pos, dtype=np.float64)
        atoms.info["pc_N"] = len(mm_pos_arr)
        atoms.info["pc_positions"] = mm_pos_arr.flatten().tolist()

        if mm_charges is not None:
            qm_set = set(qm_indices)
            mm_idx = [i for i in range(n_total) if i not in qm_set]
            atoms.info["pc_charges"] = mm_charges[mm_idx].tolist()

    return atoms


# ===================================================================
#  Main generator class
# ===================================================================

class QMMMDataGenerator:
    """
    Generate QM/MM training data using ORCA with electrostatic embedding.

    The generator runs single-point + gradient calculations for each
    frame in a trajectory.  Calculations are distributed across
    ``n_workers`` processes; each ORCA invocation can additionally
    use ``orca_nprocs`` MPI ranks.

    Args:
        method:        ORCA simple-input line (e.g. ``"B3LYP def2-SVP"``).
                       ``EnGrad`` is appended automatically.
        qm_indices:    Explicit 0-based atom indices for the QM region.
        n_qm_atoms:    Short-hand: first *N* atoms are QM.  Ignored when
                       *qm_indices* is given.
        orcaff_file:   Path to the ``.ORCAFF.prms`` force-field file
                       (from :func:`convert_amber_topology`).
        amber_prmtop:  Path to the Amber ``.prmtop`` (used to read MM
                       charges for point-charge metadata in the output).
        charge_total:  Total charge of the full system.
        charge_qm:     Charge of the QM region.
        mult_qm:       Spin multiplicity of the QM region.
        orca_command:   Path or name of the ORCA binary.
        orca_nprocs:   MPI processes per ORCA calculation.
        extra_blocks:  Additional ORCA input blocks (free text).
    """

    def __init__(
        self,
        method: str = "B3LYP def2-SVP",
        qm_indices: Optional[Sequence[int]] = None,
        n_qm_atoms: Optional[int] = None,
        orcaff_file: Optional[str] = None,
        amber_prmtop: Optional[str] = None,
        charge_total: int = 0,
        charge_qm: int = 0,
        mult_qm: int = 1,
        orca_command: Optional[str] = None,
        orca_nprocs: int = 1,
        extra_blocks: str = "",
    ):
        # ---- method ---------------------------------------------------
        if "engrad" not in method.lower():
            method += " EnGrad"
        self.method = method

        # ---- QM region ------------------------------------------------
        self.qm_indices: Optional[list[int]] = (
            list(qm_indices) if qm_indices is not None else None
        )
        self.n_qm_atoms = n_qm_atoms

        # ---- force field ----------------------------------------------
        self.orcaff_file = (
            os.path.abspath(orcaff_file) if orcaff_file else None
        )

        # ---- MM charges from topology ---------------------------------
        self.mm_charges: Optional[np.ndarray] = None
        self.amber_prmtop = amber_prmtop
        if amber_prmtop is not None:
            self.mm_charges = parse_amber_charges(amber_prmtop)
            logger.info(
                f"Loaded {len(self.mm_charges)} charges from {amber_prmtop}"
            )

        # ---- ORCA settings --------------------------------------------
        self.charge_total = charge_total
        self.charge_qm = charge_qm
        self.mult_qm = mult_qm
        self.orca_nprocs = orca_nprocs
        self.extra_blocks = extra_blocks

        # Resolve ORCA binary to an absolute path so subprocess workers
        # always find it regardless of PATH differences.
        cmd = orca_command or os.environ.get("ASE_ORCA_COMMAND", "")
        if not cmd:
            cmd = shutil.which("orca") or "orca"
        elif not os.path.isabs(cmd):
            resolved = shutil.which(cmd)
            if resolved:
                cmd = resolved
        self.orca_command = cmd

    # ---------------------------------------------------------------
    #  Internal helpers
    # ---------------------------------------------------------------

    def _resolve_qm_indices(self, n_atoms: int) -> list[int]:
        """Return QM indices, resolving ``n_qm_atoms`` if needed."""
        if self.qm_indices is not None:
            return self.qm_indices
        if self.n_qm_atoms is not None:
            return list(range(min(self.n_qm_atoms, n_atoms)))
        raise ValueError(
            "Either qm_indices or n_qm_atoms must be specified."
        )

    # ---------------------------------------------------------------
    #  Public API
    # ---------------------------------------------------------------

    def run_frames(
        self,
        frames: List[Atoms],
        n_workers: int = 1,
    ) -> List[Atoms]:
        """
        Run QM/MM single-point + gradient on each frame.

        Args:
            frames:    Full-system Atoms objects (QM + MM atoms).
            n_workers: Number of parallel ORCA calculations.

        Returns:
            List of Atoms containing **only the QM atoms** with
            ``info["energy"]`` (eV), ``arrays["forces"]`` (eV/Å),
            ``arrays["charges"]`` (e), and point-charge metadata.
            Failed frames are silently dropped.
        """
        if self.orcaff_file is None or not os.path.isfile(self.orcaff_file):
            raise FileNotFoundError(
                f"ORCA FF file not found: {self.orcaff_file}\n"
                "Generate one with convert_amber_topology() or pass "
                "--orcaff-file to the CLI."
            )

        if not frames:
            return []

        n_atoms_first = len(frames[0])
        qm_indices = self._resolve_qm_indices(n_atoms_first)
        atoms_dicts = [_atoms_to_dict(f) for f in frames]
        scratch_root = tempfile.mkdtemp(prefix="orca_qmmm_")

        results: list[Optional[dict]] = [None] * len(frames)

        try:
            worker_kwargs = dict(
                method=self.method,
                orca_command=self.orca_command,
                orca_nprocs=self.orca_nprocs,
                qm_indices=qm_indices,
                orcaff_file=self.orcaff_file,
                charge_total=self.charge_total,
                charge_qm=self.charge_qm,
                mult_qm=self.mult_qm,
                extra_blocks=self.extra_blocks,
                scratch_root=scratch_root,
            )

            if n_workers <= 1:
                for i, ad in enumerate(atoms_dicts):
                    res = _run_qmmm_frame(
                        atoms_dict=ad, frame_idx=i, **worker_kwargs,
                    )
                    results[i] = res
                    if res is not None:
                        logger.info(
                            f"Frame {i}: E = {res['energy']:.6f} eV"
                        )
                    else:
                        logger.warning(f"Frame {i}: FAILED")
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = {}
                    for i, ad in enumerate(atoms_dicts):
                        fut = pool.submit(
                            _run_qmmm_frame,
                            atoms_dict=ad, frame_idx=i, **worker_kwargs,
                        )
                        futures[fut] = i

                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            res = fut.result()
                            results[idx] = res
                            if res is not None:
                                logger.info(
                                    f"Frame {idx}: "
                                    f"E = {res['energy']:.6f} eV"
                                )
                        except Exception as exc:
                            logger.warning(
                                f"Frame {idx}: worker exception: {exc}"
                            )
        finally:
            shutil.rmtree(scratch_root, ignore_errors=True)

        # Assemble successful results
        successful: list[Atoms] = []
        for r in results:
            if r is not None:
                successful.append(
                    _result_to_atoms(
                        r, self.mm_charges, qm_indices, n_atoms_first,
                    )
                )

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
        Full pipeline: read trajectory → QM/MM → write extended XYZ.

        Args:
            input_path:  XYZ file with full-system geometries.
            output_path: Output extended XYZ (QM atoms + metadata).
            n_workers:   Parallel frame calculations.

        Returns:
            Number of successfully computed frames.
        """
        frames = ase_read(input_path, index=":")
        if isinstance(frames, Atoms):
            frames = [frames]

        if not frames:
            raise ValueError(
                f"No frames found in {input_path} — nothing to compute."
            )

        n_atoms = len(frames[0])
        qm_idx = self._resolve_qm_indices(n_atoms)

        print(f"Loaded {len(frames)} frames from {input_path}")
        print(f"  Total atoms per frame : {n_atoms}")
        print(f"  QM atoms              : {len(qm_idx)}  (indices {qm_idx[0]}–{qm_idx[-1]})")
        print(f"  MM atoms              : {n_atoms - len(qm_idx)}")
        print(f"  Method                : {self.method}")
        print(f"  ORCA binary           : {self.orca_command}")
        print(f"  ORCA nprocs           : {self.orca_nprocs}")
        print(f"  ORCA FF               : {self.orcaff_file}")
        print(f"  Workers               : {n_workers}")
        print()

        results = self.run_frames(frames, n_workers=n_workers)

        if results:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            ase_write(output_path, results, format="extxyz")
            print(f"\nWrote {len(results)} QM frames to {output_path}")
        else:
            print("\nNo frames succeeded — nothing written.")

        return len(results)

