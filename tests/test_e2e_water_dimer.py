#!/usr/bin/env python
"""
End-to-end integration test: water-dimer pipeline.

Generates 20 water-dimer configurations, runs ORCA single-point
calculations (energy + gradients), prepares data for every framework,
trains one model of each type, wraps them for NAMD, and verifies the
output contract.

Requirements
~~~~~~~~~~~~
* ORCA 6 must be available (``module load orca`` or ``$ASE_ORCA_COMMAND``).
* Python packages for all ML frameworks:
      pip install mace-torch nequip allegro schnetpack torchani h5py ase pyyaml
* Marked ``@pytest.mark.slow`` — skipped by default in normal ``pytest``
  runs.  Use ``pytest -m slow`` or ``pytest --run-slow`` to include it.

Standalone
~~~~~~~~~~
::

    python tests/test_e2e_water_dimer.py          # full pipeline
    python tests/test_e2e_water_dimer.py --skip-orca   # reuse existing data.xyz
    python tests/test_e2e_water_dimer.py --workdir /tmp/my_run
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

# Ensure the project root is on sys.path so ``src`` imports work
# regardless of how the script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ===================================================================
#  Constants
# ===================================================================

# Equilibrium water-dimer geometry (Å) — Cs symmetry, OO ~ 2.91 Å
_WATER_DIMER_POSITIONS = np.array([
    # Water 1 (acceptor)
    [-1.551007,  -0.114520,   0.000000],   # O
    [-1.934259,   0.762503,   0.000000],   # H
    [-0.599677,   0.040712,   0.000000],   # H
    # Water 2 (donor)
    [ 1.350625,   0.111469,   0.000000],   # O
    [ 1.680398,  -0.373741,  -0.758561],   # H
    [ 1.680398,  -0.373741,   0.758561],   # H
], dtype=np.float64)

_WATER_DIMER_SYMBOLS = ["O", "H", "H", "O", "H", "H"]

# ===================================================================
#  Step 0 — Generate 20 displaced water dimers
# ===================================================================

def generate_water_dimers(
    n_samples: int = 20,
    amplitude: float = 0.08,
    seed: int = 42,
) -> list[Atoms]:
    """
    Create *n_samples* water-dimer geometries by adding small Gaussian
    displacements to the equilibrium structure.  The undisplaced
    equilibrium is always included as the first frame.
    """
    rng = np.random.default_rng(seed)
    frames = []

    # Include the equilibrium geometry
    eq = Atoms(symbols=_WATER_DIMER_SYMBOLS, positions=_WATER_DIMER_POSITIONS)
    frames.append(eq)

    for _ in range(n_samples - 1):
        disp = rng.normal(0.0, amplitude, size=_WATER_DIMER_POSITIONS.shape)
        new = Atoms(
            symbols=_WATER_DIMER_SYMBOLS,
            positions=_WATER_DIMER_POSITIONS + disp,
        )
        frames.append(new)

    return frames


# ===================================================================
#  Step 1 — Run ORCA single-point calculations
# ===================================================================

def run_orca_calculations(
    frames: list[Atoms],
    output_xyz: str | Path,
    method: str = "B3LYP def2-SVP EnGrad",
    orca_nprocs: int = 1,
    n_workers: int = 1,
) -> Path:
    """
    Run ORCA single-point energy + gradient calculations on *frames*
    and write the results as an extended XYZ file.

    ORCA must be available on ``$PATH`` (e.g. via ``module load orca``)
    or set via ``$ASE_ORCA_COMMAND``.

    Returns the output path.
    """
    from src.datagen.orca_generator import OrcaDataGenerator

    output_xyz = Path(output_xyz)
    output_xyz.parent.mkdir(parents=True, exist_ok=True)

    gen = OrcaDataGenerator(
        method=method,
        orca_nprocs=orca_nprocs,
    )

    results = gen.run_frames(frames, n_workers=n_workers)

    if not results:
        raise RuntimeError("All ORCA calculations failed — no data generated")

    ase_write(str(output_xyz), results, format="extxyz")
    print(f"[ORCA] Wrote {len(results)}/{len(frames)} frames → {output_xyz}")
    return output_xyz


# ===================================================================
#  Step 2 — Prepare data for all frameworks
# ===================================================================

def prepare_data(xyz_path: str | Path, output_dir: str | Path) -> Path:
    """Run prepare_data to split and convert the dataset."""
    from src.training.prepare_data import main as prepare_main

    output_dir = Path(output_dir)
    prepare_main([
        "--xyz", str(xyz_path),
        "--output-dir", str(output_dir),
        "--train-ratio", "0.7",
        "--val-ratio", "0.15",
        "--test-ratio", "0.15",
        "--seed", "42",
    ])
    print(f"[prepare] Data prepared in {output_dir}")
    return output_dir


# ===================================================================
#  Step 3 — Train each model type
# ===================================================================

def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def train_mace(data_dir: Path, output_dir: Path) -> Path | None:
    """Train MACE (1–2 epochs) and return path to compiled model."""
    if not _has_command("mace_run_train"):
        print("[MACE] mace_run_train not found — SKIPPING")
        return None

    from src.training.train_mace import main as train_main

    out = output_dir / "mace"
    try:
        train_main([
            "--data-dir", str(data_dir),
            "--output-dir", str(out),
            "--r-max", "5.0",
            "--max-epochs", "2",
            "--batch-size", "5",
            "--device", "cpu",
        ])
    except SystemExit as e:
        if e.code != 0:
            print(f"[MACE] Training failed (exit {e.code})")
            return None

    compiled = out / "mace_compiled.pt"
    if not compiled.exists():
        # Try alternate names
        for p in out.glob("*.pt"):
            compiled = p
            break
        for p in out.glob("*.model"):
            compiled = p
            break

    if compiled.exists():
        print(f"[MACE] Trained model: {compiled}")
        return compiled
    else:
        print("[MACE] No compiled model found after training")
        return None


def train_nequip(data_dir: Path, output_dir: Path) -> Path | None:
    """Train NequIP (1–2 epochs) and return path to deployed model."""
    if not _has_command("nequip-train"):
        print("[NequIP] nequip-train not found — SKIPPING")
        return None

    from src.training.train_nequip import main as train_main

    out = output_dir / "nequip"
    try:
        train_main([
            "--data-dir", str(data_dir),
            "--output-dir", str(out),
            "--chemical-symbols", "H", "O",
            "--r-max", "5.0",
            "--max-epochs", "2",
            "--batch-size", "5",
        ])
    except SystemExit as e:
        if e.code != 0:
            print(f"[NequIP] Training failed (exit {e.code})")
            return None

    deployed = out / "nequip_deployed.pth"
    packaged = out / "nequip_packaged.nequip.zip"
    if deployed.exists():
        print(f"[NequIP] Deployed model: {deployed}")
        return deployed
    elif packaged.exists():
        print(f"[NequIP] Packaged model (no TorchScript): {packaged}")
        return packaged
    else:
        print("[NequIP] No model produced after training")
        return None


def train_allegro(data_dir: Path, output_dir: Path) -> Path | None:
    """Train Allegro (1–2 epochs) and return path to deployed model."""
    if not _has_command("nequip-train"):
        print("[Allegro] nequip-train not found — SKIPPING")
        return None
    try:
        import allegro  # noqa: F401
    except ImportError:
        print("[Allegro] allegro package not installed — SKIPPING")
        return None

    from src.training.train_allegro import main as train_main

    out = output_dir / "allegro"
    try:
        train_main([
            "--data-dir", str(data_dir),
            "--output-dir", str(out),
            "--chemical-symbols", "H", "O",
            "--r-max", "5.0",
            "--max-epochs", "2",
            "--batch-size", "5",
        ])
    except SystemExit as e:
        if e.code != 0:
            print(f"[Allegro] Training failed (exit {e.code})")
            return None

    deployed = out / "allegro_deployed.pth"
    packaged = out / "allegro_packaged.nequip.zip"
    if deployed.exists():
        print(f"[Allegro] Deployed model: {deployed}")
        return deployed
    elif packaged.exists():
        print(f"[Allegro] Packaged model (no TorchScript): {packaged}")
        return packaged
    else:
        print("[Allegro] No model produced after training")
        return None


def train_schnetpack(data_dir: Path, output_dir: Path) -> Path | None:
    """Train SchNetPack (2 epochs) and return path to scripted model."""
    try:
        import schnetpack  # noqa: F401
    except ImportError:
        print("[SchNetPack] schnetpack not installed — SKIPPING")
        return None

    from src.training.train_schnetpack import main as train_main

    out = output_dir / "schnetpack"
    try:
        train_main([
            "--data-dir", str(data_dir),
            "--output-dir", str(out),
            "--r-max", "5.0",
            "--n-atom-basis", "32",
            "--n-interactions", "2",
            "--max-epochs", "2",
            "--batch-size", "5",
            "--device", "cpu",
        ])
    except SystemExit as e:
        if e.code != 0:
            print(f"[SchNetPack] Training failed (exit {e.code})")
            return None

    scripted = out / "schnet_scripted.pt"
    if scripted.exists():
        print(f"[SchNetPack] Scripted model: {scripted}")
        return scripted
    else:
        print("[SchNetPack] No scripted model found after training")
        return None


def train_torchani(data_dir: Path, output_dir: Path) -> Path | None:
    """Train TorchANI (2 epochs) and return path to scripted model."""
    try:
        import torchani  # noqa: F401
    except ImportError:
        print("[TorchANI] torchani not installed — SKIPPING")
        return None

    from src.training.train_torchani import main as train_main

    out = output_dir / "torchani"
    try:
        train_main([
            "--data-dir", str(data_dir),
            "--output-dir", str(out),
            "--elements", "H", "O",
            "--max-epochs", "2",
            "--batch-size", "5",
            "--hidden-layers", "32", "32",
            "--device", "cpu",
        ])
    except SystemExit as e:
        if e.code != 0:
            print(f"[TorchANI] Training failed (exit {e.code})")
            return None

    scripted = out / "torchani_scripted.pt"
    if scripted.exists():
        print(f"[TorchANI] Scripted model: {scripted}")
        return scripted
    else:
        print("[TorchANI] No scripted model found after training")
        return None


# ===================================================================
#  Step 4 — Wrap and verify each model
# ===================================================================

def _make_test_input(device: str = "cpu"):
    """Return (coords, Z, pc_coords, pc_charges) for a water dimer."""
    coords = torch.tensor(_WATER_DIMER_POSITIONS, dtype=torch.float64, device=device)
    Z = torch.tensor([8, 1, 1, 8, 1, 1], dtype=torch.int64, device=device)
    pc_coords = torch.zeros((0, 3), dtype=torch.float64, device=device)
    pc_charges = torch.zeros(0, dtype=torch.float64, device=device)
    return coords, Z, pc_coords, pc_charges


def _verify_output(energy, forces, charges, N: int = 6, label: str = ""):
    """Assert the NAMD output contract."""
    prefix = f"[{label}] " if label else ""
    assert energy.dtype == torch.float64, f"{prefix}energy dtype {energy.dtype}"
    assert forces.dtype == torch.float64, f"{prefix}forces dtype {forces.dtype}"
    assert charges.dtype == torch.float64, f"{prefix}charges dtype {charges.dtype}"
    assert forces.shape == (N, 3), f"{prefix}forces shape {forces.shape}"
    assert charges.shape == (N,), f"{prefix}charges shape {charges.shape}"
    # Energy should be finite
    assert torch.isfinite(energy).all(), f"{prefix}energy not finite: {energy}"
    assert torch.isfinite(forces).all(), f"{prefix}forces not finite"
    print(f"{prefix}energy={energy.item():.4f} kcal/mol  "
          f"forces_rms={forces.pow(2).mean().sqrt().item():.6f}")


def wrap_and_verify_mace(compiled_path: Path, output_dir: Path) -> bool:
    """Wrap a trained MACE model and verify output."""
    from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper
    from src.export import export_wrapped

    try:
        wrapper = MACE_TS_Wrapper(str(compiled_path), device="cpu").eval()
        out_path = str(output_dir / "mace_namd.pt")
        export_wrapped(wrapper, out_path, model_type="MACE")

        # Verify with test input
        coords, Z, pc_c, pc_q = _make_test_input()
        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        _verify_output(energy, forces, charges, N=6, label="MACE")

        # Verify TorchScript reload
        reloaded = torch.jit.load(out_path)
        e2, f2, c2 = reloaded(coords, Z, pc_c, pc_q)
        _verify_output(e2, f2, c2, N=6, label="MACE-reloaded")
        return True
    except Exception as exc:
        print(f"[MACE] Wrapping failed: {exc}")
        return False


def wrap_and_verify_nequip(deployed_path: Path, output_dir: Path) -> bool:
    """Wrap a trained NequIP model and verify output."""
    from src.wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper
    from src.export import export_wrapped

    try:
        wrapper = NequIP_Allegro_Wrapper(str(deployed_path), device="cpu").eval()
        out_path = str(output_dir / "nequip_namd.pt")
        export_wrapped(wrapper, out_path, model_type="NequIP")

        coords, Z, pc_c, pc_q = _make_test_input()
        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        _verify_output(energy, forces, charges, N=6, label="NequIP")

        reloaded = torch.jit.load(out_path)
        e2, f2, c2 = reloaded(coords, Z, pc_c, pc_q)
        _verify_output(e2, f2, c2, N=6, label="NequIP-reloaded")
        return True
    except Exception as exc:
        print(f"[NequIP] Wrapping failed: {exc}")
        return False


def wrap_and_verify_allegro(deployed_path: Path, output_dir: Path) -> bool:
    """Wrap a trained Allegro model and verify output."""
    from src.wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper
    from src.export import export_wrapped

    try:
        wrapper = NequIP_Allegro_Wrapper(str(deployed_path), device="cpu").eval()
        out_path = str(output_dir / "allegro_namd.pt")
        export_wrapped(wrapper, out_path, model_type="Allegro")

        coords, Z, pc_c, pc_q = _make_test_input()
        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        _verify_output(energy, forces, charges, N=6, label="Allegro")

        reloaded = torch.jit.load(out_path)
        e2, f2, c2 = reloaded(coords, Z, pc_c, pc_q)
        _verify_output(e2, f2, c2, N=6, label="Allegro-reloaded")
        return True
    except Exception as exc:
        print(f"[Allegro] Wrapping failed: {exc}")
        return False


def wrap_and_verify_schnetpack(
    scripted_path: Path, output_dir: Path, r_max: float = 5.0,
) -> bool:
    """Wrap a trained SchNetPack model and verify output."""
    from src.wrappers.wrap_schnetpack import SchNetPack_Wrapper
    from src.export import export_wrapped

    try:
        wrapper = SchNetPack_Wrapper(
            model_path=str(scripted_path), r_max=r_max, device="cpu",
        ).eval()
        out_path = str(output_dir / "schnet_namd.pt")
        export_wrapped(wrapper, out_path, model_type="SchNetPack")

        coords, Z, pc_c, pc_q = _make_test_input()
        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        _verify_output(energy, forces, charges, N=6, label="SchNetPack")

        reloaded = torch.jit.load(out_path)
        e2, f2, c2 = reloaded(coords, Z, pc_c, pc_q)
        _verify_output(e2, f2, c2, N=6, label="SchNetPack-reloaded")
        return True
    except Exception as exc:
        print(f"[SchNetPack] Wrapping failed: {exc}")
        return False


def wrap_and_verify_torchani(scripted_path: Path, output_dir: Path) -> bool:
    """Wrap a trained TorchANI model and verify output."""
    from src.wrappers.wrap_torchani import TorchANI_Wrapper
    from src.export import export_wrapped

    try:
        wrapper = TorchANI_Wrapper(
            model_path=str(scripted_path), device="cpu",
            element_list=[1, 8],   # H, O
        ).eval()
        out_path = str(output_dir / "torchani_namd.pt")
        export_wrapped(wrapper, out_path, model_type="TorchANI")

        coords, Z, pc_c, pc_q = _make_test_input()
        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        _verify_output(energy, forces, charges, N=6, label="TorchANI")

        reloaded = torch.jit.load(out_path)
        e2, f2, c2 = reloaded(coords, Z, pc_c, pc_q)
        _verify_output(e2, f2, c2, N=6, label="TorchANI-reloaded")
        return True
    except Exception as exc:
        print(f"[TorchANI] Wrapping failed: {exc}")
        return False


# ===================================================================
#  Full pipeline orchestrator
# ===================================================================

def run_full_pipeline(
    workdir: str | Path,
    skip_orca: bool = False,
    orca_method: str = "B3LYP def2-SVP EnGrad",
    orca_nprocs: int = 1,
    n_workers: int = 1,
) -> dict:
    """
    Run the complete end-to-end pipeline.

    Returns a dict mapping model name → success (True/False/None=skipped).
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ------------------------------------------------------------------
    #  Step 0: Generate water dimers
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 0: Generate 20 water dimer configurations")
    print("=" * 70)
    frames = generate_water_dimers(n_samples=20, amplitude=0.08, seed=42)
    geoms_xyz = workdir / "water_dimer_geometries.xyz"
    ase_write(str(geoms_xyz), frames, format="extxyz")
    print(f"  Wrote {len(frames)} geometries → {geoms_xyz}")

    # ------------------------------------------------------------------
    #  Step 1: ORCA single-point calculations
    # ------------------------------------------------------------------
    data_xyz = workdir / "data.xyz"

    if skip_orca:
        print("\n" + "=" * 70)
        print("  STEP 1: ORCA calculations — SKIPPED (--skip-orca)")
        print("=" * 70)
        if not data_xyz.exists():
            raise FileNotFoundError(
                f"--skip-orca was set but {data_xyz} does not exist. "
                f"Run the pipeline once without --skip-orca first."
            )
        print(f"  Reusing existing {data_xyz}")
    else:
        print("\n" + "=" * 70)
        print("  STEP 1: Run ORCA DFT calculations (B3LYP/def2-SVP)")
        print("=" * 70)
        data_xyz = run_orca_calculations(
            frames,
            output_xyz=data_xyz,
            method=orca_method,
            orca_nprocs=orca_nprocs,
            n_workers=n_workers,
        )

    # Sanity check
    computed_frames = ase_read(str(data_xyz), index=":")
    if not isinstance(computed_frames, list):
        computed_frames = [computed_frames]
    print(f"  {len(computed_frames)} frames with energy+forces available")

    # ------------------------------------------------------------------
    #  Step 2: Prepare data for all frameworks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 2: Prepare data (split + convert for all frameworks)")
    print("=" * 70)
    prepared_dir = workdir / "prepared"
    prepare_data(data_xyz, prepared_dir)

    # ------------------------------------------------------------------
    #  Step 3: Train each model type
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 3: Train models (minimal epochs)")
    print("=" * 70)
    trained_dir = workdir / "trained"
    trained_dir.mkdir(parents=True, exist_ok=True)

    trained_models = {}

    print("\n--- MACE ---")
    trained_models["mace"] = train_mace(prepared_dir, trained_dir)

    print("\n--- NequIP ---")
    trained_models["nequip"] = train_nequip(prepared_dir, trained_dir)

    print("\n--- Allegro ---")
    trained_models["allegro"] = train_allegro(prepared_dir, trained_dir)

    print("\n--- SchNetPack ---")
    trained_models["schnetpack"] = train_schnetpack(prepared_dir, trained_dir)

    print("\n--- TorchANI ---")
    trained_models["torchani"] = train_torchani(prepared_dir, trained_dir)

    # ------------------------------------------------------------------
    #  Step 4: Wrap and verify
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  STEP 4: Wrap trained models for NAMD and verify output")
    print("=" * 70)
    wrapped_dir = workdir / "wrapped"
    wrapped_dir.mkdir(parents=True, exist_ok=True)

    for name, model_path in trained_models.items():
        print(f"\n--- {name.upper()} ---")
        if model_path is None:
            print(f"  [{name}] No model to wrap (training skipped or failed)")
            results[name] = None
            continue

        if name == "mace":
            results[name] = wrap_and_verify_mace(model_path, wrapped_dir)
        elif name == "nequip":
            # Only wrap .pth files (TorchScript); skip .nequip.zip
            if str(model_path).endswith(".pth"):
                results[name] = wrap_and_verify_nequip(model_path, wrapped_dir)
            else:
                print(f"  [NequIP] Model is packaged (.zip), not TorchScript — "
                      f"cannot wrap directly")
                results[name] = None
        elif name == "allegro":
            if str(model_path).endswith(".pth"):
                results[name] = wrap_and_verify_allegro(model_path, wrapped_dir)
            else:
                print(f"  [Allegro] Model is packaged (.zip), not TorchScript — "
                      f"cannot wrap directly")
                results[name] = None
        elif name == "schnetpack":
            results[name] = wrap_and_verify_schnetpack(model_path, wrapped_dir)
        elif name == "torchani":
            results[name] = wrap_and_verify_torchani(model_path, wrapped_dir)

    # ------------------------------------------------------------------
    #  Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, status in results.items():
        if status is True:
            icon = "✓"
        elif status is False:
            icon = "✗"
        else:
            icon = "—"
        print(f"  {icon}  {name:<12s}  {'PASS' if status else 'SKIP' if status is None else 'FAIL'}")

    n_pass = sum(1 for v in results.values() if v is True)
    n_fail = sum(1 for v in results.values() if v is False)
    n_skip = sum(1 for v in results.values() if v is None)
    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print("=" * 70)

    return results


# ===================================================================
#  pytest entry points
# ===================================================================

@pytest.fixture(scope="module")
def e2e_workdir(tmp_path_factory):
    """Shared workdir for the entire E2E test module."""
    return tmp_path_factory.mktemp("e2e_water_dimer")


@pytest.fixture(scope="module")
def e2e_data_xyz(e2e_workdir):
    """
    Generate water dimers and run ORCA to produce data.xyz.

    If ORCA is not available, falls back to synthetic mock data so that
    the training + wrapping steps can still be tested.
    """
    # Generate geometries
    frames = generate_water_dimers(n_samples=20, amplitude=0.08, seed=42)
    geoms_path = e2e_workdir / "water_dimer_geometries.xyz"
    ase_write(str(geoms_path), frames, format="extxyz")

    data_path = e2e_workdir / "data.xyz"

    # Try ORCA
    orca_cmd = os.environ.get("ASE_ORCA_COMMAND", shutil.which("orca"))
    if orca_cmd and shutil.which(orca_cmd.split()[0] if orca_cmd else "orca"):
        try:
            run_orca_calculations(
                frames, output_xyz=data_path,
                method="B3LYP def2-SVP EnGrad",
                orca_nprocs=1, n_workers=1,
            )
            return str(data_path)
        except Exception as exc:
            print(f"ORCA failed ({exc}); falling back to mock data")

    # Fallback: synthetic data (random energy/forces — good enough for
    # testing the pipeline mechanics).
    print("[fixture] Using synthetic mock data (ORCA not available)")
    rng = np.random.default_rng(12345)
    mock_frames = []
    for atoms in frames:
        a = atoms.copy()
        a.info["energy"] = -152.0 + rng.normal(0, 1.0)  # ~eV for 2 waters
        a.arrays["forces"] = rng.normal(0, 0.1, (6, 3))
        mock_frames.append(a)
    ase_write(str(data_path), mock_frames, format="extxyz")
    return str(data_path)


@pytest.fixture(scope="module")
def e2e_prepared_dir(e2e_data_xyz, e2e_workdir):
    """Prepare data for all frameworks."""
    out = e2e_workdir / "prepared"
    prepare_data(e2e_data_xyz, out)
    return out


# --- Individual model tests (each marked slow) ---

@pytest.mark.slow
class TestE2E_MACE:
    def test_train_and_wrap(self, e2e_prepared_dir, e2e_workdir):
        if not _has_command("mace_run_train"):
            pytest.skip("mace_run_train not found")
        trained_dir = e2e_workdir / "trained"
        model = train_mace(e2e_prepared_dir, trained_dir)
        assert model is not None, "MACE training failed"
        ok = wrap_and_verify_mace(model, e2e_workdir / "wrapped")
        assert ok, "MACE wrapping/verification failed"


@pytest.mark.slow
class TestE2E_NequIP:
    def test_train_and_wrap(self, e2e_prepared_dir, e2e_workdir):
        if not _has_command("nequip-train"):
            pytest.skip("nequip-train not found")
        trained_dir = e2e_workdir / "trained"
        model = train_nequip(e2e_prepared_dir, trained_dir)
        assert model is not None, "NequIP training failed"
        if str(model).endswith(".pth"):
            ok = wrap_and_verify_nequip(model, e2e_workdir / "wrapped")
            assert ok, "NequIP wrapping/verification failed"


@pytest.mark.slow
class TestE2E_Allegro:
    def test_train_and_wrap(self, e2e_prepared_dir, e2e_workdir):
        if not _has_command("nequip-train"):
            pytest.skip("nequip-train not found")
        try:
            import allegro  # noqa: F401
        except ImportError:
            pytest.skip("allegro not installed")
        trained_dir = e2e_workdir / "trained"
        model = train_allegro(e2e_prepared_dir, trained_dir)
        assert model is not None, "Allegro training failed"
        if str(model).endswith(".pth"):
            ok = wrap_and_verify_allegro(model, e2e_workdir / "wrapped")
            assert ok, "Allegro wrapping/verification failed"


@pytest.mark.slow
class TestE2E_SchNetPack:
    def test_train_and_wrap(self, e2e_prepared_dir, e2e_workdir):
        try:
            import schnetpack  # noqa: F401
        except ImportError:
            pytest.skip("schnetpack not installed")
        trained_dir = e2e_workdir / "trained"
        model = train_schnetpack(e2e_prepared_dir, trained_dir)
        assert model is not None, "SchNetPack training failed"
        ok = wrap_and_verify_schnetpack(model, e2e_workdir / "wrapped")
        assert ok, "SchNetPack wrapping/verification failed"


@pytest.mark.slow
class TestE2E_TorchANI:
    def test_train_and_wrap(self, e2e_prepared_dir, e2e_workdir):
        try:
            import torchani  # noqa: F401
        except ImportError:
            pytest.skip("torchani not installed")
        trained_dir = e2e_workdir / "trained"
        model = train_torchani(e2e_prepared_dir, trained_dir)
        assert model is not None, "TorchANI training failed"
        ok = wrap_and_verify_torchani(model, e2e_workdir / "wrapped")
        assert ok, "TorchANI wrapping/verification failed"


# ===================================================================
#  Standalone CLI
# ===================================================================

def _cli():
    parser = argparse.ArgumentParser(
        description="End-to-end water dimer pipeline: "
                    "ORCA → data → train → wrap → verify",
    )
    parser.add_argument(
        "--workdir", default=None,
        help="Working directory (default: auto temp dir)",
    )
    parser.add_argument(
        "--skip-orca", action="store_true",
        help="Skip ORCA step; reuse existing data.xyz in workdir",
    )
    parser.add_argument(
        "--orca-method", default="B3LYP def2-SVP EnGrad",
        help="ORCA method line (default: 'B3LYP def2-SVP EnGrad')",
    )
    parser.add_argument(
        "--orca-nprocs", type=int, default=1,
        help="MPI processes per ORCA job (default: 1)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Parallel ORCA workers (default: 1)",
    )
    parser.add_argument(
        "--load-orca-module", action="store_true",
        help="Run 'module load orca' before starting (HPC clusters)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load ORCA module if requested
    if args.load_orca_module:
        print("Loading ORCA module...")
        # 'module' is typically a shell function; we need to source the
        # init script, then run 'module load orca', and capture the
        # resulting environment so we can apply it to this process.
        init_scripts = [
            "/etc/profile.d/modules.sh",
            "/usr/share/modules/init/bash",
            "/opt/modules/init/bash",
            "/usr/share/Modules/init/bash",
        ]
        source_cmd = ""
        for init in init_scripts:
            if os.path.isfile(init):
                source_cmd = f"source {init} 2>/dev/null; "
                break

        result = subprocess.run(
            ["bash", "-c",
             f"{source_cmd}module load orca 2>&1 >&2; env -0"],
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout:
            # Parse NUL-delimited env vars from the subshell
            for entry in result.stdout.split(b"\x00"):
                if b"=" in entry:
                    key, _, val = entry.partition(b"=")
                    os.environ[key.decode()] = val.decode()
            # Verify
            orca_bin = shutil.which("orca")
            if orca_bin:
                print(f"  ORCA module loaded: {orca_bin}")
                os.environ.setdefault("ASE_ORCA_COMMAND", orca_bin)
            else:
                print("  WARNING: 'module load orca' ran but 'orca' "
                      "still not on PATH")
        else:
            print(f"  WARNING: 'module load orca' failed: "
                  f"{result.stderr.decode().strip()}")

    # Determine workdir
    if args.workdir:
        workdir = Path(args.workdir)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="e2e_water_dimer_"))
        print(f"Workdir: {workdir}")

    results = run_full_pipeline(
        workdir=workdir,
        skip_orca=args.skip_orca,
        orca_method=args.orca_method,
        orca_nprocs=args.orca_nprocs,
        n_workers=args.n_workers,
    )

    # Exit code: nonzero if any model FAILED (not skipped)
    n_fail = sum(1 for v in results.values() if v is False)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    _cli()

