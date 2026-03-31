"""
Tests for the QM data generation module.

Uses a mock ASE calculator (no real ORCA needed) to verify that the
data generator:
  - Produces extended XYZ with correct energy/forces keys
  - Handles calculation failures gracefully (skips bad frames)
  - Sampling utilities generate the right number of structures
"""

import numpy as np
import pytest

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes, CalculationFailed


# ===================================================================
#  Mock calculator
# ===================================================================

class MockOrcaCalculator(Calculator):
    """
    Fake ASE calculator that returns random energy/forces.
    Optionally raises CalculationFailed for specific frame indices.
    """
    implemented_properties = ["energy", "forces"]

    def __init__(self, fail_indices=None, **kwargs):
        super().__init__(**kwargs)
        self.fail_indices = set(fail_indices or [])
        self._call_count = 0

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        idx = self._call_count
        self._call_count += 1

        if idx in self.fail_indices:
            raise CalculationFailed("Mock SCF failure")

        rng = np.random.default_rng(idx)
        n = len(self.atoms)
        self.results["energy"] = -76.0 + rng.normal(0, 0.5)
        self.results["forces"] = rng.normal(0, 0.1, (n, 3))


# ===================================================================
#  Fixtures
# ===================================================================

@pytest.fixture
def water_frames():
    """5 slightly displaced water molecules."""
    rng = np.random.default_rng(99)
    frames = []
    for _ in range(5):
        pos = np.array([
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ]) + rng.normal(0, 0.02, (3, 3))
        frames.append(Atoms("OHH", positions=pos))
    return frames


# ===================================================================
#  Tests: OrcaDataGenerator core logic
# ===================================================================

class TestOrcaDataGeneratorLogic:
    """
    Test the data generator's frame processing WITHOUT actually calling
    ORCA.  We monkey-patch ``_run_single_point`` to use the mock.
    """

    def test_run_frames_basic(self, water_frames):
        """All frames succeed → output has energy and forces."""
        from src.datagen import orca_generator as og

        # Patch the worker to use our mock calculator
        original_fn = og._run_single_point

        def mock_worker(atoms_dict, method, orca_command, orca_nprocs,
                        extra_blocks, scratch_root, frame_idx):
            rng = np.random.default_rng(frame_idx)
            n = len(atoms_dict["symbols"])
            return {
                "frame_idx": frame_idx,
                "symbols": atoms_dict["symbols"],
                "positions": atoms_dict["positions"],
                "energy": -76.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (n, 3)).tolist(),
            }

        og._run_single_point = mock_worker
        try:
            gen = og.OrcaDataGenerator(method="B3LYP def2-SVP")
            results = gen.run_frames(water_frames, n_workers=1)

            assert len(results) == 5
            for atoms in results:
                assert "energy" in atoms.info
                assert "forces" in atoms.arrays
                assert atoms.arrays["forces"].shape == (3, 3)
                assert isinstance(atoms.info["energy"], float)
        finally:
            og._run_single_point = original_fn

    def test_run_frames_with_failures(self, water_frames):
        """Failed frames are silently dropped."""
        from src.datagen import orca_generator as og

        original_fn = og._run_single_point

        def mock_worker(atoms_dict, method, orca_command, orca_nprocs,
                        extra_blocks, scratch_root, frame_idx):
            if frame_idx in (1, 3):
                return None  # simulate failure
            rng = np.random.default_rng(frame_idx)
            n = len(atoms_dict["symbols"])
            return {
                "frame_idx": frame_idx,
                "symbols": atoms_dict["symbols"],
                "positions": atoms_dict["positions"],
                "energy": -76.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (n, 3)).tolist(),
            }

        og._run_single_point = mock_worker
        try:
            gen = og.OrcaDataGenerator(method="B3LYP def2-SVP")
            results = gen.run_frames(water_frames, n_workers=1)

            assert len(results) == 3  # 5 input, 2 failed
        finally:
            og._run_single_point = original_fn

    def test_engrad_auto_appended(self):
        """EnGrad is automatically added if not in method string."""
        from src.datagen.orca_generator import OrcaDataGenerator

        gen = OrcaDataGenerator(method="B3LYP def2-SVP")
        assert "EnGrad" in gen.method

        gen2 = OrcaDataGenerator(method="B3LYP def2-SVP EnGrad")
        assert gen2.method.count("EnGrad") == 1

    def test_run_writes_xyz(self, water_frames, tmp_path):
        """The run() method writes a valid extended XYZ file."""
        from src.datagen import orca_generator as og
        from ase.io import read as ase_read, write as ase_write

        # Write input
        input_path = str(tmp_path / "input.xyz")
        ase_write(input_path, water_frames, format="extxyz")

        original_fn = og._run_single_point

        def mock_worker(atoms_dict, method, orca_command, orca_nprocs,
                        extra_blocks, scratch_root, frame_idx):
            rng = np.random.default_rng(frame_idx)
            n = len(atoms_dict["symbols"])
            return {
                "frame_idx": frame_idx,
                "symbols": atoms_dict["symbols"],
                "positions": atoms_dict["positions"],
                "energy": -76.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (n, 3)).tolist(),
            }

        og._run_single_point = mock_worker
        try:
            gen = og.OrcaDataGenerator(method="B3LYP def2-SVP")
            output_path = str(tmp_path / "output.xyz")
            n = gen.run(input_path, output_path, n_workers=1)

            assert n == 5
            # Verify the output file
            frames = ase_read(output_path, index=":")
            assert len(frames) == 5
            for f in frames:
                # ASE may store energy/forces in calculator or info
                has_energy = (
                    "energy" in f.info
                    or (f.calc is not None and "energy" in getattr(f.calc, "results", {}))
                )
                has_forces = (
                    "forces" in f.arrays
                    or (f.calc is not None and "forces" in getattr(f.calc, "results", {}))
                )
                assert has_energy, "No energy found in frame"
                assert has_forces, "No forces found in frame"
        finally:
            og._run_single_point = original_fn


# ===================================================================
#  Tests: Sampling utilities
# ===================================================================

class TestSampling:

    def test_random_displacements(self):
        from src.datagen.sampling import random_displacements

        atoms = Atoms("OHH", positions=[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ])

        frames = random_displacements(atoms, n_samples=20, amplitude=0.1)
        assert len(frames) == 20
        for f in frames:
            assert len(f) == 3
            # Positions should differ from original
            assert not np.allclose(f.get_positions(), atoms.get_positions())

    def test_random_displacements_seed_reproducibility(self):
        from src.datagen.sampling import random_displacements

        atoms = Atoms("OHH", positions=[
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ])

        frames1 = random_displacements(atoms, n_samples=10, seed=42)
        frames2 = random_displacements(atoms, n_samples=10, seed=42)

        for f1, f2 in zip(frames1, frames2):
            assert np.allclose(f1.get_positions(), f2.get_positions())

