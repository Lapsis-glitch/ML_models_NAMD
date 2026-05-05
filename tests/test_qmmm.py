"""
Tests for the ORCA QM/MM data generation module.

All tests are mock-based — no ORCA installation is required.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms


# ===================================================================
#  Test data
# ===================================================================

# Sample ORCA .engrad contents (3 atoms, energy in Eh, grad in Eh/bohr)
SAMPLE_ENGRAD = textwrap.dedent("""\
    #
    # Number of atoms
    #
     6
    #
    # The current total energy in Eh
    #
        -152.123456789012
    #
    # The current gradient in Eh/bohr
    #
         0.001234567890
        -0.002345678901
         0.003456789012
         0.004567890123
        -0.005678901234
         0.006789012345
         0.007890123456
        -0.008901234567
         0.009012345678
         0.000111222333
        -0.000222333444
         0.000333444555
         0.000444555666
        -0.000555666777
         0.000666777888
         0.000777888999
        -0.000888999000
         0.000999000111
    #
    # The atomic numbers and current coordinates in Bohr
    #
      8    0.00000000   0.00000000   0.22112832
      1    0.00000000   1.43052519  -0.88451328
      1    0.00000000  -1.43052519  -0.88451328
      8    5.66918711   0.00000000   0.22112832
      1    5.66918711   1.43052519  -0.88451328
      1    5.66918711  -1.43052519  -0.88451328
""")

SAMPLE_ORCA_OUTPUT = textwrap.dedent("""\
    Some header lines ...
    ...

    MULLIKEN ATOMIC CHARGES
       0 O :   -0.541234
       1 H :    0.270617
       2 H :    0.270617
    Sum of Mulliken Charges =    0.00000

    ...
    FINAL SINGLE POINT ENERGY      -152.987654321000
    ...
    TOTAL RUN TIME: 0 days 0 hours 0 minutes 42 seconds
""")

# Minimal Amber prmtop (enough to parse the CHARGE section)
SAMPLE_PRMTOP = textwrap.dedent("""\
    %VERSION  VERSION_STAMP = V0001.000  DATE = 01/01/26
    %FLAG TITLE
    %FORMAT(20a4)
    test
    %FLAG POINTERS
    %FORMAT(10I8)
           6
    %FLAG CHARGE
    %FORMAT(5E16.8)
      -1.53014232E+01   7.65071160E+00   7.65071160E+00  -1.53014232E+01
       7.65071160E+00   7.65071160E+00
    %FLAG ATOM_NAME
    %FORMAT(20a4)
    O   H1  H2  O   H1  H2
""")


# ===================================================================
#  Fixtures
# ===================================================================

@pytest.fixture
def water_dimer_frames():
    """6-atom water dimer: 3 QM atoms (water 1) + 3 MM atoms (water 2)."""
    rng = np.random.default_rng(42)
    frames = []
    base_pos = np.array([
        # Water 1 (QM)
        [0.000, 0.000, 0.117],
        [0.000, 0.757, -0.469],
        [0.000, -0.757, -0.469],
        # Water 2 (MM)
        [3.000, 0.000, 0.117],
        [3.000, 0.757, -0.469],
        [3.000, -0.757, -0.469],
    ])
    for _ in range(3):
        pos = base_pos + rng.normal(0, 0.02, base_pos.shape)
        frames.append(Atoms("OHHOHH", positions=pos))
    return frames


@pytest.fixture
def prmtop_file(tmp_path):
    """Write a minimal .prmtop file and return its path."""
    path = tmp_path / "test.prmtop"
    path.write_text(SAMPLE_PRMTOP)
    return str(path)


@pytest.fixture
def engrad_file(tmp_path):
    """Write a sample .engrad file and return its path."""
    path = tmp_path / "orca_qmmm.engrad"
    path.write_text(SAMPLE_ENGRAD)
    return str(path)


@pytest.fixture
def orca_output_file(tmp_path):
    """Write a sample ORCA output and return its path."""
    path = tmp_path / "orca_qmmm.out"
    path.write_text(SAMPLE_ORCA_OUTPUT)
    return str(path)


# ===================================================================
#  Tests: Amber topology parsing
# ===================================================================

class TestAmberParsing:

    def test_parse_charges(self, prmtop_file):
        from src.datagen.qmmm.generator import parse_amber_charges

        charges = parse_amber_charges(prmtop_file)
        assert charges.shape == (6,)
        # -15.3014232 / 18.2223 ≈ -0.8398
        assert charges[0] == pytest.approx(-0.8398, abs=0.01)
        # 7.65071160 / 18.2223 ≈ +0.4199
        assert charges[1] == pytest.approx(0.4199, abs=0.01)
        # Total should be ~0 for neutral system
        assert abs(charges.sum()) < 0.01

    def test_parse_charges_bad_file(self, tmp_path):
        from src.datagen.qmmm.generator import parse_amber_charges

        bad = tmp_path / "bad.prmtop"
        bad.write_text("nothing useful here\n")
        with pytest.raises(ValueError, match="Could not parse"):
            parse_amber_charges(str(bad))


# ===================================================================
#  Tests: ORCA output parsing
# ===================================================================

class TestOrcaParsing:

    def test_parse_engrad(self, engrad_file):
        from src.datagen.qmmm.generator import _parse_engrad, HARTREE_TO_EV

        energy_ev, grads = _parse_engrad(engrad_file)

        # Energy check
        expected_ev = -152.123456789012 * HARTREE_TO_EV
        assert energy_ev == pytest.approx(expected_ev, rel=1e-10)

        # Gradient shape
        assert grads.shape == (6, 3)

        # Gradients should be non-zero
        assert np.any(np.abs(grads) > 0)

    def test_parse_final_energy(self, orca_output_file):
        from src.datagen.qmmm.generator import _parse_final_energy

        energy_ev = _parse_final_energy(orca_output_file)
        assert energy_ev is not None
        # -152.987654321 Eh → eV
        assert energy_ev == pytest.approx(-152.987654321 * 27.211386, rel=1e-6)

    def test_parse_mulliken_charges(self, orca_output_file):
        from src.datagen.qmmm.generator import _parse_mulliken_charges

        qm_indices = [0, 1, 2]
        charges = _parse_mulliken_charges(orca_output_file, qm_indices)

        assert charges is not None
        assert len(charges) == 3
        assert charges[0] == pytest.approx(-0.541234)
        assert charges[1] == pytest.approx(0.270617)
        assert charges[2] == pytest.approx(0.270617)

    def test_parse_mulliken_returns_none_if_missing(self, tmp_path):
        from src.datagen.qmmm.generator import _parse_mulliken_charges

        empty = tmp_path / "empty.out"
        empty.write_text("no mulliken data here\n")
        assert _parse_mulliken_charges(str(empty), [0, 1, 2]) is None


# ===================================================================
#  Tests: ORCA input generation
# ===================================================================

class TestInputGeneration:

    def test_format_qm_atoms_contiguous(self):
        from src.datagen.qmmm.generator import _format_qm_atoms

        assert _format_qm_atoms([0, 1, 2, 3, 4, 5]) == "{0:5}"
        assert _format_qm_atoms([3, 4, 5, 6]) == "{3:6}"

    def test_format_qm_atoms_noncontiguous(self):
        from src.datagen.qmmm.generator import _format_qm_atoms

        result = _format_qm_atoms([0, 2, 5, 7])
        assert result == "{0 2 5 7}"

    def test_build_orca_input(self):
        from src.datagen.qmmm.generator import _build_orca_input

        text = _build_orca_input(
            xyz_filename="system.xyz",
            method="B3LYP def2-SVP EnGrad",
            orca_nprocs=4,
            qm_indices=[0, 1, 2],
            orcaff_basename="test.ORCAFF.prms",
            charge_total=0,
            charge_qm=0,
            mult_qm=1,
            extra_blocks="",
        )

        assert "! B3LYP def2-SVP EnGrad" in text
        assert "%pal nprocs 4 end" in text
        assert "QMAtoms {0:2} end" in text
        assert 'ORCAFFFilename "test.ORCAFF.prms"' in text
        assert "* xyzfile 0 1 system.xyz" in text

    def test_build_orca_input_with_extra_blocks(self):
        from src.datagen.qmmm.generator import _build_orca_input

        text = _build_orca_input(
            xyz_filename="system.xyz",
            method="B3LYP def2-SVP EnGrad",
            orca_nprocs=1,
            qm_indices=[0, 1, 2],
            orcaff_basename="test.ORCAFF.prms",
            charge_total=0,
            charge_qm=-1,
            mult_qm=2,
            extra_blocks="%scf MaxIter 300 end",
        )

        assert "%scf MaxIter 300 end" in text
        assert "Charge_QM -1" in text
        assert "* xyzfile -1 2 system.xyz" in text

    def test_write_xyz_file(self, tmp_path):
        from src.datagen.qmmm.generator import _write_xyz_file

        path = str(tmp_path / "test.xyz")
        symbols = ["O", "H", "H"]
        positions = np.array([
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ])
        _write_xyz_file(path, symbols, positions)

        with open(path) as f:
            lines = f.readlines()

        assert lines[0].strip() == "3"
        assert "O" in lines[2]


# ===================================================================
#  Tests: QMMMDataGenerator core logic (mocked ORCA)
# ===================================================================

class TestQMMMDataGeneratorLogic:
    """
    Test the generator pipeline without calling ORCA.
    Monkey-patches the worker function with a mock.
    """

    def test_run_frames_basic(self, water_dimer_frames, tmp_path):
        """All frames succeed → QM atoms with energy, forces."""
        from src.datagen.qmmm import generator as gen_mod

        # Create a dummy ORCAFF file
        orcaff = str(tmp_path / "test.ORCAFF.prms")
        Path(orcaff).write_text("dummy ff\n")

        original_fn = gen_mod._run_qmmm_frame

        def mock_worker(**kwargs):
            idx = kwargs["frame_idx"]
            ad = kwargs["atoms_dict"]
            qi = kwargs["qm_indices"]
            n = len(ad["symbols"])
            rng = np.random.default_rng(idx)

            qm_set = set(qi)
            mm_idx = [i for i in range(n) if i not in qm_set]
            pos = np.array(ad["positions"])

            return {
                "frame_idx": idx,
                "qm_symbols": [ad["symbols"][i] for i in qi],
                "qm_positions": pos[qi].tolist(),
                "energy": -152.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (len(qi), 3)).tolist(),
                "charges": rng.normal(0, 0.3, len(qi)).tolist(),
                "mm_symbols": [ad["symbols"][i] for i in mm_idx],
                "mm_positions": pos[mm_idx].tolist(),
            }

        gen_mod._run_qmmm_frame = mock_worker
        try:
            gen = gen_mod.QMMMDataGenerator(
                method="B3LYP def2-SVP",
                n_qm_atoms=3,
                orcaff_file=orcaff,
            )
            results = gen.run_frames(water_dimer_frames, n_workers=1)

            assert len(results) == 3
            for atoms in results:
                # Only QM atoms (3, not 6)
                assert len(atoms) == 3
                assert set(atoms.get_chemical_symbols()) <= {"O", "H"}
                assert "energy" in atoms.info
                assert "forces" in atoms.arrays
                assert atoms.arrays["forces"].shape == (3, 3)
                assert "charges" in atoms.arrays
                assert atoms.arrays["charges"].shape == (3,)
        finally:
            gen_mod._run_qmmm_frame = original_fn

    def test_run_frames_with_mm_charges(
        self, water_dimer_frames, prmtop_file, tmp_path,
    ):
        """MM charge metadata is stored when amber_prmtop is given."""
        from src.datagen.qmmm import generator as gen_mod

        orcaff = str(tmp_path / "test.ORCAFF.prms")
        Path(orcaff).write_text("dummy ff\n")

        original_fn = gen_mod._run_qmmm_frame

        def mock_worker(**kwargs):
            idx = kwargs["frame_idx"]
            ad = kwargs["atoms_dict"]
            qi = kwargs["qm_indices"]
            n = len(ad["symbols"])
            rng = np.random.default_rng(idx)

            qm_set = set(qi)
            mm_idx = [i for i in range(n) if i not in qm_set]
            pos = np.array(ad["positions"])

            return {
                "frame_idx": idx,
                "qm_symbols": [ad["symbols"][i] for i in qi],
                "qm_positions": pos[qi].tolist(),
                "energy": -152.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (len(qi), 3)).tolist(),
                "mm_symbols": [ad["symbols"][i] for i in mm_idx],
                "mm_positions": pos[mm_idx].tolist(),
            }

        gen_mod._run_qmmm_frame = mock_worker
        try:
            gen = gen_mod.QMMMDataGenerator(
                method="B3LYP def2-SVP",
                n_qm_atoms=3,
                orcaff_file=orcaff,
                amber_prmtop=prmtop_file,
            )
            results = gen.run_frames(water_dimer_frames, n_workers=1)

            assert len(results) == 3
            for atoms in results:
                assert "pc_N" in atoms.info
                assert atoms.info["pc_N"] == 3
                assert "pc_charges" in atoms.info
                assert len(atoms.info["pc_charges"]) == 3
                assert "pc_positions" in atoms.info
                assert len(atoms.info["pc_positions"]) == 9  # 3 atoms × 3 coords
        finally:
            gen_mod._run_qmmm_frame = original_fn

    def test_run_frames_with_failures(self, water_dimer_frames, tmp_path):
        """Failed frames are silently dropped."""
        from src.datagen.qmmm import generator as gen_mod

        orcaff = str(tmp_path / "test.ORCAFF.prms")
        Path(orcaff).write_text("dummy ff\n")

        original_fn = gen_mod._run_qmmm_frame

        def mock_worker(**kwargs):
            idx = kwargs["frame_idx"]
            if idx == 1:
                return None  # simulate failure
            ad = kwargs["atoms_dict"]
            qi = kwargs["qm_indices"]
            n = len(ad["symbols"])
            rng = np.random.default_rng(idx)
            qm_set = set(qi)
            mm_idx = [i for i in range(n) if i not in qm_set]
            pos = np.array(ad["positions"])
            return {
                "frame_idx": idx,
                "qm_symbols": [ad["symbols"][i] for i in qi],
                "qm_positions": pos[qi].tolist(),
                "energy": -152.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (len(qi), 3)).tolist(),
                "mm_symbols": [ad["symbols"][i] for i in mm_idx],
                "mm_positions": pos[mm_idx].tolist(),
            }

        gen_mod._run_qmmm_frame = mock_worker
        try:
            gen = gen_mod.QMMMDataGenerator(
                method="B3LYP def2-SVP",
                n_qm_atoms=3,
                orcaff_file=orcaff,
            )
            results = gen.run_frames(water_dimer_frames, n_workers=1)
            assert len(results) == 2  # 3 input, 1 failed
        finally:
            gen_mod._run_qmmm_frame = original_fn

    def test_engrad_auto_appended(self):
        """EnGrad is automatically added if not in method string."""
        from src.datagen.qmmm.generator import QMMMDataGenerator

        gen = QMMMDataGenerator(
            method="B3LYP def2-SVP",
            n_qm_atoms=3,
            orcaff_file="/dev/null",
        )
        assert "EnGrad" in gen.method

        gen2 = QMMMDataGenerator(
            method="B3LYP def2-SVP EnGrad",
            n_qm_atoms=3,
            orcaff_file="/dev/null",
        )
        assert gen2.method.count("EnGrad") == 1

    def test_run_writes_xyz(self, water_dimer_frames, tmp_path):
        """The run() method writes a valid extended XYZ file."""
        from src.datagen.qmmm import generator as gen_mod
        from ase.io import read as ase_read, write as ase_write

        orcaff = str(tmp_path / "test.ORCAFF.prms")
        Path(orcaff).write_text("dummy ff\n")

        # Write input
        input_path = str(tmp_path / "input.xyz")
        ase_write(input_path, water_dimer_frames, format="extxyz")

        original_fn = gen_mod._run_qmmm_frame

        def mock_worker(**kwargs):
            idx = kwargs["frame_idx"]
            ad = kwargs["atoms_dict"]
            qi = kwargs["qm_indices"]
            n = len(ad["symbols"])
            rng = np.random.default_rng(idx)
            qm_set = set(qi)
            mm_idx = [i for i in range(n) if i not in qm_set]
            pos = np.array(ad["positions"])
            return {
                "frame_idx": idx,
                "qm_symbols": [ad["symbols"][i] for i in qi],
                "qm_positions": pos[qi].tolist(),
                "energy": -152.0 + rng.normal(0, 0.5),
                "forces": rng.normal(0, 0.1, (len(qi), 3)).tolist(),
                "mm_symbols": [ad["symbols"][i] for i in mm_idx],
                "mm_positions": pos[mm_idx].tolist(),
            }

        gen_mod._run_qmmm_frame = mock_worker
        try:
            gen = gen_mod.QMMMDataGenerator(
                method="B3LYP def2-SVP",
                n_qm_atoms=3,
                orcaff_file=orcaff,
            )
            output_path = str(tmp_path / "output.xyz")
            n = gen.run(input_path, output_path, n_workers=1)

            assert n == 3
            frames = ase_read(output_path, index=":")
            assert len(frames) == 3
            for f in frames:
                assert len(f) == 3  # only QM atoms
                has_energy = (
                    "energy" in f.info
                    or (f.calc is not None and "energy" in getattr(f.calc, "results", {}))
                )
                has_forces = (
                    "forces" in f.arrays
                    or (f.calc is not None and "forces" in getattr(f.calc, "results", {}))
                )
                assert has_energy, "No energy in output frame"
                assert has_forces, "No forces in output frame"
        finally:
            gen_mod._run_qmmm_frame = original_fn


# ===================================================================
#  Tests: CLI argument parsing
# ===================================================================

class TestCLIParsing:

    def test_parse_qm_indices_comma(self):
        from src.datagen.qmmm.cli import _parse_qm_indices

        assert _parse_qm_indices("0,1,2,3") == [0, 1, 2, 3]

    def test_parse_qm_indices_range(self):
        from src.datagen.qmmm.cli import _parse_qm_indices

        assert _parse_qm_indices("0-5") == [0, 1, 2, 3, 4, 5]

    def test_parse_qm_indices_mixed(self):
        from src.datagen.qmmm.cli import _parse_qm_indices

        result = _parse_qm_indices("0-2,5,8-10")
        assert result == [0, 1, 2, 5, 8, 9, 10]

