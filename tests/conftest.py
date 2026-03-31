"""
Shared pytest fixtures for ML_models_NAMD tests.

Provides dummy molecular data that every wrapper can be tested against,
without requiring actual trained models.
"""

import pytest
import torch
import numpy as np

# ---- e3nn / PyTorch ≥ 2.6 compatibility patch ----
# e3nn 0.4.4's o3/_wigner.py calls torch.load("constants.pt") at import
# time without weights_only=False.  PyTorch ≥ 2.6 defaults to
# weights_only=True, rejecting the `slice` and other builtins stored
# in that file.  Patch torch.load so e3nn can import successfully.
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # If loading e3nn's constants file, force weights_only=False
    if args and isinstance(args[0], str) and "e3nn" in args[0] and "constants.pt" in args[0]:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load


# -------------------------------------------------------------------
#  Small water-trimer-like dummy data (9 atoms: 3×H₂O)
# -------------------------------------------------------------------

@pytest.fixture
def dummy_coords():
    """[9, 3] float64 – roughly a water trimer."""
    return torch.tensor([
        # Molecule 0 (water)
        [ 0.000,  0.000,  0.117],  # O
        [ 0.000,  0.757, -0.469],  # H
        [ 0.000, -0.757, -0.469],  # H
        # Molecule 1
        [ 3.000,  0.000,  0.117],
        [ 3.000,  0.757, -0.469],
        [ 3.000, -0.757, -0.469],
        # Molecule 2
        [ 6.000,  0.000,  0.117],
        [ 6.000,  0.757, -0.469],
        [ 6.000, -0.757, -0.469],
    ], dtype=torch.float64)


@pytest.fixture
def dummy_Z():
    """[9] int64 – O, H, H repeated."""
    return torch.tensor([8, 1, 1, 8, 1, 1, 8, 1, 1], dtype=torch.int64)


@pytest.fixture
def dummy_pc_coords():
    """Empty point-charge coordinates [0, 3]."""
    return torch.zeros((0, 3), dtype=torch.float64)


@pytest.fixture
def dummy_pc_charges():
    """Empty point charges [0]."""
    return torch.zeros(0, dtype=torch.float64)


# -------------------------------------------------------------------
#  Batch fixtures (3 water molecules, 3 atoms each)
# -------------------------------------------------------------------

@pytest.fixture
def dummy_batch():
    """[9] int64 – molecule index per atom."""
    return torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int64)


@pytest.fixture
def dummy_ptr():
    """[4] int64 – molecule boundaries for 3 molecules of 3 atoms."""
    return torch.tensor([0, 3, 6, 9], dtype=torch.int64)


# -------------------------------------------------------------------
#  Single molecule fixtures (first water only)
# -------------------------------------------------------------------

@pytest.fixture
def single_coords(dummy_coords):
    """[3, 3] float64 – single water molecule."""
    return dummy_coords[:3].clone()


@pytest.fixture
def single_Z(dummy_Z):
    """[3] int64 – O, H, H."""
    return dummy_Z[:3].clone()


# -------------------------------------------------------------------
#  Synthetic extended-XYZ dataset for integration tests
# -------------------------------------------------------------------

@pytest.fixture
def synthetic_xyz(tmp_path):
    """
    Write a tiny synthetic extended XYZ file (10 water frames) to a
    temp directory.  Returns the path to the file.

    Each frame has 3 atoms (O, H, H) with random energy (~-76 eV)
    and random forces.  The format matches what prepare_data.py expects.
    """
    from ase import Atoms
    from ase.io import write as ase_write

    rng = np.random.default_rng(12345)
    frames = []
    for _ in range(10):
        pos = np.array([
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ]) + rng.normal(0, 0.05, (3, 3))

        atoms = Atoms("OHH", positions=pos)
        atoms.info["energy"] = -76.0 + rng.normal(0, 0.5)
        atoms.arrays["forces"] = rng.normal(0, 0.1, (3, 3))
        frames.append(atoms)

    xyz_path = str(tmp_path / "synthetic.xyz")
    ase_write(xyz_path, frames, format="extxyz")
    return xyz_path


@pytest.fixture
def prepared_data_dir(synthetic_xyz, tmp_path):
    """
    Run prepare_data on the synthetic XYZ and return the output
    directory.  Provides xyz/, schnetpack/, torchani/ subdirs.
    """
    from src.training.prepare_data import main as prepare_main

    out_dir = str(tmp_path / "prepared")
    prepare_main([
        "--xyz", synthetic_xyz,
        "--output-dir", out_dir,
        "--train-ratio", "0.6",
        "--val-ratio", "0.2",
        "--test-ratio", "0.2",
        "--seed", "42",
    ])
    return out_dir

