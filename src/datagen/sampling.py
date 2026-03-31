"""
Geometry sampling utilities for QM data generation.

Provides methods to generate diverse molecular configurations from a
single equilibrium structure, which can then be fed to
:class:`~src.datagen.orca_generator.OrcaDataGenerator` for single-point
QM calculations.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional

from ase import Atoms, units


def random_displacements(
    atoms: Atoms,
    n_samples: int = 50,
    amplitude: float = 0.1,
    seed: int = 42,
) -> List[Atoms]:
    """
    Generate structures by adding random Gaussian displacements.

    This is the simplest sampling method — useful for generating an
    initial rough dataset when no Hessian or force field is available.

    Args:
        atoms:     Equilibrium structure.
        n_samples: Number of displaced structures to generate.
        amplitude: Standard deviation of displacements in Å.
        seed:      Random seed.

    Returns:
        List of displaced Atoms objects.
    """
    rng = np.random.default_rng(seed)
    frames = []
    pos0 = atoms.get_positions()
    for _ in range(n_samples):
        disp = rng.normal(0, amplitude, size=pos0.shape)
        new = atoms.copy()
        new.set_positions(pos0 + disp)
        frames.append(new)
    return frames


def normal_mode_displacements(
    atoms: Atoms,
    hessian: np.ndarray,
    temperature: float = 300.0,
    n_samples: int = 50,
    seed: int = 42,
) -> List[Atoms]:
    """
    Generate structures displaced along normal modes at a given
    temperature, weighted by the Boltzmann distribution.

    Args:
        atoms:       Equilibrium structure.
        hessian:     Hessian matrix [3N, 3N] in eV/Å².
        temperature: Temperature in Kelvin.
        n_samples:   Number of structures to generate.
        seed:        Random seed.

    Returns:
        List of displaced Atoms objects.
    """
    rng = np.random.default_rng(seed)
    masses = atoms.get_masses()
    N = len(atoms)
    pos0 = atoms.get_positions()

    # Mass-weighted Hessian
    mass_vec = np.repeat(masses, 3)  # [3N]
    sqrt_m = np.sqrt(mass_vec)
    H_mw = hessian / np.outer(sqrt_m, sqrt_m)

    # Diagonalise
    eigenvalues, eigenvectors = np.linalg.eigh(H_mw)

    # Skip translations/rotations (first 6 modes, or modes with eigenvalue ≤ 0)
    kT = units.kB * temperature  # eV
    frames = []

    for _ in range(n_samples):
        displacement = np.zeros(3 * N)
        for i in range(6, 3 * N):
            if eigenvalues[i] <= 0:
                continue
            freq = np.sqrt(eigenvalues[i])
            # Thermal amplitude: sigma = sqrt(kT / eigenvalue)
            sigma = np.sqrt(kT / eigenvalues[i]) if eigenvalues[i] > 1e-6 else 0.0
            q = rng.normal(0, sigma)
            displacement += q * eigenvectors[:, i] / sqrt_m

        new = atoms.copy()
        new.set_positions(pos0 + displacement.reshape(N, 3))
        frames.append(new)

    return frames


def md_snapshots(
    atoms: Atoms,
    calculator,
    temperature: float = 300.0,
    n_steps: int = 1000,
    interval: int = 10,
    timestep: float = 1.0,
    seed: int = 42,
) -> List[Atoms]:
    """
    Run a short MD trajectory and collect snapshots.

    The calculator can be any ASE-compatible calculator (ORCA for
    ab-initio MD, or a cheap force field for initial sampling).

    Args:
        atoms:       Starting structure.
        calculator:  ASE calculator to use for forces.
        temperature: Target temperature in Kelvin.
        n_steps:     Total MD steps.
        interval:    Collect a snapshot every *interval* steps.
        timestep:    MD timestep in femtoseconds.
        seed:        Random seed for initial velocities.

    Returns:
        List of snapshot Atoms objects (without energy/forces —
        those are computed later by the QM generator).
    """
    from ase.md.langevin import Langevin

    work = atoms.copy()
    work.calc = calculator

    # Initialise velocities
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    MaxwellBoltzmannDistribution(work, temperature_K=temperature, rng=np.random.default_rng(seed))

    dyn = Langevin(
        work,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=0.01,
    )

    frames = []
    for step in range(n_steps):
        dyn.run(1)
        if (step + 1) % interval == 0:
            snap = work.copy()
            # Strip calculator to avoid serialisation issues
            snap.calc = None
            frames.append(snap)

    return frames

