"""Optimised inner models for NAMD: same weights, same maths, faster kernels.

Each ``build_*`` function rebuilds a model with custom GPU kernels and exact
rewrites, checks it against the stock model, and saves the TorchScript inner
that ``src.cli`` wraps.  The user-facing entry points are the ``--fast`` flags
of ``src.compile_<model>`` and ``src.build``.
"""

import math

import torch


def water_box(n_atoms: int, seed: int = 0, jitter: float = 0.02) -> torch.Tensor:
    """``[n_atoms, 3]`` float64 coordinates (Å): water molecules on a 3.1 Å
    cubic grid with a small random jitter.  Parity-check geometry only."""
    n_mol = (n_atoms + 2) // 3
    k = math.ceil(n_mol ** (1.0 / 3.0))
    mol = torch.tensor([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                       dtype=torch.float64)
    idx = torch.arange(n_mol)
    grid = torch.stack([idx // (k * k), (idx // k) % k, idx % k], dim=1).to(torch.float64) * 3.1
    xyz = (grid[:, None, :] + mol[None]).reshape(-1, 3)[:n_atoms]
    g = torch.Generator().manual_seed(seed)
    return xyz + jitter * torch.randn(xyz.shape, generator=g, dtype=torch.float64)


def report_parity(tag: str, n: int, e_ref, e, f_ref, f, unit: str) -> float:
    """Print |dE|, max |dF| and return max |dF| / max(1, |F|max).

    The checks cycle every element of the model onto the water geometry, which
    gives large forces, so force errors are judged relative to the force scale.
    """
    dE = (e - e_ref).abs().max().item()
    dF = (f - f_ref).abs().max().item()
    fmax = f_ref.abs().max().item()
    print(f"  [{tag}] N={n}  |dE| {dE:.2e} {unit}  max|dF| {dF:.2e} {unit}/A  |F|max {fmax:.1f}")
    return dF / max(1.0, fmax)
