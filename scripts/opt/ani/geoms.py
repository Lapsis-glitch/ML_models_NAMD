"""Deterministic parity geometries shared by ref_fp64.py (torchani, fp64) and parity_native.py (artifacts only).
case = (n_atoms, walkers, periodic). Walker w of geometry g: base water box + N(0, 0.05 A) noise, seed (g, w).
Periodic cells are cubic: droplet extent + 3 A, so there are cross-boundary pairs within the 5.1 A cutoff;
geometry g=1 of periodic cases is translated so atoms sit outside the central cell."""
import os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import bench_common as bc

CASES = [(30, 1, False), (30, 4, False), (300, 1, False), (300, 4, False), (3000, 1, False),
         (30, 1, True), (300, 1, True), (300, 2, True), (900, 1, True), (3000, 1, True)]
N_GEOM = 3


def case_inputs(n, walkers, periodic, g):
    xyz, Z = bc.water_system(n)
    xs = []
    for w in range(walkers):
        gen = torch.Generator().manual_seed(1000 * g + w + 7)
        xs.append(xyz + 0.05 * torch.randn(xyz.shape, generator=gen, dtype=torch.float64))
    if periodic:
        L = float((xyz.max(0).values - xyz.min(0).values).max()) + 3.0
        cell = torch.eye(3, dtype=torch.float64) * L
        if g == 1:   # rigid translation: many atoms outside the central cell -> exercises the wrapping
            xs = [x + torch.tensor([7.3, -13.1, 25.7 + L], dtype=torch.float64) for x in xs]
    else:
        cell = torch.zeros(3, 3, dtype=torch.float64)
    return xs, Z, cell


def key(n, walkers, periodic, g):
    return f"{n}x{walkers}{'_pbc' if periodic else ''}_g{g}"
