"""
Shared virial helpers, so every wrapper hands NAMD the same thing.

Models differ a lot in what they can give you.  MACE and X-MACE take
compute_virials flags and return a virial themselves.  NequIP and SchNetPack
have stress code but no runtime switch.  TorchANI has none at all.  Rather than
teach NAMD about five different capabilities, each wrapper produces the virial
whichever way suits its model and then normalises through the same function
here, so the C++ side only ever sees one shape, one unit and one sign.

THE CONTRACT NAMD EXPECTS
    shape   [3, 3] from forward, [B, 3, 3] from forward_batch
    units   kcal/mol
    sign    W = -dE/d(strain), which is the same as sum_i r_i (x) f_i
    zeros   when the system is not periodic
    always  symmetric

That sign was not guessed.  Finite difference against MACE puts the returned
virial at -dE/d(strain) to a relative 1.4e-09, and NAMD accumulates
sum_i f_i (x) r_i, which is the transpose.  Both are symmetric, so they are
equal and no transpose is needed on either side.

HOW THE STRAIN TRICK WORKS
The virial is how the energy responds to squashing the box, so introduce a
small symmetric strain D, apply it to the positions AND the cell together, and
differentiate.  Applying it to only one of the two is the classic way to get a
plausible and wrong answer, because the periodic images have to move with the
box.  Since we build the edges here, we also rebuild the shift vectors from the
strained cell, which is what carries the deformation into the imaged pairs.
"""

from typing import List, Optional, Tuple

import torch


def make_strain(cell: torch.Tensor) -> torch.Tensor:
    """
    A zeroed 3x3 to differentiate with respect to.

    It is zero so it changes nothing numerically; it exists purely to give
    autograd a handle on "how would the energy change if the box deformed".
    """
    D = torch.zeros((3, 3), dtype=cell.dtype, device=cell.device)
    return D.requires_grad_(True)


def apply_strain(
    coords: torch.Tensor,
    cell: torch.Tensor,
    unit_shifts: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deform positions, cell and shifts by the same strain.

    Returns (coords, cell, shifts).  Symmetrising D first means we only ever
    probe genuine deformations and not rigid rotations, which carry no energy
    change and would otherwise add noise to the gradient.
    """
    sym = 0.5 * (D + D.t())
    coords_s = coords + coords @ sym
    cell_s = cell + cell @ sym
    shifts_s = unit_shifts @ cell_s
    return coords_s, cell_s, shifts_s


def virial_from_strain(
    energy: torch.Tensor,
    D: torch.Tensor,
    scale: float,
    retain_graph: bool = True,
) -> torch.Tensor:
    """
    Differentiate the energy with respect to the strain.

    Use this when the model computed its own forces internally and the graph is
    still alive.  It costs an extra backward pass.  If the wrapper already calls
    autograd.grad for forces, prefer forces_and_virial below and get both out of
    one pass instead.

    `scale` converts the model's energy unit to kcal/mol.
    """
    grads: List[Optional[torch.Tensor]] = torch.autograd.grad(
        [energy.sum()], [D], create_graph=False, retain_graph=retain_graph,
        allow_unused=True,
    )
    g = grads[0]
    if g is None:
        return torch.zeros((3, 3), dtype=D.dtype, device=D.device)
    V = -g * scale
    return 0.5 * (V + V.t())


def forces_and_virial(
    energy: torch.Tensor,
    positions: torch.Tensor,
    D: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forces and virial from a single backward pass.

    This is the cheap path, and it is the one to use whenever the wrapper was
    already doing its own autograd for forces: adding D to the inputs list
    costs almost nothing, whereas a second grad call means a second traversal
    of the whole model.

    Returns (forces, virial), both scaled to kcal/mol.
    """
    grads: List[Optional[torch.Tensor]] = torch.autograd.grad(
        [energy.sum()], [positions, D], create_graph=False, retain_graph=False,
        allow_unused=True,
    )
    gp = grads[0]
    gd = grads[1]

    if gp is None:
        forces = torch.zeros_like(positions)
    else:
        forces = -gp * scale

    if gd is None:
        virial = torch.zeros((3, 3), dtype=D.dtype, device=D.device)
    else:
        V = -gd * scale
        virial = 0.5 * (V + V.t())

    return forces, virial


def zero_virial(like: torch.Tensor) -> torch.Tensor:
    """What a non-periodic system reports.  NAMD ignores it, but the return
    arity has to stay constant for TorchScript."""
    return torch.zeros((3, 3), dtype=torch.float64, device=like.device)


def zero_virials(B: int, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros((B, 3, 3), dtype=torch.float64, device=like.device)


def finalize(V: torch.Tensor) -> torch.Tensor:
    """
    Last step before handing a virial back: force float64 and symmetry.

    Symmetrising twice is harmless and cheap, and it means a wrapper that took
    a model's native virial gets the same treatment as one that derived its
    own, so the two paths cannot disagree about convention.
    """
    V64 = V.to(torch.float64)
    return 0.5 * (V64 + V64.transpose(-1, -2))
