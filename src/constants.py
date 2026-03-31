"""
Unit conversion constants for ML interatomic potential wrappers.

All wrappers output energy in kcal/mol, forces in kcal/mol/Å, and
charges in elementary charge units (e).  Each model's native unit
system is converted using the appropriate factor below.
"""

import torch

# ---------------------------------------------------------------------------
#  Scalar constants
# ---------------------------------------------------------------------------

EV_TO_KCAL: float = 23.0621
"""eV  →  kcal/mol   (MACE, NequIP, Allegro, SchNetPack default)"""

HARTREE_TO_KCAL: float = 627.509474
"""Hartree  →  kcal/mol   (TorchANI)"""

HARTREE_TO_EV: float = 27.211386
"""Hartree  →  eV"""

BOHR_TO_ANGSTROM: float = 0.529177
"""Bohr  →  Å"""


# ---------------------------------------------------------------------------
#  Tensor factories (for on-device multiplication)
# ---------------------------------------------------------------------------

def ev_to_kcal_tensor(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(EV_TO_KCAL, dtype=dtype)


def hartree_to_kcal_tensor(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(HARTREE_TO_KCAL, dtype=dtype)

