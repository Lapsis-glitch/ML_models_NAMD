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


# ---------------------------------------------------------------------------
#  Periodic table: chemical symbol -> atomic number  (Z = 1..118)
# ---------------------------------------------------------------------------

PERIODIC_TABLE: list = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn",
    "Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd",
    "Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
    "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]

SYMBOL_TO_Z: dict = {s: i + 1 for i, s in enumerate(PERIODIC_TABLE)}

