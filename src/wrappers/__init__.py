"""
Wrapper modules for individual ML potentials.

Each wrapper is an ``nn.Module`` that exposes:
    forward(coords, Z, pc_coords, pc_charges) -> (energy, forces, charges)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges) -> (energies, forces, charges)

All outputs are in kcal/mol (energy, forces) and elementary charges (charges).
"""

from .wrap_compiled_mace import MACE_TS_Wrapper
from .wrap_compiled_nequip import NequIP_Allegro_Wrapper
from .wrap_schnetpack import SchNetPack_Wrapper
from .wrap_torchani import TorchANI_Wrapper

__all__ = [
    "MACE_TS_Wrapper",
    "NequIP_Allegro_Wrapper",
    "SchNetPack_Wrapper",
    "TorchANI_Wrapper",
]

