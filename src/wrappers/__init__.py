"""
Wrapper modules for individual ML potentials.

Each wrapper is an ``nn.Module`` that exposes:
    forward(coords, Z, pc_coords, pc_charges, cell) -> (energy, forces, charges, virial)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        -> (energies, forces, charges, virials)

All outputs are float64: kcal/mol (energy, virial), kcal/mol/Å (forces) and
elementary charges.  See README.md in this directory for how to write one.
"""

from .wrap_compiled_mace import MACE_TS_Wrapper
from .wrap_compiled_nequip import NequIP_Allegro_Wrapper
from .wrap_schnetpack import SchNetPack_Wrapper
from .wrap_torchani import TorchANI_Wrapper
from .wrap_xmace import XMACE_TS_Wrapper

__all__ = [
    "MACE_TS_Wrapper",
    "NequIP_Allegro_Wrapper",
    "SchNetPack_Wrapper",
    "TorchANI_Wrapper",
    "XMACE_TS_Wrapper",
]

