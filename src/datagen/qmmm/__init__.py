"""
ORCA QM/MM data generation with electrostatic embedding.

Runs ORCA QM/MM single-point + gradient calculations for explicit-solvent
systems.  The QM region is treated with DFT while the MM region uses
force-field parameters derived from an Amber topology via ``orca_mm``.

Typical workflow::

    # 1. Convert Amber topology → ORCA force field
    from src.datagen.qmmm import convert_amber_topology
    ff = convert_amber_topology("system.prmtop", "system.rst7", "./ff")

    # 2. Generate training data
    from src.datagen.qmmm import QMMMDataGenerator
    gen = QMMMDataGenerator(
        method="B3LYP def2-SVP",
        n_qm_atoms=6,
        orcaff_file=ff,
    )
    gen.run("trajectory.xyz", "qmmm_data.xyz", n_workers=8)
"""

from .generator import QMMMDataGenerator, convert_amber_topology, parse_amber_charges

__all__ = [
    "QMMMDataGenerator",
    "convert_amber_topology",
    "parse_amber_charges",
]

