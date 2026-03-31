"""
QM data generation for ML interatomic potential training.

Provides ORCA 6 (via ASE's calculator interface) single-point
calculations and geometry sampling utilities.  Output is an extended
XYZ file directly consumable by ``src.training.prepare_data``.
"""

