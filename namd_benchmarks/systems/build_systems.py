#!/usr/bin/env python
"""Build TIP3P water boxes of exact molecule counts for the NAMD QM benchmark.

For each requested water-molecule count K it produces, under
  systems/w<K>_<3K>atoms/
the files NAMD needs:
  water.prmtop      AMBER topology  (parmfile)
  water.pdb         coordinates     (coordinates)   -- with element columns
  qm.pdb            QM selection    (qmParamPDB)     -- beta=1.00 for every atom

Pipeline per size:  packmol (exact count, ~1 g/cm^3 cube)  ->  tleap (TIP3P
params)  ->  parmed (PDBs with element symbols + beta column).

Run inside the cpptraj conda env (has packmol, tleap, parmed):
  conda run -n cpptraj python build_systems.py 10 20 60 100 300 600 1000 2000
"""
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "water_template.pdb")
NUMBER_DENSITY = 0.03338  # water molecules per Angstrom^3 (~1 g/cm^3)


def build_one(k_waters: int) -> str:
    n_atoms = 3 * k_waters
    name = f"w{k_waters}_{n_atoms}atoms"
    outdir = os.path.join(HERE, name)
    os.makedirs(outdir, exist_ok=True)

    # Cube side for ~1 g/cm^3, with a small margin so packmol has room to pack
    # tiny boxes without clashes that would explode the ML dynamics.
    side = (k_waters / NUMBER_DENSITY) ** (1.0 / 3.0) + 2.0

    packed = os.path.join(outdir, "packed.pdb")
    pmin = os.path.join(outdir, "packmol.inp")
    with open(pmin, "w") as fh:
        fh.write(
            f"tolerance 2.0\nfiletype pdb\nseed 12345\noutput {packed}\n\n"
            f"structure {TEMPLATE}\n  number {k_waters}\n"
            f"  inside box 0. 0. 0. {side:.3f} {side:.3f} {side:.3f}\nend structure\n"
        )
    with open(pmin) as fh:
        subprocess.run(["packmol"], stdin=fh, cwd=outdir, check=True,
                       stdout=subprocess.DEVNULL)

    prmtop = os.path.join(outdir, "water.prmtop")
    inpcrd = os.path.join(outdir, "water.inpcrd")
    tin = os.path.join(outdir, "tleap.in")
    with open(tin, "w") as fh:
        fh.write(
            "source leaprc.water.tip3p\n"
            f"sys = loadpdb {packed}\n"
            f"saveamberparm sys {prmtop} {inpcrd}\n"
            "quit\n"
        )
    subprocess.run(["tleap", "-f", tin], cwd=outdir, check=True,
                   stdout=subprocess.DEVNULL)

    # parmed: emit element-tagged coordinate PDB and the beta=1 QM-selection PDB.
    import parmed as pmd
    s = pmd.load_file(prmtop, xyz=inpcrd)
    s.save(os.path.join(outdir, "water.pdb"), format="pdb", overwrite=True)
    for a in s.atoms:
        a.occupancy = 0.0   # qmBondColumn (occ) -> no QM-MM bonds
        a.bfactor = 1.0     # QMColumn (beta)    -> all atoms in QM group 1
    s.save(os.path.join(outdir, "qm.pdb"), format="pdb", overwrite=True)

    print(f"[built] {name}: {k_waters} waters / {n_atoms} atoms, "
          f"box {side:.1f} A")
    return outdir


if __name__ == "__main__":
    counts = [int(x) for x in sys.argv[1:]] or [10]
    for k in counts:
        build_one(k)
