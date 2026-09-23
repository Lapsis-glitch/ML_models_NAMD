#!/usr/bin/env python
"""Convert a built enzyme droplet (system.prmtop + system.pdb) to Tinker-indexed
xyz for FeNNol's native MD driver (fennol_md, xyz_input{indexed yes}).
Usage: pdb_to_tinker_xyz.py <system_dir> <out.xyz>"""
import sys
import parmed as pmd

sysdir, out = sys.argv[1], sys.argv[2]
s = pmd.load_file(f"{sysdir}/system.prmtop", xyz=f"{sysdir}/system.pdb")

# atomic number -> element symbol (parmed exposes a Z-indexed table)
try:
    from parmed.periodic_table import Element            # list: Element[Z] -> 'C'
    sym = lambda z: Element[z]
except Exception:
    from parmed.periodic_table import AtomicNum
    znum = {v: k for k, v in AtomicNum.items()}
    sym = lambda z: znum.get(z, "X")

with open(out, "w") as f:
    f.write(f"{len(s.atoms)}\n")
    for i, a in enumerate(s.atoms, 1):
        f.write(f"{i:7d} {sym(a.atomic_number):2s} {a.xx:12.6f} {a.xy:12.6f} {a.xz:12.6f}\n")
print(f"wrote {out}: {len(s.atoms)} atoms")
