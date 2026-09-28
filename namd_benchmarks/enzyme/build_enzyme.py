#!/usr/bin/env python
"""Build finite solvated OMP-decarboxylase droplets for the NAMD full-ML FeNNiX benchmark.

Source: PDB **1X1Z** — orotidine-5'-monophosphate decarboxylase (OMPDC), a homodimer
(chains A/B, 221 residues each) with the inhibitor **BMP** (a UMP analog, contains P)
bound at each active site, plus glycerol (GOL) and crystal waters (dropped).

Each variant is a *finite, non-periodic* cluster: tleap solvates a shell of TIP3P around
the protein and neutralises with ions, then the PBC box is **stripped** so NAMD treats the
whole thing as one finite ML region (`qmReplaceAll on`). This mirrors exactly how the water
benchmark treats a finite water cube, and matches the FeNNiX StableHLO export, which is a
fixed-shape, non-periodic graph over a fixed atom list.

Pipeline (per variant):
  RCSB 1X1Z  ->  prefilter (selected protein chains, standard residues, altloc A/' ')
             ->  pdb4amber (clean names/termini, detect S-S bonds)
             ->  tleap (ff14SB + TIP3P solvateShell + neutralising ions)
             ->  parmed (strip box; emit element-tagged system.pdb + beta=1 qm.pdb)
             +   meta.json (n_atoms, net charge -> export --total-charge)

Outputs under systems/<name>/:
  system.prmtop   AMBER topology (box stripped)        -> NAMD parmfile
  system.pdb      coordinates with element column       -> NAMD coordinates
  qm.pdb          beta=1.00 / occ=0.00 for every atom   -> NAMD qmParamPDB (all-QM)
  meta.json       {name, chains, shell_A, n_atoms, net_charge, ...}

Run in the cpptraj conda env (AmberTools: pdb4amber, tleap, parmed):
  conda run -n cpptraj python build_enzyme.py mono_dry
  conda run -n cpptraj python build_enzyme.py --chains AB --shell 10 --name dimer_shell10
  conda run -n cpptraj python build_enzyme.py all          # the full preset ladder
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_DIR = os.path.join(HERE, "systems")
SRC_DIR = os.path.join(SYS_DIR, "_src")
PDB_ID = "1X1Z"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    # protonation/his variants tleap also understands on input:
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
}

# Preset ladder: ascending atom count.  Small neutral solvated monomer fits an 8 GB GPU
# (FeNNiX compile + run); the solvated dimer is the "full system" and targets the 48 GB box.
# Note: dry/charged variants score sanely but the FeNNiX energy drifts fast in MD (see README);
# the neutral solvated variants are the in-distribution ones.
PRESETS = {
    "mono_dry":     dict(chains="A",  shell=0),    # 3281 atoms, q -8   (path/correctness check)
    "mono_shell5":  dict(chains="A",  shell=5),    # 4525 atoms, q  0   (local 8 GB demo)
    "dimer_dry":    dict(chains="AB", shell=0),    # 6562 atoms, q -16  (biological unit, dry)
    "dimer_shell6": dict(chains="AB", shell=6),    # ~30-40k,   q  0    (remote 48 GB: full solvated)
    "dimer_shell8": dict(chains="AB", shell=8),    # ~40-55k,   q  0    (remote 48 GB: fuller shell)
}


def fetch_source() -> str:
    os.makedirs(SRC_DIR, exist_ok=True)
    dst = os.path.join(SRC_DIR, f"{PDB_ID.lower()}.pdb")
    if not os.path.exists(dst):
        print(f"[fetch] {PDB_URL}")
        urllib.request.urlretrieve(PDB_URL, dst)
    return dst


def prefilter(src: str, chains: str, out: str) -> None:
    """Keep protein ATOM records for the selected chains (standard residues, altloc A/' ')."""
    chains = set(chains)
    kept = 0
    with open(src) as fh, open(out, "w") as oh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "TER"):
                continue
            if rec == "TER":
                oh.write(line)
                continue
            chain = line[21]
            resname = line[17:20].strip()
            altloc = line[16]
            if chain not in chains or resname not in STANDARD_AA:
                continue
            if altloc not in (" ", "A"):
                continue
            # normalise altloc to blank so downstream tools don't choke
            oh.write(line[:16] + " " + line[17:])
            kept += 1
    oh_end = open(out, "a")
    oh_end.write("END\n")
    oh_end.close()
    print(f"[prefilter] chains={''.join(sorted(chains))}: kept {kept} protein atoms")


def run(cmd, **kw):
    print("   $", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def parse_sslink(path: str) -> list[tuple[int, int]]:
    pairs = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                toks = line.split()
                if len(toks) >= 2 and toks[0].isdigit() and toks[1].isdigit():
                    pairs.append((int(toks[0]), int(toks[1])))
    return pairs


def build(name: str, chains: str, shell: float) -> None:
    outdir = os.path.join(SYS_DIR, name)
    os.makedirs(outdir, exist_ok=True)
    src = fetch_source()

    filt = os.path.join(outdir, "filtered.pdb")
    prefilter(src, chains, filt)

    # pdb4amber: clean atom names / termini, detect disulfides (-> <base>_sslink)
    clean = os.path.join(outdir, "clean.pdb")
    run(["pdb4amber", "-i", filt, "-o", clean], cwd=outdir)
    ss = parse_sslink(os.path.join(outdir, "clean_sslink"))
    print(f"[pdb4amber] disulfide bonds: {ss or 'none'}")

    import parmed as pmd
    ss_lines = [f"bond mol.{i}.SG mol.{j}.SG" for i, j in ss]
    preamble = ["source leaprc.protein.ff14SB", "source leaprc.water.tip3p",
                f"mol = loadpdb {clean}"] + ss_lines

    # Pass 1 (charge probe): dry protein -> read net charge so solvation can pick the
    # correct single neutralising counter-ion (tleap neutralise takes ONE opposite ion).
    dry_prm = os.path.join(outdir, "dry.prmtop")
    dry_crd = os.path.join(outdir, "dry.inpcrd")
    tin0 = os.path.join(outdir, "tleap_dry.in")
    with open(tin0, "w") as fh:
        fh.write("\n".join(preamble + [f"saveamberparm mol {dry_prm} {dry_crd}", "quit"]) + "\n")
    run(["tleap", "-f", tin0], cwd=outdir)
    if not os.path.exists(dry_prm):
        raise SystemExit(f"tleap (dry) failed — see {outdir}/leap.log")
    prot_charge = round(sum(a.charge for a in pmd.load_file(dry_prm).atoms))
    print(f"[charge] dry protein net charge = {prot_charge}")

    # Pass 2: solvate a finite shell + neutralise (skipped for dry variants).
    prmtop = os.path.join(outdir, "system.prmtop")
    inpcrd = os.path.join(outdir, "system.inpcrd")
    if shell <= 0:
        prmtop, inpcrd = dry_prm, dry_crd       # dry variant: reuse pass-1 topology
    else:
        tin = os.path.join(outdir, "tleap.in")
        solv = [f"solvateShell mol TIP3PBOX {shell:.1f}"]
        if prot_charge < 0:
            solv.append(f"addIonsRand mol Na+ 0")    # add cations to neutralise
        elif prot_charge > 0:
            solv.append(f"addIonsRand mol Cl- 0")    # add anions to neutralise
        with open(tin, "w") as fh:
            fh.write("\n".join(preamble + solv +
                               [f"saveamberparm mol {prmtop} {inpcrd}", "quit"]) + "\n")
        run(["tleap", "-f", tin], cwd=outdir)
        if not os.path.exists(prmtop):
            raise SystemExit(f"tleap (solvated) failed — see {outdir}/leap.log")

    # parmed: strip PBC box (finite cluster), emit element-tagged coords + all-QM selection.
    # Always write the final topology to system.prmtop (dry variants reuse dry.prmtop above).
    sys_prm = os.path.join(outdir, "system.prmtop")
    s = pmd.load_file(prmtop, xyz=inpcrd)
    s.box = None
    net_charge = round(sum(a.charge for a in s.atoms))
    s.save(sys_prm, format="amber", overwrite=True)          # finite cluster, IFBOX=0
    s.save(os.path.join(outdir, "system.pdb"), format="pdb", overwrite=True)
    for a in s.atoms:
        a.occupancy = 0.0    # qmBondColumn (occ) -> no QM-MM boundary bonds
        a.bfactor = 1.0      # QMColumn (beta)    -> every atom in QM group 1
    s.save(os.path.join(outdir, "qm.pdb"), format="pdb", overwrite=True)

    meta = dict(name=name, pdb_id=PDB_ID, chains="".join(sorted(set(chains))),
                shell_A=shell, n_atoms=len(s.atoms), net_charge=net_charge,
                disulfides=len(ss), water_model="TIP3P", ff="ff14SB",
                periodic=False, all_qm=True)
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[built] {name}: {len(s.atoms)} atoms, net charge {net_charge}, "
          f"{len(ss)} S-S, shell {shell} A  ->  {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", default="mono_dry",
                    help="preset name (%s) or 'all', or use --name with --chains/--shell"
                         % ", ".join(PRESETS))
    ap.add_argument("--chains", default=None, help="chain IDs, e.g. A or AB")
    ap.add_argument("--shell", type=float, default=None, help="TIP3P shell thickness (A); 0 = dry")
    ap.add_argument("--name", default=None, help="output system name (custom build)")
    args = ap.parse_args()

    if args.name:
        if args.chains is None or args.shell is None:
            ap.error("--name requires --chains and --shell")
        build(args.name, args.chains, args.shell)
    elif args.preset == "all":
        for nm, cfg in PRESETS.items():
            build(nm, **cfg)
    elif args.preset in PRESETS:
        build(args.preset, **PRESETS[args.preset])
    else:
        ap.error(f"unknown preset {args.preset!r}; choose from {list(PRESETS)} or 'all'")


if __name__ == "__main__":
    main()
