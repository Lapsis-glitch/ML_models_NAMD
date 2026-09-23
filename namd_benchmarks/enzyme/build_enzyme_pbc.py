#!/usr/bin/env python
"""Build a *periodic* solvated OMP-decarboxylase box for the NAMD full-ML FeNNiX run.

This is the PBC companion to ``build_enzyme.py``.  Where that script makes a
finite, non-periodic droplet (``solvateShell`` + box stripped) to match the
non-periodic FeNNiX StableHLO export, this one follows the **CHARMM/VMD recipe
of the QM/MM String-eABF tutorial** (``QMMM_String_eABF_Tutorial``) to make a
genuinely periodic water box with a NAMD ``.xsc`` cell:

  1X1Z  ->  prefilter protein chain(s) (standard residues, altloc A/' ', drop OXT/H)
        ->  psfgen  (CHARMM36 ``top_all36_prot`` + water/ions; HIS->HSD, ILE CD1->CD,
                     one segment per chain, S-S via ``patch DISU``, guesscoord builds H)
        ->  VMD     center at origin
        ->  solvate cubic TIP3P box (side = solute extent + 2*pad)
        ->  autoionize -neutralize [-sc <salt>]   (SOD/CLA)
        ->  write system.psf / system.pdb         (NAMD structure + coordinates)
        ->  write system.xsc                       (cubic PBC cell, origin 0)
        ->  write qm.pdb  (beta=1/occ=0 for every atom = whole system is the ML region,
                           element columns filled via ``topo guessatom element mass`` so
                           the FeNNiX exporter resolves CHARMM names SOD/CLA/TIP3 correctly)
        ->  meta.json  {periodic:true, box, n_atoms, net_charge:0, ...}

Note on the physics: the FeNNiX graph takes only coordinates (no cell input), so
the *model* is still non-periodic — the periodic box + PME + wrapAll only keep the
system contained and give a thick, properly sized hydration shell (no droplet
surface tension).  This is exactly the "full-ML, bigger periodic box" mode the
benchmark targets; true minimum-image forces would require extending the export.

Outputs under systems/<name>/ :
  system.psf      CHARMM topology               -> NAMD ``structure``
  system.pdb      solvated+ionized coordinates  -> NAMD ``coordinates``
  system.xsc      periodic cell                 -> NAMD ``extendedSystem``
  qm.pdb          beta=1/occ=0, element column  -> NAMD ``qmParamPDB`` + FeNNiX export
  meta.json       {name, chains, pad_A, salt_M, n_atoms, net_charge, box_A, periodic}

Needs VMD (``module load vmd/2.0``) with psfgen/solvate/autoionize/topotools, and
the CHARMM toppar copied into ``toppar/`` (par_all36_prot, top_all36_prot,
toppar_water_ions_namd.str).  No AmberTools / conda env required.

  module load vmd/2.0
  python build_enzyme_pbc.py mono            # chain A   (tutorial monomer)
  python build_enzyme_pbc.py dimer           # chains AB (biological unit)
  python build_enzyme_pbc.py --chains A --pad 15 --salt 0.15 --name mono_pad15
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SYS_DIR = os.path.join(HERE, "systems")
SRC_DIR = os.path.join(SYS_DIR, "_src")
TOPPAR = os.path.join(HERE, "toppar")
PDB_ID = "1X1Z"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN",
}

PRESETS = {                       # name -> (chains, pad_A, salt_M)
    "mono":  dict(chains="A",  pad=12.0, salt=0.15),   # tutorial monomer
    "dimer": dict(chains="AB", pad=12.0, salt=0.15),   # biological unit
}


def resolve_vmd() -> str:
    cand = [os.environ.get("VMD"), shutil.which("vmd"),
            "/usr/local/bin/vmd_2.0/vmd_2.0", "/usr/local/bin/vmd_1.9.4/vmd_1.9.4"]
    for c in cand:
        if c and os.path.exists(c) and os.access(c, os.X_OK):
            return c
    raise SystemExit("VMD not found — run `module load vmd/2.0` (sets the alias) or set $VMD "
                     "to the vmd binary, e.g. /usr/local/bin/vmd_2.0/vmd_2.0")


def fetch_source() -> str:
    os.makedirs(SRC_DIR, exist_ok=True)
    dst = os.path.join(SRC_DIR, f"{PDB_ID.lower()}.pdb")
    if not os.path.exists(dst):
        print(f"[fetch] {PDB_URL}")
        urllib.request.urlretrieve(PDB_URL, dst)
    return dst


def prefilter_chain(src: str, chain: str, out: str) -> list[tuple[str, float, float, float]]:
    """Write protein ATOM records for one chain; drop OXT/hydrogens/altloc!=A.

    Returns the list of (resid, x, y, z) for CYS SG atoms (for S-S detection)."""
    sg = []
    kept = 0
    with open(src) as fh, open(out, "w") as oh:
        for line in fh:
            if line[:6].strip() != "ATOM":
                continue
            if line[21] != chain:
                continue
            resname = line[17:20].strip()
            if resname not in STANDARD_AA:
                continue
            if line[16] not in (" ", "A"):
                continue
            atom = line[12:16].strip()
            element = line[76:78].strip()
            if atom == "OXT":                       # CTER patch rebuilds OT1/OT2
                continue
            if element == "H" or (atom and atom[0] == "H"):   # crystal usually has none
                continue
            if resname == "CYS" and atom == "SG":
                sg.append((line[22:26].strip(),
                           float(line[30:38]), float(line[38:46]), float(line[46:54])))
            oh.write(line[:16] + " " + line[17:21] + chain + line[22:])   # normalise altloc
            kept += 1
        oh.write("END\n")
    print(f"[prefilter] chain {chain}: kept {kept} protein atoms, {len(sg)} CYS SG")
    return sg


def detect_disulfides(sg_by_chain: dict[str, list]) -> list[tuple[str, str, str, str]]:
    """Return (chainA, residA, chainB, residB) for SG-SG pairs within 2.5 A."""
    flat = [(ch, rid, x, y, z) for ch, lst in sg_by_chain.items() for (rid, x, y, z) in lst]
    pairs = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            ci, ri, xi, yi, zi = flat[i]
            cj, rj, xj, yj, zj = flat[j]
            d = math.dist((xi, yi, zi), (xj, yj, zj))
            if d < 2.5:
                pairs.append((ci, ri, cj, rj))
    if pairs:
        print(f"[disulfide] {len(pairs)} S-S: " +
              ", ".join(f"{a}:{ra}-{b}:{rb}" for a, ra, b, rb in pairs))
    else:
        print("[disulfide] none")
    return pairs


VMD_TCL = r"""
# ---- generated by build_enzyme_pbc.py ; CHARMM/VMD periodic solvation ----
set toppar  "@TOPPAR@"
set pad     @PAD@
set salt    @SALT@
set outdir  "@OUTDIR@"
set boxmode "@BOXMODE@"

# ===== 1) psfgen: protein -> system.psf/pdb (CHARMM36) =====================
package require psfgen
resetpsf
topology $toppar/top_all36_prot.rtf
topology $toppar/toppar_water_ions_namd.str
pdbalias residue HIS HSD
pdbalias atom ILE CD1 CD
pdbalias residue HOH TIP3
@SEGMENTS@
@PATCHES@
@COORDPDBS@
guesscoord
regenerate angles dihedrals
writepsf  $outdir/system_dry.psf
writepdb  $outdir/system_dry.pdb

# ===== 2) center the solute at the origin ==================================
mol new $outdir/system_dry.psf
mol addfile $outdir/system_dry.pdb
set all [atomselect top all]
set c [measure center $all]
$all moveby [vecscale -1.0 $c]
$all writepdb $outdir/system_dry.pdb
# per-axis solute extent + 2*pad; 'cube' grows every side to the max (tutorial style),
# 'rect' keeps each side tight (far fewer corner waters for an elongated solute).
set mm [measure minmax $all]
foreach {lo hi} $mm break
set Lx [expr {[lindex $hi 0] - [lindex $lo 0] + 2.0*$pad}]
set Ly [expr {[lindex $hi 1] - [lindex $lo 1] + 2.0*$pad}]
set Lz [expr {[lindex $hi 2] - [lindex $lo 2] + 2.0*$pad}]
if {"$boxmode" eq "cube"} {
    set Lmax $Lx
    if {$Ly > $Lmax} { set Lmax $Ly }
    if {$Lz > $Lmax} { set Lmax $Lz }
    set Lx $Lmax; set Ly $Lmax; set Lz $Lmax
}
set hx [expr {$Lx/2.0}]; set hy [expr {$Ly/2.0}]; set hz [expr {$Lz/2.0}]
puts "BUILDPBC box=$boxmode Lx=$Lx Ly=$Ly Lz=$Lz"
$all delete
mol delete top

# ===== 3) solvate a TIP3P box (centered at origin) ========================
package require solvate
solvate $outdir/system_dry.psf $outdir/system_dry.pdb \
    -minmax [list [list -$hx -$hy -$hz] [list $hx $hy $hz]] \
    -o $outdir/system_solv

# ===== 4) neutralise (+ optional salt) =====================================
# autoionize: -sc already neutralises + adds salt to the given conc; -neutralize
# is its own mode (no bulk salt).  The two are mutually exclusive in v1.x.
package require autoionize
if {$salt > 0.0} {
    autoionize -psf $outdir/system_solv.psf -pdb $outdir/system_solv.pdb \
        -sc $salt -cation SOD -anion CLA -o $outdir/system
} else {
    autoionize -psf $outdir/system_solv.psf -pdb $outdir/system_solv.pdb \
        -neutralize -cation SOD -anion CLA -o $outdir/system
}
if {![file exists $outdir/system.psf]} {
    puts "BUILDPBC ERROR autoionize did not produce system.psf"
    exit 1
}

# ===== 5) finalise: element column, qm.pdb (whole system = ML region) ======
mol delete all
mol new $outdir/system.psf
mol addfile $outdir/system.pdb
package require topotools
topo guessatom element mass
set all [atomselect top all]
set nat [$all num]
set q   [vecsum [$all get charge]]
# coordinates PDB with elements (used by NAMD coordinates AND FeNNiX export order)
$all writepdb $outdir/system.pdb
# QM/ML selection: every atom in QM group 1, no QM-MM boundary bonds
$all set beta 1.0
$all set occupancy 0.0
$all writepdb $outdir/qm.pdb
puts "BUILDPBC natoms=$nat netcharge=$q"

# ===== 6) periodic cell file (.xsc): orthorhombic, origin at 0 =============
set fp [open $outdir/system.xsc w]
puts $fp "# NAMD extended system configuration output file"
puts $fp "#\$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z s_x s_y s_z s_u s_v s_w"
puts $fp "0 $Lx 0 0 0 $Ly 0 0 0 $Lz 0 0 0 0 0 0 0 0 0"
close $fp
$all delete
exit
"""


def build(name: str, chains: str, pad: float, salt: float, box_mode: str = "cube") -> None:
    for f in ("top_all36_prot.rtf", "toppar_water_ions_namd.str", "par_all36_prot.prm"):
        if not os.path.exists(os.path.join(TOPPAR, f)):
            raise SystemExit(f"missing CHARMM file {TOPPAR}/{f} — copy the tutorial toppar in first")
    vmd = resolve_vmd()
    outdir = os.path.join(SYS_DIR, name)
    os.makedirs(outdir, exist_ok=True)
    src = fetch_source()

    # per-chain prefilter + S-S detection
    sg_by_chain: dict[str, list] = {}
    seg_lines, coord_lines, patch_lines = [], [], []
    for ch in chains:
        pf = os.path.join(outdir, f"enzyme_{ch}.pdb")
        sg_by_chain[ch] = prefilter_chain(src, ch, pf)
        seg_lines.append(f"segment {ch} {{ auto angles dihedrals\n    pdb {pf}\n}}")
        coord_lines.append(f"coordpdb {pf} {ch}")
    ss_pairs = detect_disulfides(sg_by_chain)
    for a, ra, b, rb in ss_pairs:
        patch_lines.append(f"patch DISU {a}:{ra} {b}:{rb}")

    tcl = (VMD_TCL
           .replace("@TOPPAR@", TOPPAR)
           .replace("@PAD@", f"{pad:.2f}")
           .replace("@SALT@", f"{salt:.4f}")
           .replace("@OUTDIR@", outdir)
           .replace("@BOXMODE@", box_mode)
           .replace("@SEGMENTS@", "\n".join(seg_lines))
           .replace("@PATCHES@", "\n".join(patch_lines))
           .replace("@COORDPDBS@", "\n".join(coord_lines)))
    tcl_path = os.path.join(outdir, "build_pbc.tcl")
    with open(tcl_path, "w") as fh:
        fh.write(tcl)

    log = os.path.join(outdir, "build_pbc.log")
    print(f"[vmd] {vmd} -dispdev text -e {tcl_path}  (log: {log})")
    with open(log, "w") as lf:
        rc = subprocess.run([vmd, "-dispdev", "text", "-e", tcl_path],
                            stdout=lf, stderr=subprocess.STDOUT).returncode

    # parse the BUILDPBC markers the tcl printed
    box = {"Lx": None, "Ly": None, "Lz": None}
    nat = q = None
    with open(log) as lf:
        for line in lf:
            if line.startswith("BUILDPBC"):
                for tok in line.split()[1:]:
                    k, _, v = tok.partition("=")
                    if k in box:
                        box[k] = float(v)
                    elif k == "natoms":
                        nat = int(v)
                    elif k == "netcharge":
                        q = float(v)
    need = [os.path.join(outdir, f) for f in ("system.psf", "system.pdb", "system.xsc", "qm.pdb")]
    if rc != 0 or not all(os.path.exists(p) for p in need) or nat is None or box["Lx"] is None:
        sys.stderr.write(f"\n[FAILED] {name}: VMD rc={rc}; see {log}\n")
        subprocess.run(["tail", "-25", log])
        raise SystemExit(1)

    net_charge = int(round(q)) if q is not None else 0
    box_dims = [round(box["Lx"], 3), round(box["Ly"], 3), round(box["Lz"], 3)]
    meta = dict(name=name, pdb_id=PDB_ID, chains="".join(sorted(set(chains))),
                pad_A=pad, salt_M=salt, box_mode=box_mode, n_atoms=nat, net_charge=net_charge,
                box_A=box_dims, water_model="TIP3", ff="CHARMM36",
                periodic=True, all_qm=True, disulfides=len(ss_pairs))
    with open(os.path.join(outdir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[built] {name}: {nat} atoms, net charge {net_charge}, {box_mode} box "
          f"{box_dims[0]:.1f}×{box_dims[1]:.1f}×{box_dims[2]:.1f} A (pad {pad} A, salt {salt} M)  ->  {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", default="mono",
                    help="preset (%s) or 'all', or use --name with --chains" % ", ".join(PRESETS))
    ap.add_argument("--chains", default=None, help="chain IDs, e.g. A or AB")
    ap.add_argument("--pad", type=float, default=None, help="solute->box padding (A), default 12")
    ap.add_argument("--salt", type=float, default=None, help="NaCl conc (M); 0 = neutralise only")
    ap.add_argument("--name", default=None, help="output system name (custom build)")
    ap.add_argument("--box", choices=("cube", "rect"), default="cube",
                    help="cube (every side = max extent, tutorial style) or rect "
                         "(each side tight = far fewer corner waters for an elongated solute)")
    args = ap.parse_args()

    if args.name:
        if args.chains is None:
            ap.error("--name requires --chains")
        build(args.name, args.chains,
              12.0 if args.pad is None else args.pad,
              0.15 if args.salt is None else args.salt, box_mode=args.box)
    elif args.preset == "all":
        for nm, cfg in PRESETS.items():
            build(f"pbc_{nm}", box_mode=args.box, **cfg)
    elif args.preset in PRESETS:
        cfg = dict(PRESETS[args.preset])
        if args.chains is not None:
            cfg["chains"] = args.chains
        if args.pad is not None:
            cfg["pad"] = args.pad
        if args.salt is not None:
            cfg["salt"] = args.salt
        build(f"pbc_{args.preset}", box_mode=args.box, **cfg)
    else:
        ap.error(f"unknown preset {args.preset!r}; choose from {list(PRESETS)} or 'all'")


if __name__ == "__main__":
    main()
