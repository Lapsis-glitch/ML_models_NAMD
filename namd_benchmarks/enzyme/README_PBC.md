# Periodic (PBC + PME) full-ML enzyme run — OMP decarboxylase

This is the **periodic water-box** companion to the finite-droplet workflow in
`README.md`.  It builds a *properly sized, neutralised, periodic* TIP3P box around
the enzyme (NAMD `.xsc` cell + PME + `wrapAll`) instead of a stripped-box droplet,
following the CHARMM/VMD recipe of the **QM/MM String-eABF tutorial**
(`QMMM_String_eABF_Tutorial`), which targets this very enzyme (PDB 1X1Z).

The whole solvated system is still the **full-ML region** (FeNNiX drives every
atom).  This is the *"full-ML, bigger periodic box"* mode.

## What "PBC" does and does not do here (read this)

The FeNNiX StableHLO graph takes **only coordinates** — there is no cell input in
the export or the PJRT shim (verified).  So:

* **PME + minimum-image reach only the MM bookkeeping** NAMD does at start-up.
* **The ML forces are still computed on the finite cluster** NAMD hands the engine
  each step.  Water at the very edge of the box still sees vacuum on the outside.

What the periodic box *does* buy versus the droplet: a **properly sized, contained
hydration shell with bulk-like water density and no droplet surface tension**
(the droplet's surface was the thing that pulled waters in and pushed FeNNiX into
high-energy geometries).  True periodic ML forces would require extending the
export + shim + `ComputeFennix` to pass cell vectors and do a minimum-image
neighbor list — a separate project, not this setup.

## Pipeline

```
build_enzyme_pbc.py     1X1Z -> CHARMM psfgen -> VMD solvate(cubic) + autoionize
                        -> systems/pbc_<name>/{system.psf, system.pdb, system.xsc,
                                               qm.pdb, meta.json}
toppar/                 CHARMM36 top/par (copied from the tutorial)
templates/enzyme_pbc.conf.tmpl   periodic NAMD config: paraTypeCharmm + extendedSystem
                                 + PME + wrapAll + full-ML QMForces
run_enzyme_pbc.sh       MM equilibration (SMOKE_MM=1) and full-ML driver
export_enzyme_fennix.sh per-system FeNNiX StableHLO artifact (unchanged; reads qm.pdb)
```

## Build → equilibrate → export → run

```bash
module load vmd/2.0                      # VMD 2.0 (psfgen/solvate/autoionize/topotools)

# 1) build the periodic box (CHARMM/VMD; no conda env needed)
python build_enzyme_pbc.py mono          # chain A  (~39.6k atoms, 75 A cube, 0.15 M NaCl)
python build_enzyme_pbc.py dimer         # chains AB (biological unit; larger)
#   knobs: --chains A|AB  --pad 12  --salt 0.15   (--salt 0 = neutralise only)

# 2) MM-equilibrate the fresh box (relaxes solvate/H clashes with PME/PBC).
#    Promotes the relaxed restart to systems/<S>/system_equil.{coor,xsc}, which the
#    full-ML run then starts from (FeNNiX drifts if started from a clashing geometry).
SMOKE_MM=1 MM_MIN=500 STEPS=5000 SYSTEMS=pbc_mono ./run_enzyme_pbc.sh   # NVT, 2 fs

# 3) export the FeNNiX artifact (GPU compile of a fixed-shape graph).
#    ~39.6k atoms -> needs the 48 GB remote box (8 GB OOMs at compile).
./export_enzyme_fennix.sh pbc_mono

# 4) full-ML periodic run (starts from system_equil.* automatically)
SYSTEMS=pbc_mono MODELS=fennol WALKERS=0 STEPS=60 OUTPUTFREQ=5 TIMESTEP=0.5 ./run_enzyme_pbc.sh
#   -> runs_pbc/fennol/pbc_mono/walk0/ : ENERGY/TIMING + gpu_mem.txt + status.txt
```

### Knobs (`run_enzyme_pbc.sh`)
`SYSTEMS MODELS WALKERS STEPS OUTPUTFREQ TIMESTEP QMREPLACEALL RIGIDBONDS START
TIMEOUT MINIMIZE` plus `SMOKE_MM=1`/`MM_MIN` for the MM-equilibration pass.
* `QMREPLACEALL` defaults **on** (full-ML: the engine supplies all forces).  `off`
  was the pre-fix multi-patch workaround in `ROOTCAUSE_NAMD_FORCE_BUG.md`.  The
  `namd_fennix` build now used has the fix — `on` runs stably (verified below).
* `START` = `equil` (default; start from the MM-relaxed `system_equil.*`) or
  `asbuilt` (start from the fresh solvate, clashes and all).  For a *periodic* box
  the MM relaxation gives bulk-density water that is in-distribution for FeNNiX
  (unlike the droplet case the README gotchas warn about), so `equil` is the
  sane default — but confirm on the remote with the 1-step check below.
* `RIGIDBONDS` = `all` (default, RATTLE on H) or `none` (native FeNNol uses none).
* `MODELS` may include `mace`/`ani2x` for the droplet systems, but **not for the
  periodic boxes** — those contain Na/Cl ions, which the MACE-OFF/ANI2x wrappers
  (H/C/N/O/S only) do not cover.  Use `fennol` (FENNIX-BIO1 covers Na/Cl) here.

### Verify the equilibrated-vs-as-built start on the remote (1 step each)
```bash
STEPS=0 OUTPUTFREQ=1 START=equil   SYSTEMS=pbc_mono ./run_enzyme_pbc.sh   # step-0 POT
STEPS=0 OUTPUTFREQ=1 START=asbuilt SYSTEMS=pbc_mono ./run_enzyme_pbc.sh   # step-0 POT
# grep '^ENERGY: *0' runs_pbc/fennol/pbc_mono/walk0/out.0.log  -> col 14 (POT) = FeNNiX energy
# equil should NOT score far higher than as-built; if it does, use START=asbuilt.
```

## Validated locally (8 GB RTX 3070)

* **Build** `pbc_mono`: 39 609 atoms, net charge 0, 75.2 Å cubic box, 0.15 M NaCl.
  Element columns resolve CHARMM names correctly (SOD→Na, CLA→Cl, TIP3→O/H); the
  FeNNiX exporter parses all 39 609 atoms with no unsupported elements.
* **NAMD PBC/PME start-up** (`SMOKE_MM=1`): `PERIODIC CELL BASIS 75.157` cube, PME
  grid 80³ @ 1 Å, `wrapAll` on, minimisation relaxes the fresh-build clashes,
  `End of program`. The full-ML render then starts from `system_equil.{coor,xsc}`.
* **FeNNiX GPU compile of the 39.6k-atom box and the full-ML run are remote/48 GB**
  (the 8 GB card OOMs at compile), matching the droplet `dimer_shell6` regime.
* **FeNNiX-under-PBC: step-0 feed + 50 fs stability verified** on the 4.5k-atom
  `mono_shell5` (fits 8 GB; `diagnostics/pbc_feedtest/`).  With `qmReplaceAll on`:
  * **200 steps @ 0.25 fs = 50 fs, stable & energy-conserving** under PBC+PME:
    TEMP 228–318 K around 300 (no runaway), POT bounded, TOTAL conserved.  This is
    ~4.5× past the **old ~11 fs force-corruption blow-up** (which hit 3464 K by
    step ~45) — that bug is **gone**; the `namd_fennix` build's QM-force fix is in.
    This is what "the bug is fixed" means; the whole periodic deliverable rests on it.
  * step-0 FeNNiX POT is **the same with and without PBC**: −25837.27 (no PBC) vs
    −25837.25 (cell contains the cluster) vs −25837.49 (cell face through the
    cluster).  The model ignores the cell, so this proves NAMD's `wrapAll` +
    QM re-imaging feed FeNNiX the same geometry at t=0 — no boundary-split mangling
    (any atom inside the cell is already its own minimum image).  The ~0.2 kcal/mol
    shift with a face through the cluster is the only edge effect, negligible.
  * **Still to confirm on the remote (the user-accepted caveat):** *sustained*
    dynamics once edge waters of the space-filling box actually reach/cross a face —
    the "thick shell vs teleport" question a short droplet run can't reach.  Watch
    the first ~hundreds of fs of the real `pbc_mono`/`pbc_dimer` run for any energy
    jump as waters cross the boundary.

## Notes / gotchas

* Cubic box side = solute extent + 2·`pad`; water fills it edge-to-edge and the
  `.xsc` cell matches exactly (origin 0), as in the tutorial (80 Å cube).
* `cutoff 14` < L/2 (37.6) — required for minimum-image; fine for these boxes.
* CHARMM force field here is **only** for NAMD start-up + PME; `qmReplaceAll on`
  discards every MM force in favour of the ML potential.
* Same `env.sh` caveats as the parent sweep (FeNNiX CUDA libs, replica election).
```
