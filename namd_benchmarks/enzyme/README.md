# Full-ML enzyme benchmark — OMP decarboxylase (PDB 1X1Z) with FeNNiX

Runs a **whole solvated enzyme** as a single full-ML region (no QM/MM split — every
atom's forces come from the ML potential). The enzyme is orotidine-5'-monophosphate
decarboxylase (OMPDC), built from PDB **1X1Z** (a homodimer, 221 residues/chain, with
the inhibitor BMP bound). This is the "whole system in full ML" companion to the water
sweep in `../` — same `namd_fennix` NAMD3 build, same `env.sh`, same `lib/`.

> **Periodic (PBC + PME) variant:** this file covers the original *finite droplet*
> setup. For a **properly sized, periodic water box** (NAMD `.xsc` cell + PME +
> `wrapAll`, built with VMD/CHARMM following the `QMMM_String_eABF_Tutorial`), see
> **`README_PBC.md`** + `build_enzyme_pbc.py` / `run_enzyme_pbc.sh`. The model is
> still non-periodic (graph takes only coordinates), so PBC there buys a contained
> bulk-density hydration shell, not true minimum-image ML forces.

## Two engines (read this first)

The system can be driven by either backend, and **they behave differently**:

| path | driver | dynamics | per-step cost | 8 GB GPU | 48 GB GPU |
|------|--------|----------|---------------|----------|-----------|
| **NAMD + FeNNiX** | `run_enzyme.sh` (StableHLO/PJRT shim) | **drifts → explodes ~11 fs** | measurable | ✓ runs | ✓ runs |
| **native FeNNol** | `run_native_fennol.sh` (`fennol_md`) | **stable, energy-conserving** | n/a here | ✗ OOM/segfault | ✓ runs |

The FeNNiX **energy** is identical across all paths (mono_shell5: −25837 kcal/mol =
−1120.6 eV, to 4 digits), but **the forces NAMD applies to the integrator are corrupted**
— uncorrelated with the model's true forces. Diagnosis (mono_shell5, exhaustively
isolated; scripts in `diagnostics/`):

* **The model & the shim are correct.** The exported StableHLO forces match a fresh
  recompute and are true −∇E; the **standalone PJRT probe** (`cpp/pjrt_shim/.../fennix_pjrt_probe`)
  extracts them correctly (forces max abs diff 4e-3 eV/Å vs reference); and **native
  FeNNol MD with the same model conserves energy** and is stable at 0.5 fs.
* **NAMD applies a different force.** From a clean **T=0 quench**, the force that actually
  moves the atoms (= m·Δx from the validated position DCD, *independent* of any force
  output) has **correlation ≈0.02** with the model force at the same geometry — same rms
  (~9.9 kcal/mol/Å) but a different vector field. NAMD's own force DCD confirms it
  (corr 0.9998 with m·Δx), and FeNNiX *minimization* bounces (energy rises on step 1),
  which correct forces cannot do.
* **Not config / sign / model / shim.** Matching FeNNol (`rigidbonds none`, 0.5 fs, γ=10),
  NVE, heavy damping (γ=50,200), and 1.0/0.5/0.25 fs all explode (at a fixed *physical*
  time ~11 fs). Re-exporting with **negated** forces explodes *identically* (sign ruled
  out). The standalone shim is correct (extraction ruled out).

So the corruption is in **NAMD's runtime FeNNiX QM-force application** — between the
(verified-correct) shim output and the integrator — and it hits the libtorch MLFF
backends too (the water sweep documents MACE/fennol "drift"; soft water drifts slowly and
survives 500 steps, the stiff protein explodes in ~11 fs). The static path
(`ComputeFennix` force fill + the generic `ComputeQMMgr` distribution shared with the
*working* ORCA backend) looks correct on inspection, so pinning the exact line needs an
instrumented rebuild (log `forcesKcal` in `ComputeFennix.C` vs the final per-atom force).
NB: an earlier "one-step lag" hypothesis was **disproven** by the direct force comparison.

**Consequence.** No `.conf` setting fixes corrupted forces. Until the C++ path is fixed:
use the **NAMD path for per-step timing only** (cost is graph-fixed, independent of
trajectory physicality — harvest TIMING before the blow-up), and **native FeNNol for
stable dynamics**.

## Layout

```
build_enzyme.py          1X1Z -> finite solvated droplets (AmberTools; cpptraj env)
systems/<name>/          system.prmtop, system.pdb, qm.pdb (beta=1 all), meta.json
  _src/1x1z.pdb          cached RCSB download
templates/enzyme.conf.tmpl   non-periodic NAMD config (whole system = QM, qmReplaceAll on)
export_enzyme_fennix.sh  per-system FeNNiX StableHLO artifact (fennix env)
models/fennix_enzyme/<name>/manifest.json   exported artifacts
run_enzyme.sh            NAMD full-ML driver (per-step timing)   <- mirrors ../run_benchmark.sh
run_native_fennol.sh     native fennol_md driver (stable dynamics, the "standard run")
native_fennol/<name>/    Tinker xyz + input.fnl + run.log for the native path
pre_minimize_enzyme.sh   MM pre-min (NB: counterproductive for FeNNiX — see below)
```

## Systems (the preset ladder, ascending atom count)

| name           | chains | shell | atoms | net q | notes |
|----------------|--------|-------|-------|-------|-------|
| `mono_dry`     | A      | 0     | 3281  | −8    | path/correctness check (charged, OOD) |
| `mono_shell5`  | A      | 5 Å   | 4525  | 0     | **local 8 GB demo** (solvated, neutral) |
| `dimer_dry`    | AB     | 0     | 6562  | −16   | biological unit, dry |
| `dimer_shell6` | AB     | 6 Å   | 9791  | 0     | **full solvated enzyme** (remote 48 GB) |
| `dimer_shell8` | AB     | 8 Å   | ~12k  | 0     | fuller hydration (build on demand) |

Each droplet is a **finite, non-periodic cluster**: tleap solvates a TIP3P shell and
neutralises with ions, then parmed **strips the PBC box** so NAMD treats it as one
finite ML region (`qmReplaceAll on`) — matching FeNNiX's fixed-shape, non-periodic
StableHLO export. Net charge is recorded in `meta.json` and threaded into both the
FeNNiX export (`--total-charge`) and the NAMD `QMCharge`. FENNIX-BIO1 covers all
elements here (H/C/N/O/P/S + Na/Cl via grouped heads) plus a charge embedding.

## Build → export → run

```bash
# 1) build droplets (cpptraj env: tleap/pdb4amber/parmed)
conda run -n cpptraj python build_enzyme.py mono_shell5          # one
conda run -n cpptraj python build_enzyme.py all                  # the ladder

# 2) export the per-system FeNNiX artifact (fennix env; heavy/fragile, GPU compile)
./export_enzyme_fennix.sh mono_shell5                            # 8 GB ok (~4.5k atoms)
./export_enzyme_fennix.sh dimer_shell6                           # 48 GB box (full solvated)

# 3a) NAMD full-ML timing run (per-step cost; expect the energy drift)
SYSTEMS=mono_shell5 MODELS=fennol WALKERS=0 STEPS=60 OUTPUTFREQ=5 TIMESTEP=0.25 ./run_enzyme.sh
#   -> runs/fennol/mono_shell5/walk0/ : TIMING lines + gpu_mem.txt + status.txt

# 3b) native FeNNol stable dynamics (the "standard FeNNiX run")
DEVICE=cpu    ./run_native_fennol.sh mono_shell5                 # local (8 GB OOMs on GPU)
DEVICE=cuda:0 ./run_native_fennol.sh dimer_shell6               # remote 48 GB
```

### Knobs
* `run_enzyme.sh`: `SYSTEMS MODELS WALKERS STEPS OUTPUTFREQ TIMESTEP TIMEOUT MINIMIZE`.
  `MODELS` may also include `mace`/`ani2x` (general TorchScript wrappers cover H/C/N/O/S).
  `WALKERS 0` = plain `namd3`; `1..N` = `charmrun +replicas N` (cross-walker batching).
* `run_native_fennol.sh`: `DEVICE NSTEPS DT TEMP GAMMA NPRINT`.

## Measured (local, 8 GB RTX 3070 Laptop)

* **mono_shell5** (4525-atom solvated enzyme droplet), NAMD + FeNNiX, walk0:
  **~40 ms/step** steady-state wall, **GPU peak 2.9 GB**, step-0 energy −25837 kcal/mol
  (exact). Drift → RATTLE failure after ~10–40 steps depending on timestep.
* native FeNNol, CPU: stable, ~1 s/step on CPU (timing not comparable to GPU).

## Gotchas

* **Do not MM-pre-minimize for the FeNNiX path.** A non-periodic, T=0 MM relax distorts
  the droplet's water network into a geometry FeNNiX scores as *high* energy (+1180 eV
  vs the as-built −1120 eV), making the drift worse. Start from the **as-built**
  `system.pdb` (this is why `run_enzyme.sh` no longer auto-uses `system_min.coor`).
  `pre_minimize_enzyme.sh` is kept only for non-FeNNiX (MM/MACE/ANI) experiments.
* **native fennol_md GPU memory** > the NAMD StableHLO path (it JITs the integrator at
  runtime). ~4.5k atoms OOM/segfault (rc139) on 8 GB but run on 48 GB; tiny systems run
  on 8 GB. Use `DEVICE=cpu` locally.
* Same `env.sh` caveats as the parent sweep (FFTW2 libs, fennol cuDNN prepend, replica
  server-election hygiene). `env.sh` derives all paths from `$HOME`, so it works on the
  remote box unedited.
```
