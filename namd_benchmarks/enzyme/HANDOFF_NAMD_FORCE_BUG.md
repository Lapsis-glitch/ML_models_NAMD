# Handoff: NAMD full-ML (FeNNiX/MLFF) applies corrupted forces

**Audience:** an agent/engineer who can read & instrument the `namd_fennix` C++ source.
**Goal:** find the exact line where the per-atom QM force gets corrupted between the
(verified-correct) model/shim output and NAMD's integrator. Static reading is exhausted;
this needs an instrumented rebuild. Everything below is reproducible.

---

## 1. Symptom

Running a whole solvated enzyme as a full-ML region (`qmReplaceAll on`, `QMSoftware fennol`)
in NAMD3 blows up: the **energy is correct every step**, but the **forces NAMD applies to
the integrator are uncorrelated with the model's true forces** (same rms, wrong direction).
Soft systems (water) drift slowly and survive; the stiff protein explodes in ~11 fs
(RATTLE failures / energy runaway). The libtorch MLFF backends (MACE/ANI via
`ComputeQMMACE.C`) show the same — the file-based ORCA/MOPAC path does **not**.

This is **not** the model, the export, the PJRT shim, the integrator config, the force
sign, or a one-step lag — all ruled out below. It is in NAMD's runtime QM-force handling.

---

## 2. Environment

- NAMD source + binary: `/home/rat/compile_NAMD_MACE/namd_fennix` (binary
  `Linux-x86_64-g++/namd3`). `ComputeFennix.C` and `fennix_pjrt/*.cpp` are `#include`d
  into `ComputeQM.C` and compiled into `namd3` (see `ComputeQM.C` bottom).
- Runtime env: `source /home/rat/PycharmProjects/ML_models_NAMD/namd_benchmarks/env.sh`
  then for fennol prepend JAX CUDA libs: `export LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"`.
  (env.sh sets `$NAMD3`, `$NAMD_FENNIX_PLUGIN`, FFTW2/libtorch paths.)
- Conda envs: `fennix` (FeNNol/JAX, has `fennol_md`, the model), `cpptraj` (AmberTools/parmed).
- GPU: RTX 3070 Laptop 8 GB (the 4.5k-atom system fits; ~2.9 GB peak).

## 3. Test system (small, fits 8 GB, reproduces the bug)

`namd_benchmarks/enzyme/systems/mono_shell5/` — OMP-decarboxylase monomer + 5 Å TIP3P
shell + Na+, **4525 atoms, net charge 0**, finite/non-periodic.
- `system.prmtop`, `system.pdb` (coords), `qm.pdb` (beta=1 all = full-ML selection).
- FeNNiX artifact: `models/fennix_enzyme/mono_shell5/` (`manifest.json`,
  `fennix_bio1_eval.stablehlo.mlir`, `reference.npz`, `reference_runtime.txt`).
  Model: `/home/rat/PycharmProjects/ML_models_NAMD/models/fennix-bio1S.fnx`.

## 4. Reproduce the proof (no rebuild needed)

Scripts in `namd_benchmarks/enzyme/diagnostics/`:

```bash
cd namd_benchmarks/enzyme/diagnostics
# (1) a T=0, NVE, rigidbonds-off run that dumps per-step positions AND applied forces.
#     probe.conf already points at mono_shell5; outputs pos.dcd + force.dcd.
source ../../env.sh; export LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"
"$NAMD3" probe.conf > probe.log 2>&1
# (2) compare what NAMD applied vs the model's true forces (fennix env has the model)
JAX_PLATFORMS=cpu /home/rat/miniconda3/envs/fennix/bin/python lag_proof.py
```

`lag_proof.py` reads the DCDs (minimal numpy DCD reader), computes the FeNNiX force with
the real model, and prints the comparisons in §5. (`probemm.conf` is the MM-only control.)

## 5. What is PROVEN (and what is ruled out)

| claim | evidence | number |
|---|---|---|
| Energy is correct every step | NAMD `QMENERGY` vs model | −25835 kcal/mol = −1120.6 eV, matches to 4 digits |
| **Applied force ⟂ model force** | pure kinematics: `m·(x₁−x₀)` from rest vs `F_model(x₀)` | **corr = 0.02** (same rms ~9.9 kcal/mol/Å) |
| (DCD-independent) position DCD is faithful | `corr(x₁, x₀)=1.000`, `rms|x₁−x₀|=0.0003 Å` | atom order identical to qm.pdb |
| NAMD force DCD = the force that moved atoms | `corr(force.dcd, m·Δx) = 0.9998` | so the corruption is real, not a DCD artifact |
| FeNNiX **minimization bounces** (energy ↑ at step 1) | impossible with correct forces | corroborates wrong forces |
| Model forces are correct | `corr(my F_model, reference forces_jit)=1.0`, same geometry | true −∇E (stepping along F lowers E) |
| **Standalone PJRT shim is correct** | `cpp/pjrt_shim/build/fennix_pjrt_probe` parity check | **forces max abs diff 4e-3 eV/Å** (tol 2.9e-2) |
| Native `fennol_md` is stable | same model, 0.5 fs, LGV | energy conserved, no explosion |
| NOT config | matched FeNNol (`rigidbonds none`,0.5fs,γ=10) + NVE + γ=50/200 + dt 1/0.5/0.25 | all explode at fixed *physical* time ~11 fs |
| NOT force sign | re-export with `-forces_ev_a`, run NAMD | explodes *identically* (first ~7 steps bit-for-bit) |
| NOT a one-step lag | applied force ⟂ `F_model(x_{n-1})` too (corr 0.02) | DISPROVEN — wrong at the same step, not stale |

**Conclusion:** the shim returns correct forces; NAMD applies a different per-atom force.
Corruption is in NAMD's runtime FeNNiX/MLFF QM-force path, *after* `backend->evaluate()`.

## 6. Code map (namd_fennix/src)

**`ComputeFennix.C`** — in-process FeNNiX backend (dispatched per step from
`ComputeQMMgr::recvPntChrg`, `ComputeFennix.C:280`):
- gather coords/Z from `atmP[i]` → `coords[3i+..]`, `Z[i]` — lines **316–337**
- **z-list/order validation** `Z[i]==spec.z_list[i]` (else `NAMD_die`) — lines **362–377**
- `backend->evaluate(coords, energyKcal, forcesKcal)` — line **385**
- fill `resForce[i].force = forcesKcal[3i+..]`, `resForce[i].id = atmP[i].id` — **425–445**
- `FennixBackend::evaluate` (outputs[0]=energy, outputs[1]=forces, `*ev_to_kcal`) — **152–175**

**`fennix_pjrt/pjrt_plugin.cpp`** — PJRT runtime:
- `execute_compiled` (awaits exec event, copies each output) — **340–448**
- `copy_buffer_to_host` (awaits the D2H copy; `host_layout=nullptr`) — **522–562**.
  NB `host_layout=nullptr` ⇒ device layout, *but the standalone probe using this same code
  is verified correct*, so this is not the bug (checked).

**`ComputeQM.C`** — generic QM force routing (SHARED with the working ORCA path):
- `storeQMRes`: `force[fres[i].id].force += fres[i].force` — **2469–2471** (does NOT set
  `homeIndx` here)
- `force[qmCoord[i].id].homeIndx = qmCoord[i].homeIndx` (id→home-patch slot) — **1196**
- distribute: `fmsg->force[forceIter] = force[qmmsg->coord[i].id]` — **2624**
- `recvForce` → `saveResults`: `oldForces[results_ptr->homeIndx].force += ...` — **2732**
- file-based ORCA/MOPAC store `resForce.force = -1*gradient` (true force) — **3261** (so the
  sign convention NAMD expects = true force; FeNNiX/MACE supply `+model_force` = same — OK)

**`ComputeQMMACE.C`** — libtorch MACE backend, force fill at **2145** (identical pattern to
ComputeFennix). Also affected ⇒ the bug is shared by the in-process backends or their
common downstream, and absent in the file-based ORCA path.

## 7. The puzzle for you to crack

Every link above looks correct in isolation, *and* the `storeQMRes`/distribute/`saveResults`
chain is shared with ORCA which works — yet the integrated force is provably wrong, and only
for the in-process backends (FeNNiX, MACE). So look at **what differs between the in-process
QM dispatch and the file-based one**, and at any **ordering/aliasing/threading** assumption
the in-process path violates. Concrete hypotheses to test by instrumentation:

- **H-A (atom order):** does the runtime `atmP`/`qmCoord` order actually equal the export
  `z_list` (qm.pdb/global) order? The validation only checks *elements* match positionally,
  so a same-element permutation passes. Print `Z[0..20]` vs `spec.z_list[0..20]` AND the
  `atmP[i].id` sequence; confirm `atmP[i].id == i` (global order). (I showed the *position*
  DCD is in global order, but that's NAMD's output remap, not necessarily `atmP`'s order.)
- **H-B (message marshalling):** is `QMForce`/`QMForceMsg` packed/unpacked correctly for this
  path (Charm++ `varsize` message; `resForce` count = numQMAtoms + numRealPntChrgs)? With
  4525 QM atoms and 0 PCs vs ORCA's typical small QM region, an off-by-PC-slot or size bug
  could surface only here.
- **H-C (homeIndx staleness):** `force[id].homeIndx` is set at 1196 during gather; if the
  in-process path reuses a stale `force[]`/`homeIndx` from a prior step or wrong group, 2732
  scatters to wrong atoms — *right magnitudes, wrong atoms* = exactly the observed signature.
- **H-D (multiple QM groups / single group):** confirm there is exactly one QM group here and
  the per-group routing isn't mis-binning.

## 8. Instrumentation plan (the decisive test)

Pick one global atom id (e.g. `J=100`). Add prints and rebuild `namd3`, then run §4's probe
for a single step:

1. In `ComputeFennix.C` right after line 385: print, for the slot `k` with `atmP[k].id==J`,
   `forcesKcal[3k+0..2]` and `atmP[k].position`. → call this **B** (what the model gave for J).
2. Independently compute the model force on atom J offline at the same coords (use
   `diagnostics/lag_proof.py`'s `Fmodel`). → **A** (ground truth).
3. In `ComputeQM.C::saveResults` (after line 2732): print the force written for the
   `oldForces[]` slot whose atom is J. → **C** (what the integrator gets).

Then: **A vs B** tells you if `evaluate` returned the right force for J (expected: equal —
the shim is verified). **B vs C** tells you if the distribution scattered it correctly
(suspected: NOT equal). Whichever transition breaks is the bug. If A==B but B≠C, walk the
`storeQMRes → fmsg pack (2624) → recvForce → saveResults (2732)` chain checking the
`id`/`homeIndx` used at each hop for atom J.

## 9. Success criterion

After a candidate fix, rerun §4. The bug is fixed iff:
`corr( m·(x₁−x₀), F_model(x₀) ) → ~1.0` (currently 0.02), the T=0 NVE quench **conserves
total energy**, and `SYSTEMS=mono_shell5 MODELS=fennol WALKERS=0 STEPS=200 TIMESTEP=0.5
../run_enzyme.sh` runs 200 steps with bounded energy (matching native `fennol_md`).

## 10. Build note

`ComputeFennix.C`/`fennix_pjrt/*` are `#include`d into `ComputeQM.C`, so editing them means
recompiling `ComputeQM.o` and relinking `namd3`. Use the existing build in
`/home/rat/compile_NAMD_MACE/namd_fennix` (check its `Make.config`/`arch` for the
`--with-fennix-pjrt` flags: `-DNAMD_FENNIX`, the `src/fennix_pjrt/third_party` include path,
`-ldl`). Don't `dlclose` the PJRT plugin (known crash). A full `namd3` relink is enough; no
need to rebuild Charm++/FFTW.
