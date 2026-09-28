# FeNNiX (FENNIX-BIO1, JAX → StableHLO → PJRT) inference optimisation — FINAL (2026-09-23)

The model is unchanged: FENNIX-BIO1 (`models/fennix-bio1S.fnx`). The recommended path uses the shipped per-size StableHLO artifacts (`namd_benchmarks/models/fennix/w<K>/`) **as is**. FeNNiX runs inside namd3 through NAMD's native C++ PJRT backend (`qmSoftware fennol`), with **no Python in NAMD**. There is no `src/wrappers` file for this model: the "wrapper" is the export plus the manifest plus `ComputeFennix.C`.

## Step 0: does it run on the RTX 5080 (sm_120)? Yes, nothing needed
- **`fennix` env:** jax/jaxlib 0.7.2 + `jax-cuda12-plugin` 0.7.2 (CUDA 12.9 build, sm_120 OK), cuDNN 9.14. JAX works on the GPU (matmul and cuDNN conv tested); FeNNiX eager and jit both run.
- **Standalone probe:** the earlier "DNN library initialization failed" does not reproduce. `cpp/pjrt_shim/build/fennix_pjrt_probe` compiles and executes the 300-atom artifact and validates against the exported JIT reference (dE 1.4e-4 eV, dF 3.1e-3 eV/Å).
- **namd3:** the `namd_fennix` build (libtorch 2.11) runs fennol cells at 300/3000/6000 atoms with the existing `env.sh` recipe (`$FENNIX_CUDA_LIBS` prepended; it is non-empty on this box). No new env and no re-export were needed.

## TL;DR

namd3, walk0, 300 steps; wall ms/step over steps 250–300 (`logs/namd_final.txt`, copied to `results/fennix_namd_smoke.txt`).

| atoms | stock namd3 (baseline) | patched backend | **patched + `QMNoPntChrg on` + `NAMD_QM_FAST_INDEX=1`** | EXTRA true-fp32 artifact (all levers) |
|---|---|---|---|---|
| 300 | 6.28 | 2.39 (2.6x) | **1.91 (3.3x)** | 3.66 |
| 3000 | 12.90 | 9.95 (1.3x) | **4.10 (3.1x)** | 6.04 |
| 6000 | 37.33 | 25.58 (1.5x) | **7.51 (5.0x)** | 9.94 |

- **GPU peak** (nvidia-smi, whole namd3 process): 819 / 995 / 1493 MiB at 300 / 3000 / 6000 atoms. The patches do not change it.
- **Energies:** QMENERGY at step 0 agrees within FeNNiX's own run-to-run noise. At 6000 atoms: -23431.94 (stock), -23431.95 (patched), -23431.96 (all levers).

### The three levers

1. **Patched FeNNiX backend** (`namd_patch/fennix_backend.patch`). It affects fennol only and is on by default in the patched binary; `FENNIX_LEGACY_EXEC=1` switches back to the old path for A/B.
   - **New `PjrtPlugin::execute_into()`:**
     - one upload with `kImmutableOnlyDuringCall`;
     - Execute without a separate completion await;
     - outputs copied straight into persistent host arrays, so there is no size-query round trip and no per-step allocation;
     - all device-to-host copies awaited together, and the device handle cached.
   - **Effect inside namd3** (same binary, legacy vs fast, measured with `FENNIX_TIMING`):

     | atoms | legacy call | fast call |
     |---|---|---|
     | 300 | 5.49 ms | 1.77 ms |
     | 3000 | 5.91 ms | 3.88 ms |
     | 6000 | 8.98 ms | 6.80 ms |

     The old path's ~6 serial `PJRT_Event_Await` calls cost far more inside namd3 than in a standalone loop.
   - **Charge restore:** the per-step linear search, O(N × numQMAtoms), is replaced by an id→index hash built once. This saves ~4 ms/step at 6000 atoms. First-match semantics are the same, and charges are still read every step.
   - **Timers:** `FENNIX_TIMING=1` (cadence `FENNIX_TIMING_EVERY`, default 100) prints per-phase ms (pre / infer / post). These are the counters the MLFF performance audit asked for.
2. **`QMNoPntChrg on`** (NAMD config; affects all engines).
   - With `qmElecEmbed off`, NAMD still re-selects point charges every step: `ComputeQM::processFullQM`, PCMODEUPDATESEL, an O(N × numQMAtoms) distance loop plus linear searches.
   - For a whole-box QM region there are no MM point charges, so this is pure overhead. It was found with gdb stack sampling: `processFullQM` appeared in 14 of 25 samples at 6000 atoms.
   - The physics is identical here, and FeNNiX ignores point charges anyway.
   - Alone, on the **stock** binary, it takes 6000 atoms from 37 to 19 ms/step.
   - `namd_benchmarks/templates/bench.conf.tmpl` is user-owned, so it was not edited (see follow-ups).
3. **`NAMD_QM_FAST_INDEX=1`** (`namd_patch/computeqm_fast_index.patch`; affects all engines; opt-in, default off).
   - `ComputeQM::doWork` did a linear search of the global QM table for every local QM atom every step: ~18M comparisons at 6000 atoms. It is now an id→index hash.
   - On top of levers 1 and 2, 6000 atoms go from 11.5 to 7.3–7.5 ms/step.

The shipped artifacts are unchanged, so all three levers are numerically neutral: it is the same executable (see Parity).

## Kernel-level benchmark (C++ PJRT bench, same runtime sources as NAMD, no Python)

Method (`bench/bench_pjrt`):
- models are interleaved and see identical jittered geometries (±0.02 Å);
- 100 iterations, one invocation per size;
- results in `results/fennix_final_w<N>.json`.

Times are medians in ms.

| atoms | base (orig. exec path) | **fast (patched path)** | EXTRA fp32 HIGHEST | EXTRA w4 (4 walkers, one call) | 4 × fast |
|---|---|---|---|---|---|
| 30 | 5.10 | **2.15** | 4.01 | 1.72 | 8.6 |
| 300 | 3.23 | **2.09** | 4.18 | 2.00 | 8.4 |
| 900 | 3.15 | **2.18** | 3.83 | 3.22 | 8.7 |
| 3000 | 4.56 | **3.14** | 5.44 | 10.20 | 12.6 |
| 6000 | 7.99 | **5.64** | 8.88 | 26.36 | 22.6 |

Device MiB is the compiler's footprint of each executable: code + arguments + outputs + temporaries.

| atoms | 30 | 300 | 900 | 3000 | 6000 |
|---|---|---|---|---|---|
| device MiB, W=1 (base = fast) | 6.2 | 16.0 | 39.7 | 164.9 | 417.0 |
| device MiB, fp32 HIGHEST | 5.5 | 14.2 | 40.8 | 165.6 | 420.2 |
| device MiB, w4 | 9.3 | 49.5 | 179.0 | 745.7 | 1741.7 |

- **Fast path is at the GPU floor:** JAX `jit` of the same function measures 2.0 / 3.1 / 5.4 ms at 300 / 3000 / 6000 atoms (`jax/nl_cost.py`).
- **W=4 batching (vmap export):** a large win below ~1000 atoms (4.2x per walker at 300 atoms), neutral or negative at ≥3000.
- **NAMD cannot use W=4 batching:** the fennol backend has no batched multi-walker path. Each replica runs its own backend, and walk≥2 is broken in the Jul-31 build anyway. These numbers are only for the cross-model W=4 table.

## Where the time goes
- **One executable = one CUDA graph.** Each XLA executable is a single XLA command buffer of **203 kernels** at 300 atoms: 49 Triton GEMMs, 43 scatters, the rest elementwise ops and fusions (`prof/dump_w100/*thunk_sequence*`). There is no host work between kernels. At ≤900 atoms the ~2 ms is bound by launch latency of small kernels.
- **Neighbour list:** built inside the graph as an all-pairs (triu) search with fixed capacities. It costs ~0.5 ms at 300 atoms, ~1.0 ms at 3000 and ~1.6–1.8 ms at 6000 (30 % of the call; `jax/nl_cost.py`).
- **Inside namd3, before the patches:**
  - At 300 atoms the host-side PJRT syncs dominated: 5.5 of 6.3 ms.
  - At 6000 atoms NAMD's generic QM bookkeeping dominated: ≈20 of 37 ms, spread over point-charge re-selection, linear-search charge lookups and the FeNNiX charge restore.
- **Floor after all levers:** pure-MM NAMD for the same box (`QMForces off`) costs 3.1 / 7.5 ms/step at 3000 / 6000 atoms. That bounds what is left: ≈1–2 ms/step at 6000 atoms after all levers.

## Parity
- **fast vs base:** they use the *same compiled executable*. `BENCH_CHECK_EXEC=1` runs one input through both host paths; the difference is within the model's own run-to-run floor.
  - At 3000 atoms, namd-vs-fast: dE 2.0e-3 eV, dF 4.3e-3 eV/Å. Namd-vs-namd (the floor): dE 2.6e-3 eV, dF 5.2e-3 eV/Å.
  - FeNNiX is **non-deterministic** from run to run (fp32 atomics in scatters): about 0.1 kcal/mol/Å in forces.
- **Precision finding: the shipped artifacts run their matmuls in TF32.** JAX lowers fp32 `dot_general` with DEFAULT precision, and XLA executes that as TF32 on sm_80+. Compared against an fp64 evaluation of the same model (`jax/precision_check.py`, fresh compiles):

  | atoms | 30 | 300 | 900 | 3000 | 6000 |
  |---|---|---|---|---|---|
  | shipped (TF32): dE / max dF | 0.056 / 0.072 | 0.29 / 0.092 | 0.60 / 0.116 | 1.26 / 0.090 | 2.82 / 0.093 |
  | EXTRA fp32 HIGHEST: dE / max dF | 0.000 / 1.0e-4 | 0.0005 / 2.1e-4 | 0.023 / 4.3e-4 | 0.089 / 5.0e-4 | 0.70 / 1.9e-3 |

  Units are kcal/mol and kcal/mol/Å.
  - **The TF32 error changes per compile**, because the autotuner picks different GEMM algorithms. The shipped 6000-atom export's own jit-vs-eager check is 0.45 eV = 10.5 kcal/mol. In namd3, its step-0 energy differs from the fp32-HIGHEST artifact by 8 kcal/mol.
  - Forces stay around 0.1 kcal/mol/Å.
  - **Fairness:** the other models in this study run without TF32 (NequIP rejected TF32 for exactly this reason). For a precision-matched comparison, use `models/opt/fennix_fp32/w<K>`. It is 1.4–2x slower: namd3 goes from 1.91 to 3.66 ms at 300 atoms and from 7.5 to 9.9 ms at 6000.
  - It is kept as a labelled EXTRA; the required deliverable keeps the shipped precision.
- **Neighbour-list capacity (pre-existing risk, not changed):**
  - The exported graph has fixed pair capacities: 1.05 × the pair count of the export geometry.
  - The overflow flag is not exported. If a run ever compresses the system past +5 % pairs, the extra pairs are silently dropped.
  - In the smoke runs the droplets expand, so no overflow occurred. At 300 atoms, pairs within 7.5 Å went from 10731 to 9905, against a capacity of 11268.

## What was tried

| idea | result |
|---|---|
| patched PJRT execute path (fewer syncs, no size query, persistent host buffers) | **3.1x on the FeNNiX call inside namd3 at 300 atoms**, 1.3–1.5x at 3000–6000; numerically neutral |
| charge-restore hash (FeNNiX backend) | ~4 ms/step saved at 6000 atoms |
| `QMNoPntChrg on` (config) | 6000 atoms: stock 37 → 19 ms/step; patched 26 → 11.5 |
| `NAMD_QM_FAST_INDEX` (generic ComputeQM patch, opt-in) | 6000 atoms: a further 11.5 → 7.3–7.5 ms/step |
| XLA options via `compile_options.pb` `env_option_overrides` | none better than the default (details below) |
| MLIR bytecode instead of text (298 MB at 6000 atoms) | compile time unchanged (only an ordering effect); dropped |
| cell-list neighbour list inside the graph | not done: at most ~1 ms of 7.5 ms/step at 6000 atoms, and it needs dynamic grid bounds for an evaporating droplet, which brings a fixed-capacity overflow risk |
| W=4 vmap export | 4.2x per walker at 300 atoms, worse at 6000; NAMD has no batched fennol path |
| fp32 HIGHEST (EXTRA) | 200–400x smaller force error vs fp64, 1.4–2x slower |
| nsys / ncu | the installed 2024.5 tools see no CUDA activity on this driver/sm_120; used XLA dumps and gdb sampling instead |

XLA option details:
- `XLA_FLAGS` is ignored: `compile_options.pb` carries a full DebugOptions frozen at export time. Per-artifact options go in `env_option_overrides` (`make_compile_options.py`).
- Command buffers are already on; turning them off is 2x slower.
- Triton GEMM is already on; turning it off is 2x slower.
- `CONCURRENT` scheduling, concurrent regions and exhaustive tiling are neutral.
- cuBLASLt crashes the XLA compiler (heap_simulator CHECK).

## NAMD loadability / exact env
- **Patched binary:** `/home/rat/compile_NAMD_MACE/namd_fennix_fxopt/Linux-x86_64-g++/namd3`. It is a copy of `namd_fennix` with the two patches applied; the original tree and binary are untouched. It uses the same `env.sh`:
  ```bash
  source namd_benchmarks/env.sh
  export LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"     # fennol cells only (as run_benchmark.sh does)
  export NAMD_QM_FAST_INDEX=1                                     # optional, all engines
  # config: add  "QMNoPntChrg on"  (whole-box QM, no point charges)
  /home/rat/compile_NAMD_MACE/namd_fennix_fxopt/Linux-x86_64-g++/namd3 bench.conf
  ```
- **Artifacts:** the recommended path uses the unchanged `namd_benchmarks/models/fennix/w<K>/manifest.json`. The EXTRA `models/opt/fennix_fp32/w<K>/manifest.json` loads in both the stock and the patched namd3.

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
bash scripts/opt/fennix/build.sh            # bench tool, fp32 + w4 exports, patched namd3 copy (~5 min)
bash scripts/opt/fennix/run_grid.sh         # kernel-level grid -> scripts/opt/results/fennix_final_w*.json
N3=/home/rat/compile_NAMD_MACE/namd_fennix_fxopt/Linux-x86_64-g++/namd3
flock scripts/opt/.gpu_bench.lock scripts/opt/fennix/namd_smoke/run_smoke.sh stock namd_benchmarks/models/fennix/w100/manifest.json 100 300
FENNIX_TIMING=1 NAMD_QM_FAST_INDEX=1 EXTRA_CONF="QMNoPntChrg on" flock scripts/opt/.gpu_bench.lock \
  scripts/opt/fennix/namd_smoke/run_smoke.sh all namd_benchmarks/models/fennix/w100/manifest.json 100 300 $N3
source scripts/opt/fennix/env_fennix.sh
BENCH_CHECK_EXEC=1 scripts/opt/fennix/bench/bench_pjrt --iters 3 base=namd_benchmarks/models/fennix/w1000/manifest.json   # exec-path parity
/home/rat/miniconda3/envs/fennix/bin/python scripts/opt/fennix/jax/precision_check.py 100                                   # TF32 vs fp32 vs fp64
```

Files (all under `scripts/opt/fennix/`):
- **Bench and runners:** `bench/{bench_pjrt.cpp,build.sh}`, `env_fennix.sh`, `run_grid.sh`, `build.sh`.
- **NAMD patches:** `namd_patch/{fennix_backend.patch,computeqm_fast_index.patch}`.
- **NAMD smoke tests:** `namd_smoke/run_smoke.sh`, which uses private run dirs in `namd_smoke/runs/`; `namd_benchmarks/runs/` was not touched.
- **Tools:** `make_compile_options.py` (XLA option variants), `to_bytecode.py`, `jax/{nl_cost.py,precision_check.py}`.
- **Output:** logs in `logs/`, trimmed XLA dump in `prof/dump_w100/`.

Other notes:
- **Tools installed:** conda env `dbgtools` (gdb only, for stack sampling). The `fennix` and `allegro` envs are untouched.
- **Tests:** no repo Python source changed (exporter and wrappers untouched), so no test run was needed.

## Follow-ups / user decisions
1. **Adopt the NAMD patches.** Review `namd_patch/*.patch`; they apply cleanly to `namd_fennix/src` with `patch -p1`.
   - `fennix_backend.patch` affects fennol only.
   - `computeqm_fast_index.patch` touches the generic `ComputeQM.C` (opt-in env var) and helps every engine at large N.
2. **Add `QMNoPntChrg on` to `namd_benchmarks/templates/bench.conf.tmpl`** (whole-box QM). It removes ~8–18 ms/step of NAMD overhead at 3000–6000 atoms for *every* model. Without it, the large-N columns of the sweep mostly measure NAMD's point-charge bookkeeping, not the ML model.
3. **Same fix for the TorchScript backend:** `ComputeMLFF.C:2795` has the same O(N × numQMAtoms) charge-restore search, and the same hash fix applies (infra/shim owner).
4. **Precision fairness:** the shipped FeNNiX artifacts use TF32 matmuls. Decide whether the cross-model comparison uses the shipped artifacts or `models/opt/fennix_fp32` (true fp32, like the other models).
5. **Optional robustness:** export the neighbour-list `overflow` flag as a 3rd output and `NAMD_die` on it. This needs a change to the manifest/backend contract.
