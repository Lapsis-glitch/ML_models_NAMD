# Handoff: integrate the working FENNIX PJRT path into NAMD

## Purpose

This note is for a follow-on model or engineer implementing the already-proven
native C++ PJRT execution path for FeNNol `FENNIX-BIO1` into NAMD.

The key point is:

- the standalone probe in `cpp/pjrt_shim/` already proves **no-runtime-Python**
  execution of exported FENNIX StableHLO through the PJRT C API
- this should be integrated into NAMD as a **parallel backend path**, not forced
  into the existing TorchScript wrapper/export contract

---

## Executive summary

### What already works

The standalone executable `cpp/pjrt_shim/build/fennix_pjrt_probe` successfully:

1. `dlopen()`s the installed CUDA PJRT plugin
2. resolves `GetPjrtApi`
3. initializes the plugin
4. creates a PJRT client
5. enumerates CUDA devices
6. compiles exported StableHLO MLIR
7. uploads host coordinates as an F32 input buffer
8. executes the loaded executable
9. downloads energy and forces
10. validates against an offline-exported JAX-jitted reference for the reference input

### What does **not** exist yet

- no NAMD-side backend yet
- no dynamic-shape support
- no general arbitrary-composition runtime
- no validated atomic-charge output path

### Recommended next move inside NAMD

Transplant the reusable native runtime from `cpp/pjrt_shim/src/` into a new
NAMD backend that:

- loads one manifest/artifact bundle
- creates one PJRT client
- compiles once at startup
- caches the loaded executable
- accepts coordinates from NAMD
- returns energy and forces converted to kcal/mol and kcal/mol/Å

---

## Why this is a separate backend, not a normal repo wrapper

This repository’s normal path is built around TorchScript `torch.nn.Module`
wrappers with the fixed contract documented in `AGENTS.md` / `CLAUDE.md`.
That path works for MACE, NequIP, Allegro, SchNetPack, TorchANI, X-MACE, etc.

`FENNIX-BIO1` is different:

- it is FeNNol/JAX/Flax-based
- it is not naturally TorchScript-compatible
- the existing `src/export.py` TorchScript export flow is therefore not the
  right integration surface

Conclusion:

- **do not try to stuff FENNIX through the current TorchScript wrapper path**
- instead, add a **PJRT/StableHLO backend** beside the TorchScript backend in NAMD

---

## Current artifact bundle that is known-good

Verified export bundle:

- `models/fennix_bio1_stablehlo_n3/manifest.json`
- `models/fennix_bio1_stablehlo_n3/fennix_bio1_eval.stablehlo.mlir`
- `models/fennix_bio1_stablehlo_n3/fennix_bio1_eval.hlo.txt`
- `models/fennix_bio1_stablehlo_n3/compile_options.pb`
- `models/fennix_bio1_stablehlo_n3/reference_runtime.txt`
- `models/fennix_bio1_stablehlo_n3/reference.npz`

### Manifest facts

From `models/fennix_bio1_stablehlo_n3/manifest.json`:

- `model_type = FENNIX-BIO1`
- `n_atoms = 3`
- `z_list = [8, 1, 1]`
- `total_charge = 0`

### Input contract

One input only:

- `coordinates`
- shape: `[3, 3]`
- dtype: `float32`
- units: `Angstrom`

### Output contract

Two outputs only:

1. `energy`
   - shape: `[1]`
   - dtype: `float32`
   - units: `eV`
2. `forces`
   - shape: `[3, 3]`
   - dtype: `float32`
   - units: `eV/Angstrom`

### Conversion factor

From the manifest:

- `ev_to_kcal = 23.0621`

This is currently what the probe uses to print NAMD-like units.

---

## Offline/export path

The exporter is:

- `scripts/export_fennix_bio1_stablehlo.py`

What it does:

- loads the `.fnx` FeNNol model
- warms FeNNol preprocessing once
- uses the lowerable preprocessing path:
  - `model.preprocessing.process(state, raw)`
- computes:
  - `model._energy_and_forces(model.variables, pre)`
- lowers a fixed-shape function with JAX
- writes StableHLO MLIR and sidecars

### Important exporter detail

The exporter does **not** rely on direct lowering through the raw Python-facing
preprocessing path because that hit JAX tracing failures earlier.

The usable pattern is the current one in
`scripts/export_fennix_bio1_stablehlo.py`:

- call `model.preprocess(**raw_np)` once
- use `model.preproc_state`
- lower a function that uses `model.preprocessing.process(state, raw)`

Do not casually rewrite this without re-validating lowering.

---

## Native runtime code to transplant into NAMD

### Files to inspect first

- `cpp/pjrt_shim/src/artifact_bundle.h`
- `cpp/pjrt_shim/src/artifact_bundle.cpp`
- `cpp/pjrt_shim/src/pjrt_plugin.h`
- `cpp/pjrt_shim/src/pjrt_plugin.cpp`
- `cpp/pjrt_shim/src/main.cpp`
- `cpp/pjrt_shim/CMakeLists.txt`
- `cpp/pjrt_shim/README.md`

### Current reusable library boundary

`cpp/pjrt_shim/CMakeLists.txt` now builds:

- `libfennix_pjrt_runtime.a`
- `fennix_pjrt_probe`

That means the runtime has already been refactored toward a library-first shape.
The CLI in `src/main.cpp` is just a probe/demo layer.

### Key runtime entry points

From `cpp/pjrt_shim/src/pjrt_plugin.cpp` and `.h`:

- `PjrtPlugin::load()`
- `PjrtPlugin::initialize()`
- `PjrtPlugin::create_client()`
- `PjrtPlugin::compile_mlir(...)`
- `PjrtPlugin::execute_compiled(...)`
- `PjrtPlugin::destroy_compiled(...)`

From `cpp/pjrt_shim/src/artifact_bundle.cpp`:

- `load_artifact_spec(...)`
- `load_runtime_reference(...)`
- `load_flat_float_values(...)`

These are the most natural NAMD integration surfaces.

---

## Current probe behavior

The current standalone probe in `cpp/pjrt_shim/src/main.cpp` does this:

1. parses arguments
2. loads `manifest.json`
3. resolves artifact-relative paths
4. loads `reference_runtime.txt`
5. optionally loads a custom same-shape coordinate file
6. loads the PJRT plugin
7. creates a client
8. reads StableHLO and compile options
9. compiles the MLIR
10. performs **one warmup execution**
11. performs one measured execution
12. checks output shapes against the manifest
13. prints energy/forces in both native and converted units
14. validates against the exported JAX-jitted reference only when using the
    baked reference coordinates

### Why the warmup exists

A single compile+execute path showed enough first-run GPU variance to sometimes
exceed the exported tolerance. Adding one warmup execution before the measured
execution made validation pass reliably in the verified run.

If integrating into NAMD, expect that some startup warmup behavior may still be
useful or necessary.

---

## Runtime environment assumptions

### Runtime Python is not used

At runtime, the probe uses only:

- native C++
- vendored `pjrt_c_api.h`
- the site-installed CUDA PJRT plugin

Python/JAX/FeNNol are only required **offline** for export.

### Current plugin path used by the probe

The current local default is:

- `/home/rat/miniconda3/envs/fennix/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so`

Do **not** hardcode that path in NAMD.
Make plugin discovery/configuration explicit.

---

## Known risks / gotchas

### 1) Vendored header vs installed plugin ABI skew

The repo vendors:

- `cpp/pjrt_shim/third_party/xla/pjrt/c/pjrt_c_api.h`

because the local `jaxlib`/plugin installation does not ship a matching header.

Important consequence:

- the vendored header is newer than the installed plugin’s exported
  `PJRT_Api` size
- the runtime therefore checks only the minimum `PJRT_Api` prefix needed for
  the APIs it actually uses

This is a real ABI fragility point. Be conservative when expanding PJRT API usage.

### 2) The destructor intentionally does not `dlclose()` the plugin

In the probe, unloading after partial or mismatched initialization was
crash-prone, so the short-lived probe intentionally avoids `dlclose()`.

If you change lifecycle handling inside NAMD, re-test carefully.

### 3) Fixed-shape artifact only

The currently proven artifact is specialized to:

- `n_atoms = 3`
- `z_list = [8,1,1]`
- `total_charge = 0`

It is not a general arbitrary system runtime.

### 4) Same-shape reuse only is currently proven

The probe now supports alternate coordinates of the **same** shape via:

- `--coords cpp/pjrt_shim/examples/water_perturbed_coords.txt`

That proves reuse of one compiled artifact for multiple coordinate sets of the
same shape, but **not** support for different atom counts or compositions.

### 5) Charges are not part of the current runtime contract

The native probe currently handles:

- energy
- forces

It does **not** expose a validated atomic-charge output path suitable for NAMD.

If NAMD expects charges from this backend, that is separate future work.

### 6) Validation target is the exported JAX-jitted reference

The current validation is intentionally against the exported JIT outputs in
`reference_runtime.txt`, not against runtime Python evaluation.

That is correct for this phase because it validates the native runtime against
exactly what was exported.

---

## Recommended NAMD integration architecture

Add a new NAMD backend roughly like:

- `FennixPjrtBackend`
  - owns plugin path / handle
  - owns PJRT API pointer
  - owns client
  - owns one or more artifact specs
  - owns a cache of compiled executables
  - exposes `evaluate(coords) -> energy, forces`

### Strong recommendation

Do this as a **parallel backend** beside the TorchScript backend, not by
rewriting the existing TorchScript wrapper path.

---

## Minimum viable NAMD integration plan

### Phase 1: single known-good artifact inside NAMD

Implement a backend that:

- loads one manifest
- loads plugin
- creates one client
- compiles one executable at startup
- accepts one `3x3` coordinate tensor
- returns one energy and one `3x3` force tensor
- converts eV / eV/Å to kcal/mol / kcal/mol/Å

This should mirror the current probe as closely as possible.

### Phase 2: artifact caching

Cache compiled executables keyed by artifact identity.
At minimum, use something equivalent to manifest path or resolved artifact key.

### Phase 3: multi-artifact selection

Add an artifact-family mechanism so NAMD can choose among multiple fixed-shape
exports.

A practical selection key is:

- model family
- `n_atoms`
- `z_list`
- `total_charge`

---

## Shape strategy options for NAMD

### Option A: artifact family per fixed shape/composition

Export one artifact per:

- atom count
- atom identity list
- total charge

Pros:

- closest to the currently proven path
- least risky
- easiest near-term implementation

Cons:

- can lead to many artifacts

### Option B: padded fixed-max-N export

Pros:

- fewer artifacts

Cons:

- requires redesign of export/runtime assumptions
- masking semantics must be defined
- not yet proven here

### Option C: true dynamic-shape support

Pros:

- best long-term elegance

Cons:

- highest risk
- not yet proven with current FeNNol preprocessing/export setup

### Recommended choice now

Start with **Option A**.

---

## Unit handling for NAMD

Current native outputs are:

- `float32` energy in eV
- `float32` forces in eV/Å

For NAMD, likely convert immediately to:

- energy in kcal/mol
- forces in kcal/mol/Å
- likely promote to double at the NAMD boundary if that is what the surrounding
  code expects

Current conversion factor already present in the manifest:

- `23.0621`

---

## Concrete first coding target inside NAMD

A good first internal deliverable is:

> A small NAMD backend class that loads one `manifest.json`, compiles the
> artifact once at startup, accepts the fixed `3x3` coordinate input, returns
> converted energy/forces, and is covered by a tiny NAMD-side smoke test.

That keeps the first NAMD milestone maximally close to the already working probe.

---

## Commands that currently work

### Rebuild the standalone shim

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD/cpp/pjrt_shim
cmake -S . -B build
cmake --build build -j
```

### Run the validated reference case

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD/cpp/pjrt_shim
./build/fennix_pjrt_probe
```

### Run with alternate same-shape coordinates

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD/cpp/pjrt_shim
./build/fennix_pjrt_probe \
  --coords ./examples/water_perturbed_coords.txt
```

### Regenerate the artifact bundle offline

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python \
  scripts/export_fennix_bio1_stablehlo.py \
  --model models/fennix-bio1S.fnx \
  --out-dir models/fennix_bio1_stablehlo_n3 \
  --z-list 8,1,1 \
  --total-charge 0
```

For a full fixed-composition NAMD region, the exporter can now specialize
directly from a PDB instead of a manual `--z-list`:

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python \
  scripts/export_fennix_bio1_stablehlo.py \
  --model models/fennix-bio1S.fnx \
  --pdb namd_water_test/water.qm.pdb \
  --out-dir models/fennix_bio1_stablehlo_water_pdb
```

That keeps the current fixed-shape strategy, but derives `n_atoms`, `z_list`,
and reference coordinates from the full PDB atom ordering automatically. When
`--total-charge` is omitted in this mode, formal PDB charge columns are summed
if present; otherwise the export defaults to net charge `0`.

The exporter also now has an **experimental** same-system multi-walker mode for
replica / walker workloads where every walker has the same atom count, `z_list`
ordering, and total charge but different coordinates. Supplying
`--walker-coords-npy coords.npy` with shape `[n_walkers, n_atoms, 3]` produces a
batched artifact whose manifest advertises:

- input `coordinates`: `[n_walkers, n_atoms, 3]`
- output `energy`: `[n_walkers, 1]`
- output `forces`: `[n_walkers, n_atoms, 3]`

This keeps the fixed-artifact strategy intact — the artifact is still bound tomorning
one composition and one walker count — but makes it possible to experiment with
multiple identical walkers sharing one compiled StableHLO entrypoint. Treat
this as an exporter-side experiment until the CUDA PJRT probe has been validated
end-to-end on such a batched manifest.

---

## Final guidance to the next model

1. Do **not** touch the existing TorchScript model wrappers unless necessary.
2. Treat `cpp/pjrt_shim/` as the source of truth for the working native path.
3. Transplant the runtime library, not the probe CLI.
4. Start with one fixed artifact inside NAMD.
5. Cache compiled executables.
6. Convert units at the NAMD boundary.
7. Defer dynamic shapes and charges until after the first NAMD smoke test passes.

