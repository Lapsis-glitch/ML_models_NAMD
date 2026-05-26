# FENNIX PJRT native shim probe

Standalone no-Python-runtime C++ probe for the FENNIX-BIO1 StableHLO path.

Current status: the probe now builds and successfully loads the installed CUDA
PJRT plugin, initializes it, creates a CUDA client, compiles the exported
StableHLO artifact, warms up the loaded executable once, uploads a same-shape
input, executes it, downloads the outputs, and validates the reference-input
case against an offline-exported JAX reference sidecar.

This is intentionally outside NAMD first. It verifies that native C++ can:

1. `dlopen` the JAX CUDA PJRT plugin (`xla_cuda_plugin.so`),
2. retrieve `GetPjrtApi`,
3. initialize the plugin,
4. create a PJRT client, and
5. see CUDA devices.

This is the first end-to-end no-Python-runtime execution of
`models/fennix_bio1_stablehlo_n3/fennix_bio1_eval.stablehlo.mlir` through the
PJRT C API. The probe is now manifest-driven and can also run alternate
same-shape coordinates via `--coords`, which is a better stepping stone toward
NAMD than a single hard-coded water input. The next step is generalizing from
one fixed-shape artifact to an artifact family (or padded shape strategy) and
then transplanting the working path into NAMD.

## Prerequisites

Generate the offline FENNIX-BIO1 StableHLO artifact first:

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python \
  scripts/export_fennix_bio1_stablehlo.py \
  --model models/fennix-bio1S.fnx \
  --out-dir models/fennix_bio1_stablehlo_n3 \
  --z-list 8,1,1 \
  --total-charge 0
```

That export now writes these sidecars alongside the StableHLO MLIR:

- `compile_options.pb` — serialized minimal JAX compile options used by the
  native `PJRT_Client_Compile` call.
- `reference_runtime.txt` — simple line-based reference file containing the
  fixed input coordinates, exported JIT outputs, and tolerances used by the C++
  probe for validation.

## Build

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD/cpp/pjrt_shim
cmake -S . -B build
cmake --build build -j
```

This now builds:

- `libfennix_pjrt_runtime.a` — reusable native PJRT helper library
- `fennix_pjrt_probe` — small CLI on top of that library

## Run

```bash
cd /home/rat/PycharmProjects/ML_models_NAMD/cpp/pjrt_shim
./build/fennix_pjrt_probe
```

The probe reads `manifest.json` first and resolves `stablehlo_mlir`,
`compile_options_pb`, and `reference_runtime` from the manifest automatically.
Explicit path flags still override those derived paths when needed.

Optional explicit paths:

```bash
./build/fennix_pjrt_probe \
  --plugin /home/rat/miniconda3/envs/fennix/lib/python3.11/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so \
  --manifest /home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/manifest.json \
  --stablehlo /home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/fennix_bio1_eval.stablehlo.mlir \
  --compile-options /home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/compile_options.pb \
  --reference /home/rat/PycharmProjects/ML_models_NAMD/models/fennix_bio1_stablehlo_n3/reference_runtime.txt
```

Run the same artifact with alternate coordinates of the same shape:

```bash
./build/fennix_pjrt_probe \
  --coords /home/rat/PycharmProjects/ML_models_NAMD/cpp/pjrt_shim/examples/water_perturbed_coords.txt
```

## Notes

- Runtime Python is not used by this C++ probe.
- The installed `jaxlib` wheel exposes the CUDA PJRT plugin but does not ship
  `pjrt_c_api.h`, so this probe vendors the upstream OpenXLA PJRT C API header.
- The vendored PJRT header is newer than the installed plugin's exported
  `PJRT_Api` size. The probe therefore checks only the minimum `PJRT_Api`
  prefix needed by its current diagnostic calls instead of requiring the full
  newest header size.
- Validation is currently against the exported JAX-jitted reference for the
  exact fixed-shape artifact when the input coordinates match the reference
  sidecar. For custom `--coords`, value validation is skipped but shape/type
  checks and execution still run.
- The probe now compiles once and executes twice (warmup + measured execution).
  That extra warmup reduced first-run validation jitter observed when compiling
  and executing only once per process.
- The current probe is still intentionally specialized to the fixed water-like
  `tensor<3x3xf32>` export and should be treated as a proven execution shim for
  one artifact plus alternate same-shape coordinates, not yet a general FENNIX
  runtime.

