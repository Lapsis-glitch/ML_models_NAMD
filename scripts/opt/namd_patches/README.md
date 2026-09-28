# NAMD source changes from the 2026-09 inference-optimisation work

These are ALL source changes made to the NAMD tree (`namd_fennix`, Jul-31 source) during this work.
Apply from the NAMD source root (the directory that contains `src/`):

```bash
cd <namd_source_root>
for p in /path/to/namd_patches/0*.patch; do patch -p1 --dry-run < "$p" && patch -p1 < "$p"; done
```
Verified 2026-09-23: all three apply cleanly (no fuzz) to the unmodified sources, and the result is byte-identical
to the tested builds (`namd_fennix` shim + `namd_fennix_fxopt`).

| patch | files | affects | default behaviour |
|---|---|---|---|
| `01_mlff_shim_extralibs_knobs.patch` | `src/mlff_shim/{mlff_shim.cpp,build_and_test.sh,BUILD.local.md}` | TorchScript MLFF shim (`libnamd_mlff.so`, dlopen'd by namd3) | unchanged unless env vars set |
| `02_fennix_backend.patch` | `src/ComputeFennix.C`, `src/fennix_pjrt/pjrt_plugin.{h,cpp}` | FeNNiX / `qmSoftware fennol` only | fast path ON (`FENNIX_LEGACY_EXEC=1` reverts) |
| `03_computeqm_fast_index.patch` | `src/ComputeQM.C` | every QM engine | OFF unless `NAMD_QM_FAST_INDEX=1` |

## 01 — MLFF shim (needed for the optimised MACE / NequIP / ANI artifacts)
- `NAMD_MLFF_EXTRA_LIBS=/a.so:/b.so` — each library is `dlopen(RTLD_NOW|RTLD_GLOBAL)`'d in order before
  `torch::jit::load`. This is how the native custom-op libraries (cuEquivariance `uniform_1d`, OpenEquivariance,
  cuAEV) get registered. Without it the optimised MACE/NequIP/ANI artifacts fail to load ("Unknown builtin op").
- Model object is intentionally leaked at process exit (fixes an abort at the end of a finished run with libtorch 2.11).
- Opt-in JIT knobs, all off by default: `NAMD_MLFF_JIT_PROFILING`, `NAMD_MLFF_JIT_OPTIMIZE`, `NAMD_MLFF_JIT_TEXPR`,
  `NAMD_MLFF_JIT_FUSION`, `NAMD_MLFF_JIT_PROFILED_RUNS`, `NAMD_MLFF_FREEZE`, `NAMD_MLFF_TF32` (TF32 changes numerics).
- `build_and_test.sh`: defaults point at libtorch 2.11.0+cu130 and CUDA 13 headers on this box (`TORCH=`, `CUDA=` overridable);
  drops `-lcudart` (libtorch's own cudart is used). On another machine just pass `TORCH=<libtorch> CUDA=<cuda root>`.

## 02 — FeNNiX backend (fennol only)
- `PjrtPlugin::execute_into()`: one host->device upload, no separate execute await, outputs copied straight into
  persistent host arrays (no size query, no per-step allocation), all device->host copies awaited together, device handle cached.
  FeNNiX call inside namd3 at 300 atoms: 5.5 -> 1.8 ms. `FENNIX_LEGACY_EXEC=1` switches back to the old path.
- Charge restore: per-step O(N x numQMAtoms) linear search -> id->index hash built once (same first-match semantics).
- `FENNIX_TIMING=1` (cadence `FENNIX_TIMING_EVERY`, default 100): per-phase ms (pre / infer / post).

## 03 — ComputeQM fast index (all QM engines, opt-in)
- `ComputeQM::doWork` linear-searched the global QM table for every local QM atom every step (~18M comparisons/step at
  6000 atoms). With `NAMD_QM_FAST_INDEX=1` it uses an id->index hash (same first-match semantics). ~4 ms/step saved at 6000 atoms.

## Not patched (config / known issues)
- `QMNoPntChrg on` in the NAMD config removes per-step point-charge re-selection for whole-box QM (no MM point charges):
  6000 atoms 37 -> 19 ms/step on the stock binary. Config only, no source change.
- `src/ComputeMLFF.C:2795` has the same O(N x numQMAtoms) charge-restore search as 02 fixed for FeNNiX — not patched yet.
- Multi-walker (walk >= 2) segfault on client connect is a pre-existing regression in the Jul-31 `ComputeMLFF.C`; not patched.
- Build config: `Make.config` TORCHDIR must point at the same libtorch the shim is built against (one libtorch per process).
