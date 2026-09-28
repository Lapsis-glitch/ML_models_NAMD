# ANI-2x (TorchANI) inference optimisation — FINAL (2026-09-23)

The model is unchanged: ANI-2x, all 8 ensemble members, same weights (checked tensor by tensor against the deployed `models/compiled_ani2x.pt`), same fp32 network math. Every artifact loads in NAMD's **default shim** (`namd_benchmarks/lib/libnamd_mlff.so`, the C++ libtorch 2.11 zip). **No Python runs inside NAMD.**

## TL;DR

| artifact (models/opt/) | what it is | NAMD requirements |
|---|---|---|
| `ani_baseline.pt` | current wrapper (defaults) + deployed inner `models/compiled_ani2x.pt` (BASELINE) | none |
| **`ani_fast.pt`** | **FastANI (recommended):** cuAEV + fused ensemble + grouped elements + species cache + fp64 energy sum + cell-list periodic path; wrapper `lean=True` | cuAEV lib |
| `ani_fast_nocache.pt` | same without the species cache and the grouped pass (safe if atom types change at a fixed size) | cuAEV lib |
| `ani_fast_acc32.pt` | EXTRA: like `ani_fast` but sums energies in fp32 like the stock model (energies identical to baseline in the parity cases) | cuAEV lib |
| `ani_baseline_lean.pt` | reference: stock inner + wrapper `lean=True` only | none |

```bash
export NAMD_MLFF_EXTRA_LIBS=/home/rat/PycharmProjects/ML_models_NAMD/scripts/opt/ani/cuaev_native/libcuaev_native_precise.so
```
That is the only extra. The library needs only the libtorch zip, plus the zip's cudart through a symlink in `cuaev_native/lib/`. It needs neither libtorch_python nor libpython.

namd3 runs (walk0, seeded, private run dirs):

| system | base | **fast** | speedup |
|---|---|---|---|
| 300 atoms (steps 250–300) | 14.0 ms/step | **2.09 ms/step** | **6.7x** |
| 3000 atoms (steps 50–100 base / 250–300 fast) | 25.0 ms/step | **10.2 ms/step** | 2.5x |

At 3000 atoms NAMD's own ~6.5 ms/step queue wait dominates.

QMENERGY of fast vs base shows a constant offset: ~0.6 kcal/mol at 300 atoms, 2.2 at 3000. That offset is the baseline's fp32 energy rounding (see Parity). The trajectories track each other (step 100: -4794081.05 vs -4794080.46).

## What was done (all exact rewrites; fp32 network math unchanged)

1. **Native cuAEV** (`cuaev_native/`).
   - torchani 2.7.9 ships the cuAEV CUDA sources (`torchani/csrc/aev.cu`, `cuaev.cpp`) but no binary. They already register `TORCH_LIBRARY(cuaev)` with a C++ Autograd kernel.
   - The only source change is `<torch/extension.h>` → `<torch/torch.h>`, which drops pybind11/Python. The library is built against the libtorch zip with nvcc 13.0 for sm_120.
   - The system nvcc is 12.0/12.6, which has no sm_120. nvcc 13.0.88 plus cccl/crt/nvvm were pip-installed with `--target scripts/opt/ani/toolchain --no-deps` (264 MB; allegro untouched).
   - Variant `precise` (default) uses plain `expf`/`cosf` and no `-use_fast_math`. `VARIANT=upstream` (torchani's CMake flags: `-use_fast_math` + `__expf` intrinsics) also builds but is not used: no measurable end-to-end gain, and AEV is only ~20% of GPU time.
   - The single-molecule path that does all pairs inside the kernel (`torch.ops.cuaev.run`) replaces torchani's PyTorch all-pairs neighbour list, radial/angular terms and index_add (~150 ops). cuAEV vs pyAEV: max|dAEV| 1.3e-6, max|d grad| 1e-5 (fp32 noise).
   - Walkers (B > 1) of ≤ 1024 atoms go through cuAEV's batch mode in one call, 1.6–2.6x faster than one call per walker at W=4. Larger walkers get one call each: the batch kernel uses one thread block per molecule and holds it in shared memory (≤ ~2400 atoms).
2. **Fused ensemble** (`fast_ani.py: ElementNet`).
   - Per element, layer 1 of all 8 members is ONE GEMM ([8·h1, 1008] @ [1008, n]); layers 2–4 are batched GEMMs over members.
   - Everything is kept transposed (features × atoms), so there are no layout copies.
   - Atoms are grouped by element with one bincount + stable argsort. The stock model does 8 members × 7 elements × (eq + nonzero with a host **sync** + index_select + MLP + index_add).
3. **Grouped element pass** (opt-in, on in `ani_fast`).
   - All present elements run in one chain of batched GEMMs over (element × member).
   - Hidden widths are zero-padded to the widest element; padded units stay exactly 0 through CELU.
   - Atoms are padded to the most populated element and masked out of the energy.
   - This means fewer launches but more FLOPs, so it is used only while (#elements present) × (max atoms per element) ≤ 1024. Gains: 1.2x on water 30/300, 1.9x on a 32-atom peptide, 1.4x on a 334-atom peptide. Above the threshold the per-element loop is used.
4. **Species cache** (opt-in, on in `ani_fast`).
   - The element grouping (and the grouped plan) is reused while the species tensor and size are unchanged.
   - The check is a device-side `torch._assert_async`, so a changed topology at the same size aborts the run instead of producing wrong numbers. NAMD never changes Z.
   - `ani_fast_nocache` is the variant without it (one small device-to-host sync per call).
5. **fp64 energy accumulation** (in `ani_fast` and `ani_fast_nocache`).
   - Per-atom network outputs (fp32, as computed) and the fp32-stored self energies are summed in fp64. The stock model sums a −10⁴…−10⁵ Ha total in fp32.
   - Forces are unaffected (∂sum/∂xᵢ = 1).
   - `ani_fast_acc32` keeps the stock fp32 sum.
6. **Periodic path** (the wrapper calls `inner.ani((species, coords), cell, pbc, …)`).
   - The stock path uses torchani's `AllPairs`, which enumerates every pair in all image cells within the cutoff: O(27 N²), **3.1 s and 9 GB at 3000 atoms**.
   - FastANI uses torchani's own `CellList` above 190 atoms when every axis has ≥ 3 cutoff-sized buckets, otherwise `AllPairs` (torchani's `AdaptiveList` rule). AEVs come from torchani's pure-PyTorch terms.
   - Coordinates are first wrapped into the central cell using the *differentiable* cell. CellList's own wrap uses `cell.detach()`, which would drop the wrap term from the virial.
   - cuAEV is not used for periodic systems: its half-neighbour-list op returns no gradient for the pair vectors, so the strain virial comes out wrong (measured dV ≈ 60–240 kcal/mol). Rejected.
7. **Wrapper `lean=True`** (opt-in flag in `src/wrappers/wrap_torchani.py`, default unchanged).
   - The Z→species table stays on the device instead of being copied host→device every call.
   - The 3 element-range checks become `torch._assert_async`: no host syncs, but a bad Z aborts instead of raising.
   - The Hartree→kcal factor moves to the device once.
   - Outputs are bit-identical (base vs base_lean: dE 0).

## Benchmarks (bench_common, jitter 0.02 Å, one (system, walkers) per invocation, 15 GiB cap, cuAEV via the native lib only)

Median ms (peak alloc MiB). Results are in `results/ani_final_w<N>_W<W>.json`, `ani_final_prot_*_W1.json` and `ani_final_pbc.txt`.

| atoms × W | base | **fast** | speedup | fast_nocache |
|---|---|---|---|---|
| 30×1 | 12.63 (19) | **1.56** (22) | 8.1x | 2.08 (1) |
| 300×1 | 13.22 (34) | **1.64** (32) | 8.0x | 2.09 (8) |
| 900×1 | 14.25 (76) | **2.02** (46) | 7.1x | 2.19 (25) |
| 3000×1 | 17.38 (267) | **3.42** (105) | 5.1x | 3.64 (85) |
| 6000×1 | 25.22 (910) | **6.09** (191) | 4.1x | 6.54 (170) |
| 30×4 | 14.22 (22) | **2.01** (25) | 7.1x | 2.40 (3) |
| 300×4 | 13.95 (85) | **2.42** (54) | 5.8x | 2.60 (33) |
| 900×4 | 16.35 (250) | **4.27** (121) | 3.8x | 4.52 (101) |
| 3000×4 | 36.70 (1019) | **12.71** (360) | 2.9x | 13.17 (338) |
| 6000×4 | 75.62 (3588) | **24.67** (701) | 3.1x | 25.49 (679) |

Protein, W=1: OMP decarboxylase monomer (`namd_benchmarks/enzyme/systems/mono_dry/qm.pdb`, H C N O S) and two fragments (residues 1–2 and 1–20).

| system | base | **fast** | speedup | fast_nocache |
|---|---|---|---|---|
| 32 atoms (res 1–2) | 18.76 | **1.47** | 12.8x | 2.95 |
| 334 atoms (res 1–20) | 21.21 | **2.18** | 9.7x | 3.07 |
| 3281 atoms (monomer) | 24.46 | **4.63** | 5.3x | 4.86 |

Periodic path (`bench_pbc.py`, cubic cell = extent + 3 Å, W=1; peak = one call):

| water, cell | base | **fast** | speedup | peak MiB base → fast |
|---|---|---|---|---|
| 30, 11.7 Å | 15.6 | **7.0** | 2.2x | 341 → 341 |
| 300, 19.5 Å | 20.8 | **9.2** | 2.2x | 428 → 360 |
| 900, 26.0 Å | 58.1 | **10.0** | 5.8x | 1133 → 398 |
| 3000, 36.1 Å | 3119 | **13.2** | 236x | 9024 → 469 |
| 6000, 44.2 Å | (not run: > 16 GB) | **19.6** | – | → 692 |

## Parity (`parity.txt`; fresh process, torchani never imported, called like the shim)

The model is fp32, so the reference is an **fp64 evaluation of the same torchani model** (`ref_fp64.py`: whole model `.double()`, pyAEV, fp64 strain virial with the wrapper's convention). Each case uses 3 geometries (water + 0.05 Å noise); for periodic cases, geometry g=1 is translated so atoms sit outside the central cell. Worst over all cases (kcal/mol, kcal/mol/Å, kcal/mol):

| model | max dE vs base | max dF vs base | max dV vs base | **dE vs fp64** | **dF vs fp64** | **dV vs fp64** |
|---|---|---|---|---|---|---|
| base | 0 | 0 | 0 | 4.15 | 1.80e-2 | 2.6e-2 |
| **fast** | 5.44 | 1.69e-2 | 8.9e-2 | **6.7e-3** | **1.12e-2** | **3.1e-2** |
| fast_nocache | 5.44 | 1.69e-2 | 1.1e-1 | 6.7e-3 | 1.13e-2 | 2.9e-2 |
| fast_acc32 | **0** | 1.69e-2 | 9.8e-2 | 4.15 | 1.12e-2 | 2.7e-2 |

- **Forces, non-periodic:** fast vs base dF ≤ 7.5e-4. The baseline's own dF vs fp64 is 8e-4 at 30 atoms and 5.7e-3 at 3000.
- **Forces, periodic:** fast vs base dF reaches 1.7e-2 at 3000 atoms. The baseline's own dF vs fp64 is 1.8e-2 at 900 atoms, because the fp32 coordinates are 20–60 Å in magnitude.
- **Summary:** every fast variant is **at or below the baseline's own error** against fp64.
- **Energies:** the large "dE vs base" numbers are the **baseline's** fp32 rounding of the total energy: 0.054 / 0.74 / 4.15 kcal/mol at 30 / 300 / 3000 atoms vs fp64. This is deterministic, a quantisation rather than noise. `fast` is within 7e-3 kcal/mol of fp64; `fast_acc32` reproduces the baseline energies exactly.
- **Virial:** |V| is O(10³–10⁴) kcal/mol for these boxes, so dV ≈ 1e-2…1e-1 is ~1e-5 relative, i.e. fp32 noise. The baseline already differs from itself run to run by up to 7e-3 because of atomic index_add ordering.

## Where the time goes

| | baseline | fast |
|---|---|---|
| kernels per call (30 / 300 / 3000 atoms) | 913 / 956 / 949 | 79 / 79 / 110 |
| host syncs per call (300 atoms) | ~70 (a `nonzero` per member × element) | ~4 (2 inside cuAEV, cell check, output copy) |
| wall per call (profiler, 30 / 300 / 3000) | 13.2 / 13.1 / 17.0 ms | 1.5 / 1.8 / 3.4 ms |

- **Baseline:** host-bound up to ~3000 atoms. It runs 8 members × 7 elements of mask/nonzero/index_select/4 Linear/index_add, forward and backward, plus the PyTorch AEV (~150 ops).
- **Fast, 30–900 atoms:** still host-bound, ~80 launches.
  - The cuAEV forward costs ~0.22 ms of host time (2 internal syncs, cub calls, ~20 launches).
  - TorchScript knobs (`--jit-profiling 0`, `--jit-optimize 0`, `--jit-texpr 0`) are within noise.
- **Fast, 3000 atoms (GPU-bound):**

  | part | share of GPU time |
  |---|---|
  | layer-1 GEMM (fp32 SIMT) | ~40% |
  | cuAEV forward + backward | ~23% |
  | layers 2–4 (bmm/baddbmm) | ~20% |
  | elementwise / copies | rest |

## What was tried / rejected

| idea | result |
|---|---|
| native cuAEV (precise math) | exact to fp32 noise; the core of the ~6x |
| fused ensemble per element | ~20 → ~10 GEMM launches per element |
| grouped element pass | +15–20% on water ≤ 300 atoms, +40–90% on peptides; slower above ~1000 padded atoms, so size-gated |
| species cache (guarded by async assert) | +10–20% at ≤ 300 atoms |
| wrapper `lean` | ~0.2 ms saved at small N, bit-identical |
| cuAEV batch mode for W > 1, ≤ 1024 atoms | W=4: 30 atoms 2.6 → 1.7 ms, 300 atoms 3.3 → 2.4 ms |
| cuAEV `upstream` fast-math build | built, no measurable end-to-end gain; not used |
| first layer on a transposed view (no copy) | neutral or slightly slower at 3000 atoms; reverted |
| cuAEV for the periodic path (`--pbc-cuaev`) | forces OK, **virial wrong** (no gradient to pair vectors); rejected |
| TorchScript JIT knobs | within noise |
| MNP extension (OpenMP multi-stream) | not built: it runs the per-element nets in parallel CPU threads, and the fused/grouped GEMMs remove the same launches without threads |
| CUDA graphs | not possible: cuAEV has 2 data-dependent host syncs per call, and the graph shim is not installed |

## Tests
- `pytest tests/ -k "ani or ANI or torchani or TorchANI"`: 24 passed, 1 failed. The failure is `TestE2E_TorchANI::test_train_and_wrap`, the known pre-existing one (`TorchANI_Wrapper.forward() missing 1 required positional argument: 'cell'`; the test predates the periodic signature).
- `tests/test_interface_compliance.py`: 21 passed.
- The only change to `src/wrappers/wrap_torchani.py` is the new opt-in `lean` flag. The pre-edit file is backed up as `scripts/opt/ani/wrap_torchani.py.orig`.

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
bash scripts/opt/ani/build.sh      # nvcc wheels (if missing) + cuAEV lib + all artifacts + parity + mlff_shim_test (~5 min)
bash scripts/opt/ani/run_grid.sh   # bench grid + proteins + periodic (~10 min)
flock scripts/opt/.gpu_bench.lock bash scripts/opt/ani/namd_smoke/run_smoke.sh fast models/opt/ani_fast.pt native 100 300
```
Files:
- Native lib: `cuaev_native/{build.sh,src/,lib/,libcuaev_native_precise.so}`
- Model and build: `fast_ani.py`, `build_fast.py`, `ani_env.py` (build-time only: loads the native lib and tells torchani cuAEV is installed), `wrap.py`
- Parity: `geoms.py`, `ref_fp64.py`, `parity_native.py`
- Benchmarks and profiling: `bench_pbc.py`, `run_grid.sh`, `prof.py`, `namd_smoke/run_smoke.sh`, `geoms_prot/`
- Logs: `parity.txt`, `shim_test.txt`, `grid_log.txt`

## Follow-ups / user decisions
1. **Pre-existing bug — FIXED 2026-09-23 by the main session** (defaults in wrap_torchani.py, cli.py, compile_torchani.py, docs; Z tables patched in models/ani2x_wrapped_identical*.pt, trpcage_ani2x_*.pt, opt/ani_baseline*.pt with .pre_elemfix.bak backups; regression test tests/test_ani_element_order.py). Original note: `wrap_torchani.py`'s default `element_list` `[1,6,7,8,16,17]` (and `src/cli.py`'s `--elements` default) does not match ANI-2x's species order H C N O S **F Cl** = `[1,6,7,8,16,9,17]`.
   - Every Cl atom is evaluated with the **F** network, and F is rejected.
   - Water and proteins (H C N O S) are unaffected. The deployed `namd_benchmarks/models/ani2x.pt` has the same issue.
   - My artifacts pass the correct list explicitly. Fixing the default changes results for Cl-containing systems, so it is your call.
2. To sweep with the fast artifact, repoint `namd_benchmarks/models/ani2x.pt` (your symlink, not changed) at `models/opt/ani_fast.pt` and export the `NAMD_MLFF_EXTRA_LIBS` line above.
3. `ani_fast` aborts (device assert) if the atom types change at a fixed atom count within one process. NAMD never does that; use `ani_fast_nocache` if something else might.
4. The stock periodic path is unusable beyond ~1000 atoms (O(27 N²) memory). If you run periodic ANI, use the fast artifact.
5. The cuAEV lib's RPATH points to `/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130`. Rebuild it (`cuaev_native/build.sh`) if the zip moves.
