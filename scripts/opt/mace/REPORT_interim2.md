# MACE-OFF23 (medium) inference optimisation — INTERIM 2 (stopped for reboot, 2026-09-22 ~01:25)

## Status
- cuEq loadability: blocker CONFIRMED and WORKED AROUND (see below). Baseline built. Exact FLOP cuts done (FastMACE).
- Not done yet: native C++ uniform_1d op (Route B, mapping captured), wrapper flags, CUDA graphs, fp32 extra, build.sh, final bench grid.
- src/wrappers/wrap_compiled_mace.py is UNEDITED.
- Tests: `-k "mace or MACE"` 25 passed / 2 skipped; test_interface_compliance.py 21 passed.

## Inputs / previous work
- Rebuild in allegro (rebuild.py): ScaleShiftMACE on mace 0.3.15 / e3nn 0.6 from models/opt/mace_off23_medium_state.pt. Parity vs old compiled inner: dE 9e-10 eV, dF 7e-14 eV/Å.
- cuEq fixes: U-basis rebase (rebase_U) for products.0 contractions.1 U_matrix_3 (stored basis differs from mace 0.3.15's; naive conversion silently gives dE 0.015 eV); TorchScript fixes (cueq_fusion.py) for run-time hasattr(cueq_config) branches and the MethodType conv-fusion patch (scripted stock cuEq path was ~63 eV off).
- Inner artifacts: models/opt/mace_inner_{e3nn,cueq,cueqf}_f64.pt.

## Artifacts (all wrapped with the CURRENT wrapper source via scripts/opt/mace/wrap.py)
| file | contents |
|---|---|
| models/opt/mace_baseline.pt | current wrapper + models/compiled_mace_off23_medium.pt (BASELINE) |
| models/opt/mace_e3nn.pt | rebuilt e3nn inner |
| models/opt/mace_cueqf.pt | cuEquivariance + fused conv TP |
| models/opt/mace_fast_cueqf.pt | FastMACE on cuEq (half_radial + species_skip + plain_linear). Rebuild with --no-plain-linear (faster). |

## NAMD loadability (mlff_shim_test, GPU)
- Default zip shim: mace_baseline.pt and mace_e3nn.pt pass (E = -47995.022275 kcal/mol). RSS 1.6 GB.
- Stock INFRA recipe (pip-torch shim + EXTRA_LIBS): mace_cueqf.pt FAILS with `Unknown builtin op: cuequivariance::uniform_1d` (op is a Python torch.library.custom_op; the .so files register nothing).
- Fix: scripts/opt/mace/cueq_pyinit/libmace_cueq_pyinit.so (pyinit.cpp): static constructor embeds CPython (home = allegro env, override MACE_CUEQ_PYHOME), imports cuequivariance_torch, releases the GIL, never finalises; if Python already runs it only imports. Result: ALL CHECKS PASSED, same energy as baseline, RSS 3.5 GB.
  ```bash
  unset NAMD_MLFF_LIB; export LIBTORCH_ROOT=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/torch
  source namd_benchmarks/env.sh; SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
  export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$LD_LIBRARY_PATH
  export NAMD_MLFF_EXTRA_LIBS=/home/rat/miniconda3/envs/allegro/lib/libpython3.12.so.1.0:$PWD/scripts/opt/mace/cueq_pyinit/libmace_cueq_pyinit.so
  flock scripts/opt/.gpu_bench.lock namd_benchmarks/lib/mlff_shim_test $NAMD_MLFF_LIB models/opt/mace_cueqf.pt 0
  ```
  Caveats: not yet run inside namd3 (multi-thread GIL behaviour untested); each uniform_1d call re-enters Python (~0.65 ms host; 4 fwd + 4 bwd per step); no CUDA-graph capture on this path; cuEq prints "pynvml: Not Supported, using RTX A6000 defaults" on WSL (kernel tuning only).
- Planned Route B: native TORCH_LIBRARY `cuequivariance::uniform_1d` calling libcue_ops `run_uniform_1d_cuda` — no Python, default zip shim, possibly graph-capturable. Flat→nested argument mapping captured in scripts/opt/mace/cueq_native/spy_mapping.log (LD_PRELOAD spy in spy.cpp). Op library not written yet.

## Results so far (bench_common, jitter 0.02, W=1, fp64)
| system | base | e3nn | cueqf | fast_cueqf | fast, no plain-linear |
|---|---|---|---|---|---|
| water30 | 36.4 ms | 37.1 | 13.8 (2.6x) | ~14.4 | – |
| water300 | 298.6 | 299.0 | 27.8 (10.7x) | 20.4 (14.6x) | 20.2 |
| water3000 | – | – | 266 | 194 | 184 |

- Separate invocations; compare within a row group only. base/e3nn/cueqf at 30/300: scripts/opt/results/mace_early_w{30,300}.json. fast vs cueqf: in-process ablation runs (not saved).
- Parity vs base: dF ≤ 1.6e-13 kcal/mol/Å; dE ≤ 7e-9 kcal/mol at 3000 atoms.
- Peak memory at water3000: 4.8 GB (cueqf) → 3.3 GB (fast).
- Ablation water3000: fast 194, no half_radial 244, no species_skip 234, no plain_linear 184 ms.

## Where the time goes
- e3nn fp64: products.0 symmetric contraction ~100 of 300 ms at 300 atoms (einsum over all 10 elements via one-hot).
- cuEq fp64, 3000 atoms, ~187 ms GPU/step after FastMACE: fp64 GEMMs (radial MLP on half edges + linears) ~70 ms; uniform_1d fwd+bwd ~50 ms; block linears ~33 ms. RTX 5080 Laptop fp64 ≈ 1/64 fp32.
- cuEq, 30 atoms: ~2.4 ms GPU vs ~14 ms wall — host-bound on Python custom-op round trips + TorchScript interpreter (~400 launches/step).

## Exact rewrites (scripts/opt/mace/fast_mace.py, build_fast.py)
1. half_radial: Bessel/cutoff embedding + conv_tp_weights once per undirected pair (i<j), gathered to directed edges. Sync-free (nonzero_static(size=E//2), sort + searchsorted, device-side _assert_async). Non-periodic only; virial/stress/training calls fall back to stock model.
2. species_skip: one-hot FCTP skip_tp → per-species block matrices (extracted by probing, verified 1e-12 rel). One D2H sync for species counts; cache_species=True caches per N.
3. plain_linear: cuEq Linear → plain matmul. Slower at large N — leave off.
Scripted FastMACE vs stock: dF ≤ 8e-15 eV/Å, dE ≤ 6e-11 eV at 300 atoms.

## Resume steps
1. `export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$LD_LIBRARY_PATH` — required even to BUILD cuEq models (else cuEq silently falls back to Naive and fails).
2. Rebuild recommended inner with --no-plain-linear (commands below).
3. Route B: write cueq_native/uniform1d_op.cpp — schema verbatim from `torch.ops.cuequivariance.uniform_1d.default._schema`; CUDA kernel + Autograd kernel (torch::autograd::Function; TensorList args flatten; needs_input_grad indexes tensors); include -I$SP (cuequivariance_ops/equivariance/uniform1d/api.hh), link libcue_ops.so. Mapping (spy_mapping.log): batch_dim vector<vector<>> per buffer incl. index buffers (Python BATCHED 1→kBatched 0, SHARED 0→kShared 1, INDEXED -1→kIndexed 2; index buffers [kBatched]); index_cfg [[idx or -1]] per buffer; dtypes per buffer; batch_sizes [batch_size]; buffer_bytes = nbytes; zero_out false (caller zeroes SHARED/INDEXED outputs); ignore_first true. See cuequivariance_ops_torch/uniform_1d.py for output alloc and _do_bwd_jit. Parity-check in a fresh process WITHOUT importing cuequivariance_torch, then mlff_shim_test on default zip shim with EXTRA_LIBS=libcue_ops.so:<op.so>.
4. Then: wrapper opt-in flags, CUDA-graph API if native op capturable, fp32 extra (build_fast.py --dtype float32), scripts/opt/mace/build.sh, final bench_common per system 30/300/900/3000/6000 × W=1,4 with --extra-lib (pyinit or native lib), `-k mace` + interface tests (NOT the full suite).

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
export LD_LIBRARY_PATH=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
PY=/home/rat/miniconda3/envs/allegro/bin/python
for k in e3nn cueq cueqf; do $PY scripts/opt/mace/compile_inner.py $k models/opt/mace_inner_${k}_f64.pt; done
$PY scripts/opt/mace/wrap.py models/compiled_mace_off23_medium.pt models/opt/mace_baseline.pt
$PY scripts/opt/mace/wrap.py models/opt/mace_inner_e3nn_f64.pt models/opt/mace_e3nn.pt
$PY scripts/opt/mace/wrap.py --cueq models/opt/mace_inner_cueqf_f64.pt models/opt/mace_cueqf.pt
flock scripts/opt/.gpu_bench.lock $PY scripts/opt/mace/build_fast.py cueqf models/opt/mace_inner_fast_cueqf_f64.pt --no-plain-linear
$PY scripts/opt/mace/wrap.py --cueq models/opt/mace_inner_fast_cueqf_f64.pt models/opt/mace_fast_cueqf.pt
(cd scripts/opt/mace/cueq_pyinit && E=/home/rat/miniconda3/envs/allegro && g++ -O2 -shared -fPIC pyinit.cpp -I$E/include/python3.12 -L$E/lib -lpython3.12 -Wl,-rpath,$E/lib -o libmace_cueq_pyinit.so)
$PY scripts/opt/bench_common.py --extra-lib scripts/opt/mace/cueq_pyinit/libmace_cueq_pyinit.so \
  --model base=models/opt/mace_baseline.pt --model cueqf=models/opt/mace_cueqf.pt --model fast=models/opt/mace_fast_cueqf.pt \
  --systems 300 --walkers 1 --jitter 0.02 --out scripts/opt/results/mace_w300.json
```
Tools: prof.py (torch.profiler of a wrapped artifact), breakdown.py (per-module synced timing).
