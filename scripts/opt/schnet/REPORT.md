# SchNet + shared neighbour list + CUDA graphs — optimisation report (schnet-nl)

Model: `models/compiled_schnet_default.pt` (SchNetPack 2.x SchNet, 3 interactions, 128 features, 20 radial basis functions, r_max 5 Å, fp32 weights). Env: allegro (torch 2.11.0+cu130), RTX 5080 Laptop (sm_120). The model and its weights are unchanged; everything below changes how the same maths is evaluated.

## Artifacts
| artifact | what it is |
|---|---|
| `models/opt/schnet_baseline.pt` | Wrapper and `edges.py` as they were before this work (user's PBC version). Rebuilt byte-identically from the `.pre_schnet_nl.bak` copies. |
| `models/opt/schnet_default_new.pt` | Current source, default flags. Only the two shared `edges.py` fixes apply. |
| `models/opt/schnet_fast_fullfilter.pt` | `--fast --no-half-filter` (ablation). |
| `models/opt/schnet_fast.pt` | **Recommended.** `--fast`: fast energy route, half-list filter from 1500 atoms, cell-list neighbour list for large systems, CUDA-graph API. |

## Results

### Plain `forward` / `forward_batch`
- Run with `bench_common.py --jitter 0.02`, one invocation per system (`scripts/opt/results/schnet_final_jitter_w{30,300,900,3000,6000}.json`).
- Speedup is vs base. Peak memory is in MiB.

| system | W | base ms | default_new ms | fast_fullfilter ms | **fast ms** | speedup | peak MiB base → fast |
|---|---|---|---|---|---|---|---|
| water30 | 1 | 3.21 | 2.98 | 2.87 | **2.82** | 1.14× | 21 → 4 |
| water30 | 4 | 3.53 | 3.26 | 3.00 | **3.01** | 1.17× | 16 → 16 |
| water300 | 1 | 3.80 | 3.55 | 3.28 | **3.35** | 1.13× | 96 → 79 |
| water300 | 4 | 4.77 | 4.43 | 4.10 | **4.04** | 1.18× | 316 → 316 |
| water900 | 1 | 4.36 | 4.14 | 3.87 | **3.91** | 1.11× | 294 → 276 |
| water900 | 4 | 11.89 | 10.68 | 10.51 | **7.92** | 1.50× | 1106 → 751 |
| water3000 | 1 | 11.11 | 10.95 | 10.76 | **8.20** | 1.36× | 1125 → 749 |
| water3000 | 4 | 52.48 | 40.89 | 39.32 | **28.17** | 1.86× | 4405 → 3003 |
| water6000 | 1 | 23.60 | 23.49 | 21.38 | **15.86** | 1.49× | 2348 → 1578 |
| water6000 | 4 (each model alone) | 421.5 | 91.2 | 80.5 | **57.6** | 7.3× | 15400 → 6345 |

- **p10/p90** are in the JSON files. The spread is ±10–15% at small N and ±2–4% at large N.
- **Why 6000×4 is reported "alone":** the base model's block-diagonal neighbour list peaks at 15.4 GB on a 16 GB card. When models are interleaved, every other model's timing thrashes the allocator (default_new 377 ms, fast_fullfilter 271 ms). The `schnet_6000x4_*_alone.json` runs are the real numbers; `fast` is 57.7 ms in both modes.
- **Why one invocation per system:** the TorchScript profiling executor only optimises the branches taken during profiling. `fast` switches route by system size. If one process benchmarks water30 first, its large-N branch runs unfused later (8.2 → 9.2 ms at 3000 atoms). NAMD runs one N per process, so the per-system invocations are representative. `schnet_final_jitter_allsys_oneprocess.json` is the all-systems single invocation, kept for completeness.

### CUDA-graph path, through the NAMD C ABI
- Measured with infra's `shim_bench` (dlopen, `mlff_eval`, jitter 0.02, 200 iterations, mean of 2 rounds); the file is `scripts/opt/results/schnet_shim_bench_graph.txt`.
- "fast+graph" is the patched shim with `NAMD_MLFF_CUDA_GRAPH=1`. "base" and "fast" use the installed shim.
- The Python equivalent (`schnet_graph_jitter.json`) agrees: 0.52 / 1.18 / 2.12 / 4.53 ms.

| system | base ms | fast ms | fast+graph ms | graph vs base |
|---|---|---|---|---|
| water30 | 3.25 | 2.92 | **0.50** | 6.5× |
| water300 | 3.73 | 3.38 | **1.20** | 3.1× |
| water900 | 4.46 | 4.02 | **2.06** | 2.2× |
| water1800 | 6.90 | 5.35 | **4.52** | 1.5× |
| water3000 | 11.24 | 8.31 | 8.30 | no graph: above `graph_max_atoms`=2048, falls back to `forward` |

- The first call to the graph path costs about 1 s (capture). After that there are no re-captures under jitter (one capture per system).
- Graph memory is +35–125 MiB (static pool plus 25% edge headroom).

## Where the time goes
Profile from `scripts/opt/schnet/profile_breakdown.py` on the baseline:
- **water30:** 3.0 ms wall but only about 0.3 ms of GPU work, so about 90% is host overhead. That is roughly 170 ops per step through the TorchScript interpreter and autograd, plus 3 host syncs:
  - `linalg_det` inside `cell_is_periodic`;
  - `nonzero` in the neighbour list;
  - `int(idx_m[-1])` inside SchNetPack's Atomwise, in the middle of the step.
  - Only CUDA graphs remove this overhead.
- **water3000:** about 9 of 11 ms is GPU:
  - mm/addmm about 27%;
  - NNC-fused elementwise work (radial basis, softplus, filter × cutoff products) about 30%;
  - `index_add` and the indexing backward about 15%;
  - dense neighbour list about 1 ms;
  - most per-edge work is the filter network running on E ≈ 38×N directed edges.
- **water3000 ×4:** the block-diagonal [12000,12000,3] neighbour list alone was 16.8 ms of 52 ms.

## What worked
1. **`cell_is_periodic` short-circuit** (`src/edges.py`, shared, on by default). Return False straight away on an all-zero cell before calling `torch.linalg.det`. The det cost about 250 µs of host time on every non-periodic step. Results are identical for all inputs (zero / box / flat / NaN cells checked).
2. **Per-molecule batched neighbour list** (`build_edges_batched`, shared, on by default).
   - With equal-size walkers it computes [B,n,n] instead of block-diagonal [BN,BN]. That is B× less work and memory.
   - Output is bit-identical (edge_index, vecs and lengths checked with `torch.equal` for 30–6000 atoms × W, plus unequal `ptr`). Unequal sizes keep the old path.
   - Effect: 3000×4 52 → 41 ms; 6000×4 15.4 → 9.3 GB and 421 → 91 ms.
3. **Fast energy route** (`fast=True`, opt-in). Calls SchNetPack's own submodules but drops:
   - the Forces postprocessor and the dict plumbing around it;
   - the Atomwise `int(idx_m[-1])` host sync (my own `index_add`, the same op SchNetPack uses internally);
   - the per-call H2D copy of `conv_factor`.
   - Forces come from one `autograd.grad` on an fp32 leaf. Gain is 6–12% at small N. It is also what makes the step capturable as a graph.
4. **Half-list filter** (`half_filter`, on by default within `--fast`).
   - SchNet evaluates W(d_ij) for both (i,j) and (j,i). In the non-periodic case these are bit-identical, so the wrapper computes the filter once per pair and scatters both messages.
   - A numerical self-check against the stock route runs when the wrapper is constructed.
   - Gain is 1.36–1.5× and about 30% less memory from about 3000 atoms. Below about 1500 total atoms it is slower in plain `forward` (extra selection kernels while the step is launch-bound), hence `half_min_atoms=1500`. `graph_step` always uses it (900 atoms: 2.64 → 2.12 ms).
5. **CUDA graphs** (opt-in on both the model and the shim side).
   - `graph_step` builds the neighbour list inside the graph with `torch.nonzero_static`, padded to a capacity. Padding entries are self-edges spread over all atoms with a 2·r_max offset, so the cosine cutoff is exactly 0 and they contribute exactly 0 to energy and forces.
   - The step has no host syncs. Its energy, forces and `n_edges` outputs are static.
   - Spreading the padding matters: putting all padding on atom 0 serialised the scatter atomics (900 atoms: 11.7 ms instead of 2.7 ms).
   - Graphs win up to about 2000 atoms and tie at 3000, hence `graph_max_atoms=2048`.
6. **Cell-list neighbour list** (`build_edges_cell` and `build_edges_cell_batched` in `src/edges.py`, opt-in, not used by any default path).
   - Same edges in the same row-major order as the dense builders. Checked eager and scripted (including the NNC-fused 3rd call), CPU and GPU, 3–6000 atoms, jitter, W=2/4, unequal `ptr`, edge-of-cutoff pairs.
   - One host sync (fixed-stride cell ids plus `searchsorted`, so no grid-shape sync and no `bincount`).
   - Fixed cost about 0.9 ms, so it only wins above about 4500 atoms ×1 or about 2500 atoms ×4 (6000 atoms: 2.45 → 1.83 ms; 6000×4 neighbour list: 10.8 → 3.4 ms and 2.2 GB → 0.42 GB).
   - `--fast` uses it when B·n² ≥ 16e6 (`nl_cell_min_pairs`).

## What didn't / not pursued
- **Graphs above about 2000 atoms:** the 25% edge padding and the in-graph O(N²) neighbour list cancel the launch saving.
- **vesin:** `vesin-torch` is not installed in allegro, it does its search on the CPU (a round trip from GPU), and it is pinned to a libtorch minor version and needs `NAMD_MLFF_EXTRA_LIBS`. The pure-torch cell list above covers the same need without those costs.
- **JIT knobs:** infra already showed none is a clear win.
- **TF32:** not used; it changes numerics.

## Parity (vs baseline; fp32 model)
- **Forces:** worst |dF| = 3.3e-5 kcal/mol/Å across all cases (`scripts/opt/schnet/check_fast_parity.out`). The noise floor is the baseline against itself: 1.3e-5 on CPU between successive calls, because the profiled/fused graph differs from the first call, and ~1e-5 on GPU from atomic ordering.
- **Energy:** |dE| up to 0.08 kcal/mol at 6000 atoms. That is about 3e-6 relative (E ≈ 3.1e4 kcal/mol at 6000 atoms), the same size the unchanged-maths `default_new` shows.
- **Other paths checked:**
  - `graph_step` vs `forward`: same noise floor.
  - Unequal `ptr` batches: fine.
  - Periodic path: untouched, and its virial differs only within its own run-to-run noise (3e-4).
  - Output shapes: identical to the baseline.

## Tests
- `python -m pytest tests/ -q`: 120 passed / 16 skipped / 2 failed, identical before and after these edits.
- The 2 failures (`TestE2E_SchNetPack`, `TestE2E_TorchANI::test_train_and_wrap`) come from the uncommitted PBC work: the test calls `forward()` without `cell`. They were not touched.

## NAMD shim loadability
- **Installed shim** (`namd_benchmarks/lib/libnamd_mlff.so`, libtorch 2.11): `mlff_shim_test ... 0` gives **ALL CHECKS PASSED** for `schnet_default_new`, `schnet_fast` and `schnet_fast_fullfilter`. No `NAMD_MLFF_EXTRA_LIBS` needed for any artifact. Output in `scripts/opt/results/schnet_shim_test.txt`.
- **Patched shim** (`scripts/opt/schnet/cudagraph/shim/libnamd_mlff.so`, built from a copy; the installed shim and its source are untouched): with `NAMD_MLFF_CUDA_GRAPH=1` it passes on GPU (prints "CUDA graph path ON") and on CPU (graph path disabled there).
- **Overflow / re-capture** (`cudagraph/test_overflow.py`) was exercised in both Python and C++ (through ctypes into the patched shim). Evaluating A, then B = 0.85·A (more pairs than the captured capacity), then B, then A agrees with the non-graph handle (dF ≤ 2.2e-5).
- **What the shim patch does** (`cudagraph/shim/apply_cuda_graph_patch.py`, about 130 lines):
  - Only serial, non-periodic `mlff_eval` calls with ≤ `graph_max_atoms` atoms use it (override with `NAMD_MLFF_CUDA_GRAPH_MAX_ATOMS`).
  - Capacity comes from `model.graph_capacity(coords)`. It warms up 4× on a side stream, then captures with `cudaStreamCaptureModeThreadLocal` so NAMD's other CUDA threads are unaffected.
  - Each step: H2D copy of coords into a static buffer, replay, D2H copy of energy, forces and `n_edges`, one sync. If `n_edges > cap`, it re-captures and re-runs.
  - Any exception disables the graph path for that model and falls back to `forward()`.
  - Batched `forward_batch` and PBC are not graphed (multi-walker NAMD is currently broken anyway).
- **Model API contract** for any model wanting the graph path:
  - attributes `graph_capable: bool` and `graph_max_atoms: int`;
  - `graph_capacity(coords) -> int`;
  - `graph_step(coords, Z, cap) -> (E[1] f64 kcal/mol, F[N,3] f64 kcal/mol/Å, n_edges)`;
  - `n_edges > cap` means the output is invalid and the caller must re-capture.
- **Caveat:** `NAMD_MLFF_FREEZE=1` keeps only `forward_batch`, so it would strip `graph_step`/`graph_capacity`, and the graph path silently falls back to `forward()` (with a stderr note). To combine the two knobs, freeze must also preserve those two methods.

## Follow-ups
- `src/cli.py --model-type schnet` does not expose the new flags; build the fast artifact via `python -m src.wrappers.wrap_schnetpack --fast ...` (build.sh does this).
- Installing the graph-capable shim (copy into `namd_benchmarks/lib` after backing up, or fold the patch into the shim source) is the user's decision; off by default either way.

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
export LD_LIBRARY_PATH=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
PY=/home/rat/miniconda3/envs/allegro/bin/python
BUILD_SHIM=1 bash scripts/opt/schnet/build.sh              # 4 artifacts + patched shim copy
# correctness
flock scripts/opt/.gpu_bench.lock $PY scripts/opt/schnet/nl/check_batched_identity.py
flock scripts/opt/.gpu_bench.lock $PY scripts/opt/schnet/nl/check_cell_nl.py
flock scripts/opt/.gpu_bench.lock $PY scripts/opt/schnet/check_fast_parity.py
flock scripts/opt/.gpu_bench.lock $PY scripts/opt/schnet/cudagraph/test_overflow.py py
(source namd_benchmarks/env.sh; flock scripts/opt/.gpu_bench.lock $PY scripts/opt/schnet/cudagraph/test_overflow.py shim)
$PY -m pytest tests/ -q
# benchmarks (bench_common takes the lock itself)
for S in 30 300 900 3000 6000; do $PY scripts/opt/bench_common.py --model base=models/opt/schnet_baseline.pt \
  --model default_new=models/opt/schnet_default_new.pt --model fast_fullfilter=models/opt/schnet_fast_fullfilter.pt \
  --model fast=models/opt/schnet_fast.pt --systems $S --walkers 1,4 --jitter 0.02 \
  --out scripts/opt/results/schnet_final_jitter_w$S.json; done
for m in base=models/opt/schnet_baseline.pt default_new=models/opt/schnet_default_new.pt \
         fast_fullfilter=models/opt/schnet_fast_fullfilter.pt fast=models/opt/schnet_fast.pt; do
  $PY scripts/opt/bench_common.py --model $m --systems 6000 --walkers 4 --jitter 0.02 --iters 15 --rounds 2 --warmup 8 \
    --out scripts/opt/results/schnet_6000x4_${m%%=*}_alone.json; done
$PY scripts/opt/schnet/cudagraph/bench_graph.py --base models/opt/schnet_baseline.pt --fast models/opt/schnet_fast.pt \
    --systems 30,300,900,1800,3000 --jitter 0.02 --out scripts/opt/results/schnet_graph_jitter.json
bash scripts/opt/schnet/cudagraph/shim_bench_graph.sh "30 300 900 1800 3000" 2
# NAMD loadability
source namd_benchmarks/env.sh
flock scripts/opt/.gpu_bench.lock namd_benchmarks/lib/mlff_shim_test $NAMD_MLFF_LIB models/opt/schnet_fast.pt 0
NAMD_MLFF_CUDA_GRAPH=1 flock scripts/opt/.gpu_bench.lock namd_benchmarks/lib/mlff_shim_test \
    scripts/opt/schnet/cudagraph/shim/libnamd_mlff.so models/opt/schnet_fast.pt 0
```
Other files: source edits in `src/edges.py` and `src/wrappers/wrap_schnetpack.py` (pre-edit copies `scripts/opt/schnet/nl/edges.py.pre_schnet_nl.bak`, `scripts/opt/schnet/wrap_schnetpack.py.pre_schnet_nl.bak`); tools `profile_breakdown.py`, `nl/time_nl.py`, `cudagraph/graph_ceiling.py`. `bench_baseline_initial.json` is the failed NVRTC run from the dead session.
