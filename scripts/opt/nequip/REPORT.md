# NequIP-OAM-L inference optimisation — FINAL (2026-09-23)

The model is unchanged: NequIP-OAM-L 0.1 (`mir-group/NequIP-OAM-L:0.1`, package `~/.nequip/model_cache/039e56…nequip.zip`), 9.6 M params, fp32 as shipped, TF32 off. Every recommended artifact loads in NAMD's **default shim** (`namd_benchmarks/lib/libnamd_mlff.so`, the C++ libtorch 2.11 zip). **No Python runs inside NAMD.**

## TL;DR

| artifact (models/opt/) | what it is | NAMD requirements | 300 atoms | 900 atoms | 3000 atoms |
|---|---|---|---|---|---|
| `nequip_baseline.pt` | current wrapper + deployed inner `models/compiled_nequip_oam_l.nequip.pth` (BASELINE) | none | 142.8 ms | 484.6 | OOM (>15 GiB) |
| `nequip_recompiled_cuda.pt` | same e3nn model recompiled in torch 2.11 (fp32 noise-floor reference) | none | 142.9 | 487.1 | OOM |
| `nequip_oeq.pt` | OpenEquivariance fused TP-conv kernels | liboeq_native.so | 25.9 (5.5x) | 43.5 (11.1x) | 130.7 |
| **`nequip_fast_oeq.pt`** | **OEQ + exact FastNequIP rewrites (recommended)** | **liboeq_native.so** | **24.9 (5.7x)** | **39.4 (12.3x)** | **115.4** |
| `nequip_cueq.pt` | cuEquivariance; its ops are Python-only | NOT NAMD-loadable | 98 | – | 1203 |

The runtime env for the OEQ artifacts is the default shim plus one library:
```bash
export NAMD_MLFF_EXTRA_LIBS=/home/rat/PycharmProjects/ML_models_NAMD/scripts/opt/nequip/oeq_native/liboeq_native.so
```
- Nothing else is needed.
- The library's RUNPATH points cudart and nvrtc at `oeq_native/lib/lib{cudart,nvrtc}.so.13`. Those are symlinks to the libtorch zip's hashed copies, so the process loads the same files libtorch already uses. cublas and the driver also come from the zip / WSL.

In a namd3 run (300 atoms, walk0, seeded, 300 steps), steps 100–300 take:

| artifact | ms/step |
|---|---|
| base | 141 |
| oeq | 27.8 |
| fast_oeq | **26.5 (5.3x)** |

## OpenEquivariance, native (the kernel route)

OAM-L was trained with `enable_OpenEquivariance` (its package config says so). NequIP swaps the e3nn TensorProductScatter for OEQ's fused `TensorProductConv` (`torch.ops.libtorch_tp_jit.jit_conv_forward`). The TorchScript archive stores the kernel description as a JSON-bytes buffer plus a hash. OEQ NVRTC-compiles the kernels on the first call and caches them per process.

- **Install:** OEQ 0.7.0 was pip-installed into allegro. The dry run showed only openequivariance + pybind11 would change; torch, e3nn and numpy are unchanged.
- **Why a new library was needed:** OEQ registers the op schemas and CUDA kernels in C++ (`extension/torch_core.hpp`, `TORCH_LIBRARY(libtorch_tp_jit)`), but the autograd formulas are Python-only (`TensorProductConv.register_autograd`). Neither shipped .so fits NAMD: the JIT build embeds pybind11 and links libpython, and the precompiled one targets CUDA 12.
- **What was built:** `scripts/opt/nequip/oeq_native/oeq_native.cpp` compiles the upstream `torch_core.hpp` unchanged against the libtorch zip and drops pybind11. It adds `TORCH_LIBRARY_IMPL(libtorch_tp_jit, Autograd)` kernels that mirror the Python ones: `jit_conv_forward` → `jit_conv_backward` → `jit_conv_double_backward`, and the same for `jit_tp_*`. The backward re-dispatches through the op, so double backward works; the Python triple backward is left out.
- **Build:** `oeq_native/build.sh`, about 20 s.

Acceptance:
1. **Parity in a fresh process that never imports openequivariance** (`parity_native.py` and `parity_fast.py` assert this on `sys.modules`). All cells OK; see the parity section.
2. **`mlff_shim_test` on the default shim:** baseline, oeq and fast_oeq all print ALL CHECKS PASSED.
   - Single-structure E: -290.890484 (base), -290.890420 (oeq), -290.890439 (fast) kcal/mol.
   - Batch energies agree to 5e-5.
   - RSS is 2.17 GB (base) vs 2.05 GB (oeq).
3. **namd3 smoke** (`namd_smoke/run_smoke.sh`, private run dirs under `scripts/opt/nequip/namd_smoke/runs/`; `namd_benchmarks/runs/` was not touched):

| cell (300 atoms) | libs | ms/step (steps 100–300) | first 50 steps | GPU peak | QMENERGY step 100 |
|---|---|---|---|---|---|
| base | default shim | 141 | 276 ms/step | 3035 MiB | -32954.5153 |
| oeq | default + liboeq_native.so | 27.8 | 141 ms/step (NVRTC + JIT warm-up) | 1225 MiB | -32954.5154 |
| fast_oeq | default + liboeq_native.so | **26.5** | 132 ms/step | 1051 MiB | -32954.5142 |

No crashes. Steps 0/1/2/10/50 agree to ≤1.5e-3 kcal/mol.

## FastNequIP: exact rewrites (`fast_nequip.py`, `build_fast.py`)

The rewrites are graph surgery on the eager model after `enable_OpenEquivariance` and before scripting. The weights are unchanged. The packaged model runs nequip 0.14 code, so the surgery handles that version's `scatter_norm_factor` and the ZBL `edge_cutoff` key.

1. **half_edges.** The radial MLP (8→128→1312 per layer), the Bessel/cutoff embedding and the spherical harmonics are computed once per undirected pair. The canonical half is `i<j`, or `i==j` with a positive cell shift for periodic self-images, picked sync-free with `nonzero_static(E/2)` plus a device-side assert.
   - The OEQ conv is called twice with the same per-pair weights: (dst=i, src=j, Y) and (dst=j, src=i, (−1)^l·Y). The sign flip is bit-exact, and no [E,1312] gather is materialised.
   - ZBL scatters each pair energy onto both atoms, so per-atom energies are identical to the stock model.
2. **species_sc.** FCTP(x, 48-d type embedding) becomes one dense matmul per atom type present, using the `kron(W_l[type], I_{2l+1})` blocks.
   - The blocks were probed in fp64; the reconstruction error is 0.0 in every layer.
   - Each call costs one small device-to-host copy (the type counts).
3. The no-op `[:num_local_nodes]` slices were dropped.

Ablation at 3000 atoms: half_edges alone gives 1.05x, species_sc alone 1.07x, both 1.13x. Memory drops 6.4 → 3.7 GiB.

## Benchmarks (bench_common, jitter 0.02 Å, one (system, walkers) per invocation, 15 GiB VRAM cap, expandable_segments, OEQ via the native lib only)

Median ms (peak alloc MiB). Results are in `results/nequip_final_w<N>_W<W>.json`.

| atoms × W | base | recomp | oeq | **fast_oeq** |
|---|---|---|---|---|
| 30×1 | 44.4 (122) | 45.4 (106) | 20.5 (51) | **19.4 (33)** |
| 300×1 | 142.8 (1755) | 142.9 (1739) | 25.9 (463) | **24.9 (270)** |
| 900×1 | 484.6 (6323) | 487.1 (6307) | 43.5 (1604) | **39.4 (926)** |
| 3000×1 | OOM¹ | – | 130.7 (6441) | **115.4 (3705)** |
| 6000×1 | – | – | 268.3 (13559) | **252.8 (7811)** |
| 30×4 | 50.2 (358) | 49.8 (342) | 22.2 (115) | **21.9 (70)** |
| 300×4 | 526.5 (6886) | 526.8 (6869) | 45.7 (1771) | **41.3 (1019)** |
| 900×4 | –² | – | 131.5 (6347) | **116.7 (3640)** |
| 3000×4 | – | – | OOM¹ | **615.6 (14763)** |
| 6000×4 | – | – | OOM | OOM |

1. Verified CUDA OOM under the cap: "Tried to allocate 3.90 GiB" for base at 3000; 3.74 GiB for oeq at 3000×4.
2. The e3nn models run only where N·W ≤ 1200, at about 1.8 GiB per 300 atoms. Beyond that, WSL2 spills to host RAM instead of raising OOM.

### Parity (fp32 model, so the reference is the fp32 noise floor)

`parity_native.py`: 5 jittered geometries per cell, compared against the baseline. `recomp` (the same e3nn model, recompiled) shows the floor. Values are max|dE| (kcal/mol) / max|dF| (kcal/mol/Å).

| cell | recomp | oeq |
|---|---|---|
| 30×1 | 1.3e-4 / 7.6e-5 | 9.4e-5 / 8.3e-5 |
| 30×4 | 2.1e-4 / 1.3e-4 | 1.5e-4 / 9.6e-5 |
| 300×1 | 3.5e-4 / 1.1e-4 | 4.1e-4 / 1.1e-4 |
| 300×4 | 4.4e-4 / 1.2e-4 | 4.7e-4 / 1.2e-4 |

`parity_fast.py` includes periodic boxes: the droplet extent + 3 Å, so there are cross-boundary edges, and the virial is non-zero. Values are dE / dF / dVirial.

| cell | fast vs oeq | oeq vs oeq (run-to-run floor) | oeq vs base |
|---|---|---|---|
| 30 open | 1.3e-4 / 6.4e-5 / 0 | 5.9e-5 / 1.0e-4 / 0 | 2.2e-5 / 5.8e-5 / 0 |
| 300 open | 6.7e-4 / 2.0e-4 / 0 | 3.7e-3 / 1.7e-4 / 0 | 8.0e-4 / 8.5e-5 / 0 |
| 300×4 open | 2.1e-4 / 2.6e-4 / 0 | 1.8e-4 / 9.9e-5 / 0 | 6.9e-4 / 1.1e-4 / 0 |
| 300 PBC (\|V\| 2566) | 7.2e-4 / 2.0e-4 / 1.8e-3 | 5.2e-4 / 7.7e-5 / 1.4e-3 | 5.0e-4 / 8.8e-5 / 7.0e-4 |
| 300×2 PBC | 4.4e-3 / 3.5e-4 / 7.4e-3 | 3.4e-4 / 8.8e-5 / 3.0e-3 | 3.9e-3 / 2.6e-4 / 4.8e-3 |
| 900 PBC (\|V\| 7395) | 2.2e-3 / 2.2e-4 / 4.2e-3 | 3.0e-5 / 1.1e-4 / 2.2e-3 | 6.6e-4 / 1.1e-4 / 1.4e-3 |

- OEQ sits at the floor.
- FastNequIP is about 2x the floor in dF, which comes from fp32 summation order.
- At 3000/6000 atoms fast vs oeq dE is 6e-3 / 1.6e-2 kcal/mol, about 3 µkcal/mol per atom. It is a coherent fp32 offset from the self-connection matmul; forces stay at 3.6e-4.

## Where the time goes (oeq)

- **3000 atoms (191k edges, 64 per atom, GPU-bound):**

  | part | share |
  |---|---|
  | `aten::mm` (edge MLP, fp32 SIMT sgemm) | 38% |
  | OEQ conv backward | 25% |
  | OEQ conv forward | 16% |
  | self-connection FCTP | ~8% |

  Memory is dominated by the per-edge TP weights: [E,1312] fp32 is 1 GiB per layer, kept for backward. The half-edge rewrite halves it.
- **30–300 atoms (host-bound):**
  - At 30 atoms: ~19 ms wall vs ~2.5 ms GPU.
  - About 3500 ATen ops/step (forward + backward): e3nn codegen Linear ~650, FCTP ~590 and Gate ~430 in the forward, plus autograd.
  - TorchScript knobs (profiling 0 / optimize 0 / texpr 0) all land in 21–25 ms, i.e. noise.
- **Wrapper:**
  - The dense neighbour list costs 0.2–2.3 ms from 30 to 6000 atoms. The cell-list version only wins at 6000 (2.0 vs 2.3 ms).
  - `edge_transpose_perm` costs 0.14–0.36 ms and is read by neither inner model.
  - Neither is worth changing the wrapper shared with Allegro.

## What was tried

| idea | result |
|---|---|
| recompile the e3nn model in torch 2.11 | 1.00x; parity = fp32 floor |
| OEQ (native lib, C++ autograd) | 2.2x at 30, 5.5x at 300, 11x at 900, 11.5x at 300×4; enables 3000 and 6000 atoms; loads in NAMD |
| FastNequIP half_edges + species_sc | a further 1.13x at ≥900×4/3000 and −42% memory (3000×4 fits); neutral at 30 |
| cuEquivariance (`enable_CuEquivariance`) | 0.24x of OEQ at 300, 0.11x at 3000; needs 2 more Python-only ops (`fused_tensor_product`, `segmented_transpose`). Dropped |
| TF32 (EXTRA) | −10% at 3000 but dE 0.94 / 8.1 / 76 kcal/mol at 30 / 300 / 3000 atoms and dF 0.17–0.29. Rejected; don't set NAMD_MLFF_TF32 for this model |
| TorchScript JIT knobs | no win |
| wrapper NL / transpose perm | <1.5%; not changed |

## Tests
`pytest tests/ -k "nequip or NequIP or allegro or Allegro"`: 22 passed, 4 skipped. `tests/test_interface_compliance.py`: 21 passed. `src/wrappers/wrap_compiled_nequip.py` is unchanged.

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
bash scripts/opt/nequip/build.sh      # native OEQ lib + inner compiles + FastNequIP + wraps + parity + mlff_shim_test (~3 min)
bash scripts/opt/nequip/run_grid.sh   # bench grid (~8 min)
flock scripts/opt/.gpu_bench.lock bash scripts/opt/nequip/namd_smoke/run_smoke.sh fast_oeq models/opt/nequip_fast_oeq.pt native 100 300
```
Files:
- `oeq_native/{oeq_native.cpp,build.sh,lib/}`
- `compile_ts.py`, `fast_nequip.py`, `build_fast.py`, `wrap.py`
- `parity_native.py`, `parity_fast.py`, `parity_tf32.py`
- `bench_nq.py` (bench_common + VRAM cap), `run_grid.sh`, `namd_smoke/run_smoke.sh`
- `prof.py`, `breakdown.py`, `opcount.py`
- Results: `scripts/opt/results/nequip_final_w*_W*.json`, `nequip_tf32extra_*.json`

## Follow-ups / user decisions
1. To sweep NequIP with OEQ in NAMD, repoint `namd_benchmarks/models/nequip_oam.pt` (user-owned symlink, not changed here) at `models/opt/nequip_fast_oeq.pt` and export the `NAMD_MLFF_EXTRA_LIBS` line above.
2. Below about 1000 atoms the model is host-bound. CUDA graphs (the SchNet graph shim, not installed) or AOTInductor (needs a change to the shim's model loader) are the remaining levers.
3. `liboeq_native.so` needs the allegro OEQ and nvidia/cu13 headers only at build time; at run time it needs only the libtorch zip. Rebuild it if the zip moves.
4. Do not load `liboeq_native.so` into a Python process that also imports openequivariance (duplicate `TORCH_LIBRARY(libtorch_tp_jit)`); `wrap.py` refuses that combination.
