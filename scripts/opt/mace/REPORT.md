# MACE-OFF23 (medium) inference optimisation — FINAL (2026-09-23)

(Interim report preserved as REPORT_interim2.md.)

The model is unchanged (same weights and architecture, fp64 as shipped). Every artifact below loads in NAMD's **default shim** (`namd_benchmarks/lib/libnamd_mlff.so`, the C++ libtorch 2.11 zip). **No Python runs inside NAMD.**

## TL;DR

| artifact (models/opt/) | what it is | NAMD requirements | 300 atoms | 3000 atoms |
|---|---|---|---|---|
| `mace_baseline.pt` | current wrapper + deployed compiled inner | none | 301 ms | does not fit in 16 GB |
| `mace_fast_e3nn.pt` | exact FastMACE rewrites, stock e3nn kernels | none (fallback) | 296 ms (1.02x) | does not fit |
| `mace_cueqf.pt` | cuEquivariance + fused conv TP | native op libs (below) | 29.0 ms (10.4x) | 267 ms |
| **`mace_fast_cueqf.pt`** | **cuEq + FastMACE (recommended)** | **native op libs** | **20.8 ms (14.4x)** | **184 ms** |
| `mace_fast_cueqf_f32.pt` | EXTRA: same in fp32 (changes numerics) | native op libs | 11.3 ms (26x) | 33 ms |

The runtime env for the cuEq artifacts is the default shim plus three libraries:
```bash
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:/home/rat/PycharmProjects/ML_models_NAMD/scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so
```
- `libnvrtc.so.13` has to come first. `libcue_ops.so` needs that soname, and the libtorch zip only ships a hash-named nvrtc. LD_LIBRARY_PATH stays untouched.
- `libcueq_uniform1d_native.so` is new (Route B). It links only libtorch + libcue_ops. It has no Python and no libtorch_python.

In a namd3 run (300 atoms, walk0, seeded Langevin, 300 steps) the recommended artifact takes **19.5 ms/step vs 293 ms/step for the baseline (15x)**. Its QMENERGY matches the baseline to all 4 printed decimals at every step (0–300).

## Route B: native `cuequivariance::uniform_1d` (done, and the only way cuEq is allowed in NAMD)

Upstream, `cuequivariance::uniform_1d` is a Python `torch.library.custom_op`. Its compiled `_ext` .so registers nothing and needs libtorch_python, so a cuEq TorchScript artifact can't load in a pure-libtorch process. `scripts/opt/mace/cueq_native/uniform1d_op.cpp` ports that op to C++, line by line:
- The schema is copied verbatim from `torch.ops.cuequivariance.uniform_1d.default._schema` (cuequivariance-ops-torch 0.11.1).
- **CUDA kernel.** This is the custom_op body: `_handle_batch_dim_auto`, allocation of the outputs (SHARED/INDEXED outputs zeroed), and the flat→nested argument mapping of the pybind layer. That mapping was recorded with the LD_PRELOAD spy (`spy.cpp` → `spy_mapping.log`): index buffers get `kBatched` and index_cfg −1, `batch_sizes=[batch_size]`, `buffer_bytes=nbytes`, `zero_out=false`, `ignore_first=true`. The work itself is one call to `kernelcatcher::equivariance::uniform_1d::run_uniform_1d_cuda` on the current torch stream.
- **Autograd kernel.** A `torch::autograd::Function` port of `_do_bwd_jit`: the backward is the same op, renamed `*_bwd` with rewired operations, and it is dispatched through the op again, so double backward works too. `needs_input_grad(i)` indexes tensor inputs only; the backward returns 20 undefined grads for the non-tensor args plus one per tensor.
- Build: `bash scripts/opt/mace/cueq_native/build.sh` (about 11 s, against `/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130`).

Acceptance:
1. **Parity in a fresh process that never imports cuequivariance python** (asserted: `sys.modules` has no cuequivariance*). `cueq_native/parity_native.py` compared baseline, cueqf and fast_cueqf at N=30/300 × W=1/4: max dE 9.3e-10 kcal/mol, max dF 1.9e-13 kcal/mol/Å — ALL OK.
2. **`mlff_shim_test` on the default shim** with only the native EXTRA_LIBS: `mace_fast_cueqf.pt` and `mace_cueqf.pt` print ALL CHECKS PASSED. E = -47995.022275 kcal/mol, identical to the baseline; batch energies are identical too. RSS is 2.33 GB (fast_cueqf) / 2.18 GB (cueqf) vs 1.72 GB (baseline).
3. **namd3 smoke** (`scripts/opt/mace/namd_smoke/run_smoke.sh`, private run dirs under `scripts/opt/mace/namd_smoke/runs/`; `namd_benchmarks/runs/` was not touched):

| cell | shim / libs | ms/step (wall, steps 250–300) | GPU peak (nvidia-smi) | QMENERGY vs base |
|---|---|---|---|---|
| 300 atoms, base | default, none | 293 | 2541 MiB | — |
| 300 atoms, fast_cueqf | default + native | **19.5** | 998 MiB | identical, steps 0–300 |
| 300 atoms, fast_cueqf_f32 (EXTRA) | default + native | 12.0 | 877 MiB | +0.05 kcal/mol offset |
| 3000 atoms, fast_cueqf (100 steps) | default + native | 184 | 5233 MiB | (base does not fit) |
| 300 atoms, fast_cueqf via pyinit (**REJECTED**) | pip-torch shim + libpython | 20.7 | 1030 MiB | identical |

No crashes and no deadlocks. The native path is also faster than pyinit because it has no Python round trip per op.

A compiled-handle cache (`compile_uniform_1d_cuda` + `execute_uniform_1d_cuda`, keyed on the full problem) was tried. In three alternating A/B pairs at 30 atoms there was no measurable gain (9.0–10.0 ms both ways), so it was reverted: `run_uniform_1d_cuda`'s own lookup is cheap.

### Rejected: pyinit (Python embedded in NAMD)
`scripts/opt/mace/cueq_pyinit/libmace_cueq_pyinit.so` (pip-torch shim + libpython + an in-process `import cuequivariance_torch`) worked, but it is **rejected by user requirement: no Python inside NAMD**. It is kept only as a record. None of the numbers in this report depend on it; every bench number uses the native op.

## Benchmarks (bench_common, jitter 0.02 Å, one (system, walkers) per invocation, cuEq via native op)

Median ms (peak alloc MiB). Parity is against the first model in each invocation.

**W = 1**

| atoms | base | fast_e3nn | cueqf | **fast_cueqf** | fast_cueqf_f32 EXTRA¹ |
|---|---|---|---|---|---|
| 30 | 37.5 (150) | 38.3 (94) | 11.5 (28) | **11.6 (15)** | 11.1 (8) |
| 300 | 300.7 (1506) | 295.6 (1347) | 29.0 (377) | **20.8 (239)** | 11.3 (119) |
| 900 | 922 (4997) | 890 (4550) | 77.4 (1257) | **53.7 (824)** | 13.6 (412) |
| 3000 | –² | –² | 267 (4789) | **184 (3280)** | 33.3 (1641) |
| 6000 | –² | –² | 550 (9937) | **403 (6904)** | 72.0 (3453) |

**W = 4 (forward_batch)**

| atoms/walker | base | fast_e3nn | cueqf | **fast_cueqf** | fast_cueqf_f32 EXTRA¹ |
|---|---|---|---|---|---|
| 30 | 114.0 (525) | 114.0 (350) | 12.8 (107) | **10.8 (54)** | 10.4 (27) |
| 300 | 1212 (5933) | 1194 (5358) | 92.6 (1502) | **62.3 (946)** | 14.3 (473) |
| 900 | –² | –² | 284 (5032) | **194 (3294)** | 33.7 (1648) |
| 3000 | –² | –² | OOM (>15 GiB) | **741 (13113)³** | 135 (6551) |
| 6000 | –² | –² | OOM | OOM | 277 (13818) |

1. The fp32 rows up to 900 atoms (W=1) and 300×4 come from `results/mace_f32parity_w*_W*.json`, i.e. the final fp32 build with fp64 energy accumulation. The fp32 rows in `mace_final_w{30,300,900}_W1.json` / `w30_W4` / `w300_W4` are from an earlier fp32 build (fp32 energy summation: dE 13–38 kcal/mol) and are superseded. The larger cells used the final build.
2. The e3nn models need about 5 GiB per 900 atoms. Beyond about 1200 total atoms they exceed the 16 GB card. On WSL2 the driver then spills silently into shared host RAM instead of raising OOM (a 900×4 base run was crawling after 20 min and was killed). The grid therefore runs them only where N·W ≤ 1200. Large cells use `bench_capped.py` (hard 15 GiB cap → clean OOM row).
3. Needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`results/mace_final_w3000_W4_expseg.json`). Without it, the first call fits (12.2 GiB) but later calls OOM because 5.7 GiB is reserved but unallocated (fragmentation). **Recommend setting it for NAMD runs with large MACE systems.** It is already an optional line in `namd_benchmarks/env.sh`.

**Parity (fp64 artifacts):** fast_e3nn / cueqf / fast_cueqf vs baseline give max dF ≤ 2.6e-13 kcal/mol/Å and max dE ≤ 1.9e-9 kcal/mol (fp64 rounding). At 6000 atoms fast_cueqf vs cueqf is dF 1.6e-13, dE 8.9e-8 kcal/mol (relative 1e-15).

**fp32 EXTRA parity** (vs fp64 baseline, or vs fp64 cueqf where the baseline does not fit):

| atoms | 30 | 300 | 300×4 | 900 | 3000 | 6000 |
|---|---|---|---|---|---|---|
| max dE (kcal/mol) | 5.5e-3 | 5.4e-2 | 5.5e-2 | 0.16 | 0.54 | 1.09 |
| max dF (kcal/mol/Å) | 4.8e-4 | 1.2e-3 | 1.7e-3 | 2.4e-3 | 3.8e-3 | 5.7e-3 |

The dE is essentially an offset from fp32 rounding of the per-species E0 table and of the per-atom energies. With fp32 summation it was 13 kcal/mol at 300 atoms; `fast_mace.py` now accumulates E0 and the interaction energy in fp64, which is a no-op for fp64 builds. Speed: 1.8x over fp64 fast at 300 atoms, 4x at 900, 5.5x at 3000 (the laptop RTX 5080 runs fp64 at about 1/64 of fp32).

## Where the time goes

- **e3nn baseline, fp64:** the products.0 symmetric contraction takes about 100 of the 300 ms at 300 atoms. It is an opt_einsum over all 10 elements via the one-hot, with intermediates of [atoms, 16³, 128]. That is also why memory scales at about 5 GiB per 900 atoms. FastMACE on e3nn only saves 10–35% of memory, because it doesn't touch that contraction.
- **cuEq, fp64, 3000 atoms, ~184 ms/step after FastMACE:**

  | part | share |
  |---|---|
  | fp64 GEMMs (radial MLP on half edges + linears) | ~70 ms |
  | uniform_1d fwd+bwd (4 fwd + 4 bwd calls) | ~50 ms |
  | block linears | ~33 ms |

  The ablation at 3000 atoms: fast 194 ms; without half_radial 244; without species_skip 234.
- **30 atoms (all cuEq variants, ~9.5–11.5 ms):** host-bound. GPU time is about 2.4–2.8 ms per call. The rest is the TorchScript interpreter plus about 430 kernel launches and 7 small syncs per call. The only real lever left here is a CUDA graph (see follow-ups).

## What was tried

| idea | result |
|---|---|
| cuEq + fused conv TP (Route B native op) | 10–12x at ≥300 atoms, exact |
| FastMACE half_radial (radial MLP once per undirected pair, sync-free) | exact, −20% at 3000 atoms |
| FastMACE species_skip (one-hot FCTP skip → per-species block matmul) | exact, −17% at 3000 atoms |
| FastMACE plain_linear (cuEq Linear → matmul) | slower at large N; off (the old on-disk fast artifact had it on, now rebuilt) |
| fp64 energy accumulation | exact for fp64; fixes fp32 energy offset from 13 → 0.05 kcal/mol at 300 atoms |
| uniform_1d handle cache | no measurable gain, reverted |
| pyinit (embedded Python) | works, 20.7 ms/step in namd3, REJECTED (no Python in NAMD) |
| JIT knobs / TF32 | infra: no win / changes numerics |
| wrapper flags | not needed; `src/wrappers/wrap_compiled_mace.py` unchanged by this agent |

Two fixes carried over from earlier work: the U-basis rebase (`rebuild.rebase_U`; the naive cuEq conversion is off by 0.015 eV) and the TorchScript fixes in `cueq_fusion.py` (the scripted stock cuEq path is about 60 eV off). The fp32 rebuild now constructs and rebases in fp64, then casts. Constructing in fp32 had made the U-span check fail at 2.5e-8.

## Tests
`pytest tests/ -k "mace or MACE"`: 25 passed, 2 skipped. `tests/test_interface_compliance.py`: 21 passed.

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
bash scripts/opt/mace/build.sh              # native op lib + all inners + wraps + parity + mlff_shim_test (~10 min)
bash scripts/opt/mace/run_grid.sh           # bench grid (N*W<=1200 includes e3nn models), capped at 15 GiB VRAM
flock scripts/opt/.gpu_bench.lock bash scripts/opt/mace/namd_smoke/run_smoke.sh fast models/opt/mace_fast_cueqf.pt native 100 300
```
Files:
- `cueq_native/{uniform1d_op.cpp,build.sh,parity_native.py,spy.cpp,spy_mapping.log}`
- `{rebuild,cueq_fusion,fast_mace,build_fast,compile_inner,wrap}.py`
- `bench_capped.py`, `run_grid.sh`, `namd_smoke/run_smoke.sh`, `prof.py` (`--native`), `breakdown.py`
- Results: `scripts/opt/results/mace_final_w{N}_W{W}.json`, `mace_final_w3000_W4_expseg.json`, `mace_f32parity_w*_W*.json`

## Follow-ups / user decisions
1. Run cuEq MACE in NAMD with the `NAMD_MLFF_EXTRA_LIBS` line above. `run_benchmark.sh` hard-codes `models/mace_off23.pt`; to sweep the fast artifact, copy it in or add a model entry. That file belongs to the user and is not edited here.
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for large MACE systems (≥3000 atoms × 4 walkers).
3. At small N the model is host-bound. A CUDA-graph step would need the schnet-nl graph shim (not installed; user decision) plus a sync-free capture path in FastMACE: static species counts, fixed-capacity edges. Whether cuEq kernels are capturable has not been tested.
4. `libcue_ops.so` still comes from the allegro pip wheel (cuequivariance-ops-torch-cu13 0.11.1). NAMD needs that path to stay. Copy the .so next to the op lib if the env might change.
5. Do not load `libcueq_uniform1d_native.so` into a Python process that also imports cuequivariance_torch (duplicate op registration).
