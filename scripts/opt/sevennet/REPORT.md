# SevenNet inference optimisation (2026-09-30)

The model is unchanged: SevenNet-0 (`7net-0`, 11July2024), fp32 as shipped.
- 842,623 parameters, 89 elements, 5 Å cutoff.
- 5 interaction layers: channels 128, lmax 2, linear self-connection, e3nn `uvu` convolution, radial MLP [8→64→64→paths].
- The same tools were also checked on `7net-l3i5` (lmax 3) and the multi-fidelity `7net-mf-0`.

**Any SevenNet checkpoint, the usual two steps** (compile, then wrap):
```bash
python -m src.compile_sevennet --checkpoint 7net-0 --fast --out models/opt/my_sevennet_inner_fast.pt   # [--modal M]
python -m src.cli --model-type sevennet --compiled models/opt/my_sevennet_inner_fast.pt [--d3] --out mlff_model.pt
```
`--fast` falls back to the OEQ kernels alone, and says so, if a rewrite doesn't fit a model. `src.cli` loads the OEQ op library itself and prints the `NAMD_MLFF_EXTRA_LIBS` line for NAMD.

Every recommended artifact loads in NAMD's **default shim** (`namd_benchmarks/lib/libnamd_mlff.so`, the C++ libtorch 2.11 zip). **No Python runs inside NAMD.**

## TL;DR

| artifact (models/opt/) | what it is | 30 atoms | 300 atoms | 3000 atoms | namd3, 300 atoms |
|---|---|---|---|---|---|
| `sevennet_baseline.pt` | current wrapper + SevenNet's stock serial deployment `models/compiled_sevennet_0.pt` (BASELINE) | 27.7 ms | 46.1 | 511 | 40.5 ms/step |
| `sevennet_oeq.pt` | SevenNet's own OpenEquivariance deployment (`deploy(use_oeq=True)`), same weights | 14.3 (1.9x) | 14.4 (3.2x) | 45.2 (11.3x) | 16.2 |
| **`sevennet_fast.pt`** | **OEQ + FastSevenNet exact rewrites (`--fast`, `src/sevennet_fast.py`) — RECOMMENDED** | **4.9 (5.7x)** | **6.0 (7.7x)** | **41.3 (12.4x)** | **8.6 (4.7x)** |
| `sevennet_baseline_d3.pt` | baseline + D3(BJ, PBE) (`--d3`) | 30.9 | 47.6 | 525 | 46.2 |
| **`sevennet_fast_d3.pt`** | **FastSevenNet + D3** | **6.8** | **8.0** | **56.7** | **9.5 (4.9x)** |

The table gives median ms per call (bench_common, W = 1) and the namd3 wall time per step.

The OEQ and fast artifacts need the default shim plus the **same** library as the NequIP artifacts:
```bash
export NAMD_MLFF_EXTRA_LIBS=/home/rat/PycharmProjects/ML_models_NAMD/scripts/opt/nequip/oeq_native/liboeq_native.so
```
- Without it the shim refuses the model with an unknown-op error, `libtorch_tp_jit::jit_conv_forward`.
- SevenNet's OEQ deployment calls OEQ's `TensorProductConv` op, exactly as NequIP does.
- The native library (`scripts/opt/nequip/REPORT.md`, "OpenEquivariance, native") has the upstream C++ op schemas and kernels plus C++ autograd, with no pybind11 or libpython. It covers SevenNet unchanged.

## 1. OpenEquivariance (the kernel route)

SevenNet 0.13 supports OEQ natively: `sevenn/nn/oeq_helper.py` swaps each `IrrepsConvolution` for a scatter/gather-fused `TensorProductConv(torch_op=True, deterministic=False)`, and `deploy(use_oeq=True)` scripts and freezes it like the stock deployment.
- **Graph:** the frozen archive calls exactly 5 custom ops, one `libtorch_tp_jit.jit_conv_forward` per layer. Backward and double backward go through the native library's C++ autograd.
- **Build:** `python -m src.compile_sevennet --checkpoint 7net-0 --oeq --out models/opt/sevennet_inner_oeq.pt`. The new `--oeq` flag calls SevenNet's `deploy(..., use_oeq=True)`. The build needs a GPU and `openequivariance` (0.7.0, already in allegro).
- **Wrap:** `python -m src.cli --model-type sevennet --compiled <inner> --out ...`. `src.cli` sees the deployment's `oeq` key and loads the native library itself, or you can pass `--extra-libs` explicitly.
- **Scope:** the kernels-only route works for every SevenNet checkpoint, multi-fidelity ones included (`--modal`).

## 2. FastSevenNet (`src.compile_sevennet --fast`, `src/sevennet_fast.py`)

With OEQ the model is **host-bound** below ~1000 atoms. At 30 atoms an OEQ call takes 13.6 ms wall but only 2.6 ms of GPU time, spread over ~930 kernel launches.

In the eager model, forward only, the CPU time splits as follows (backward roughly doubles it):

| module | share of forward CPU |
|---|---|
| the three e3nn Linears per layer | 39% |
| convolution (radial MLP + OEQ) | 22% |
| gate | 20% |
| edge embedding | 12% |

FastSevenNet is graph surgery on the eager model inside SevenNet's own `deploy()`. A context manager (`rewriting_deploy`) wraps the one `e3nn.util.jit.script` call that `deploy()` makes, so the rewrites run after SevenNet has removed the force module and fixed the modality, and before it scripts and freezes the model. So the output is an ordinary SevenNet deployment with the same `_extra_files`, and the wrapper loads it unchanged.

The weights are unchanged. Each rewrite is checked in fp64 against what it replaces before scripting. A model whose structure a rewrite doesn't know raises `RewriteNotApplicable`, and `compile_sevennet --fast` then falls back to the OEQ kernels alone:

| rewrite | what it does | 30-atom call (cumulative) |
|---|---|---|
| (OEQ only) | | 14.1 ms |
| `linear` | every e3nn `Linear` (18) → one dense matmul with the equivalent block matrix, probed from the module in fp64 (`kron(W_l, I_{2l+1})` blocks). A multi-fidelity linear appends the modality one-hot to its input. The modality is fixed at deploy time, so those rows become a constant bias (`addmm`) | 8.8 |
| `gate` | e3nn `Gate` → `split`, act(scalars), gated × act(gates)`.index_select`; normalize2mom and path constants folded into one scale | 6.7 |
| `radial` | the 5 radial MLPs run once, fused: one matmul for the shared first layer, one `bmm` for the hidden layer, one matmul per conv. act constants and the conv's 1/denominator are folded into the weights (the conv is linear in them). The convs call the OEQ op with int64 indices directly, with no int32 round trip | 5.2 |
| `fuse` | per layer, the self-connection and self-interaction-1 linears read the same features, so they become one matmul + `split`. Each gate's output constants are folded into the rows of the matrix that consumes it | **4.9** |

Tried and dropped: spherical harmonics as one monomial matmul (exact, but no gain).

Coverage:
- `7net-0`: works (the benchmark artifact).
- `7net-l3i5` (lmax 3): builds and sits at the floor.

  | system | fast vs stock dF (kcal/mol/Å) | floor dF |
  |---|---|---|
  | H/C/N/O/S/Cl molecule | 3.0e-5 | 3.0e-5 |
  | 300-atom water | 7.1e-5 | 1.2e-4 |
  | 300-atom water, PBC | 1.0e-4 (dV 7e-4 at \|V\| 2284) | 1.5e-4 |

  Speed (`results/sevennet_l3i5_*.json`, W = 1):

  | atoms | stock | fast |
  |---|---|---|
  | 30 | 42.6 ms | 4.5 ms (9.5x) |
  | 300 | 123 ms | 8.6 ms (14.3x) |
  | 3000 | OOM | 79 ms |

- Multi-fidelity checkpoints (`7net-mf-0`, `7net-mf-ompa`, `7net-omni`, …): supported. Deploy with `--modal`; the modality becomes a bias.
  - `7net-mf-0` at its second fidelity (`PBE`, modal index 1) matches its stock deployment (`test_fast_multifidelity_matches_stock`).
  - Indicative speed (`results/sevennet_mf0_*.json`, measured while another job loaded the CPU, so ratios only): 4.9x at 30 atoms, 8.8x at 300.

## Parity

Setup:
- `parity.py`: a fresh process with the native library only, asserting that `openequivariance` is never imported.
- 3 jittered geometries per cell. Periodic boxes are the droplet extent + 3 Å, so there are cross-boundary edges and a non-zero virial.
- `*_d3` artifacts are compared with `baseline_d3`.
- Floor: the baseline run twice (fp32 atomic scatter).
- Energies are summed in fp64 by the wrapper (see "Energy summation").
- Values are max |dE| kcal/mol / max |dF| kcal/mol/Å / max |dVirial| kcal/mol.

| cell | floor | oeq | **fast** | fast_d3 |
|---|---|---|---|---|
| 30 open | 2.2e-5 / 1.9e-5 / 0 | 1.1e-5 / 2.0e-5 / 0 | 3.3e-5 / 2.5e-5 / 0 | 5.5e-5 / 4.8e-5 / 0 |
| 300 open | 6.6e-5 / 2.5e-5 / 0 | 1.2e-4 / 2.4e-5 / 0 | 1.1e-3 / 8.5e-5 / 0 | 9.9e-4 / 6.8e-5 / 0 |
| 300×4 open | 1.4e-4 / 3.6e-5 / 0 | 1.4e-4 / 4.3e-5 / 0 | 8.1e-4 / 1.1e-4 / 0 | 8.3e-4 / 1.0e-4 / 0 |
| 3000 open | 2.1e-4 / 4.1e-5 / 0 | 4.3e-4 / 4.4e-5 / 0 | 1.3e-2 / 1.4e-4 / 0 | 1.3e-2 / 1.3e-4 / 0 |
| 300 PBC (\|V\| 2170) | 5.5e-5 / 3.5e-5 / 1.5e-4 | 6.6e-5 / 3.5e-5 / 1.4e-4 | 1.1e-3 / 7.6e-5 / 5.6e-4 | 1.2e-3 / 7.7e-5 / 6.3e-4 |
| 300×2 PBC | 4.4e-5 / 3.2e-5 / 1.6e-4 | 8.8e-5 / 3.1e-5 / 1.9e-4 | 2.8e-4 / 7.3e-5 / 5.9e-4 | 3.0e-4 / 7.2e-5 / 5.5e-4 |
| 900 PBC (\|V\| 7222) | 8.8e-5 / 3.5e-5 / 1.9e-4 | 9.9e-5 / 3.8e-5 / 2.1e-4 | 3.4e-3 / 1.1e-4 / 1.4e-3 | 3.4e-3 / 9.6e-5 / 1.5e-3 |

- OEQ sits at the floor in energy, forces and virial.
- FastSevenNet forces are 2–3x the floor, about 1e-6 of \|F\|max, from fp32 summation order in the dense matmuls. FastNequIP showed the same.
- The FastSevenNet energy differs by up to 4e-8 of the total energy (1.3e-2 of -3.3e5 kcal/mol at 3000 atoms). That is below one fp32 ulp per atom, and coherent across atoms: the folded weights (normalisation, gate and 1/denominator constants multiplied in fp64, then rounded once to fp32) round differently from SevenNet's runtime fp32 products. Forces and dynamics are unaffected.

Other checks:
- Inputs with no edges give the same energy (to 1e-4 kcal/mol) and zero forces in base, oeq and fast: one atom, two atoms 20 Å apart, and a batch of a water plus an isolated O (`FastConv` keeps SevenNet's zero-edge branch).
- `mlff_shim_test` on the default shim printed ALL CHECKS PASSED for base, oeq, fast and fast_d3. Single-structure E: -324.816247 (base), -324.816241 (oeq), -324.816236 (fast) kcal/mol.
- `tests/test_sevennet.py::test_optimised_deployment_matches_stock` builds the OEQ and fast deployments from `7net-0` in the test. It compares them with the stock model on the H/C/N/O/S/Cl molecule and a triclinic water box (energy, forces, virial). `test_fast_multifidelity_matches_stock` does the same for `7net-mf-0` at modal index 1. Both run only with CUDA and openequivariance.

## Energy summation (fp64)

The wrapper now sums SevenNet's `atomic_energy` in fp64, in both `forward` and `forward_batch`. Before, `forward` used SevenNet's `inferred_total_energy`, a fp32 sum, and `forward_batch` used an fp32 `index_add`.
- **Interface:** outputs are unchanged (float64 kcal/mol). Forces are unchanged: the gradient of a sum doesn't depend on the precision it is accumulated in.
- **Effect:** the run-to-run energy noise of the same model (the floor column) dropped from up to 2.3e-2 to 2.2e-4 kcal/mol.
- **Cost:** one cast and one sum.
- **Existing files:** `.pt` files wrapped before this change (e.g. `models/sevennet_0_mlff.pt`, `models/sevennet_0_d3_mlff.pt`) keep the old fp32 sum until they are re-wrapped with `src.cli`.

## Benchmarks

Setup: `run_grid.sh` (bench_common through `nequip/bench_nq.py`, 15 GiB VRAM cap, expandable_segments, jitter 0.02 Å, one (N, W) per invocation, OEQ via the native library only). All six artifacts are interleaved in each invocation. Results are in `results/sevennet_final_w<N>_W<W>.json`.

These timings predate the fp64 energy summation, which adds one cast and one sum per call. They were not re-timed, because another job was loading the CPU when the change was made.

Median ms per call (peak alloc MiB):

| atoms × W | base | oeq | **fast** | base_d3 | oeq_d3 | fast_d3 |
|---|---|---|---|---|---|---|
| 30×1 | 27.7 (63) | 14.3 (10) | **4.9 (10)** | 30.9 | 16.6 | 6.8 |
| 30×4 | 29.7 (197) | 14.4 (38) | **5.2 (39)** | 39.0 | 22.9 | 13.6 |
| 300×1 | 46.1 (885) | 14.4 (174) | **6.0 (181)** | 47.6 | 17.1 | 8.0 |
| 300×4 | 149.2 (3486) | 20.2 (691) | **13.6 (722)** | 157.7 | 29.0 | 21.9 |
| 900×1 | 133.0 (3071) | 18.3 (606) | **12.0 (631)** | 135.3 | 20.7 | 14.3 |
| 900×4 | –¹ | 45.8 (2433) | **41.7 (2524)** | – | 54.9 | 50.5 |
| 3000×1 | 510.8 (12278) | 45.2 (2418) | **41.3 (2517)** | 524.5 | 59.9 | 56.7 |
| 3000×4 | – | 169.6 (9679) | **163.1 (10061)** | – | 229.3 | 221.0 |
| 6000×1 | – | 91.3 (5114) | **87.0 (5303)** | – | 148.9 | 143.5 |
| 6000×4 | – | OOM² | OOM | – | OOM | OOM |

1. The e3nn baseline needs about 4 GiB per 1000 atoms and is run only where N·W ≤ 3000.
2. Verified CUDA OOM under the 15 GiB cap ("Tried to allocate 236 MiB" with 14.75 GiB allocated).

Where the time goes after the rewrites:
- **30 atoms:** 4.9 ms wall against 1.4 ms of GPU time. The rest is the TorchScript interpreter and autograd dispatch. The wrapper (neighbour list, strain, output casts) is ~0.3 ms of it.
- **300 atoms:** GPU-bound. `aten::mm` takes 51% of GPU time (dense linears + radial MLP) and the OEQ conv 34%.
- **3000 atoms:**

  | part | share of GPU time |
  |---|---|
  | radial MLP last layer ([E,64]×[64,960] per layer, and its backward) + dense linears | ~40% |
  | OEQ conv backward | 26% |
  | OEQ conv forward | 18% |

  The dense linears do ~6x the FLOPs of the block form, but fewer launches. The result is neutral at 3000 atoms and costs ~3% at 6000, which the fused radial MLP more than pays back.
- **D3:** about 2 ms at 30–900 atoms and 15 ms at 3000. At 6000 it costs 57 ms, 40% of fast_d3, and becomes the bottleneck. At 30×4 the batched D3 costs 8.5 ms, more than the model. `src/d3.py` was not changed here.

## What was tried

| idea | result |
|---|---|
| OEQ (SevenNet's own `use_oeq` deploy + the native lib) | 1.9x at 30 atoms, 3.2x at 300, 7.3x at 900, 11.3x at 3000. Memory 12.3 → 2.4 GB at 3000. Enables 3000×4 and 6000 atoms |
| FastSevenNet rewrites (see table above) | a further 2.9x at 30, 2.4x at 300, 1.5x at 900, 1.1x at 3000; exact up to fp32 order |
| spherical harmonics as one monomial matmul | exact, no gain; dropped |
| cuEquivariance | not tried: `deploy()` hard-codes `enable_cueq=False`, and cuEq's ops are Python-only (see the NequIP report) |
| FlashTP | `flashTP_e3nn` isn't installed, and it would need its own native library |
| TF32 | not tried: NequIP measured dE of 1–76 kcal/mol with it |
| wrapper changes | the wrapper is ~0.3 ms per call. The energy is now summed in fp64 (a precision change, see above). `read_sevennet_metadata` reads `_extra_files` from the zip, so the model isn't loaded twice and custom ops aren't needed to read it. An OEQ deployment without its op library fails with a message that names the library, and `src.cli` loads the library itself |

## Tests
`pytest tests/test_interface_compliance.py tests/test_sevennet.py tests/test_d3.py`: 50 passed, including the optimised-deployment tests (4 cases for 7net-0, 2 for 7net-mf-0).

## Reproduce
```bash
cd /home/rat/PycharmProjects/ML_models_NAMD
bash scripts/opt/sevennet/build.sh      # OEQ + fast deploys, wraps (+D3), parity, mlff_shim_test (~1.5 min)
bash scripts/opt/sevennet/run_grid.sh   # bench grid (~4.5 min)
flock scripts/opt/.gpu_bench.lock bash scripts/opt/sevennet/namd_smoke/run_smoke.sh fast models/opt/sevennet_fast.pt native 100 300
```
Files:
- the rewrites: `src/sevennet_fast.py`; the entry point: `src/compile_sevennet.py --fast`
- here: `build.sh`, `parity.py`, `prof.py`, `run_grid.sh`, `namd_smoke/run_smoke.sh`
- results in `scripts/opt/results/sevennet_final_w*_W*.json`, `sevennet_l3i5_w*_W1.json` and `sevennet_mf0_w*_W1.json`

Your own checkpoint: the two commands at the top of this report.

## Follow-ups / user decisions
1. To use it in NAMD, point `QMExecPath` at `models/opt/sevennet_fast.pt` (or `_d3`) and export `NAMD_MLFF_EXTRA_LIBS` as above. The untracked `models/sevennet_0_*mlff.pt` files are the reference builds and were left alone.
2. Done (2026-09-30, the user's decision): energies are summed in fp64; see "Energy summation". Re-wrap any older SevenNet `.pt` files to pick it up.
3. D3 dominates above ~3000 atoms and in batched small systems. The next lever for SevenNet+D3 is `src/d3.py`: a cell-list pair search and fewer ops in the batched path.
4. At large N, a half-edge rewrite would halve the radial-MLP work and weight memory, as FastNequIP does (+13% there). That means the OEQ conv called twice on i<j pairs with sign-flipped odd-l harmonics. It was not done: its gain is limited to ≥ 3000 atoms.
5. Do not load `liboeq_native.so` into a Python process that also imports `openequivariance`, whether directly or through SevenNet model building (`sevenn.model_build`, used by `deploy` and the ASE calculator). That registers the same op namespace twice. `import sevenn`, `src.cli` and `--d3` are safe.
