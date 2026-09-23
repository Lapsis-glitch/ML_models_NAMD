# Cross-model inference comparison: optimised vs baseline (2026-09-23)

Hardware: RTX 5080 Laptop (sm_120, 16 GB), WSL2. Models: SchNet, MACE-OFF23-medium, NequIP-OAM-L, ANI-2x and FeNNiX-BIO1. X-MACE was skipped (the user: "basically MACE"; the MACE route applies).

Every optimised artifact loads in NAMD **without Python**:
- the TorchScript models through the installed default shim (`namd_benchmarks/lib/libnamd_mlff.so`, libtorch 2.11 zip), with native op libs where needed;
- FeNNiX through NAMD's C++ PJRT backend.

Per-model details, parity and how each was built: `scripts/opt/<model>/REPORT.md`.

## Tests
`pytest tests/` (allegro env, full suite, 2026-09-23): **121 passed, 16 skipped, 2 failed**. Both failures are the known pre-existing ones:
- `TestE2E_SchNetPack::test_train_and_wrap`
- `TestE2E_TorchANI::test_train_and_wrap`

In both, the e2e test calls `forward()` without the `cell` argument, which the PBC wrapper signature now requires. There are no new failures.

## How it was measured
- **TorchScript models** (`compare/run_compare.sh` → `results/compare_w<N>_W<W>.json`):
  - `bench_common.py` with a 15 GiB VRAM cap (`mace/bench_capped.py`), `expandable_segments`, jitter 0.02 Å.
  - One (N, W) per invocation, with all artifacts interleaved in that one process so ratios are same-clock.
  - All native op libs are loaded together; they were verified to coexist, with no cuEq/OEQ/torchani Python imported.
  - e3nn baselines (MACE, NequIP) are included only where N·W ≤ 1200. Beyond that they exceed 16 GB, and WSL2 spills to host RAM instead of raising OOM.
- **FeNNiX** (`compare/run_fennix.sh` → `results/compare_fennix_w<N>.json`):
  - The C++ `bench_pjrt` (the same PJRT runtime code NAMD uses), 100 iterations, jitter 0.02 Å.
  - **This is a different harness.** It times the PJRT execute call only, with no TorchScript/autograd wrapper, and its memory figure is the compiled executable's device footprint, not the peak allocation. Compare it with the others with care.
- **Units:** median ms per call (all W walkers in one call); peak allocated MiB in parentheses.

## Cross-model: optimised artifacts only

**W = 1** (median ms per call)

| atoms | ANI-2x `ani_fast` | FeNNiX fast (TF32, shipped) | FeNNiX fp32 (extra) | SchNet `schnet_fast` | NequIP `nequip_fast_oeq` | MACE `mace_fast_cueqf` (fp64) | MACE fp32 (extra) |
|---|---|---|---|---|---|---|---|
| 30 | **1.63** | 1.93 | 4.06 | 2.86 | 21.5 | 9.99 | 10.5 |
| 300 | **1.53** | 2.07 | 4.05 | 3.09 | 22.4 | 21.2 | 10.5 |
| 900 | **1.95** | 2.19 | 4.38 | 3.84 | 38.0 | 52.2 | 12.2 |
| 3000 | 3.43 | **3.18** | 4.97 | 8.80 | 117 | 181 | 34.8 |
| 6000 | 6.17 | **5.81** | 8.73 | 16.2 | 254 | 395 | 73.0 |

**W = 4** (ms per call covering all 4 walkers)

| atoms/walker | ANI-2x | FeNNiX w4 vmap (extra; not usable in NAMD) | SchNet | NequIP | MACE fp64 | MACE fp32 (extra) |
|---|---|---|---|---|---|---|
| 30 | **1.83** | 2.07 | 2.97 | 20.3 | 10.2 | 10.1 |
| 300 | 2.50 | **2.09** | 4.02 | 40.4 | 62.7 | 14.3 |
| 900 | 4.31 | **3.26** | 8.27 | 118 | 195 | 38.4 |
| 3000 | 12.9 | **10.1** | 28.8 | 631 | 728¹ | 136 |
| 6000 | **25.0** | 26.4 | 59.3 | OOM | OOM | 813² |

**Memory at 3000 atoms, W = 1:**

| model | MiB |
|---|---|
| ANI | 105 |
| FeNNiX | 164 (executable footprint) |
| SchNet | 746 |
| MACE fp32 | 1640 |
| MACE fp64 | 3278 |
| NequIP | 3705 |

**Precision:**

| model | precision |
|---|---|
| MACE | **fp64** (as shipped) |
| NequIP | fp32, TF32 off |
| ANI | fp32 networks, energy summed in fp64 |
| SchNet | fp32 |
| FeNNiX | shipped artifacts run matmuls in **TF32** (dF ≈ 0.1 kcal/mol/Å vs fp64); the fp32-HIGHEST extra is 200–400× more accurate in forces and 1.4–2× slower |

For a precision-matched comparison, use the FeNNiX fp32 column. MACE is the only fp64 model: on this laptop GPU fp64 runs at about 1/64 of the fp32 rate, and the fp32 extra shows what that costs.

Notes:
1. `mace_fast` at 3000×4 is taken from a run alone (`compare_alone_mace_fast_w3000_W4.json`). Interleaved it measured 868 ms (p90 1067), because allocator pressure from the neighbouring models distorts it at 13 GB peak.
2. `mace_fast_f32` at 6000×4 measured 802 ms interleaved and 813 ms alone, at a 13.8 GB peak. That is near the card's limit and probably a WSL2 shared-memory spill. The MACE REPORT's 277 ms did **not** reproduce, so treat this cell as unreliable.

## Per model: baseline vs optimised (this run)

### SchNet: `schnet_baseline.pt` vs `schnet_fast.pt`

| N×W | base | fast | speedup |
|---|---|---|---|
| 30×1 | 3.35 (21) | 2.86 (4) | 1.17× |
| 300×1 | 3.59 (95) | 3.09 (77) | 1.16× |
| 900×1 | 4.36 (291) | 3.84 (273) | 1.13× |
| 3000×1 | 11.4 (1118) | 8.80 (746) | 1.29× |
| 6000×1 | 24.3 (2345) | 16.2 (1577) | 1.50× |
| 30×4 | 3.45 (33) | 2.97 (16) | 1.16× |
| 300×4 | 4.76 (327) | 4.02 (309) | 1.18× |
| 900×4 | 12.3 (1111) | 8.27 (742) | 1.49× |
| 3000×4 | 54.6 (4422) | 28.8 (2985) | 1.89× |
| 6000×4 | 421.5 (15400)³ | 59.3 (6328) | 7.1× |

3. The base value at 6000×4 comes from the SchNet REPORT (model run alone; 15.4 GB peak, so it cannot share a process).

The SchNet CUDA-graph path (patched shim, not installed) gives 0.50 / 1.20 / 2.06 ms at 30 / 300 / 900 atoms through the C ABI (SchNet REPORT).

### MACE-OFF23: `mace_baseline.pt` vs `mace_fast_cueqf.pt` (fp64) and `mace_fast_cueqf_f32.pt` (extra)

| N×W | base | fast (fp64) | speedup | fp32 extra |
|---|---|---|---|---|
| 30×1 | 35.9 (133) | 9.99 (14) | 3.6× | 10.5 (7) |
| 300×1 | 292 (1484) | 21.2 (237) | 13.8× | 10.5 (119) |
| 900×1 | 897 (4975) | 52.2 (824) | 17.2× | 12.2 (412) |
| 3000×1 | does not fit | 181 (3278) | – | 34.8 (1640) |
| 6000×1 | does not fit | 395 (6904) | – | 73.0 (3454) |
| 30×4 | 110 (507) | 10.2 (53) | 10.8× | 10.1 (27) |
| 300×4 | 1183 (5903) | 62.7 (944) | 18.9× | 14.3 (472) |
| 900×4 | does not fit | 195 (3291) | – | 38.4 (1646) |
| 3000×4 | does not fit | 728 (13116)¹ | – | 136 (6553) |
| 6000×4 | does not fit | OOM | – | 813 (13826)² |

### NequIP-OAM-L: `nequip_baseline.pt` vs `nequip_fast_oeq.pt`

| N×W | base | fast_oeq | speedup |
|---|---|---|---|
| 30×1 | 48.4 (106) | 21.5 (33) | 2.3× |
| 300×1 | 140 (1739) | 22.4 (270) | 6.3× |
| 900×1 | 483 (6307) | 38.0 (926) | 12.7× |
| 3000×1 | does not fit | 117 (3705) | – |
| 6000×1 | does not fit | 254 (7813) | – |
| 30×4 | 45.3 (342) | 20.3 (70) | 2.2× |
| 300×4 | 524 (6869) | 40.4 (1019) | 13.0× |
| 900×4 | does not fit | 118 (3640) | – |
| 3000×4 | does not fit | 631 (14755) | – |
| 6000×4 | does not fit | OOM | – |

### ANI-2x: `ani_baseline.pt` vs `ani_fast.pt`

| N×W | base | fast | speedup |
|---|---|---|---|
| 30×1 | 13.0 (1) | 1.63 (22) | 8.0× |
| 300×1 | 13.2 (17) | 1.53 (32) | 8.7× |
| 900×1 | 14.1 (58) | 1.95 (46) | 7.2× |
| 3000×1 | 17.2 (250) | 3.43 (105) | 5.0× |
| 6000×1 | 25.8 (893) | 6.17 (191) | 4.2× |
| 30×4 | 13.2 (5) | 1.83 (25) | 7.2× |
| 300×4 | 14.5 (68) | 2.50 (54) | 5.8× |
| 900×4 | 16.7 (233) | 4.31 (121) | 3.9× |
| 3000×4 | 37.2 (1001) | 12.9 (360) | 2.9× |
| 6000×4 | 75.8 (3571) | 25.0 (701) | 3.0× |

Periodic ANI at 3000 atoms: base 3119 ms / 9 GB → fast 13.2 ms (ANI REPORT).

### FeNNiX-BIO1: C++ PJRT harness, W=1 per-call ms (device MiB = executable footprint)

| atoms | base (orig. exec path) | fast (patched path) | speedup | fp32 extra | w4 vmap extra | device MiB (TF32 / fp32 / w4) |
|---|---|---|---|---|---|---|
| 30 | 4.87 | 1.93 | 2.5× | 4.06 | 2.07 | 6.0 / 5.5 / 8.2 |
| 300 | 3.22 | 2.07 | 1.6× | 4.05 | 2.09 | 15.3 / 14.2 / 49.5 |
| 900 | 2.85 | 2.19 | 1.3× | 4.38 | 3.26 | 39.7 / 40.8 / 178 |
| 3000 | 4.91 | 3.18 | 1.5× | 4.97 | 10.1 | 164 / 166 / 746 |
| 6000 | 8.13 | 5.81 | 1.4× | 8.73 | 26.4 | 417 / 420 / 1742 |

## Inside namd3 (ms/step, walk0, taken from the per-model REPORTs; separate runs)

| model | 300 atoms: base → optimised | 3000 atoms | 6000 atoms |
|---|---|---|---|
| MACE (fp64) | 293 → **19.5** (15×); fp32 extra 12.0 | fast 184 (base does not fit) | – |
| NequIP | 141 → **26.5** (5.3×) | – | – |
| ANI-2x | 14.0 → **2.09** (6.7×) | 25.0 → **10.2** | – |
| FeNNiX (TF32), patched backend only | 6.28 → **2.39** | 12.9 → **9.95** | 37.3 → 25.6 |
| FeNNiX, + `QMNoPntChrg on` + `NAMD_QM_FAST_INDEX=1` | → 1.91 | → 4.10 | → 7.51 |
| FeNNiX fp32 extra, all levers | 3.66 | 6.04 | 9.94 |
| SchNet | not run in namd3; the C-ABI shim bench gives 3.73 → 3.38 (+graph shim 1.20) | 11.2 → 8.3 | – |

- **Same conditions:** the MACE, NequIP and ANI namd3 runs used the stock template (no `QMNoPntChrg`), so the like-for-like FeNNiX row is "patched backend only".
- **NAMD bookkeeping:** at ≥3000 atoms NAMD's own QM bookkeeping (point-charge re-selection, linear-search lookups) costs every model ~8–18 ms/step. `QMNoPntChrg on` plus the ComputeQM fast-index patch remove most of it (FeNNiX REPORT). A 3000-atom ANI step of 10.2 ms is therefore mostly NAMD, not the model.

## NAMD requirements per optimised model

```bash
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages; R=/home/rat/PycharmProjects/ML_models_NAMD
# SchNet  (models/opt/schnet_fast.pt): nothing extra. Optional CUDA graphs = patched shim
#          scripts/opt/schnet/cudagraph/shim/libnamd_mlff.so + NAMD_MLFF_CUDA_GRAPH=1 (not installed)
# MACE    (models/opt/mace_fast_cueqf.pt):
export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:$R/scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # large systems (>=3000 x4)
# NequIP  (models/opt/nequip_fast_oeq.pt):
export NAMD_MLFF_EXTRA_LIBS=$R/scripts/opt/nequip/oeq_native/liboeq_native.so
# ANI-2x  (models/opt/ani_fast.pt):
export NAMD_MLFF_EXTRA_LIBS=$R/scripts/opt/ani/cuaev_native/libcuaev_native_precise.so
# FeNNiX  (unchanged namd_benchmarks/models/fennix/w<K>/manifest.json, or models/opt/fennix_fp32/w<K>):
#   patched namd3 copy ~/compile_NAMD_MACE/namd_fennix_fxopt/Linux-x86_64-g++/namd3,
#   LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH", optional NAMD_QM_FAST_INDEX=1, config "QMNoPntChrg on"
```

## Caveats
- **Clock drift:** laptop GPU clocks swing 2–3× between runs. Compare only within one invocation, i.e. one (N, W) cell of a table. The per-model REPORT numbers come from other runs; they agree within ~10% except the one marked cell.
- **Harness differences:** FeNNiX is timed on the bare PJRT execute call (C++), while the TorchScript models include the wrapper, autograd forces and the output copy (Python calling scripted modules, like the shim). The namd3 table is the only fully like-for-like view.
- **Host-bound regime:** below ~1000 atoms every TorchScript model is host-bound (interpreter + launches). GPU time at 30 atoms is ~0.3 ms (SchNet), ~2.5 ms (MACE/NequIP) and <1 ms (ANI), against wall times of 3/10/21/1.6 ms. The remaining lever is CUDA graphs (the SchNet graph shim), and FeNNiX already runs as one XLA command buffer. Small-N rankings therefore reflect dispatch overhead more than model cost.
- **FeNNiX W=4 isn't available in NAMD:** the fennol backend has no batched multi-walker path, so the w4 column is only indicative.
- **Multi-walker NAMD:** walk ≥ 2 segfaults on client connect in the Jul-31 namd3 build (pre-existing regression); W=4 numbers are harness-only for all models.
- **MACE fp64 is not like-for-like:** it pays the laptop's 1/64 fp64 rate; use the fp32 extra column for a precision-matched view (it changes energies by 0.05–1 kcal/mol, per the MACE REPORT).
- **WSL2 spills instead of OOM-ing:** cells near 16 GB (MACE 3000×4, NequIP 3000×4, MACE-f32 6000×4) are fragile. On WSL2 an oversized allocation spills to host RAM instead of failing.

## User decisions pending (collected from all REPORTs)
1. **SchNet:** install the CUDA-graph shim (`scripts/opt/schnet/cudagraph/shim/`; 3–6× at ≤900 atoms). `src/cli.py` also lacks the new SchNet `--fast` flags.
2. **FeNNiX:** adopt the NAMD patches (`scripts/opt/fennix/namd_patch/{fennix_backend,computeqm_fast_index}.patch`; patched copy at `~/compile_NAMD_MACE/namd_fennix_fxopt`).
3. **Add `QMNoPntChrg on` to `namd_benchmarks/templates/bench.conf.tmpl`** (whole-box QM). Without it, the ≥3000-atom sweep columns mostly measure NAMD point-charge bookkeeping for every model.
4. **`ComputeMLFF.C:2795`:** apply the same charge-restore id→index hash fix there, so TorchScript models also lose the O(N × numQM) search.
5. **FeNNiX precision for the comparison:** shipped TF32 or `models/opt/fennix_fp32` (precision-matched with the others).
6. **Repoint the `namd_benchmarks/models/` symlinks** (`mace_off23.pt`, `nequip_oam.pt`, `ani2x.pt`, schnet) to the optimised artifacts, and add the per-model `NAMD_MLFF_EXTRA_LIBS` to `run_benchmark.sh` (MACE is hard-coded to `models/mace_off23.pt`).
7. **Large MACE/NequIP runs:** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
8. **Keep the `allegro` env stable** (NAMD loads `libcue_ops.so` and `libnvrtc.so.13` from it for MACE), or copy those .so files next to the op lib. The native op libs are RPATH'd to `~/compile_NAMD_MACE/libtorch-2.11.0+cu130`; rebuild them if the zip moves.
9. **Optional: FeNNiX neighbour-list overflow flag.** Export it so NAMD can die on it; the capacity is only +5% over the export geometry, and an overflow is currently silent.
