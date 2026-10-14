# ML_models_NAMD

Run machine-learning interatomic potentials inside [NAMD](https://www.ks.uiuc.edu/Research/namd/).

Every model is wrapped behind one TorchScript interface, so NAMD loads it as a single `.pt` file through its C++ `mlff` backend, with no Python at run time. FeNNiX is the exception: it is exported to StableHLO and runs through NAMD's PJRT backend ([FeNNiX](#fennix)).

| Model | `--fast` build | Speed-up on RTX 5080 (30 / 300 / 3000 atoms) |
|---|---|---|
| [MACE](https://github.com/ACEsuit/mace) (MACE-OFF, any `*.model`) | cuEquivariance + FastMACE rewrites | 3.6× / 14× / reference does not fit |
| [NequIP](https://github.com/mir-group/nequip) (NequIP-OAM, any package) | OpenEquivariance + FastNequIP rewrites | 2.3× / 6.3× / reference does not fit |
| [SevenNet](https://github.com/MDIL-SNU/SevenNet) (7net-0, -l3i5, -omat, multi-fidelity) | OpenEquivariance + FastSevenNet rewrites | 5.7× / 7.7× / 12× |
| [TorchANI](https://github.com/aiqm/torchani) (ANI-1x, ANI-1ccx, ANI-2x) | cuAEV + fused ensemble | 8.0× / 8.7× / 5.0× |
| [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) ≥ 2.0 | fast energy route, cell-list neighbour list | 1.2× / 1.2× / 1.3× |
| [Allegro](https://github.com/mir-group/allegro) | none | |
| [X-MACE](https://github.com/rhyan10/X-MACE) (excited states) | none | |
| [FeNNiX](https://github.com/thomasple/FeNNol) | NAMD-side patches | |

The optimised models keep the weights and the maths; each build is checked against the stock model before it is saved. Benchmarks and parity tables: [`scripts/opt/COMPARISON.md`](scripts/opt/COMPARISON.md).

---

## Contents

1. [Quick start](#quick-start)
2. [Environments](#environments)
3. [Building a model](#building-a-model)
4. [Running in NAMD](#running-in-namd)
5. [Checking a model](#checking-a-model)
6. [FeNNiX](#fennix)
7. [D3 dispersion](#d3-dispersion)
8. [The wrapper interface](#the-wrapper-interface)
9. [Testing](#testing)
10. [Repository layout](#repository-layout)

---

## Quick start

```bash
pip install -e ".[sevennet,fast]"
python -m src.build sevennet 7net-0 --fast --out models/sevennet_fast.pt
```

This compiles the model, builds the custom-op library it needs, wraps it, and ends with the line to export before running NAMD:

```
NAMD model: /path/to/models/sevennet_fast.pt
Before running NAMD:
  export NAMD_MLFF_EXTRA_LIBS=/path/to/src/fast/native/oeq/liboeq_native.so
```

One line per model:

```bash
python -m src.build mace     MACE-OFF23_medium.model --fast --mace-python <MACE_312 python>
python -m src.build nequip   nequip.net:mir-group/NequIP-OAM-L:0.1 --r-max 6.0 --fast
python -m src.build sevennet 7net-0 --fast
python -m src.build torchani ani2x --fast
python -m src.build schnet   schnet_scripted.pt --r-max 5.0 --fast
python -m src.build allegro  my_allegro.nequip.zip
python -m src.build xmace    fulvene.model --state 0
```

Leave out `--fast` for the **reference** model: the framework's own kernels, runs anywhere, CPU included. Use it to check an optimised build ([Checking a model](#checking-a-model)).

---

## Environments

The frameworks cannot share one Python environment. One conda env per family:

| Env | Used for | Key pins |
|---|---|---|
| `allegro` | NequIP, Allegro, SevenNet, TorchANI, SchNetPack, every `--fast` build, the tests | `torch 2.11+cu130`, `e3nn ≥ 0.6`, `nequip 0.17` |
| `MACE_312` | reading MACE `*.model` files pickled by `e3nn 0.4.4` (MACE-OFF23) | `mace-torch 0.3.x`, `e3nn 0.4.4` |
| `x_mace` | X-MACE (patched source, see [X-MACE](#x-mace)) | `torch 2.2`, `e3nn 0.5.1`, `numpy < 2` |
| `fennix` | FeNNiX export | JAX + FeNNol |

Install the repo into each env with the matching extras: `pip install -e ".[mace]"`, or `[nequip]`, `[allegro]`, `[schnet]`, `[torchani]`, `[sevennet]`, `[fast]`, `[test]`, `[all]`.

---

## Building a model

```bash
python -m src.build MODEL_TYPE SOURCE [--fast] [--out mlff_model.pt] [options]
```

| `MODEL_TYPE` | `SOURCE` | Required options |
|---|---|---|
| `mace` | `*.model` file | `--mace-python` with `--fast` if this env cannot read the file (MACE-OFF23) |
| `nequip`, `allegro` | `*.nequip.zip` package or `nequip.net:` id | `--r-max` if the model stores no cutoff (NequIP-OAM-L: `6.0`) |
| `sevennet` | pretrained name or `checkpoint.pth` | `--modal` for multi-fidelity checkpoints |
| `torchani` | `ani1x`, `ani1ccx` or `ani2x` | |
| `schnet` | scripted SchNetPack model | `--r-max` = the training cutoff |
| `xmace` | `*.model` file | `--state` (default `0`, the ground state) |

`SOURCE` can also be an already compiled TorchScript file; the compile step is then skipped (not with `--fast`, which rebuilds from the weights).

Other options: `--d3` adds D3(BJ) dispersion ([below](#d3-dispersion)); `--dtype float32` makes an fp32 MACE (faster for large systems, different numerics); `--libtorch` is described next. `python -m src.build --help` lists everything.

### What `--fast` needs

- A CUDA GPU and the `allegro` env (the only one with a CUDA build of torch for sm_120 here).
- **Native op libraries** for MACE, NequIP, SevenNet and ANI. `src.build` compiles them the first time they are needed; pass `--libtorch` (NAMD's libtorch). See [Native op libraries](#native-op-libraries) to build them by hand or when something goes wrong.

If a model's architecture does not fit the rewrites (MACE, NequIP, SevenNet), the build says so and falls back to the faster kernels alone.

### Native op libraries

The optimised MACE, NequIP, SevenNet and ANI models call custom GPU ops. NAMD's shim runs no Python, so these ops are registered by small C++ libraries that the shim loads from `NAMD_MLFF_EXTRA_LIBS` (NAMD patch `01`). Each library is built once per machine.

| Library | Used by | Built from | Output |
|---|---|---|---|
| `cueq` | MACE | `uniform1d_op.cpp` + `libcue_ops.so` from `cuequivariance-ops-torch-cu13` | `src/fast/native/cueq/libcueq_uniform1d_native.so` |
| `oeq` | NequIP, SevenNet | the `openequivariance` package's C++ extension sources | `src/fast/native/oeq/liboeq_native.so` |
| `cuaev` | ANI | the installed `torchani`'s `csrc/` (cuAEV) | `src/fast/native/cuaev/libcuaev_native_precise.so` |

**Requirements** (in the `allegro` env):

- `g++` with C++17, and `pip install -e ".[fast]"` (cuEquivariance, OpenEquivariance and the CUDA 13 runtime wheels they pull in).
- **The libtorch your NAMD shim is built with.** The libraries link against it, and a library built against a different libtorch fails to load in NAMD. If it is the same torch version as the `allegro` env, the default (the torch of the active Python) works.
- `cuaev` only: `nvcc` for CUDA ≥ 12.8 (needed for sm_120), and `torchani` installed with its `csrc/` directory (`ls $(python -c 'import torchani, os; print(os.path.dirname(torchani.__file__))')/csrc`).

**Build** from the repository root:

```bash
export TORCH=/path/to/libtorch          # NAMD's libtorch; omit to use the torch of `python`
bash src/fast/native/cueq/build.sh      # MACE
bash src/fast/native/oeq/build.sh       # NequIP, SevenNet
bash src/fast/native/cuaev/build.sh     # ANI
```

| Variable | Default | Meaning |
|---|---|---|
| `TORCH` | the torch of `$PY` | libtorch to link against |
| `PY` | `python` | Python of the build env (locates site-packages) |
| `CUDA` | `<site-packages>/nvidia/cu13` | CUDA headers and runtime |
| `NVCC_HOME` | `src/fast/native/cuaev/toolchain/nvidia/cu13` | `cuaev`: CUDA toolkit with `bin/nvcc`; if unset, the CUDA 13.0 compiler wheels are pip-installed there once (about 260 MB) |
| `ARCH` | `120` | `cuaev`: GPU compute capability (`89` for Ada, `90` for Hopper, `120` for Blackwell) |
| `VARIANT` | `precise` | `cuaev`: `upstream` uses torchani's `-use_fast_math` build instead |
| `OUTLIB` | see table above | output path |

`src.build` looks for the libraries at these paths and only builds the missing ones. To rebuild one (for example after changing libtorch), delete its `.so` and run its `build.sh` again.

**Use** in NAMD, colon-separated, in this order:

```bash
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
# MACE
export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:$PWD/src/fast/native/cueq/libcueq_uniform1d_native.so
# NequIP, SevenNet
export NAMD_MLFF_EXTRA_LIBS=$PWD/src/fast/native/oeq/liboeq_native.so
# ANI
export NAMD_MLFF_EXTRA_LIBS=$PWD/src/fast/native/cuaev/libcuaev_native_precise.so
```

`src.build` prints the right line for the model it built. To check a library loads without Python, run the shim's test program with `NAMD_MLFF_EXTRA_LIBS` exported: `source namd_benchmarks/env.sh; namd_benchmarks/lib/mlff_shim_test $NAMD_MLFF_LIB mlff_model.pt 0` (`Unknown builtin op` means a library is missing; an undefined-symbol error means it was built against a different libtorch).

### Step by step

`src.build` chains two scripts you can also run yourself:

1. **Compile** the checkpoint to TorchScript: `python -m src.compile_<model>` with `mace`, `nequip` (also Allegro), `sevennet`, `torchani` or `xmace`. Each takes `--fast` for the optimised inner model. SchNetPack models are already scripted by training (`src/training/train_schnetpack.py`); `src.compile_schnetpack` only makes random-weight models for timing.
2. **Wrap** it for NAMD:

   ```bash
   python -m src.cli --model-type mace --compiled inner.pt --out mlff_model.pt \
       [--extra-libs "$NAMD_MLFF_EXTRA_LIBS"]
   ```

   `--extra-libs` is required to wrap an optimised inner model. The ablation switches of the optimised builds are in `python -m src.fast.{mace,nequip,ani}`.

### X-MACE

Upstream X-MACE (`rhyan10/X-MACE`, branch `X-MACE_socs`) cannot be TorchScript-compiled as published. The `x_mace` env carries a patched copy installed editable; **do not `pip install mace-torch` over it**. The patches fix a `zip()` over five `ModuleList`s that TorchScript silently skips, `None` entries in `socs_readouts`, untyped accumulators and `torch.tensor([])` placeholders, and `Optional[Tensor]` handling in `compute_forces`. Run `src.build xmace` in that env.

---

## Running in NAMD

NAMD loads the model through its QM interface with the `mlff` backend (a libtorch shim, `libnamd_mlff.so`). The atoms with `beta = 1` in `qm.pdb` are the ML region:

```tcl
QMForces                on
QMSoftware              mlff
QMExecPath              /path/to/mlff_model.pt
QMColumn                beta
qmParamPDB              qm.pdb
qmBondColumn            occ
QMBaseDir               /tmp/mlff_run
QMChargeMode            none
QMElecEmbed             off
QMPointChargeScheme     none
QMSwitching             off
QMVdWParams             off
QMMult                  1 1
QMCharge                1 0
# on = the model supplies all forces (whole-system ML)
qmReplaceAll            off
# whole-system ML with no MM point charges only: skips per-step point-charge selection
# QMNoPntChrg           on
```

For an optimised model, export the `NAMD_MLFF_EXTRA_LIBS` line `src.build` printed before starting NAMD. Without it the model fails to load with `Unknown builtin op`.

- The shim must be built against the same libtorch as NAMD, and the native op libraries against that libtorch too.
- NAMD source changes live as patches in [`scripts/opt/namd_patches/`](scripts/opt/namd_patches/README.md): `01` (`NAMD_MLFF_EXTRA_LIBS`, JIT options), `02` (FeNNiX PJRT fast path), `03` (faster QM atom indexing for large QM regions, `NAMD_QM_FAST_INDEX=1`).
- A complete config: `namd_benchmarks/templates/bench.conf.tmpl`; runtime environment: `namd_benchmarks/env.sh`.

---

## Checking a model

Load the wrapped model with nothing but torch and the native libraries, as NAMD does, and evaluate one water molecule. Run it for the reference and the optimised model and compare:

```python
import os, torch
for so in filter(None, os.environ.get("NAMD_MLFF_EXTRA_LIBS", "").split(":")):
    torch.ops.load_library(so)

dev = "cuda"
coords = torch.tensor([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                      dtype=torch.float64, device=dev)                       # Å
Z = torch.tensor([8, 1, 1], device=dev)
pc = torch.zeros(0, 3, dtype=torch.float64, device=dev), torch.zeros(0, dtype=torch.float64, device=dev)
no_box = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)            # all-zero cell = not periodic

m = torch.jit.load("mlff_model.pt", map_location=dev)
for _ in range(3):                    # the 3rd call runs the optimised TorchScript graph
    e, f, q, virial = m(coords, Z, *pc, no_box)
print(f"E = {e.item():.6f} kcal/mol, forces {tuple(f.shape)} {f.dtype}")
```

ANI-2x gives `E ≈ -47934 kcal/mol` (−76.38 Ha). A wrong `--r-max` or element order shows up here as a bad energy. Expected optimised − reference differences: fp64 MACE ΔE = 0, ΔF ≈ 1e-13; SevenNet and NequIP (fp32) ΔF of 1e-4 to 3e-4 kcal/mol/Å; ANI 1e-3 to 1e-2 kcal/mol (it sums the energy in fp64, stock torchani in fp32).

---

## FeNNiX

FeNNiX-BIO1 is a JAX model. `scripts/export_fennix_bio1_stablehlo.py` (`fennix` env) exports a **fixed-shape** energy + forces function to StableHLO, which NAMD's C++ PJRT backend compiles and runs. Export once per system; `--pdb` must hold exactly the ML-region atoms in NAMD's order:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python scripts/export_fennix_bio1_stablehlo.py \
    --model models/fennix-bio1S.fnx --pdb system/qm.pdb --out-dir models/fennix/my_system
```

```tcl
QMSoftware              fennol
QMExecPath              /path/to/models/fennix/my_system/manifest.json
```

| Flag | Meaning |
|---|---|
| `--pbc` | periodic export (cell input, virial output); minimum image, so every box width must be ≥ 2 × cutoff. `--cell` gives the reference cell (default: PDB `CRYST1`) |
| `--matmul-precision` | `default` is TF32 on NVIDIA GPUs (fast; energy error up to ~10 kcal/mol at 6000 atoms); `highest` is true fp32, 1.4–2× slower. `--pbc` defaults to `highest` |
| `--nblist-margin` | headroom on the fixed neighbour-list capacity (default 1.25); if MD outgrows it NAMD stops |
| `--total-charge` | fixed total charge (default: sum of PDB formal charges) |
| `--pc-capacity P` | point-charge embedding inputs (`QMElecEmbed`), for testing the NAMD side with `--test-model` |

The FeNNiX speed-up is on the NAMD side: patch `02`, `QMNoPntChrg on` for whole-system ML, and `NAMD_QM_FAST_INDEX=1` with patch `03` for large QM regions.

---

## D3 dispersion

`--d3` adds Grimme D3 with Becke–Johnson damping to any model, as `SevenNetD3Calculator` adds it to SevenNet. SevenNet's own D3 is a CUDA library NAMD cannot load, so `src/d3.py` reimplements it in TorchScript inside the model file. Parameters are read from the installed `sevenn` package when wrapping (NAMD does not need it).

- Matches SevenNet's CUDA D3 in energy, forces and stress (`tests/test_d3.py`).
- Two-body, BJ damping. `--d3-functional` takes any name in SevenNet's BJ table (`pbe` default, `pbe0`, `b3-lyp`, `r2scan`, …).
- Cutoffs as r² in bohr² (`src.cli --d3-cutoff 9000 --d3-cn-cutoff 1600`, i.e. 50.2 Å and 21.2 Å). The cost grows with every pair within 50 Å over periodic images; lower `--d3-cutoff` for large boxes.
- Applied among the atoms the model sees (the QM region in QM/MM).

---

## The wrapper interface

Every wrapper in `src/wrappers/` is an `nn.Module` with:

```
forward(coords, Z, pc_coords, pc_charges, cell)               -> (energy, forces, charges, virial)
forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)  -> (energies, forces, charges, virials)
```

| Output | Shape (single / batch) | Units |
|---|---|---|
| energy | scalar / `[B]` | kcal/mol |
| forces | `[N, 3]` / `[N_total, 3]` | kcal/mol/Å |
| charges | `[N]` / `[N_total]` | e (zeros if the model has none) |
| virial | `[3, 3]` / `[B, 3, 3]` | kcal/mol, `Σ rᵢ ⊗ fᵢ` |

All outputs are float64. `cell` holds the lattice vectors as rows in Å; an all-zero cell means not periodic (virial zero). `pc_coords` / `pc_charges` are accepted and ignored. To add a model, see [`src/wrappers/README.md`](src/wrappers/README.md).

---

## Testing

```bash
python -m pytest tests/ -v                       # allegro env; mock models, no weights needed
bash tests/run_nequip_tests.sh                   # NequIP/Allegro in their own env
bash tests/run_e2e_water_dimer.sh [--skip-orca]  # ORCA -> train every model -> wrap
JAX_PLATFORMS=cpu conda run -n fennix python -m pytest tests/test_fennix_export.py tests/test_fennix_pbc.py tests/test_fennix_pc.py -v
```

Known failures, all older than the periodic interface: `TestE2E_SchNetPack::test_train_and_wrap` and `TestE2E_TorchANI::test_train_and_wrap` call `forward()` without `cell`; the four `*_non_periodic_path_unchanged_vs_head` tests compare against a `HEAD` that predates PBC.

---

## Repository layout

```
src/
  build.py               one call: compile + native libraries + wrap
  compile_<model>.py     checkpoint -> TorchScript (mace, nequip, sevennet, torchani, xmace, schnetpack)
  cli.py                 wrap a compiled model for NAMD
  fast/                  optimised builds: <model>/ rewrites + builders, native/ C++ op libraries
  wrappers/              one wrapper per framework; README.md = how to add one
  edges.py, virial.py, constants.py, d3.py, export.py
  training/, datagen/    training scripts and ORCA data generation
scripts/
  export_fennix_bio1_stablehlo.py
  opt/                   benchmarks, per-model reports, reproduction scripts, NAMD patches
namd_benchmarks/         NAMD benchmark systems, env.sh, config template
models/                  compiled and wrapped models
tests/
```
