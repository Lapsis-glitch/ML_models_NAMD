# ML_models_NAMD

Run machine-learning interatomic potentials inside [NAMD](https://www.ks.uiuc.edu/Research/namd/).

Every model is wrapped behind one TorchScript interface, so NAMD loads it as a single `.pt` file through its C++ `mlff` backend, with no Python at run time. FeNNiX is the exception: it is exported to StableHLO and runs through NAMD's PJRT backend ([FeNNiX](#fennix)).

| Model | `--fast` build | Speed-up on RTX 5080 (30 / 300 / 3000 atoms) |
|---|---|---|
| [MACE](https://github.com/ACEsuit/mace) (MACE-OFF, any `*.model`) | cuEquivariance + FastMACE rewrites | 3.6× / 14× / reference runs out of GPU memory |
| [NequIP](https://github.com/mir-group/nequip) (NequIP-OAM, any package) | OpenEquivariance + FastNequIP rewrites | 2.3× / 6.3× / reference runs out of GPU memory |
| [SevenNet](https://github.com/MDIL-SNU/SevenNet) (7net-0, -l3i5, -omat, multi-fidelity) | OpenEquivariance + FastSevenNet rewrites | 5.7× / 7.7× / 12× |
| [TorchANI](https://github.com/aiqm/torchani) (ANI-1x, ANI-1ccx, ANI-2x) | cuAEV + fused ensemble | 8.0× / 8.7× / 5.0× |
| [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) ≥ 2.0 | fast energy route, cell-list neighbour list (also on CPU) | 1.2× / 1.2× / 1.3× |
| [Allegro](https://github.com/mir-group/allegro) | none | |
| [X-MACE](https://github.com/rhyan10/X-MACE) (excited states) | none | |
| [FeNNiX](https://github.com/thomasple/FeNNol) | NAMD-side patches | |

The optimised models use the same weights and compute the same function as the stock models. `src.build` compares each optimised model with the stock model before saving it. Benchmarks and parity tables: [`scripts/opt/COMPARISON.md`](scripts/opt/COMPARISON.md).

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

This command compiles the model, builds the custom-op library it needs and wraps the model. Its last lines show what to export before running NAMD:

```
NAMD model: /path/to/models/sevennet_fast.pt
Before running NAMD:
  export NAMD_MLFF_EXTRA_LIBS=/path/to/src/fast/native/oeq/liboeq_native.so
```

One line per model:

```bash
python -m src.build mace     MACE-OFF23_medium.model --fast --mace-python <python that can read the .model>
python -m src.build nequip   nequip.net:mir-group/NequIP-OAM-L:0.1 --r-max 6.0 --fast
python -m src.build sevennet 7net-0 --fast
python -m src.build torchani ani2x --fast
python -m src.build schnet   schnet_scripted.pt --r-max 5.0 --fast
python -m src.build allegro  results/allegro/allegro_packaged.nequip.zip --r-max 5.0
python -m src.build xmace    fulvene.model --state 0
```

Where to get each `SOURCE`:

| Model | Source |
|---|---|
| MACE-OFF23 | `wget https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model` (also `_small`, `_large`; see [ACEsuit/mace-off](https://github.com/ACEsuit/mace-off)) |
| NequIP-OAM-L | downloaded from nequip.net on first use of the `nequip.net:` id |
| SevenNet | `7net-0`, `7net-l3i5` and `7net-mf-0` ship with the `sevenn` package; for other names, `sevenn` downloads the checkpoint on first use |
| ANI | downloaded by torchani from GitHub on first use of `ani1x` / `ani1ccx` / `ani2x` |
| Allegro, other NequIP models | a model you train ([Training your own model](#training-your-own-model)), or a published package from [nequip.net](https://www.nequip.net) |
| SchNetPack | a model you train ([Training your own model](#training-your-own-model)) |
| X-MACE | your own `.model`; `models/fulvene.model` is an example |
| FeNNiX | `models/fennix-bio1S.fnx` |

Without network access to these hosts, the first download fails with a proxy or connection error. The `models/` directory contains compiled models you can pass as `SOURCE` instead, which also skips the compile step: `compiled_mace_off23_medium.pt`, `compiled_nequip_oam_l.nequip.pth`, `compiled_ani2x.pt`, `fulvene_compiled.pt` (X-MACE), and `compiled_schnet_default.pt` (SchNet with random weights, for timing only).

Leave out `--fast` to build the **reference** model, which uses the framework's own kernels and runs anywhere, CPU included. Use it to check an optimised build ([Checking a model](#checking-a-model)).

---

## Environments

The frameworks' dependencies conflict, so you cannot install all of them in one Python environment. Use one environment per group:

| Environment | Used for | Why it is separate |
|---|---|---|
| main | NequIP, Allegro, SevenNet, TorchANI, SchNetPack, MACE models saved with a current e3nn, every `--fast` build, the tests | needs a CUDA build of torch for your GPU and `e3nn ≥ 0.6` (required by NequIP) |
| old MACE | reading MACE `*.model` files saved with an older e3nn: MACE-OFF23 was saved with `e3nn 0.4.4`, which cannot be installed next to NequIP | needs the mace-torch and e3nn versions that saved the file |
| X-MACE | X-MACE (patched source, see [X-MACE](#x-mace)) | X-MACE requires older torch and e3nn versions |
| JAX | FeNNiX export | JAX + FeNNol |

Create each environment with your usual tool (venv or conda), then install from the repository root:

```bash
# main: first install torch built for your GPU's CUDA version (pytorch.org), then
pip install -e ".[nequip,allegro,sevennet,torchani,schnet,test]"
pip install -e ".[fast]"                      # --fast builds; building openequivariance downloads libtorch from download.pytorch.org
pip install cuequivariance-ops-torch-cu12     # MACE --fast; use -cu13 if torch is a CUDA 13 build
pip install mace-torch && pip install "e3nn>=0.6"   # MACE --fast; ignore pip's warning that mace-torch wants e3nn 0.4.4
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # torch must still be your CUDA build

# old MACE: only to read MACE files saved with e3nn 0.4.4 (MACE-OFF23)
pip install -e ".[mace]"

# JAX: FeNNiX export (for a GPU, install JAX with CUDA support as the JAX docs describe)
pip install fennol jax
```

Installing an extra can replace the installed torch with another version, so check torch after each `pip install`. The `[mace]` extra installs `e3nn 0.4.4`, and NequIP and SevenNet do not work with that version: keep the extra out of the main environment. For X-MACE, see [X-MACE](#x-mace).

---

## Building a model

```bash
python -m src.build MODEL_TYPE SOURCE [--fast] [--out mlff_model.pt] [options]
```

| `MODEL_TYPE` | `SOURCE` | Required options |
|---|---|---|
| `mace` | `*.model` file | none; see the note below for MACE-OFF23 |
| `nequip`, `allegro` | `*.nequip.zip` package or `nequip.net:` id | `--r-max` with any package that stores no cutoff, including your own trained ones (use the training cutoff) and NequIP-OAM-L (`6.0`); `--compile-device` sets where the model will run (default: `cuda` if a GPU is visible, else `cpu`) |
| `sevennet` | pretrained name or `checkpoint.pth` | `--modal` for multi-fidelity checkpoints |
| `torchani` | `ani1x`, `ani1ccx` or `ani2x`, or a model you trained | for a trained model, `--elements` as the trainer prints it (without it, `src.build` assumes the ANI-2x order) |
| `schnet` | scripted SchNetPack model (from training) | `--r-max` = the training cutoff |
| `xmace` | `*.model` file | `--state` (default `0`, the ground state) |

`src.build` writes the compiled model next to the output as `<out>_inner.pt` (`_inner.nequip.pth` for NequIP and Allegro); you can delete it or pass it as `SOURCE` to rewrap.

**MACE-OFF23** was saved with `e3nn 0.4.4`, so only the old-MACE environment can read the `.model` file. Build the reference model there: `python -m src.build mace MACE-OFF23_medium.model`. For `--fast`, run `src.build` in the main environment and add `--mace-python <old-MACE environment's python>`; `src.build` uses that Python only to read the weights. In the main environment, reading the file without `--mace-python` fails with `ValueError: too many values to unpack` from e3nn's `__setstate__`.

`SOURCE` can also be an already compiled TorchScript file, in which case `src.build` skips the compile step. This does not work with `--fast`, because the optimised build needs the weights to rebuild the model. SchNet is the exception: its `--fast` is a wrapper option, works from the scripted model and also runs on CPU.

Other options: `--d3` adds D3(BJ) dispersion ([below](#d3-dispersion)); `--dtype float32` makes an fp32 MACE (faster for large systems, different numerics); `--libtorch` is described next. `python -m src.build --help` lists everything.

### What `--fast` needs

- A CUDA GPU, and the main environment with a CUDA build of torch (not needed for SchNet). `src.build` stops at once if torch reports no GPU.
- **Native op libraries** for MACE, NequIP, SevenNet and ANI. `src.build` compiles them the first time they are needed; pass `--libtorch` (NAMD's libtorch). See [Native op libraries](#native-op-libraries) to build them by hand or when something goes wrong.

If the rewrites do not apply to a model's architecture (MACE, NequIP, SevenNet), `src.build` prints a message and builds the model with the faster kernels only.

### Native op libraries

The optimised MACE, NequIP, SevenNet and ANI models call custom GPU ops. NAMD's shim runs no Python, so small C++ libraries register these ops; the shim loads them from the paths in `NAMD_MLFF_EXTRA_LIBS` (NAMD patch `01`). You build each library once per machine.

| Library | Used by | Built from | Output |
|---|---|---|---|
| `cueq` | MACE | `uniform1d_op.cpp` + `libcue_ops.so` from `cuequivariance-ops-torch-cu12` / `-cu13` | `src/fast/native/cueq/libcueq_uniform1d_native.so` |
| `oeq` | NequIP, SevenNet | the `openequivariance` package's C++ extension sources | `src/fast/native/oeq/liboeq_native.so` |
| `cuaev` | ANI | the installed `torchani`'s `csrc/` (cuAEV) | `src/fast/native/cuaev/libcuaev_native_precise.so` |

**Requirements** (in the main environment):

- `g++` with C++17, `pip install -e ".[fast]"` (cuEquivariance, OpenEquivariance) and, for MACE, `pip install cuequivariance-ops-torch-cu12` or `-cu13` to match the CUDA version of your torch.
- **The libtorch your NAMD shim is built with.** The libraries link against it, and NAMD cannot load a library built against a different libtorch. If the shim uses the same torch version as your Python environment, the default (the torch of the active Python) works.
- `cuaev` only: an `nvcc` that supports your GPU (Blackwell cards need CUDA ≥ 12.8), and `torchani` installed with its `csrc/` directory (`ls $(python -c 'import torchani, os; print(os.path.dirname(torchani.__file__))')/csrc`).

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
| `PY` | `python` | Python of the build environment, used to find its site-packages |
| `CUDA` | the nvidia CUDA wheel in site-packages, else `$CUDA_HOME`, else `/usr/local/cuda` | CUDA root with `include/` and `lib/` |
| `NVCC_HOME` | the first of `$CUDA`, the `nvcc` on `PATH`, `src/fast/native/cuaev/toolchain` whose `nvcc` supports `ARCH` | `cuaev`: CUDA toolkit with `bin/nvcc`. With CUDA 13 and no suitable `nvcc`, the script pip-installs the compiler wheels into `toolchain/` once (about 260 MB) |
| `ARCH` | the GPU of the build machine | `cuaev`: compute capability to build for (`89` Ada, `90` Hopper, `120` Blackwell) |
| `VARIANT` | `precise` | `cuaev`: `upstream` builds with torchani's `-use_fast_math` flags instead |
| `OUTLIB` | see table above | output path |

`src.build` checks these paths and builds only the libraries that are missing. If a build fails, the script prints what is missing (for example `openequivariance is not installed` or `no nvcc supporting sm_120 found`) and `src.build` stops. To rebuild one (for example after changing libtorch), delete its `.so` and run its `build.sh` again.

**Use** in NAMD: export the library paths, colon-separated. These commands print the value for each model (the paths where the libraries are, or will be, built; check that the files exist):

```bash
python -c "from src.fast.native import lib_paths; print(':'.join(lib_paths('cueq')))"   # MACE
python -c "from src.fast.native import lib_paths; print(':'.join(lib_paths('oeq')))"    # NequIP, SevenNet
python -c "from src.fast.native import lib_paths; print(':'.join(lib_paths('cuaev')))"  # ANI
```

`src.build` prints the right line for the model it built. To check that NAMD can load a library without Python, run the shim's test program with `NAMD_MLFF_EXTRA_LIBS` exported: `source namd_benchmarks/env.sh; namd_benchmarks/lib/mlff_shim_test $NAMD_MLFF_LIB mlff_model.pt 0` (`Unknown builtin op` means a library is missing; an undefined-symbol error means it was built against a different libtorch).

### Step by step

`src.build` chains two scripts you can also run yourself:

1. **Compile** the checkpoint to TorchScript: `python -m src.compile_<model>` with `mace`, `nequip` (also Allegro), `sevennet`, `torchani` or `xmace`. Each takes `--fast` for the optimised inner model. The SchNetPack trainer already saves a scripted model ([Training your own model](#training-your-own-model)); `src.compile_schnetpack` only makes random-weight models for timing.
2. **Wrap** it for NAMD:

   ```bash
   python -m src.cli --model-type mace --compiled inner.pt --out mlff_model.pt \
       [--extra-libs "$NAMD_MLFF_EXTRA_LIBS"]
   ```

   `--extra-libs` is required to wrap an optimised inner model. The ablation switches of the optimised builds are in `python -m src.fast.{mace,nequip,ani}`.

### Training your own model

Prepare the data once, then train with the script for your framework. The input is an extended XYZ file with the energy in eV in `atoms.info["energy"]` and the forces in eV/Å in `atoms.arrays["forces"]` (what ASE writes for a structure with a calculator attached):

```bash
python -m src.training.prepare_data --xyz data.xyz --output-dir prepared
python -m src.training.train_schnetpack --data-dir prepared --output-dir results/schnetpack --r-max 5.0
python -m src.training.train_allegro    --data-dir prepared --output-dir results/allegro    --r-max 5.0
python -m src.training.train_nequip     --data-dir prepared --output-dir results/nequip     --r-max 5.0
```

| Trainer | File to pass to `src.build` |
|---|---|
| `train_schnetpack` | `results/schnetpack/schnet_scripted.pt`, with the same `--r-max` as in training |
| `train_allegro` | `results/allegro/allegro_packaged.nequip.zip`, with `--r-max` as in training (or `allegro_deployed.pth`, if training wrote it, without `--r-max`) |
| `train_nequip` | `results/nequip/nequip_packaged.nequip.zip`, with `--r-max` as in training |
| `train_mace` | `results/mace/mace_compiled.pt` |
| `train_torchani` | `results/torchani/torchani_scripted.pt`, with `--elements` as printed at the end of training |

Each trainer's `--help` lists its model and training options (for example `--max-epochs` and `--batch-size`). `src.build` also accepts a model trained for only a few epochs, but its energies mean nothing; train until the validation error that the trainer prints stops falling. At the end of NequIP and Allegro training, the trainers try to export a TorchScript `.pth` as well and may print `nequip-compile --mode torchscript failed (expected on PyTorch >= 2.10)` or `Manual TorchScript also failed`. You can ignore both: `src.build` compiles the `.nequip.zip` package itself.

### X-MACE

Upstream X-MACE ([rhyan10/X-MACE](https://github.com/rhyan10/X-MACE), branch `X-MACE_socs`) cannot be TorchScript-compiled as published, and this repository does not include the patched copy. Clone that branch, apply the fixes below and install it editable (`pip install -e .`) in its own environment. pip installs X-MACE under the package name `mace-torch`, so it replaces any other mace-torch in that environment, and a later `pip install mace-torch` would overwrite the patched copy. Without the patched copy, loading a `.model` fails with `Can't get attribute 'AutoencoderExcitedMACE'`. `models/fulvene.model` was also saved from a GPU, so loading it needs one; on a CPU-only machine, use `models/fulvene_compiled.pt` as `SOURCE`. The patches fix a `zip()` over five `ModuleList`s that TorchScript silently skips, `None` entries in `socs_readouts`, untyped accumulators and `torch.tensor([])` placeholders, and `Optional[Tensor]` handling in `compute_forces`. Run `src.build xmace` in that environment.

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
# whole-system ML with no MM point charges only: NAMD then skips the per-step point-charge selection
# QMNoPntChrg           on
```

For an optimised model, export the `NAMD_MLFF_EXTRA_LIBS` line `src.build` printed before starting NAMD. Without it the model fails to load with `Unknown builtin op`.

- The shim must be built against the same libtorch as NAMD, and the native op libraries against that libtorch too.
- The changes to the NAMD source are stored as patches in [`scripts/opt/namd_patches/`](scripts/opt/namd_patches/README.md): `01` (`NAMD_MLFF_EXTRA_LIBS`, JIT options), `02` (FeNNiX PJRT fast path), `03` (faster QM atom indexing for large QM regions, `NAMD_QM_FAST_INDEX=1`).
- `namd_benchmarks/templates/bench.conf.tmpl` is a complete config, and `namd_benchmarks/env.sh` sets up the runtime environment.

---

## Checking a model

Load the wrapped model with nothing but torch and the native libraries, as NAMD does, and evaluate one water molecule. Run it for the reference and the optimised model and compare. The water test only works for a model with parameters for H and O; for another model, replace `coords` and `Z` with a molecule made of elements it knows (a model asked about unknown elements still returns a number, but a meaningless one).

```python
import os, torch
for so in filter(None, os.environ.get("NAMD_MLFF_EXTRA_LIBS", "").split(":")):
    torch.ops.load_library(so)

MODEL = "mlff_model.pt"                                                     # the --out of src.build
dev = "cuda" if torch.cuda.is_available() else "cpu"
coords = torch.tensor([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                      dtype=torch.float64, device=dev)                       # Å
Z = torch.tensor([8, 1, 1], device=dev)
pc = torch.zeros(0, 3, dtype=torch.float64, device=dev), torch.zeros(0, dtype=torch.float64, device=dev)
no_box = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)            # all-zero cell = not periodic

m = torch.jit.load(MODEL, map_location=dev)
for _ in range(3):                    # TorchScript optimises the graph by the 3rd call
    e, f, q, virial = m(coords, Z, *pc, no_box)
print(f"E = {e.item():.6f} kcal/mol, forces {tuple(f.shape)} {f.dtype}")
```

With the reference models, the snippet prints:

| Model | E (kcal/mol) |
|---|---|
| ANI-2x | -47934.33 |
| MACE-OFF23 medium | -47995.02 |
| NequIP-OAM-L | -290.90 |
| SevenNet 7net-0 | -324.82 |
| X-MACE fulvene (H, C only), state 0, on `scripts/opt/infra/fulvene1.xyz` (see below) | -143996.57 |

To use a geometry file instead of water, replace `coords` and `Z` in the snippet:

```python
import ase.io
a = ase.io.read("scripts/opt/infra/fulvene1.xyz")
coords = torch.tensor(a.positions, dtype=torch.float64, device=dev)
Z = torch.tensor(a.numbers, device=dev)
```

With a wrong `--r-max` or element order, the energy here is visibly wrong. Your own trained models have no reference value; compare their energy with the training data instead, after converting eV to kcal/mol (× 23.0605).

Two messages during wrapping are expected: TorchANI prints `cutoff (r_max): 0.0 Å`, because its cutoff is stored in the AEV computer, and SchNet `--fast` prints a `UserWarning` about converting a tensor with `requires_grad=True` to a scalar. Expected optimised − reference differences: fp64 MACE ΔE = 0, ΔF ≈ 1e-13; SevenNet and NequIP (fp32) ΔF of 1e-4 to 3e-4 kcal/mol/Å; ANI 1e-3 to 1e-2 kcal/mol (the optimised model sums the energy in fp64, stock torchani in fp32).

---

## FeNNiX

FeNNiX-BIO1 is a JAX model. `scripts/export_fennix_bio1_stablehlo.py` (JAX environment, `pip install fennol jax`) exports a **fixed-shape** energy + forces function to StableHLO, which NAMD's C++ PJRT backend compiles and runs. Export once per system. The `--pdb` file (any PDB with element columns) must contain exactly the ML-region atoms, in NAMD's order:

```bash
# first test: one water molecule (the default when neither --pdb nor --z-list is given)
python scripts/export_fennix_bio1_stablehlo.py --model models/fennix-bio1S.fnx --out-dir models/fennix/water
# your system
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/export_fennix_bio1_stablehlo.py \
    --model models/fennix-bio1S.fnx --pdb my_system/qm.pdb --out-dir models/fennix/my_system
```

```tcl
QMSoftware              fennol
QMExecPath              /path/to/models/fennix/my_system/manifest.json
```

To check an export without NAMD, read the lines `energy abs diff eV:` and `force max abs diff eV/A:` that the export script prints after evaluating the exported function against the Python reference (below 1e-6 for the water test). The export script also writes `reference.npz` and `reference_runtime.txt`, with the reference energies and forces, to the output directory.

| Flag | Meaning |
|---|---|
| `--pbc` | periodic export (cell input, virial output); minimum image, so every box width must be ≥ 2 × cutoff. `--cell` gives the reference cell (default: PDB `CRYST1`) |
| `--matmul-precision` | `default` is TF32 on NVIDIA GPUs (fast; energy error up to ~10 kcal/mol at 6000 atoms); `highest` is true fp32, 1.4–2× slower. `--pbc` defaults to `highest` |
| `--nblist-margin` | headroom on the fixed neighbour-list capacity (default 1.25); NAMD stops if the system needs more neighbours than that during MD |
| `--total-charge` | fixed total charge (default: sum of PDB formal charges) |
| `--pc-capacity P` | point-charge embedding inputs (`QMElecEmbed`), for testing the NAMD side with `--test-model` |

The FeNNiX speed-up comes from the NAMD side: patch `02`, `QMNoPntChrg on` for whole-system ML, and `NAMD_QM_FAST_INDEX=1` with patch `03` for large QM regions.

---

## D3 dispersion

`--d3` adds Grimme D3 with Becke–Johnson damping to any model, as `SevenNetD3Calculator` adds it to SevenNet. SevenNet's own D3 is a CUDA library NAMD cannot load, so `src/d3.py` reimplements it in TorchScript, inside the model file. When wrapping, `src.cli` reads the parameters from the installed `sevenn` package; NAMD does not need `sevenn`.

- The results match SevenNet's CUDA D3 in energy, forces and stress (`tests/test_d3.py`).
- It is two-body only, with BJ damping. `--d3-functional` takes any name in SevenNet's BJ table (`pbe` default, `pbe0`, `b3-lyp`, `r2scan`, …).
- The cutoffs are given as r² in bohr² (`src.cli --d3-cutoff 9000 --d3-cn-cutoff 1600`, i.e. 50.2 Å and 21.2 Å). The cost grows with the number of pairs within 50 Å over periodic images, so lower `--d3-cutoff` for large boxes.
- D3 is computed only between the atoms the model sees (the QM region in QM/MM).

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

All outputs are float64. `cell` holds the lattice vectors as rows in Å; an all-zero cell means not periodic (virial zero). The wrappers accept `pc_coords` / `pc_charges` and ignore them. To add a model, see [`src/wrappers/README.md`](src/wrappers/README.md).

---

## Testing

```bash
python -m pytest tests/ -v                       # main environment; mock models, no weights needed
bash tests/run_nequip_tests.sh                   # NequIP/Allegro
bash tests/run_e2e_water_dimer.sh [--skip-orca]  # ORCA -> train every model -> wrap
JAX_PLATFORMS=cpu python -m pytest tests/test_fennix_export.py tests/test_fennix_pbc.py tests/test_fennix_pc.py -v   # JAX environment
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
