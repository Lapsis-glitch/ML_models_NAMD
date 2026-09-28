# ML_models_NAMD

Run machine-learning interatomic potentials (MLIPs) inside [NAMD](https://www.ks.uiuc.edu/Research/namd/).

Each supported model is wrapped behind one fixed TorchScript interface, so NAMD's C++ MLFF backend can load any of them as a single `.pt` file without running Python. FeNNiX is the exception: it is exported to StableHLO and runs through NAMD's PJRT backend instead (see [FeNNiX](#fennix-stablehlo--pjrt)).

| Model | Wrapper | Native units | NAMD backend |
|---|---|---|---|
| [MACE](https://github.com/ACEsuit/mace) (incl. MACE-OFF) | `MACE_TS_Wrapper` | eV | `QMSoftware mlff` |
| [NequIP](https://github.com/mir-group/nequip) (incl. NequIP-OAM) | `NequIP_Allegro_Wrapper` | eV | `QMSoftware mlff` |
| [Allegro](https://github.com/mir-group/allegro) | `NequIP_Allegro_Wrapper` | eV | `QMSoftware mlff` |
| [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) ≥ 2.0 | `SchNetPack_Wrapper` | eV | `QMSoftware mlff` |
| [TorchANI](https://github.com/aiqm/torchani) (ANI-1x/1ccx/2x) | `TorchANI_Wrapper` | Hartree | `QMSoftware mlff` |
| [X-MACE](https://github.com/rhyan10/X-MACE) (excited states) | `XMACE_TS_Wrapper` | eV | `QMSoftware mlff` |
| [FeNNiX / FeNNol](https://github.com/thomasple/FeNNol) | StableHLO export | eV | `QMSoftware fennol` |

---

## Contents

1. [Environments](#environments)
2. [Guide: compile a model for NAMD](#guide-compile-a-model-for-namd)
3. [FeNNiX (StableHLO / PJRT)](#fennix-stablehlo--pjrt)
4. [Guide: optimised builds](#guide-optimised-builds)
5. [The wrapper interface](#the-wrapper-interface)
6. [Testing](#testing)
7. [Adding a new model](#adding-a-new-model)
8. [Repository layout](#repository-layout)

---

## Environments

The frameworks cannot share one Python environment (`mace-torch 0.3.x` pins `e3nn 0.4.4`, NequIP ≥ 0.6 needs `e3nn ≥ 0.6`, X-MACE pins `e3nn 0.5.1`, FeNNol needs JAX). We use one conda env per family:

| Env | Used for | Key pins |
|---|---|---|
| `MACE_312` | reading MACE `*.model` files (compile, or extract weights for the optimised build) | `mace-torch 0.3.x`, `e3nn 0.4.4` |
| `allegro` | NequIP/Allegro, TorchANI, SchNetPack, wrapping, tests, the only env with a working CUDA torch on sm_120 | `torch 2.11+cu130`, `nequip 0.17`, `e3nn ≥ 0.6` |
| `x_mace` | X-MACE (editable install of a patched `rhyan10/X-MACE`) | `torch 2.2`, `e3nn 0.5.1`, `numpy < 2` |
| `fennix` | FeNNiX StableHLO export | JAX + FeNNol |

Install the repo into each env with the matching extra:

```bash
pip install -e ".[mace]"        # or [nequip], [allegro], [schnet], [torchani], [test], [all]
```

Wrapping (`python -m src.cli`) only needs torch plus the file you are wrapping, and we run it from `allegro`. The exception is X-MACE, which is wrapped in `x_mace`.

---

## Guide: compile a model for NAMD

Every TorchScript model goes through the same four steps:

```
weights ──(1) compile──▶ inner TorchScript ──(2) wrap──▶ mlff_model.pt ──(3) check──▶ (4) NAMD
          model-specific   compiled_*.pt       src.cli    NAMD-ready        Python      QMSoftware mlff
```

1. **Compile.** Turn the framework's checkpoint into a TorchScript file. This step is different for every framework.
2. **Wrap.** `python -m src.cli` puts that file behind the fixed interface, converts units to kcal/mol, and saves `mlff_model.pt`.
3. **Check.** Load the result in plain PyTorch and evaluate one molecule.
4. **Run.** Point NAMD at the file.

This produces the **reference** model: the framework's own kernels behind the wrapper. It runs anywhere, CPU included. For production speed on a GPU, build the optimised variant as well ([Guide: optimised builds](#guide-optimised-builds)), then check it against this reference.

The examples below use published pretrained models. A model you trained yourself goes the same way: start from its compiled or deployed file.

### Step 1 — Compile

| Model | Env | Command | Output |
|---|---|---|---|
| MACE-OFF23 | `MACE_312` | `python -m src.compile_mace_off --weights MACE-OFF23_medium.model --out models/compiled_mace_off23_medium.pt` | fp64 TorchScript |
| NequIP-OAM-L | `allegro` | `python scripts/opt/nequip/compile_ts.py nequip.net:mir-group/NequIP-OAM-L:0.1 models/compiled_nequip_oam_l.nequip.pth --device cuda` | TorchScript `.nequip.pth` |
| ANI-2x / ANI-1x / ANI-1ccx | `allegro` | `python -m src.compile_torchani --variant ani2x --out models/compiled_ani2x.pt` | TorchScript + element order |
| X-MACE | `x_mace` | see [X-MACE](#x-mace) below | TorchScript |
| SchNetPack | `allegro` | no pretrained model; train one, or use `python -m src.compile_schnetpack --out … --r-max 5.0` for **random weights (timing only)** | TorchScript |

Notes:

- **MACE-OFF.** Get the `.model` file from the [MACE-OFF release](https://github.com/ACEsuit/mace-off) (mace-torch caches it in `~/.cache/mace/`). `--dtype float32` makes an fp32 variant, which is faster but has different numerics.
- **NequIP / Allegro.** `nequip-compile --mode torchscript` refuses to run on torch ≥ 2.10. `scripts/opt/nequip/compile_ts.py` does the same thing without that check, and accepts either a `nequip.net:` model id or a local `*.nequip.zip` package. Set `--device` to where NAMD will run the model. On older torch you can use `nequip-compile --mode torchscript --target pair_nequip <package> <out>` directly. OAM-L doesn't store its cutoff as an attribute, so pass `--r-max 6.0` when wrapping.
- **TorchANI.** The compile script prints the element order. Pass it unchanged to `--elements` when wrapping. The ANI-2x order is `1,6,7,8,16,9,17` (H C N O S F Cl), which is also the CLI default.

#### X-MACE

The upstream X-MACE code (`rhyan10/X-MACE`, branch `X-MACE_socs`) can't be TorchScript-compiled as published. We keep a patched copy installed editable in the `x_mace` env. **Don't `pip install mace-torch` over it**, because that replaces the patches. The patches fix:

- a `zip()` over five `ModuleList`s in `AutoencoderExcitedMACE.forward` that TorchScript drops without an error (the loop never runs);
- `None` entries in `socs_readouts`, untyped accumulators and bare `torch.tensor([])` placeholders;
- a `retain == False` typo and missing `Optional[Tensor]` handling in `compute_forces`.

Models saved with `nac_indices=None` / `soc_indices=None` also need a fix at load time:

```python
import torch
from copy import deepcopy
from e3nn.util import jit

m = torch.load("fulvene.model", map_location="cpu", weights_only=False).eval()
if m.nac_indices is None: m.nac_indices = 0
if m.soc_indices is None: m.soc_indices = 0
torch.jit.save(jit.compile(deepcopy(m)), "models/fulvene_compiled.pt")
```

### Step 2 — Wrap

```bash
python -m src.cli --model-type mace     --compiled models/compiled_mace_off23_medium.pt       --out mlff_model.pt
python -m src.cli --model-type nequip   --compiled models/compiled_nequip_oam_l.nequip.pth    --r-max 6.0 --out mlff_model.pt
python -m src.cli --model-type allegro  --compiled results/allegro/allegro_deployed.pth       --out mlff_model.pt
python -m src.cli --model-type schnet   --compiled results/schnetpack/schnet_scripted.pt      --r-max 5.0 --out mlff_model.pt
python -m src.cli --model-type torchani --compiled models/compiled_ani2x.pt --elements 1,6,7,8,16,9,17 --out mlff_model.pt
python -m src.cli --model-type xmace    --compiled models/fulvene_compiled.pt --state 0       --out mlff_model.pt
```

| Flag | Applies to | Meaning |
|---|---|---|
| `--model-type` | all | `mace`, `nequip`, `allegro`, `schnet`, `torchani`, `xmace` |
| `--compiled` | all | the file from step 1 |
| `--out` | all | output file (default `mlff_model.pt`) |
| `--device` | all | device to load onto while wrapping (default `cpu`) |
| `--r-max` | schnet (required), nequip (fallback) | cutoff in Å; **must equal the training cutoff** |
| `--energy-key`, `--forces-key` | schnet | output dict keys if your model uses non-default names |
| `--elements` | torchani | atomic numbers in the model's species order (default ANI-2x) |
| `--state` | xmace | electronic state to expose (0 = ground) |
| `--extra-libs` | all | colon-separated native op libraries to load first (same value as `NAMD_MLFF_EXTRA_LIBS`); needed to wrap an optimised inner model |
| `--fast` | schnet | optimised energy route ([optimised builds](#guide-optimised-builds)); tuning: `--graph-max-atoms`, `--half-min-atoms`, `--nl-cell-min-pairs`, `--no-half-filter` |
| `--lean` | torchani | low-overhead path with bit-identical outputs (optimised builds) |

The CLI scripts the wrapper with `torch.jit.script`, saves it, and prints the cutoff, batching support and internal dtype.

SchNetPack models trained in units other than eV need the `energy_units_to_kcal` argument of `SchNetPack_Wrapper`. The CLI doesn't expose it, so build the wrapper in Python and call `src.export.export_wrapped` yourself.

### Step 3 — Check the result

Load the wrapped file with nothing but torch and evaluate one water molecule, first without a box and then in one:

```python
import torch

m = torch.jit.load("mlff_model.pt", map_location="cpu")

coords = torch.tensor([[0.000,  0.000,  0.117],
                       [0.000,  0.757, -0.469],
                       [0.000, -0.757, -0.469]], dtype=torch.float64)   # Å
Z = torch.tensor([8, 1, 1], dtype=torch.long)
no_pc = torch.zeros(0, 3, dtype=torch.float64), torch.zeros(0, dtype=torch.float64)

# all-zero cell = not periodic -> virial is zero
e, f, q, w = m(coords, Z, *no_pc, torch.zeros(1, 3, 3, dtype=torch.float64))
print("E [kcal/mol]:", e.item(), "forces", tuple(f.shape), f.dtype)

# 20 Å cubic box -> same energy for an isolated molecule, virial filled in
e_pbc, _, _, w_pbc = m(coords, Z, *no_pc, 20.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0))
print("E_pbc - E:", (e_pbc - e).item(), "virial diag:", w_pbc.diagonal().tolist())
```

For ANI-2x this prints `E ≈ -47934 kcal/mol` (−76.38 Ha), float64 forces of shape `(3, 3)`, and `E_pbc - E = 0`. A wrong `--elements` order or `--r-max` shows up here as a bad energy, which is much easier to spot than inside NAMD. Set `TORCHANI_NO_WARN_EXTENSIONS=1` to silence TorchANI's extension warnings.

### Step 4 — Run in NAMD

The wrapped model is loaded by NAMD's QM interface through the `mlff` backend (a libtorch shim, `libnamd_mlff.so`, which namd3 `dlopen`s). The atoms that the model should handle are flagged in a PDB column, the same way as for any other QM engine:

```tcl
QMForces                on
QMSoftware              mlff
QMExecPath              /path/to/mlff_model.pt
# atoms with beta = 1 in qm.pdb are the ML region
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
# on = the model supplies all forces (full-ML runs)
qmReplaceAll            off
# only for whole-box ML with no MM point charges (skips per-step point-charge selection);
# leave it off for QM/MM embedding
# QMNoPntChrg           on
```

A complete working config is in `namd_benchmarks/templates/bench.conf.tmpl`, and `namd_benchmarks/env.sh` sets up the runtime environment. That includes `NAMD_MLFF_LIB` (the shim to load) and the library paths. The shim must be built against the **same libtorch** as NAMD itself. The NAMD source changes it relies on are in `scripts/opt/namd_patches/`, and their README explains each patch.

---

## FeNNiX (StableHLO / PJRT)

FeNNiX-BIO1 is a JAX model, so it doesn't go through TorchScript. `scripts/export_fennix_bio1_stablehlo.py` (run in the `fennix` env) lowers a **fixed-shape** energy + forces function to StableHLO. NAMD's C++ PJRT backend then compiles and runs it.

Each export is tied to one atom count and one composition. Export once per system. The `--pdb` you pass must contain **exactly the ML-region atoms, in NAMD's order**. For whole-box ML, that is simply the system PDB:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python scripts/export_fennix_bio1_stablehlo.py \
    --model models/fennix-bio1S.fnx \
    --pdb   system/qm.pdb \
    --out-dir models/fennix/my_system
```

| Flag | Meaning |
|---|---|
| `--pdb` / `--z-list` | atom list (and reference coordinates) that fix the shape; use one or the other |
| `--total-charge` | fixed total charge (default: sum of PDB formal charges, else 0) |
| `--pbc` | periodic export: takes the live cell as a second input and returns the virial |
| `--cell` | reference cell for `--pbc`: 3 numbers (orthorhombic) or 9 (rows a, b, c); defaults to the PDB `CRYST1` |
| `--nblist-margin` | headroom on the fixed neighbour-list capacity (default 1.25); if MD outgrows it, the artifact sets an overflow flag and NAMD stops |
| `--matmul-precision` | `default` is TF32 on NVIDIA GPUs; `--pbc` defaults to `highest` |

The output directory holds `manifest.json`, the `.stablehlo.mlir`, and reference outputs. In NAMD:

```tcl
QMSoftware              fennol
QMExecPath              /path/to/models/fennix/my_system/manifest.json
```

Under `--pbc` the model uses the minimum-image convention, so every perpendicular box width must be at least 2 × cutoff. `namd_benchmarks/export_fennix.sh` shows how to export for a series of system sizes.

---

## Guide: optimised builds

`python -m src.cli` on its own always produces the **reference** model. The optimised models are built with the tools in `scripts/opt/<model>/`. They keep the same weights and the same maths, but swap in faster GPU kernels and remove overhead. The recipes below use **the same settings as the benchmark artifacts in `models/opt/`**. Rebuilding MACE-OFF23, NequIP-OAM-L, ANI-2x and SchNet this way reproduces those artifacts exactly (ΔE = ΔF = 0). The MACE and NequIP builds assert parity with the stock model before saving. Their checks cycle through every element the model knows, so they test your model's actual species rather than only H and O. ANI is checked with the comparison step at the end of this guide.

| Model | What the optimised build changes | Works for | Extra requirement in NAMD |
|---|---|---|---|
| MACE | cuEquivariance kernels + FastMACE exact rewrites | any MACE `*.model` | 3 native libs via `NAMD_MLFF_EXTRA_LIBS` |
| NequIP | OpenEquivariance kernels + FastNequIP exact rewrites | NequIP packages (a kernels-only fallback is available if the rewrites don't fit your model) | `liboeq_native.so` via `NAMD_MLFF_EXTRA_LIBS` |
| ANI | cuAEV + fused per-element ensemble networks | torchani's pretrained ANI-1x, ANI-1ccx, ANI-2x (not custom-trained ANI) | `libcuaev_native_precise.so` via `NAMD_MLFF_EXTRA_LIBS` |
| SchNet | fast energy route, cell-list neighbour list, half-list filter | any SchNetPack model with an `Atomwise` energy head | none |
| FeNNiX | nothing on the model side; NAMD-side patches + config | any export | NAMD patch `02` (+ `03`), `QMNoPntChrg on` |
| Allegro, X-MACE | no optimised route; use the reference model | | |

Speed-ups on an RTX 5080 range from 1.1× (SchNet) to about 15× (MACE). They are in `scripts/opt/COMPARISON.md`, and each model's `scripts/opt/<model>/REPORT.md` has the details and parity numbers.

### At a glance

Every optimised build is the same four steps. Only the tools in each step change per model:

```
1. native op library  (once per machine, against NAMD's libtorch)   ->  *_native.so
2. fast inner model   (same weights, parity-checked)                ->  models/opt/<name>_inner_fast.pt
3. wrap               python -m src.cli ... --extra-libs "$NAMD_MLFF_EXTRA_LIBS"
4. run                NAMD with the same NAMD_MLFF_EXTRA_LIBS exported
```

| Model | 1. native library | 2. fast inner | 3. extra `src.cli` flags |
|---|---|---|---|
| MACE | `mace/cueq_native/build.sh` | `mace/build_fast.py cueqf OUT --no-plain-linear --state STATE` (after `extract_state.py` in `MACE_312`) | `--extra-libs` |
| NequIP | `nequip/oeq_native/build.sh` | `nequip/build_fast.py PKG OUT` | `--extra-libs` (`--r-max` if needed) |
| ANI | `ani/cuaev_native/build.sh` | `ani/build_fast.py --model ani2x --cache-species --group-max-atoms 1024 OUT` | `--extra-libs --lean --elements …` |
| SchNet | — | — (wrap the normal inner) | `--fast` |

(Paths are under `scripts/opt/`.) To rebuild the benchmark models themselves, including the NAMD loadability test, run `bash scripts/opt/<model>/build.sh`. The sections below give the exact commands for your own model.

### Before you start

- Everything runs in the **`allegro` env on a GPU**. The cuEquivariance, OpenEquivariance and cuAEV kernels are CUDA-only.
- Put CUDA 13 NVRTC on the library path, **including at build time**. Without it cuEquivariance silently falls back to its slow path:
  ```bash
  SP=$(python -c 'import site; print(site.getsitepackages()[0])')   # allegro env
  export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:$LD_LIBRARY_PATH
  ```
- The native op libraries (`*_native.so`) register the custom ops in C++, so NAMD never runs Python. They must be built against the **same libtorch as NAMD's shim**: pass `TORCH=<libtorch dir>` (and `CUDA=<cuda root>` if needed) to their `build.sh`. Build each library once per machine.
- In NAMD, `NAMD_MLFF_EXTRA_LIBS=/a.so:/b.so` makes the shim load those libraries before the model. This needs NAMD patch `01` from `scripts/opt/namd_patches/`. Without the libraries the model fails to load with `Unknown builtin op`. Wrap with the same list: `python -m src.cli … --extra-libs "$NAMD_MLFF_EXTRA_LIBS"`.
- The `scripts/opt/<model>/build.sh` scripts rebuild the benchmark models end to end, including the NAMD loadability test. They read `PY`, `SP`, `TORCH` (and `MACE_PY`, `PKG`) from the environment and fall back to the original machine's paths.

### MACE

```bash
# 1. dump config + weights to a version-neutral file
#    (MACE_312 env: e3nn 0.4.4 pickles can't be read under e3nn 0.6)
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python scripts/opt/mace/extract_state.py my_mace.model models/opt/my_mace_state.pt

# --- the rest runs in the allegro env ---
# 2. native cuEq op library (once)
TORCH=/path/to/libtorch bash scripts/opt/mace/cueq_native/build.sh
export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:$PWD/scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so

# 3. fast inner model: cuEq + fused conv + FastMACE, fp64 (asserts parity with the stock model)
python scripts/opt/mace/build_fast.py cueqf models/opt/my_mace_inner_fast.pt --no-plain-linear \
    --state models/opt/my_mace_state.pt

# 4. wrap
python -m src.cli --model-type mace --compiled models/opt/my_mace_inner_fast.pt \
    --extra-libs "$NAMD_MLFF_EXTRA_LIBS" --out models/opt/my_mace_fast.pt
```

- `--dtype float32` in step 3 gives an fp32 variant. It is faster for large systems, but the numerics change.
- For no custom ops at all, use `build_fast.py e3nn …` and wrap without `--extra-libs`. That gives FastMACE on plain e3nn: slower than cuEq, but it loads with no extra libraries.
- If `build_fast.py` stops with an assertion on an unusual MACE architecture, fall back to the kernels only: `python scripts/opt/mace/compile_inner.py cueqf inner.pt --state …`.

### NequIP

```bash
# 1. native OpenEquivariance op library (once)
TORCH=/path/to/libtorch bash scripts/opt/nequip/oeq_native/build.sh
export NAMD_MLFF_EXTRA_LIBS=$PWD/scripts/opt/nequip/oeq_native/liboeq_native.so

# 2. fast inner from a NequIP package (*.nequip.zip from `nequip-package build`;
#    nequip.net models are cached in ~/.nequip/model_cache/); asserts parity at 30 and 300 atoms
python scripts/opt/nequip/build_fast.py my_model.nequip.zip models/opt/my_nequip_inner_fast.nequip.pth

# 3. wrap (--r-max only if the model doesn't store its cutoff, e.g. OAM-L: 6.0)
python -m src.cli --model-type nequip --compiled models/opt/my_nequip_inner_fast.nequip.pth \
    --extra-libs "$NAMD_MLFF_EXTRA_LIBS" --out models/opt/my_nequip_fast.pt
```

The FastNequIP rewrites were written for NequIP-OAM-L, and `build_fast.py` stops with an assertion if a block doesn't have the structure it expects. The fallback gives you the OpenEquivariance kernels without the rewrites:

```bash
python scripts/opt/nequip/compile_ts.py my_model.nequip.zip inner.nequip.pth --device cuda --modifiers enable_OpenEquivariance
```

Then wrap it the same way.

### ANI

```bash
# 1. native cuAEV library (once). Needs an nvcc for CUDA >= 12.8 (sm_120): set NVCC_HOME, or install the
#    pip wheels into scripts/opt/ani/toolchain the way scripts/opt/ani/build.sh does
TORCH=/path/to/libtorch VARIANT=precise bash scripts/opt/ani/cuaev_native/build.sh
export NAMD_MLFF_EXTRA_LIBS=$PWD/scripts/opt/ani/cuaev_native/libcuaev_native_precise.so

# 2. fast inner (prints the element list to wrap with)
python scripts/opt/ani/build_fast.py --model ani2x --cache-species --group-max-atoms 1024 models/opt/ani_inner_fast.pt

# 3. wrap
python -m src.cli --model-type torchani --compiled models/opt/ani_inner_fast.pt --lean \
    --elements 1,6,7,8,16,9,17 --extra-libs "$NAMD_MLFF_EXTRA_LIBS" --out models/opt/ani_fast.pt
```

- For ANI-1x or ANI-1ccx, use `--model ani1x` / `--model ani1ccx` and `--elements 1,6,7,8`.
- `build_fast.py` has no parity check of its own, so compare the result with the reference using the check below. Expect ΔE of 1e-3 to 1e-2 kcal/mol: the fast path sums the energy in fp64, while stock torchani sums in fp32.

### SchNet

No extra build is needed. Wrap with `--fast`:

```bash
python -m src.cli --model-type schnet --compiled schnet_scripted.pt --r-max 5.0 --fast --out models/opt/schnet_fast.pt
```

The fast route is used for non-periodic inputs; periodic calls go through the reference path. The defaults of the tuning flags are the benchmark values. No extra libraries are needed in NAMD.

### FeNNiX

The exported artifact is already the optimised one; the speed-up is on the NAMD side:

- NAMD patch `02_fennix_backend` (faster PJRT execute path, about 3× at 300 atoms);
- `QMNoPntChrg on` for whole-box ML (config only);
- `NAMD_QM_FAST_INDEX=1` with patch `03` for large QM regions.

Non-periodic exports run their matmuls in **TF32** by default. That is fast, but the energy error reaches about 10 kcal/mol at 6000 atoms, and it changes from one compile to the next. Export with `--matmul-precision highest` for true fp32, which is 1.4–2× slower.

### Checking an optimised build

Compare the optimised model with the reference in a fresh Python process that loads **only the native libraries**. This is how NAMD will load it, so it also catches a missing library:

```python
import os, torch
for so in filter(None, os.environ.get("NAMD_MLFF_EXTRA_LIBS", "").split(":")):
    torch.ops.load_library(so)                      # what the NAMD shim does

dev = "cuda"
coords = torch.tensor([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]],
                      dtype=torch.float64, device=dev)
Z = torch.tensor([8, 1, 1], device=dev)
pc = torch.zeros(0, 3, dtype=torch.float64, device=dev), torch.zeros(0, dtype=torch.float64, device=dev)
cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)

res = {}
for name in ("mlff_model.pt", "models/opt/my_mace_fast.pt"):   # reference, optimised
    m = torch.jit.load(name, map_location=dev)
    for _ in range(3):                                          # 3rd call runs the optimised graph
        e, f, _, _ = m(coords, Z, *pc, cell)
    res[name] = (e.item(), f)
(e0, f0), (e1, f1) = res.values()
print("dE =", abs(e1 - e0), "kcal/mol   max dF =", (f1 - f0).abs().max().item(), "kcal/mol/Å")
```

Expected differences from the reference:
- fp64 MACE: exact (ΔE = 0, ΔF ≈ 1e-13);
- ANI: 1e-3 to 1e-2 kcal/mol, from the fp64 summation;
- fp32 and TF32 variants: 1e-4 to 1e-2.

The per-model `REPORT.md` files have the full parity tables.

---

## The wrapper interface

Every TorchScript wrapper in `src/wrappers/` is an `nn.Module` with:

```
forward(coords, Z, pc_coords, pc_charges, cell)
    -> (energy, forces, charges, virial)
forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)      # @torch.jit.export
    -> (energies, forces, charges, virials)
supports_batch: bool = True
supports_pbc:   bool = True
```

| Output | Shape (single / batch) | Units |
|---|---|---|
| energy | scalar / `[B]` | kcal/mol |
| forces | `[N, 3]` / `[N_total, 3]` | kcal/mol/Å |
| charges | `[N]` / `[N_total]` | e (zeros if the model has no charges) |
| virial | `[3, 3]` / `[B, 3, 3]` | kcal/mol, symmetric, `−dE/dε = Σ rᵢ ⊗ fᵢ` |

- All outputs are **float64**. The conversion from each model's native units uses the factors in `src/constants.py` and runs on the device.
- `cell` is `[1, 3, 3]` (`cells` is `[B, 3, 3]`) with the lattice vectors as rows, in Å. **An all-zero cell means not periodic**, and then the virial is zero.
- `pc_coords` / `pc_charges` are accepted for compatibility and currently ignored.
- Neighbour lists come from `src/edges.py`, in FP32, with periodic images when there is a cell. TorchANI is the exception: its AEV code builds its own.

### Per-model notes

- **MACE.** The wrapper builds the one-hot `node_attrs` and caches the constant tensors (batch, ptr, node_attrs) after the first call. Edges are built in FP32 and MACE-OFF runs in FP64.
- **NequIP / Allegro.** Both share one wrapper. `atom_types` are 0-based type indices, not Z; the Z → type table comes from the compiled model's metadata.
- **SchNetPack.** The cutoff isn't stored in the model, so `--r-max` must match training. The float32 → float64 cast happens in the wrapper, because `CastTo64` isn't scriptable.
- **TorchANI.** The model returns energies only; forces come from `torch.autograd.grad`. Batches of different-sized molecules are padded with species `-1`.
- **X-MACE.** The inner model computes every state (`energy [B, n_states]`, `forces [N, n_states, 3]`) and the wrapper returns the one chosen with `--state`. Its weights are float32 (MACE-OFF's are float64).

---

## Testing

```bash
python -m pytest tests/ -v                      # in the allegro env
```

Known failures, all left over from before the PBC interface:
- `TestE2E_SchNetPack::test_train_and_wrap` and `TestE2E_TorchANI::test_train_and_wrap` call `forward()` without the `cell` argument.
- The four `*_non_periodic_path_unchanged_vs_head` tests (`test_pbc_virial.py`, `test_virial_nequip_ani.py`, `test_virial_xmace_schnet.py`) compare against `git show HEAD:…` on the assumption that HEAD predates PBC. That stopped being true once the PBC wrappers were committed.

The unit tests use **mock inner models**, so they check the interface without any trained weights. They cover shapes, dtypes, batching, PBC and the virial, and that each wrapper survives `torch.jit.script`. The integration tests do a real train → wrap and are what catches version-pin conflicts. They skip NequIP/Allegro when `e3nn < 0.6`, which is intentional.

```bash
bash tests/run_nequip_tests.sh                  # NequIP/Allegro in their own env
bash tests/run_e2e_water_dimer.sh [--skip-orca] # ORCA -> train every model -> wrap
JAX_PLATFORMS=cpu conda run -n fennix python -m pytest tests/test_fennix_export.py tests/test_fennix_pbc.py -v
```

---

## Adding a new model

See **[`src/wrappers/README.md`](src/wrappers/README.md)**. It covers:
- what NAMD's C++ side calls and expects;
- a complete, tested template wrapper (forces and virial by autograd, batching, PBC);
- the TorchScript rules that break wrappers;
- registration, tests, loading the model through NAMD's shim, and how to make it fast.

---

## Repository layout

```
src/
  cli.py                 wrap a compiled model for NAMD (step 2)
  compile_mace_off.py    pretrained MACE-OFF   -> TorchScript (step 1)
  compile_torchani.py    pretrained ANI-*      -> TorchScript (step 1)
  compile_schnetpack.py  random-weight SchNet  -> TorchScript (timing only)
  wrappers/              one wrapper per framework; README.md = how to write a new one
  edges.py, nl_vesin.py  neighbour lists (FP32, PBC-aware)
  virial.py              shared virial normalisation
  constants.py           unit conversion factors
  export.py              torch.jit.script + save + diagnostics
  training/              prepare_data + train_<model> + default configs
  datagen/               ORCA data generation (pure QM, and QM/MM in qmmm/)
scripts/
  export_fennix_bio1_stablehlo.py   FeNNiX -> StableHLO
  opt/                              optimised builds, benchmarks, NAMD patches
namd_benchmarks/         NAMD benchmark suite (water boxes, enzyme), env.sh, config template
cpp/pjrt_shim/           C++ PJRT runtime used for FeNNiX
models/                  compiled and wrapped model artifacts
tests/
```
