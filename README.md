# ML_models_NAMD

Wrappers that make popular **machine-learning interatomic potentials** (MLIPs) compatible with [NAMD](https://www.ks.uiuc.edu/Research/namd/).

Each wrapper takes a trained model, translates NAMD's calling convention into the model's native input format, runs inference, and returns energies, forces, and charges in a **single, standardised output format** that the NAMD C++ engine can consume directly via TorchScript.

## Supported Models

| Model | Wrapper | Native Units | Builds Own Edges? | Deploy Tool |
|---|---|---|---|---|
| [MACE](https://github.com/ACEsuit/mace) | `MACE_TS_Wrapper` | eV | No | `mace` tools → TorchScript |
| [NequIP](https://github.com/mir-group/nequip) | `NequIP_Allegro_Wrapper` | eV | No | `nequip-deploy build` / `nequip-compile` |
| [Allegro](https://github.com/mir-group/allegro) | `NequIP_Allegro_Wrapper` | eV | No | `nequip-deploy build` |
| [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack) (≥ 2.0) | `SchNetPack_Wrapper` | eV (configurable) | No | `torch.jit.script` |
| [TorchANI](https://github.com/aiqm/torchani) | `TorchANI_Wrapper` | Hartree | Yes (internal) | `torch.jit.script` |
| [X-MACE](https://github.com/rhyan10/X-MACE) (excited states) | `XMACE_TS_Wrapper` | eV | No | `e3nn.util.jit.compile` |

## Standardised Output Contract

Every wrapper is a `torch.nn.Module` that exposes two methods and one attribute:

```
forward(coords, Z, pc_coords, pc_charges)        → (energy, forces, charges)
forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges) → (energies, forces, charges)
supports_batch: bool
```

| Output | Shape | Dtype | Units |
|---|---|---|---|
| `energy` / `energies` | scalar or `[B]` | `float64` | kcal/mol |
| `forces` | `[N, 3]` or `[N_total, 3]` | `float64` | kcal/mol/Å |
| `charges` | `[N]` or `[N_total]` | `float64` | e (elementary charge) |

All unit conversion (eV → kcal/mol, Hartree → kcal/mol, etc.) is fused into the wrapper and runs on-device.

Models that do not predict charges return zeros.  Point-charge arguments (`pc_coords`, `pc_charges`) are accepted for interface compatibility but currently ignored by all wrappers.

---

## Installation

```bash
# Clone
git clone <repo-url>
cd ML_models_NAMD

# Core only (torch)
pip install -e .

# With specific model support
pip install -e ".[mace]"
pip install -e ".[nequip]"
pip install -e ".[allegro]"
pip install -e ".[schnet]"
pip install -e ".[torchani]"

# Everything (all models + test deps)
pip install -e ".[all]"
```

> **The five frameworks cannot live in one Python env.** `mace-torch 0.3.x` pins `e3nn 0.4.4` while NequIP ≥ 0.6 needs `e3nn ≥ 0.6.0`; X-MACE pins `e3nn 0.5.1`. We use separate conda envs (`MACE_312`, `nequip_env` / `allegro`, `MLIP_2026`, `x_mace`) and dispatch the right interpreter per model. The exact paths are recorded in `.claude/settings.local.json` and noted in `CLAUDE.md`.

### X-MACE (special case)

X-MACE upstream is not TorchScript-clean as published. A patched copy of [rhyan10/X-MACE @ X-MACE_socs](https://github.com/rhyan10/X-MACE/tree/X-MACE_socs) lives in `~/x-mace-src` and is installed editable into the `x_mace` conda env. The patches fix:

- A `zip()` over five `ModuleList`s in `AutoencoderExcitedMACE.forward` that TorchScript silently drops (loop never runs).
- `socs_readouts.append(None)` when `compute_socs=False` — TorchScript cannot iterate a `ModuleList` containing `None`.
- Untyped accumulators in `forward` and bare `torch.tensor([])` placeholders that fail at runtime.
- A typo in `compute_forces` (`retain == False` instead of `=`) and missing `Optional[Tensor]` handling on `autograd.grad`.

Existing `*.model` files saved with `nac_indices=None`/`soc_indices=None` need a runtime fix before `e3nn.util.jit.compile`:

```python
m = torch.load("fulvene.model", map_location="cpu", weights_only=False).eval()
if m.nac_indices is None: m.nac_indices = 0
if m.soc_indices is None: m.soc_indices = 0
compiled = e3nn.util.jit.compile(deepcopy(m))
torch.jit.save(compiled, "fulvene_compiled.pt")
```

## Quick Start

### 1. Prepare your model

Each MLIP framework has its own way of producing a deployable artifact:

**MACE** — compile with MACE tools:
```bash
# Produces a TorchScript .pt file
mace_run_train --save_model_as compiled ...
```

**NequIP / Allegro** — deploy with `nequip-deploy`:
```bash
nequip-deploy build --train-dir /path/to/training deployed_model.pth
```

**SchNetPack** — script the trained model:
```python
import torch
model = torch.load("best_model.pth")
scripted = torch.jit.script(model)
scripted.save("schnet_scripted.pt")
```

**TorchANI** — use the bundled compiler for a built-in foundation model, or script a custom one yourself:
```bash
python -m src.compile_torchani --variant ani2x --out compiled_ani2x.pt
```

**X-MACE** — compile from a saved `*.model` (see the X-MACE installation section above for the patched-source requirement):
```python
import torch
from copy import deepcopy
from e3nn.util import jit
m = torch.load("fulvene.model", map_location="cpu", weights_only=False).eval()
if m.nac_indices is None: m.nac_indices = 0
if m.soc_indices is None: m.soc_indices = 0
torch.jit.save(jit.compile(deepcopy(m)), "fulvene_compiled.pt")
```

> **Foundation-model shortcut.** `src/compile_mace_off.py`, `src/compile_schnetpack.py`, and `src/compile_torchani.py` take a pretrained checkpoint (e.g. `MACE-OFF23_medium.model`, ANI-2x) and emit the compiled artifact in a single step — handy when you don't want to train from scratch.

### 2. Export for NAMD

Use the unified CLI:

```bash
# MACE
python -m src.cli --model-type mace --compiled mace_compiled.pt --out mlff_model.pt

# NequIP
python -m src.cli --model-type nequip --compiled deployed.pth --out mlff_model.pt

# Allegro
python -m src.cli --model-type allegro --compiled deployed.pth --out mlff_model.pt

# SchNetPack (--r-max is required; must match training cutoff)
python -m src.cli --model-type schnet --compiled schnet_scripted.pt --r-max 5.0 --out mlff_model.pt

# TorchANI (--elements sets species order; default is ANI-2x: H,C,N,O,S,Cl)
python -m src.cli --model-type torchani --compiled ani2x_scripted.pt --out mlff_model.pt

# X-MACE (--state K selects which electronic state to expose; default 0 = ground)
python -m src.cli --model-type xmace --compiled fulvene_compiled.pt --state 0 --out mlff_model.pt
```

All commands produce a single `mlff_model.pt` TorchScript file that NAMD can load.

### 3. Use in NAMD

Point your NAMD configuration to the exported model:

```tcl
MLForce              on
MLForceModelFile     mlff_model.pt
```

---

## Project Structure

```
ML_models_NAMD/
├── pyproject.toml                  # Package metadata & optional deps
├── src/
│   ├── __init__.py
│   ├── cli.py                      # Unified CLI entry point
│   ├── constants.py                # Unit conversion factors
│   ├── edges.py                    # Shared edge/neighbor-list builders
│   ├── export.py                   # Shared TorchScript export logic
│   ├── compile_mace_off.py         # Compile MACE-OFF foundation model → .pt
│   ├── compile_schnetpack.py       # Build random SchNetPack → .pt (timing)
│   ├── compile_torchani.py         # Compile ANI-1x/1ccx/2x → .pt
│   ├── models/                     # Pre-compiled model artifacts
│   ├── datagen/
│   │   ├── __init__.py
│   │   ├── cli.py                  # Pure-QM data generation CLI
│   │   ├── orca_generator.py       # ORCA single-point driver
│   │   ├── sampling.py             # Geometry sampling utilities
│   │   └── qmmm/                   # QM/MM data generation
│   │       ├── __init__.py
│   │       ├── generator.py        # ORCA QM/MM driver + parsers
│   │       └── cli.py              # QM/MM CLI entry point
│   ├── training/
│   │   ├── __init__.py
│   │   ├── prepare_data.py         # XYZ → per-framework data formats
│   │   ├── train_mace.py           # MACE training + compilation
│   │   ├── train_nequip.py         # NequIP training + deploy
│   │   ├── train_allegro.py        # Allegro training + deploy
│   │   ├── train_schnetpack.py     # SchNetPack training + script
│   │   ├── train_torchani.py       # TorchANI training + script
│   │   └── configs/                # Template YAML configs
│   │       ├── mace_default.yaml
│   │       ├── nequip_default.yaml
│   │       └── allegro_default.yaml
│   └── wrappers/
│       ├── __init__.py
│       ├── wrap_compiled_mace.py   # MACE wrapper
│       ├── wrap_compiled_nequip.py # NequIP & Allegro wrapper (shared)
│       ├── wrap_schnetpack.py      # SchNetPack wrapper
│       ├── wrap_torchani.py        # TorchANI wrapper
│       └── wrap_xmace.py           # X-MACE wrapper (excited states)
└── tests/
    ├── conftest.py                 # Shared fixtures
    ├── test_interface_compliance.py # Parametrised tests across all wrappers
    ├── test_datagen.py             # Pure-QM data generation tests
    ├── test_qmmm.py                # QM/MM data generation tests
    ├── test_pipeline_integration.py # Full train → wrap pipeline tests
    ├── test_e2e_water_dimer.py     # End-to-end ORCA → train → wrap pipeline
    ├── run_e2e_water_dimer.sh      # Shell wrapper (loads ORCA module)
    └── run_nequip_tests.sh         # NequIP/Allegro tests in their own env
```

### Shared Modules

| Module | Purpose |
|---|---|
| `constants.py` | `EV_TO_KCAL`, `HARTREE_TO_KCAL`, and other conversion factors |
| `edges.py` | `build_edges()` and `build_edges_batched()` — O(N²) vectorised neighbor lists in FP32 |
| `export.py` | `export_wrapped()` — `torch.jit.script` + save + diagnostics |
| `cli.py` | `--model-type {mace,nequip,allegro,schnet,torchani,xmace}` dispatcher |

---

## Model-Specific Notes

### MACE

- Input dict uses MACE-specific keys (`positions`, `atomic_numbers`, `node_attrs`, `edge_vectors`, `edge_lengths`, `shifts`, `cell`, `batch`, `ptr`, `num_nodes`).
- Requires one-hot `node_attrs` encoding (built automatically by the wrapper).
- Edge computation runs in FP32 for throughput; model runs in FP64.
- Constant tensors (batch, ptr, cell, node_attrs) are cached after the first call.

### NequIP & Allegro

- Both use the same deployed-model I/O convention (NequIP `AtomicData` dict), so they share a single wrapper class.
- Input dict uses keys: `pos`, `edge_index`, `atom_types`, `edge_cell_shift`, `cell`, `batch`, `ptr`.
- `atom_types` are **0-indexed type IDs** (not raw atomic numbers). The wrapper builds a Z → type-index lookup from the deployed model's metadata.
- Deployed via `nequip-deploy build`.

### SchNetPack (≥ 2.0)

- Input dict uses SchNetPack keys: `_positions`, `_atomic_numbers`, `_idx_i`, `_idx_j`, `_offsets`, `_cell`, `_n_atoms`, `_idx_m`.
- The cutoff radius (`--r-max`) must be provided explicitly and **must match** the training cutoff.
- Energy/forces output dict keys are configurable (`--energy-key`, `--forces-key`) for models with custom output head names.
- Default unit assumption is eV, but can be overridden via the `energy_units_to_kcal` constructor parameter for models trained in other unit systems.

### TorchANI

- The AEV computer builds its own neighbor list internally; the shared `build_edges` functions are **not** used.
- Forces are computed via `torch.autograd.grad` (the model only returns energies).
- Native units are **Hartree** (converted via `HARTREE_TO_KCAL = 627.509474`).
- Species mapping (`--elements`) must match the model's expected order (default: ANI-2x `[H, C, N, O, S, Cl]`).
- Batched evaluation pads variable-size molecules to equal length with species index `-1`.

### X-MACE (excited states)

- Wraps `AutoencoderExcitedMACE` from [rhyan10/X-MACE](https://github.com/rhyan10/X-MACE). Inner model returns `energy: [B, n_states]` and `forces: [N, n_states, 3]` (per-state forces are computed inside the inner model via `torch.autograd.grad` per state).
- The wrapper exposes a **single** electronic state via `--state K` (default 0 = ground state); slicing happens after inference, so all states are still computed.
- Inner weights are **float32** (unlike MACE-OFF, which is float64). The wrapper feeds float32 `positions`/`node_attrs`/`shifts`/`cell` and casts the model output to float64 kcal/mol.
- Input dict uses MACE-style keys (`positions`, `atomic_numbers`, `node_attrs`, `edge_index`, `shifts`, `cell`, `batch`, `ptr`) — no `edge_vectors`/`edge_lengths`/`num_nodes` (X-MACE recomputes them internally).
- Compiling X-MACE from `*.model` requires the patched source at `~/x-mace-src` (see Installation → X-MACE). Don't `pip install mace-torch` over the editable install — it would clobber the patches.

---

## Training Pipeline

The `src/training/` package provides a complete **data → train → deploy → wrap** workflow.  Starting from a single extended XYZ file, you can train any supported model and produce a NAMD-ready `.pt` file.

### End-to-End Workflow

```
data.xyz ──→ prepare_data.py ──→ train_<model>.py ──→ src.cli --model-type <model>
                │                       │                        │
                ▼                       ▼                        ▼
         prepared_data/          trained model             mlff_model.pt
         ├── xyz/                (.pt or .pth)            (NAMD-ready)
         ├── schnetpack/
         └── torchani/
```

### Step 1: Prepare Data

All training scripts expect data produced by `prepare_data.py`.  It reads an extended XYZ file (with energy and forces in the standard `info`/`arrays` fields), splits it into train/val/test, and writes all formats at once:

```bash
python -m src.training.prepare_data \
    --xyz data.xyz \
    --output-dir ./prepared_data \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
    --seed 42
```

This creates:
```
prepared_data/
├── xyz/                  # Extended XYZ (MACE, NequIP, Allegro)
│   ├── train.xyz
│   ├── val.xyz
│   └── test.xyz
├── schnetpack/           # ASE DB (SchNetPack)
│   ├── train.db
│   ├── val.db
│   └── test.db
└── torchani/             # HDF5 with Hartree units (TorchANI)
    └── data.h5
```

The script auto-detects energy/forces keys (supports `energy`, `Energy`, `REF_energy`, `dft_energy` and `forces`, `REF_forces`).  The XYZ is expected to have energies in **eV** and forces in **eV/Å** — TorchANI's HDF5 is automatically converted to Hartree.

### Step 2: Train

Each training script reads from `prepared_data/`, trains the model, and saves a deployable artifact.  Every script prints the exact command to wrap the model for NAMD at the end.

#### MACE

```bash
python -m src.training.train_mace \
    --data-dir ./prepared_data \
    --output-dir ./results/mace \
    --r-max 5.0 \
    --max-epochs 200 \
    --device cuda
```

Calls `mace_run_train` under the hood with sensible defaults (ScaleShiftMACE, correlation=3, hidden_irreps=128x0e+128x1o).  Override any parameter via `--config custom.yaml` or individual flags (`--lr`, `--batch-size`, `--forces-weight`, etc.).

Outputs `results/mace/mace_compiled.pt`.

#### NequIP

```bash
python -m src.training.train_nequip \
    --data-dir ./prepared_data \
    --output-dir ./results/nequip \
    --chemical-symbols H C N O \
    --r-max 5.0 \
    --max-epochs 200
```

Generates a YAML config from the template, runs `nequip-train`, then `nequip-deploy build`.  Chemical symbols are auto-detected from the data if omitted.

Outputs `results/nequip/nequip_deployed.pth`.

#### Allegro

```bash
python -m src.training.train_allegro \
    --data-dir ./prepared_data \
    --output-dir ./results/allegro \
    --chemical-symbols H C N O \
    --r-max 5.0 \
    --max-epochs 200
```

Same workflow as NequIP but with the Allegro model architecture (pair-wise equivariant layers).

Outputs `results/allegro/allegro_deployed.pth`.

#### SchNetPack

```bash
python -m src.training.train_schnetpack \
    --data-dir ./prepared_data \
    --output-dir ./results/schnetpack \
    --r-max 5.0 \
    --n-atom-basis 128 \
    --n-interactions 3 \
    --max-epochs 200 \
    --device cuda
```

Uses the SchNetPack Python API with PyTorch Lightning.  Builds a SchNet representation + Atomwise energy head + Forces derivative.  Trains, loads the best checkpoint, and scripts the model.

Outputs `results/schnetpack/schnet_scripted.pt`.

#### TorchANI

```bash
python -m src.training.train_torchani \
    --data-dir ./prepared_data \
    --output-dir ./results/torchani \
    --elements H C N O \
    --Rcr 5.2 --Rca 3.5 \
    --max-epochs 200 \
    --device cuda
```

Builds a TorchANI model from scratch (AEV computer + per-element networks), trains with a PyTorch loop (Adam + joint energy/forces loss), and scripts the result.

Outputs `results/torchani/torchani_scripted.pt` + `metadata.txt` with the element order.

### Step 3: Wrap for NAMD

Each training script prints the wrapping command at the end.  For example:

```bash
python -m src.cli --model-type mace    --compiled results/mace/mace_compiled.pt     --out mlff_model.pt
python -m src.cli --model-type nequip  --compiled results/nequip/nequip_deployed.pth --out mlff_model.pt
python -m src.cli --model-type allegro --compiled results/allegro/allegro_deployed.pth --out mlff_model.pt
python -m src.cli --model-type schnet  --compiled results/schnetpack/schnet_scripted.pt --r-max 5.0 --out mlff_model.pt
python -m src.cli --model-type torchani --compiled results/torchani/torchani_scripted.pt --elements 1,6,7,8 --out mlff_model.pt
```

### Training Configuration

All training scripts support three levels of configuration (highest priority wins):

1. **Default template** — sensible defaults in `src/training/configs/*.yaml`
2. **Custom YAML** — pass `--config my_config.yaml` to override any default
3. **CLI flags** — `--r-max`, `--max-epochs`, `--batch-size`, `--lr`, etc.

The default configs are intentionally conservative (small models, moderate epochs) for quick iteration.  Scale up via config overrides for production training.

### Data Format Requirements

The input extended XYZ file should follow the standard ASE convention:

```
3
energy=-76.4 pbc="F F F"
O    0.000   0.000   0.117   forces="0.0 0.0 -0.1"
H    0.000   0.757  -0.469   forces="0.0 0.3  0.05"
H    0.000  -0.757  -0.469   forces="0.0 -0.3  0.05"
```

- **Energy**: stored in the `info` dict (comment line) under key `energy` (also accepts `Energy`, `REF_energy`, `dft_energy`). Units: **eV**.
- **Forces**: stored in `arrays` under key `forces` (also accepts `REF_forces`). Units: **eV/Å**.
- Multiple frames are concatenated in a single file.

---

## QM/MM Data Generation (Explicit Solvent)

The `src/datagen/qmmm/` package generates training data for **explicit-solvent** systems using ORCA's QM/MM mode with electrostatic embedding.  Given a trajectory of full-system geometries (solute + solvent), it runs single-point + gradient calculations where a user-defined QM region is treated with DFT and the remaining atoms are described by an Amber-derived force field.

### Prerequisites

* **ORCA 6** (with `orca` and `orca_mm` on `$PATH` or specified via `--orca-command` / `--orca-mm-command`)
* An **Amber `.prmtop` topology** and `.inpcrd` / `.rst7` coordinate file for the full system
* An **XYZ trajectory** where every frame has the same atom count and ordering as the topology

### Workflow Overview

```
                                              ┌──────────────────────┐
system.prmtop ──→ orca_mm ──→ system.ORCAFF.prms     │                      │
system.rst7   ─┘                                      │  QMMMDataGenerator   │
                                                      │                      │
trajectory.xyz ──────────────────────────────────────→ │  • writes ORCA input │
                                                      │  • runs ORCA QM/MM   │
                                                      │  • parses engrad +   │
                                                      │    Mulliken charges   │
                                                      └──────────┬───────────┘
                                                                 │
                                                                 ▼
                                                         qmmm_data.xyz
                                                   (QM atoms + energy/forces/
                                                    charges + PC metadata)
```

### Step 1: Convert Amber Topology

Convert the Amber topology to ORCA's force-field format using the `orca_mm` utility:

```bash
# Manual conversion (if you prefer to do it yourself)
orca_mm --amber2orca system.prmtop system.rst7

# Or let the CLI do it automatically (pass both --amber-prmtop and --amber-inpcrd)
```

### Step 2: Run QM/MM Calculations

#### CLI

```bash
# Using a pre-converted ORCA FF file:
python -m src.datagen.qmmm.cli \
    --input trajectory.xyz \
    --output qmmm_data.xyz \
    --method "B3LYP def2-SVP" \
    --n-qm-atoms 6 \
    --orcaff-file system.ORCAFF.prms \
    --amber-prmtop system.prmtop \
    --n-workers 8 \
    --orca-nprocs 2

# With automatic topology conversion:
python -m src.datagen.qmmm.cli \
    --input trajectory.xyz \
    --output qmmm_data.xyz \
    --method "B3LYP def2-SVP" \
    --n-qm-atoms 6 \
    --amber-prmtop system.prmtop \
    --amber-inpcrd system.rst7 \
    --n-workers 8

# With explicit (non-contiguous) QM atom indices:
python -m src.datagen.qmmm.cli \
    --input trajectory.xyz \
    --output qmmm_data.xyz \
    --method "wB97X-D3 def2-TZVP" \
    --qm-indices "0-5,12,15-17" \
    --orcaff-file system.ORCAFF.prms \
    --n-workers 4
```

#### Python API

```python
from src.datagen.qmmm import QMMMDataGenerator, convert_amber_topology

# Optional: convert topology (or pass --orcaff-file directly)
ff_path = convert_amber_topology(
    "system.prmtop", "system.rst7", output_dir="./orcaff"
)

gen = QMMMDataGenerator(
    method="B3LYP def2-SVP",
    n_qm_atoms=6,                     # first 6 atoms are QM
    orcaff_file=ff_path,
    amber_prmtop="system.prmtop",     # for MM charge metadata
    charge_qm=0,
    mult_qm=1,
    orca_nprocs=2,
)

# Run on a trajectory — distributes across 8 worker processes
gen.run("trajectory.xyz", "qmmm_data.xyz", n_workers=8)
```

### QM Region Specification

The QM region can be defined in two ways:

| CLI flag | Python argument | Meaning |
|---|---|---|
| `--n-qm-atoms 6` | `n_qm_atoms=6` | First 6 atoms in each frame are QM |
| `--qm-indices "0-5,12"` | `qm_indices=[0,1,2,3,4,5,12]` | Explicit 0-based atom indices |

The remaining atoms are treated as the MM region using the force-field parameters from the Amber topology.

### Output Format

The output is an extended XYZ file containing **only the QM atoms**, with:

| Field | Location | Units | Description |
|---|---|---|---|
| `energy` | `info["energy"]` | eV | Total QM/MM energy |
| `forces` | `arrays["forces"]` | eV/Å | Forces on QM atoms |
| `charges` | `arrays["charges"]` | e | Mulliken charges on QM atoms (if available) |
| `pc_N` | `info["pc_N"]` | — | Number of MM point charges |
| `pc_charges` | `info["pc_charges"]` | e | MM partial charges (from Amber topology) |
| `pc_positions` | `info["pc_positions"]` | Å | MM atom positions (flat: x₁ y₁ z₁ x₂ …) |

This format is directly consumable by `prepare_data.py` for training.  The point-charge metadata is preserved so that ML models can learn the electrostatic environment.

### QM/MM CLI Reference

```
python -m src.datagen.qmmm.cli --input TRAJ --output OUT --n-qm-atoms N --orcaff-file FF [OPTIONS]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | ✅ | — | Input XYZ with full-system geometries |
| `--output` | ✅ | — | Output extended XYZ (QM atoms + metadata) |
| `--n-qm-atoms` | ✅* | — | Number of QM atoms (first N) |
| `--qm-indices` | ✅* | — | Explicit QM indices (e.g. `"0-5,12"`) |
| `--method` | | `B3LYP def2-SVP` | ORCA method line |
| `--orcaff-file` | ✅† | — | Pre-converted `.ORCAFF.prms` file |
| `--amber-prmtop` | | — | Amber `.prmtop` (for charges + conversion) |
| `--amber-inpcrd` | ✅† | — | Amber `.inpcrd` / `.rst7` (for conversion) |
| `--charge-total` | | `0` | Total system charge |
| `--charge-qm` | | `0` | QM region charge |
| `--mult-qm` | | `1` | QM spin multiplicity |
| `--orca-command` | | `orca` | Path to ORCA binary |
| `--orca-nprocs` | | `1` | MPI procs per ORCA run |
| `--n-workers` | | `1` | Parallel frame calculations |
| `--extra-blocks` | | — | Additional ORCA input blocks |

\* One of `--n-qm-atoms` or `--qm-indices` is required (mutually exclusive).
† Either `--orcaff-file`, or both `--amber-prmtop` and `--amber-inpcrd` for automatic conversion.

---

## CLI Reference

```
python -m src.cli --model-type TYPE --compiled PATH [OPTIONS]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model-type` | ✅ | — | `mace`, `nequip`, `allegro`, `schnet`, `torchani`, or `xmace` |
| `--compiled` | ✅ | — | Path to compiled/deployed model file |
| `--out` | | `mlff_model.pt` | Output TorchScript file |
| `--device` | | `cpu` | Device to load onto (`cpu` or `cuda`) |
| `--r-max` | schnet only | — | Cutoff radius in Å (also accepted as a NequIP fallback when the deployed model has no `r_max` attribute, e.g. NequIP-OAM-L: `--r-max 6.0`) |
| `--energy-key` | | `energy` | SchNetPack output dict key for energy |
| `--forces-key` | | `forces` | SchNetPack output dict key for forces |
| `--elements` | | `1,6,7,8,16,9,17` | TorchANI species order (atomic numbers; ANI-2x = H C N O S F Cl) |
| `--state` | | `0` | X-MACE electronic state index to expose (0 = ground) |

---

## Testing

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

Tests use **mock inner models** (no real MLIP weights needed) to verify that every wrapper:

- Returns the correct output shapes and dtypes (`float64`).
- Has `supports_batch = True`.
- Produces correct batch output dimensions.
- Can survive `torch.jit.script()`.

Edge-building utilities and unit conversion constants are also tested independently.

NequIP/Allegro pipeline tests need their own conda env (`nequip_env`) — run with the helper:

```bash
bash tests/run_nequip_tests.sh
```

The end-to-end ORCA → train → wrap pipeline (loads `module load orca`, generates 20 water dimers, trains every model, wraps for NAMD):

```bash
bash tests/run_e2e_water_dimer.sh                # full
bash tests/run_e2e_water_dimer.sh --skip-orca    # reuse cached data
```

---

## Adding a New Model

1. Create `src/wrappers/wrap_<name>.py` with a class that inherits from `nn.Module`.
2. Implement `forward(coords, Z, pc_coords, pc_charges) → (energy, forces, charges)`.
3. Implement `forward_batch(...)` decorated with `@torch.jit.export`.
4. Set `self.supports_batch = True`.
5. Convert outputs to **kcal/mol** (`float64`) and return zero charges if the model doesn't predict them.
6. Use `build_edges()` / `build_edges_batched()` from `src/edges.py` if the model needs external neighbor lists.
7. Add the wrapper to `src/wrappers/__init__.py` and `src/cli.py`.
8. Add a mock inner model and builder to `tests/test_interface_compliance.py` — the parametrised tests will pick it up automatically.
9. Add an optional dependency group in `pyproject.toml`.

