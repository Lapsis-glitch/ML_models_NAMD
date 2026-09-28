# QM/MM Data Generation (Explicit Solvent)

The `src/datagen/qmmm/` package generates training data for **explicit-solvent** systems using ORCA's QM/MM mode with electrostatic embedding.  Given a trajectory of full-system geometries (solute + solvent), it runs single-point + gradient calculations where a user-defined QM region is treated with DFT and the remaining atoms are described by an Amber-derived force field.

## Prerequisites

* **ORCA 6** (with `orca` and `orca_mm` on `$PATH` or specified via `--orca-command` / `--orca-mm-command`)
* An **Amber `.prmtop` topology** and `.inpcrd` / `.rst7` coordinate file for the full system
* An **XYZ trajectory** where every frame has the same atom count and ordering as the topology

## Workflow Overview

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

## Step 1: Convert Amber Topology

Convert the Amber topology to ORCA's force-field format using the `orca_mm` utility:

```bash
# Manual conversion (if you prefer to do it yourself)
orca_mm --amber2orca system.prmtop system.rst7

# Or let the CLI do it automatically (pass both --amber-prmtop and --amber-inpcrd)
```

## Step 2: Run QM/MM Calculations

### CLI

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

### Python API

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

## QM Region Specification

The QM region can be defined in two ways:

| CLI flag | Python argument | Meaning |
|---|---|---|
| `--n-qm-atoms 6` | `n_qm_atoms=6` | First 6 atoms in each frame are QM |
| `--qm-indices "0-5,12"` | `qm_indices=[0,1,2,3,4,5,12]` | Explicit 0-based atom indices |

The remaining atoms are treated as the MM region using the force-field parameters from the Amber topology.

## Output Format

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

## QM/MM CLI Reference

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

