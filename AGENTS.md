# AGENTS.md

## Purpose
- This repo wraps multiple ML interatomic potentials behind one TorchScript interface so NAMD can always load `mlff_model.pt`.
- The design center is the wrapper contract; most code exists to preserve interface + units across model families.

## Non-negotiable wrapper contract
- Every wrapper in `src/wrappers/` must expose:
  - `forward(coords, Z, pc_coords, pc_charges) -> (energy, forces, charges)`
  - `forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges) -> (energies, forces, charges)`
  - `supports_batch: bool = True`
- Outputs must be `float64` in kcal/mol, kcal/mol/A, and e; conversion factors live in `src/constants.py`.
- `pc_coords` / `pc_charges` are accepted for compatibility; current wrappers ignore them.
- Export path is always through `src/export.py:export_wrapped()` (`torch.jit.script` + save). Do not remove TorchScript.

## Architecture map (where to look first)
- `src/cli.py`: dispatcher for `--model-type {mace,nequip,allegro,schnet,torchani,xmace}` and model-specific args (`--r-max`, `--energy-key`, `--forces-key`, `--elements`, `--state`).
- `src/wrappers/wrap_*.py`: adapter layer from NAMD contract to each model's native input/output schema.
- `src/edges.py`: shared O(N^2) FP32 neighbor lists used by all wrappers except TorchANI.
- `src/datagen/cli.py` + `src/datagen/qmmm/`: ORCA-backed pure-QM and QM/MM data generation; both produce extended XYZ that `src/training/prepare_data.py` can ingest.
- `src/training/prepare_data.py` + `src/training/train_*.py`: data conversion and framework-specific train/deploy pipelines.
- `src/compile_mace_off.py`, `src/compile_schnetpack.py`, `src/compile_torchani.py`: foundation-model compile shortcuts.

## Data flow to keep in mind
- `data.xyz -> prepare_data.py -> prepared_data/{xyz,schnetpack,torchani} -> train_<model>.py -> compiled artifact -> src.cli -> mlff_model.pt`.
- `src.datagen.cli` (pure QM) and `src.datagen.qmmm.cli` (QM/MM from trajectory + Amber topology / ORCAFF) both feed that same path by writing extended XYZ for `src/training/prepare_data.py`.
- Registration points for any new wrapper/model type: `src/wrappers/__init__.py` and `src/cli.py`.

## Testing and validation workflow
- Contract tests are weight-free mocks in `tests/test_interface_compliance.py`; keep them mock-based.
- Integration path is in `tests/test_pipeline_integration.py`; environment/version conflicts surface here.
- ORCA-facing data-generation coverage is also mock-based in `tests/test_datagen.py` and `tests/test_qmmm.py`; use those when touching sampling, ORCA parsing, or QM/MM plumbing.
- Typical checks:
  - `python -m pytest tests/test_interface_compliance.py -v`
  - `python -m pytest tests/test_datagen.py tests/test_qmmm.py -v`
  - `python -m pytest tests/ -v`
  - `bash tests/run_nequip_tests.sh` (NequIP/Allegro env split)
  - `bash tests/run_e2e_water_dimer.sh --skip-orca` (reuse generated data)

## Project-specific gotchas
- Multi-env setup is intentional (`MACE_312`, `nequip_env`/`allegro`, `MLIP_2026`, `x_mace`); do not assume one env can run all models.
- PyTorch >=2.6 + `e3nn 0.4.4`: keep the `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` subprocess mitigation and `tests/conftest.py` torch.load patch.
- NequIP/Allegro wrapper expects 0-indexed `atom_types` built from deployed metadata (`src/training/_nequip_metadata_wrapper.py`), not raw Z; some deployed models still need CLI `--r-max` as a fallback when the artifact does not expose `r_max`.
- SchNetPack requires `--r-max` to match training cutoff; wrappers may also need `--energy-key` / `--forces-key` when the scripted model uses non-default output names.
- TorchANI builds its own AEV neighbor list and computes forces via autograd; it does not use `src/edges.py`.
- X-MACE requires patched source in `~/x-mace-src`; wrapper exposes one state via `--state` but inner model computes all states.
- `src/datagen/qmmm/generator.py` writes QM-only extended XYZ with `info['pc_N']`, `info['pc_charges']`, and `info['pc_positions']`; `prepare_data.py` can split it, but current wrappers still ignore `pc_coords` / `pc_charges` at inference time.

## When adding a new model
- Implement `src/wrappers/wrap_<name>.py` with exact contract and unit conversion.
- Add CLI wiring in `src/cli.py` and export in `src/wrappers/__init__.py`.
- Add mock inner model + builder to `tests/test_interface_compliance.py` parametrization.
- Add optional dependency group in `pyproject.toml`.

