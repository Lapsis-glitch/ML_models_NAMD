# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Wrappers around six ML interatomic potentials (MACE, NequIP, Allegro, SchNetPack, TorchANI, X-MACE) that expose a single, fixed TorchScript interface so a NAMD C++ engine can load any of them as `mlff_model.pt`. Everything in this repo exists to enforce that contract.

## The wrapper contract (the design constraint everything else serves)

Every wrapper in `src/wrappers/` is a `nn.Module` exposing exactly:

```
forward(coords, Z, pc_coords, pc_charges, cell)
    -> (energy, forces, charges, virial)
forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
    -> (energies, forces, charges, virials)
supports_batch: bool = True
supports_pbc:   bool = True
```

- Outputs are always **kcal/mol**, **kcal/mol/Å**, **e**, **kcal/mol** (virial), in **float64**, regardless of the inner model's native units. Conversion is fused into the wrapper using factors from `src/constants.py` (e.g. `EV_TO_KCAL`, `HARTREE_TO_KCAL`).
- `cell` is `[1,3,3]` (`cells` `[B,3,3]`), lattice vectors as rows, Å. **An all-zero cell means non-periodic** (`src/edges.py:cell_is_periodic`, determinant test). Periodic edges come from `build_edges_pbc` / `build_edges_batched_pbc`.
- The virial is `[3,3]` / `[B,3,3]`, symmetric, `W = -dE/d(strain) = Σ r_i ⊗ f_i`, zeros when non-periodic. Every wrapper normalises through `src/virial.py` so NAMD sees one shape/unit/sign regardless of how the model produced it (MACE/X-MACE native, others via the strain trick). Sign/normalisation are pinned by finite-difference tests (`tests/test_pbc_virial.py`, `test_virial_*.py`).
- Models that don't predict charges return zeros. `pc_coords`/`pc_charges` are accepted but currently ignored by all wrappers (interface compatibility only).
- The whole module is `torch.jit.script`-ed (or traced as fallback) by `src/export.py` because NAMD only loads TorchScript. Deprecation warnings about `torch.jit` are suppressed in `pyproject.toml` — do not "fix" by removing TorchScript.

When adding a wrapper, follow this contract exactly; the parametrised tests in `tests/test_interface_compliance.py` pick up new wrappers automatically once registered in `src/wrappers/__init__.py` and `src/cli.py`.

## High-level data flow

```
data.xyz → src/training/prepare_data.py → prepared_data/{xyz,schnetpack,torchani}/
         → src/training/train_<model>.py → trained artifact (.pt or .pth)
         → src/cli.py --model-type <model> → mlff_model.pt   (NAMD-ready)
```

`src/compile_{mace_off,schnetpack,torchani}.py` are an alternative entry point: they take a **pretrained foundation model** (e.g. `MACE-OFF23_medium.model`, `ANI2x`) and produce the compiled artifact that `src/cli.py` then wraps. Use these when you don't want to train from scratch.

Shared utilities used by all wrappers:

- `src/edges.py` — FP32 neighbor lists: O(N²) vectorised (`build_edges`, `build_edges_batched`), periodic variants (`*_pbc`), zero-cell short-circuit, opt-in cell list. `src/nl_vesin.py` is an alternative backend. Used by every wrapper *except* TorchANI, which has its own AEV neighbor list.
- `src/virial.py` — shared virial normalisation (see contract).
- `src/export.py` — `export_wrapped()` does scripting + save + parity diagnostics.

## Common commands

Install (per model):
```bash
pip install -e ".[mace]"      # or [nequip], [allegro], [schnet], [torchani], [all], [test]
```

Run all tests (uses mock inner models — no real weights required):
```bash
python -m pytest tests/ -v
```

Run a single test / class:
```bash
python -m pytest tests/test_interface_compliance.py -v
python -m pytest tests/test_pipeline_integration.py::TestNequIPPipeline -v
```

NequIP/Allegro tests must run in their own conda env (see "Environment splits" below):
```bash
bash tests/run_nequip_tests.sh
```

End-to-end ORCA → train → wrap pipeline (loads `module load orca`):
```bash
bash tests/run_e2e_water_dimer.sh                  # full
bash tests/run_e2e_water_dimer.sh --skip-orca      # reuse data
```

Wrap any compiled model for NAMD:
```bash
python -m src.cli --model-type {mace|nequip|allegro|schnet|torchani|xmace} \
    --compiled <path> --out mlff_model.pt
# schnet additionally requires --r-max (must match training cutoff)
# torchani uses --elements (default 1,6,7,8,16,9,17 = ANI-2x order H C N O S F Cl)
# xmace uses --state K to select which electronic state (default 0 = ground)
```

## Environment splits (important — non-obvious)

This project cannot be installed into a single Python env. Inspect `.claude/settings.local.json` for the canonical interpreter paths actually used:

- **MACE_312** — MACE training/compile. Pins `e3nn 0.4.4` (transitive via `mace-torch 0.3.x`), which is incompatible with NequIP ≥ 0.6.
- **allegro** / **nequip_env** — NequIP/Allegro. Needs `e3nn ≥ 0.6.0` (NequIP 0.17 requirement). Also used for TorchANI, SchNetPack, wrapping, the test suite and all `scripts/opt/` builds. `torch 2.11+cu130` — the **only env with working CUDA on the RTX 5080 (sm_120)**; GPU runs need `$SP/nvidia/cu13/lib` on `LD_LIBRARY_PATH` (NVRTC; without it cuEquivariance silently falls back).
- **fennix** — JAX + FeNNol, for the FeNNiX StableHLO export only.
- **MLIP_2026** — newer torch + e3nn experimentation env.
- **x_mace** — X-MACE (rhyan10/X-MACE, branch `X-MACE_socs`). Source lives at `~/x-mace-src` (editable install) and contains local TorchScript-compatibility patches; do **not** `pip install mace-torch` over it. Pins `e3nn 0.5.1`, `torch 2.2`, `numpy<2`. Used by `wrap_xmace.py` and to recompile `*.model` → `*_compiled.pt`.

Consequence: `tests/test_pipeline_integration.py` skips NequIP/Allegro when `e3nn < 0.6.0` is detected. That skip is intentional, not a bug.

## PyTorch ≥ 2.6 / e3nn 0.4.4 compatibility

`e3nn 0.4.4`'s `o3/_wigner.py` calls `torch.load("constants.pt")` at import time without `weights_only=False`. PyTorch ≥ 2.6 defaults to `weights_only=True` and rejects the file.

Two mitigations are already in place — keep them:

1. **Subprocess training/compile** (`nequip-train`, `nequip-compile`): the training scripts set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in the child env. If you add a new subprocess that imports e3nn, propagate this var.
2. **In-process imports** (tests): `tests/conftest.py` monkey-patches `torch.load` to force `weights_only=False` only for `e3nn/.../constants.pt`. Don't remove this patch.

## Per-wrapper gotchas to remember when editing

- **MACE** — needs one-hot `node_attrs`; wrapper builds it. Edges in FP32, model in FP64. Constants (batch, ptr, cell, node_attrs) are cached after the first call — invalidate the cache if input topology changes.
- **NequIP/Allegro** — share `wrap_compiled_nequip.py`. `atom_types` are 0-indexed type IDs (not Z); the wrapper builds Z→type from deployed-model metadata (`_nequip_metadata_wrapper.py`). Current NequIP (0.17) packages with `nequip-package build` and compiles with `nequip-compile --mode torchscript`, which refuses on torch ≥ 2.10 — `scripts/opt/nequip/compile_ts.py` does the same without that check. NequIP-OAM-L has no `r_max` attribute: pass `--r-max 6.0`.
- **SchNetPack 2.x** — `--r-max` must match training cutoff. `CastTo64` postprocessor is removed (not TorchScript-compatible) — wrapper handles the float32→float64 cast. Energy/forces dict keys are configurable (`--energy-key`, `--forces-key`).
- **SchNetPack + ASE 3.28** — `SQLite3Database.metadata` is monkey-patched to open a temp connection when accessed outside a context manager (regression in ASE 3.28).
- **TorchANI** — builds AEV (and thus its own neighbor list) internally; does **not** use `src/edges.py`. Energies only — forces come from `torch.autograd.grad`. Native units are Hartree. Built via `torchani.arch.Assembler` (not the deprecated `torchani.nn.Sequential`). `_TorchANIExportWrapper` bridges the tuple-based forward to the (species, coords) interface. Batched eval pads to equal length using species `-1`. **Element order**: the ANI-2x default is `[1,6,7,8,16,9,17]` (H C N O S F Cl, torchani's order); an older default scored Cl with the F network — `tests/test_ani_element_order.py` guards it. `lean=True` is an opt-in low-overhead path (not exposed in `src/cli.py`).
- **X-MACE** — multi-state excited-state MACE variant (`AutoencoderExcitedMACE`). Inner model returns `energy: [B, n_states]` and `forces: [N, n_states, 3]` (per-state autograd, computed inside the inner model in `compute_forces`). Wrapper exposes one state via `state_idx` (`--state K`), slicing `[:, s]` for energy and `[:, s, :]` for forces. Inner weights are **float32** (unlike MACE-OFF which is float64), so `node_attrs`/`positions`/`shifts`/`cell` all go in as float32; output is cast to float64 kcal/mol. Compiling X-MACE from `*.model` requires the patched source at `~/x-mace-src` — upstream `AutoencoderExcitedMACE.forward` has TorchScript-incompatible `zip()` over 5 ModuleLists, untyped empty lists, `None` entries in `socs_readouts`, and `torch.tensor([])` placeholders that all silently break the scripted output. Existing `*.model` files saved with `nac_indices=None` need the runtime fix `m.nac_indices = 0; m.soc_indices = 0` before `e3nn.util.jit.compile`.

## FeNNiX (not TorchScript)

FeNNiX-BIO1 is JAX. `scripts/export_fennix_bio1_stablehlo.py` (`fennix` env) lowers a **fixed-shape** (atom count + composition) evaluator to StableHLO + `manifest.json`; NAMD runs it via its C++ PJRT backend (`QMSoftware fennol`, `QMExecPath manifest.json`). Same contract as the wrappers but outputs are eV (NAMD converts), plus an `overflow` flag for fixed neighbour-list capacity (`--nblist-margin`). `--pbc` adds a cell input and virial output (minimum image: box widths ≥ 2 × cutoff). Non-periodic exports default to TF32 matmuls; `--pbc` defaults to `highest`. Tests: `tests/test_fennix_export.py`, `tests/test_fennix_pbc.py` (fennix env, `JAX_PLATFORMS=cpu`).

## Optimised builds and NAMD integration

- `src/cli.py` produces the **reference** model. Faster variants (same weights, parity-checked) are built by `scripts/opt/<model>/` tools: MACE (cuEquivariance + FastMACE), NequIP (OpenEquivariance + FastNequIP), ANI-2x (cuAEV + FastANI), SchNet (`wrap_schnetpack --fast`). The README's "Guide: optimised builds" has the per-model steps (benchmark settings, reproduce `models/opt/` bit-for-bit); `scripts/opt/COMPARISON.md` + `scripts/opt/<model>/REPORT.md` have the numbers. `src/cli.py` wraps optimised inners too: `--extra-libs` (loads native op libs, same list as `NAMD_MLFF_EXTRA_LIBS`), `--fast` (SchNet), `--lean` (TorchANI). The `build_fast.py` parity checks cycle through all of the model's species on a water geometry, with tolerances relative to |F|max.
- NAMD loads models through a libtorch shim (`libnamd_mlff.so`) with **no Python**. Custom ops must be native `TORCH_LIBRARY` libraries (`scripts/opt/*/…_native/`), loaded via `NAMD_MLFF_EXTRA_LIBS=/a.so:/b.so` (NAMD patch 01). Never route custom ops through embedded Python.
- All NAMD source changes live as patches in `scripts/opt/namd_patches/` (01 shim extra libs + JIT knobs, 02 FeNNiX PJRT fast path, 03 ComputeQM fast index), with a README.
- NAMD config uses the QM interface: `QMForces on`, `QMSoftware mlff|fennol`, `QMExecPath <model>`, `QMColumn beta`, `qmParamPDB`. Template: `namd_benchmarks/templates/bench.conf.tmpl`; runtime env: `namd_benchmarks/env.sh`.

## Adding a new model

Full guide with a tested template wrapper: `src/wrappers/README.md` (NAMD shim expectations, strain virial, batching, TorchScript pitfalls, tests, shim loading, optimisation). Summary:

1. New wrapper in `src/wrappers/wrap_<name>.py` implementing the contract above (kcal/mol, float64, `cell` input, virial via `src/virial.py`, both `forward` and `@torch.jit.export forward_batch`).
2. Register in `src/wrappers/__init__.py` and `src/cli.py`.
3. Add a mock inner model + builder to `tests/test_interface_compliance.py` — the parametrised tests pick it up automatically.
4. New optional-deps group in `pyproject.toml`.

## Testing philosophy

`tests/test_interface_compliance.py` uses **mock inner models** so the contract can be checked without any trained MLIP weights. `tests/test_pipeline_integration.py` exercises the real train→wrap path and is what surfaces version-pin conflicts. Don't replace the mocks with real models — the unit suite must stay weight-free.

Known failures (all predate the PBC contract): `TestE2E_SchNetPack::test_train_and_wrap` and `TestE2E_TorchANI::test_train_and_wrap` call `forward()` without `cell`; the four `*_non_periodic_path_unchanged_vs_head` tests diff against `git show HEAD:` assuming HEAD is pre-PBC, which stopped being true at commit 41ba530.
