# Shared brief for per-model optimisation agents

Goal: maximum INFERENCE speed (and honest memory numbers) for each ML potential as it runs inside NAMD,
WITHOUT changing the model itself (same weights, same architecture, same predictions within numerical
noise). Allowed levers: compilation/export path, TorchScript freezing/optimisation, cuEquivariance /
OpenEquivariance kernels, fused/custom CUDA ops (e.g. TorchANI cuAEV), neighbor-list implementation,
removing redundant work/syncs/allocations/dtype casts in the wrapper, batching, CUDA graphs, XLA flags, etc.

## Facts (verified)
- Repo: /home/rat/PycharmProjects/ML_models_NAMD — READ CLAUDE.md FIRST (wrapper contract, env splits, gotchas).
- GPU is an RTX 5080 Laptop (sm_120, 16 GB), WSL2. ONLY the `allegro` env
  (/home/rat/miniconda3/envs/allegro, torch 2.11.0+cu130, e3nn 0.6.0, mace 0.3.15, nequip 0.17.1,
  torchani 2.7.9, schnetpack 2.2.0, vesin 0.5.3) can use this GPU. Other envs (MACE_312 torch2.7,
  x_mace torch2.2, MLIP_2026, nequip_oam, fennix) fail on CUDA ("no kernel image"). TorchScript files
  scripted in older torch load fine in torch 2.11, so benchmark everything in `allegro`.
- TorchANI on GPU in allegro needs:
  export LD_LIBRARY_PATH=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH TORCHANI_NO_WARN_EXTENSIONS=1
- e3nn 0.4.4 envs need TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for torch>=2.6 loads (see CLAUDE.md).
- NAMD loads artifacts via a C++ libtorch shim (source /home/rat/compile_NAMD_MACE/namd_fennix/src/mlff_shim/).
  It passes float64 coords (requires_grad=True) on GPU, int64 Z, empty pc tensors, and a zero cell
  [1,3,3] if forward takes a 5th arg (arity-detected). The shim's libtorch is being upgraded to
  2.11.0+cu130 by the INFRA agent right now; it will also add env var NAMD_MLFF_EXTRA_LIBS
  (colon-separated .so list dlopen'd before torch::jit::load) so custom-op libraries work in NAMD.
  When `scripts/opt/INFRA_DONE` exists, read it: it tells you how to run
  `namd_benchmarks/lib/mlff_shim_test <lib> <model.pt> 0` — the NAMD-loadability acceptance test.
  Until then, develop and benchmark in Python.
- The CURRENT wrapper source (src/wrappers/*, uncommitted PBC/virial work) differs from the old
  deployed artifacts in models/ (e.g. forward now takes a `cell` arg and may return a virial).
  Your BASELINE must be the current wrapper source re-exported in allegro with default settings
  (plus the currently deployed artifact as a secondary reference if it loads).

## Rules
- Benchmark ONLY with `scripts/opt/bench_common.py` (read its docstring): interleaved timing,
  exclusive GPU flock (scripts/opt/.gpu_bench.lock — waiting for it is normal), parity vs first model.
  Any other GPU timing/profiling you do must also hold the lock: `flock scripts/opt/.gpu_bench.lock <cmd>`.
  Keep individual locked runs short (a few minutes) so other agents aren't starved. Don't edit bench_common.py
  (owned by infra); write your own helper scripts if you need more.
- Laptop clocks swing 2-3x between runs: only compare numbers from ONE invocation. Noise ~±20% at
  30 atoms. fp32 scatter nondeterminism gives parity diffs ~1e-5 even for identical artifacts.
- Standard reporting grid: water boxes 30, 300, 900, 3000 atoms (namd_benchmarks/systems/), walkers
  1 and 4 (forward_batch), plus 6000 atoms if it fits. Report median ms, p10/p90, peak alloc MiB,
  max|dE| (kcal/mol), max|dF| (kcal/mol/A) vs baseline.
- Precision: the REQUIRED optimized artifact keeps the model's native precision. A reduced-precision
  (fp32/TF32/bf16) variant is allowed only as a separately labeled extra with its parity table.
- Python env hygiene: never change torch/e3nn/numpy versions in `allegro` (check with
  `pip install --dry-run`). If you need conflicting packages, `conda create --clone allegro -n opt_<you>`.
- Files: you own ONLY your own wrapper file(s) (listed in your prompt) plus scripts/opt/<you>/.
  Don't modify src/edges.py, src/export.py, src/cli.py, other wrappers, or the NAMD shim — if you
  need a change there, write it up in scripts/opt/COORD.md (the `schnet-nl` agent owns src/edges.py
  and CUDA-graph work; infra owns the shim). New optional wrapper features must be opt-in flags,
  default behaviour unchanged, and the wrapper must still torch.jit.script.
- Don't overwrite/delete anything in models/ or namd_benchmarks/models. New artifacts -> models/opt/.
- Don't git commit. Don't run the full NAMD sweep (single-cell smoke tests after INFRA_DONE are fine:
  `cd namd_benchmarks && source env.sh && MODELS=<m> WATER=100 WALKERS=0 ./run_benchmark.sh --force`,
  but that writes runs/<m>/..., so only do it if useful and note it).
- Tests: after changing a wrapper, run
  `/home/rat/miniconda3/envs/allegro/bin/python -m pytest tests/ -q -x -k <yourmodel>` and the
  interface-compliance tests; they must still pass.
- Append cross-cutting findings (one line, prefixed [you]) to scripts/opt/COORD.md and read it
  occasionally for others' findings.

## Deliverables
1. scripts/opt/<you>/build.sh (or .py): reproducibly builds the optimized artifact(s) from the
   inputs listed in your prompt.
2. models/opt/<you>_*.pt (or equivalent) optimized artifacts.
3. scripts/opt/results/<you>_*.json from bench_common.py (final run: baseline + optimized in ONE invocation).
4. scripts/opt/<you>/REPORT.md: baseline vs optimized table, profile breakdown (where time goes:
   neighbor list / featurization / interaction / backward / host copies), what you tried incl. what
   didn't work and why, parity, NAMD shim loadability (mlff_shim_test result, NAMD_MLFF_EXTRA_LIBS
   needed?), and exact commands to reproduce.
Final message: short summary with the key numbers table and any follow-ups needing other agents/user.

## RAM rule (added after the parallel run exhausted RAM)
Agents now run ONE AT A TIME. The box has 30 GB RAM (~18 GB free; the user runs IDEs and their own jobs —
never touch processes you didn't start). Run a single heavy process at a time (no parallel compiles beyond
-j4), check `free -g` before big steps, kill stray python/namd3 processes you started. On resume, first
check what you already completed on disk and continue from there instead of redoing it.

## DISK rule (updated 2026-09-23)
C: was freed (143 GB free) and the user says disk is no longer an issue. Conda env clones and pip installs
of cuEq/OEQ-style packages are fine; still use --no-cache-dir, delete big intermediates you create, and
avoid needless multi-GB churn. Running the full pytest suite is allowed but prefer targeted -k runs.

## NO PYTHON INSIDE NAMD (user requirement, 2026-09-23)
NAMD artifacts must load through the CURRENT installed shim (namd_benchmarks/lib/libnamd_mlff.so, libtorch 2.11 zip)
with no embedded Python. Custom ops must be native C++/CUDA TORCH_LIBRARY .so files passed via NAMD_MLFF_EXTRA_LIBS.
Python-only paths (e.g. scripts/opt/mace/cueq_pyinit, pip-torch shim) are benchmark-only and must be labelled
"not NAMD-loadable".
