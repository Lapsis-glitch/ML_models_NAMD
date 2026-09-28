# NAMD3 ML-FF / FeNNol / xTB QM benchmark suite

Benchmarks the per-step cost of treating a whole water box as a QM region with
each supported NAMD QM engine, sweeping **system size** (30 → 6000 atoms) and
**walker count** (0 → 8 replicas) on a single GPU.

The `namd_fennix` NAMD3 build at `/home/rat/compile_NAMD_MACE/namd_fennix` runs
the QM engine in-process and prints per-eval timing; one GPU acts as the
inference server and extra replicas are batched onto it (`forward_batch`), so
the walker axis measures real cross-walker batching, not idealized scaling.

## Layout

```
env.sh                 all runtime exports (source before anything)
templates/bench.conf.tmpl   NAMD config template (whole box = QM, qmReplaceAll on)
systems/               built TIP3P water boxes  w<K>_<3K>atoms/{water.prmtop,water.pdb,qm.pdb}
  build_systems.py     packmol + tleap + parmed builder (run in `cpptraj` env)
  water_template.pdb   single TIP3P water for packmol
models/                model artifacts (symlinks): mace_off23.pt ani2x.pt nequip_oam.pt
  fennix/w<K>/manifest.json   per-size FeNNol StableHLO artifacts (see export_fennix.sh)
lib/libnamd_mlff.so    libtorch shim that namd3 dlopen's (built from the namd_fennix src)
run_benchmark.sh       the sweep driver (resumable, failure-tolerant)
gather_results.py      parse runs/ -> results/summary.csv + console table
runs/<model>/w<K>_<atoms>atoms/walk<W>/   per-cell logs + status.txt
results/summary.csv    gathered metrics
```

## What each cell measures

* **Whole water box is the QM region** (`beta=1` for every atom) with
  `qmReplaceAll on`, so the ML/QM engine supplies *all* forces. Water is kept
  rigid (`rigidbonds all`) so 500 steps stay numerically stable for any engine.
* Models: **mace** (MACE-OFF23), **ani2x** (ANI-2x), **nequip_oam** (NequIP-OAM),
  **schnet** (SchNetPack) via the TorchScript `mlff` backend; **xtb** (native
  GFN2-xTB); **fennol** (FeNNol FENNIX-BIO1 via the PJRT/StableHLO backend,
  per-size artifact).
* Sizes (waters → atoms): 10→30, 20→60, 60→180, 100→300, 300→900, 600→1800,
  1000→3000, 2000→6000.
* Walkers: `0` = a plain single `namd3` run (no replica framework, the genuine
  no-replica baseline); `1..8` = `charmrun ++local +pN namd3 +replicas N`.
* `STEPS=500`, NAMD `TIMING:`/`ENERGY:` every `OUTPUTFREQ=100` steps, and the
  native `MLFF_TIMING` per-eval breakdown enabled.

## Run it

```bash
cd namd_benchmarks
source env.sh

# whole grid (resumable; re-running skips completed cells)
./run_benchmark.sh

# preview the plan only
./run_benchmark.sh --dry-run

# subset via env vars
MODELS="mace xtb" WATER="10 20 60" WALKERS="0 1 2 4 8" ./run_benchmark.sh

# gather -> results/summary.csv and a console table
python gather_results.py
```

Knobs (env vars, see top of `run_benchmark.sh`): `MODELS WATER WALKERS STEPS
OUTPUTFREQ TIMEOUT GPU_SAMPLE`. Each cell has a wall `TIMEOUT` (default 1800 s);
OOM / timeout / error are recorded in `status.txt` and the sweep continues.

## Metrics (gather_results.py)

* `s_per_step` — steady-state NAMD wall-seconds/step (mean of intervals **after**
  the first; the first interval includes model load / StableHLO compile / GPU
  warmup, reported separately as `first_interval_s_per_step`). Universal across
  all backends.
* `ns_per_day` — derived from `s_per_step` (1 fs timestep).
* `mlff_infer_ms`, `mlff_forward_ms_batch` — steady-state per-eval inference and
  batched GPU forward (mlff backends only; derived from the cumulative MLFF
  TIMING prints).
* `gpu_peak_mib`, `wall_seconds`, `status`, `exit_code`.

## FeNNol artifacts (per-size)

The FENNIX backend is bound to a fixed atom count / composition, so each size
needs its own StableHLO artifact under `models/fennix/w<K>/manifest.json`,
derived from that system's `qm.pdb`. Cells with no artifact are skipped
(`status=no_artifact`). See `export_fennix.sh` for the export command. Large
sizes may exceed the 8 GB GPU during PJRT compile — export the sizes you need.

## Notes / gotchas

* `namd3` won't even load without the FFTW2 libs on `LD_LIBRARY_PATH` — `env.sh`
  handles it. The MLFF inference itself is the dlopen'd `lib/libnamd_mlff.so`.
* FeNNol runs prepend the JAX env's bundled CUDA libs (`$FENNIX_CUDA_LIBS`) so
  the PJRT plugin gets its own cuDNN 9.14; doing this for `mlff` runs can break
  libtorch's CUDA inference, so the driver scopes it to fennol only.
* Absolute timings depend on the GPU clock state (performance vs power-saving);
  compare within a single sweep, not across machine states.
* **Multi-walker (replicas ≥2) server election.** Earlier this raced (the MLFF
  election `unlink()`ed the socket just before `bind()`, so lockstep partitions
  could both become servers and hang at startup). The current `namd3` was
  recompiled with the bind-before-unlink fix and now elects correctly — validated
  walk 0/1/2 here (partition 0 → SERVER, partition 1 → `sawLiveListener=YES` →
  CLIENT, batched forward of N walkers). The driver still clears
  `/tmp/mlff_namd_*.sock` before each cell. Diagnostic if it ever recurs on
  another build: two `becameServer=YES` + no `TIMING:` in a walk≥1 cell == that
  race.
