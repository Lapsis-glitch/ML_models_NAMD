# Local benchmark run — partial, stopped early (record only)

**Machine:** RTX 3070 Laptop GPU (8 GB), WSL2. **Run:** 2026-06-02 23:21 → 2026-06-03, stopped manually.
**Why stopped:** the authoritative sweep is running on a separate (larger) GPU machine; this
local run was kept only as a harness / walk0 validation.

## What completed

- **42 / 432 cells**, all at the **smallest system size only (30 atoms / 10 waters)**.
  The size axis (60 → 6000 atoms) never started.
- Status: **41 `ok`, 1 `timeout_partial`** (xtb @ 30 atoms, walk5 — CPU-bound, see below).
- Models reached: `mace` (9), `ani2x` (9), `nequip_oam` (9), `schnet` (9), `xtb` (6, walk0–5).
  **`fennol` = 0** (it's last in per-size model order, never reached). No StableHLO/PJRT data here.
- Full per-cell table: `results/summary.csv`.

## Trust / data-quality caveats (important)

This run is **NOT** a clean benchmark. Two issues:

1. **Leaked-orphan contamination (multi-walker + all memory).** The driver's straggler
   cleanup (`pkill -9 namd3`) was a silent no-op: NAMD renames its Charm++ PE threads
   (process `comm` = `"NAMD masterPe"`), so `pkill namd3` matched nothing. Multi-walker
   (walk≥1) cells leaked NAMD processes that were never reaped — including one that ran
   ~7 h and was already resident when the "idle" GPU baseline (1756 MiB) was recorded.
   Consequence:
   - **walk0 (single) timing = clean.**
   - **walk≥1 timing = inflated** by GPU contention from accumulating orphans.
   - **ALL `gpu_peak_mib` / `gpu_mem_over_baseline_mib` = unreliable** (orphan residue +
     contaminated baseline). E.g. ani2x/schnet walk0 peaks of 3520/4094 MiB are leftover
     residue, not those models' real footprint.
   - **Fixed** in `run_benchmark.sh` (now `pkill -9 -f '/namd3 '`, matches the binary in
     the cmdline). The fix is in the repo and applies to the remote / any clean re-run; it
     was NOT retroactively applied to this run.

2. **xtb multi-walker = CPU contention, not GPU.** xtb is native CPU GFN2; each replica
   runs xtb on CPU, so walk≥2 collapses (0.02 → 1.05 → 2.14 → 2.97 → 4.03 s/step,
   walk5 timed out). This is real CPU oversubscription on the laptop, unrelated to issue #1.

## The one clean, quotable result — single-walker (walk0) speed @ 30 atoms

| model      | s/step | ns/day | infer_ms |
|------------|--------|--------|----------|
| schnet     | 0.0072 | 12.06  |   9.4    |
| xtb        | 0.0223 |  3.88  |   —      |
| ani2x      | 0.0353 |  2.45  |  30.5    |
| mace       | 0.1479 |  0.58  | 143.0    |
| nequip_oam | 0.1596 |  0.54  | 168.5    |

Caveat even here: 30 atoms barely loads the GPU and the clock was in power-saving state,
so treat these as relative ordering at the smallest size only — **not** representative of
larger systems. The authoritative model × size × walker grid (and all trustworthy memory
numbers) comes from the remote machine with the fixed `run_benchmark.sh`.
