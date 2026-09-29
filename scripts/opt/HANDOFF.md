# HANDOFF — per-model inference optimisation (state as of 2026-09-21)

## Goal (user's words, paraphrased)
Optimise every model for inference speed WITHOUT changing the model (compilation, cuEquivariance, CUDA
graphs, kernels, neighbor list, wrapper overhead...), one agent per model, so models can be judged fairly
on speed and memory inside NAMD.

## Hard constraints learned this session (read before doing anything)
1. **One hands-on agent at a time.** Parallel agents exhausted the 30 GB RAM and killed the session. The
   user also runs IDEs + their own jobs (e.g. `21_novel_reaction.py` NEB) — never touch their processes.
2. **Disk:** the WSL disk is `C:\WSL\ext4.vhdx`; Windows C: is ~full (dropped to 41 MB at one point, 5.6 GB
   at handoff). Check `df -h /mnt/c` before any big write. No conda env clones, no big downloads; ask before
   anything >500 MB. User may compact the vhdx or move the distro to D: (`wsl --manage Ubuntu-24 --move D:\WSL`).
3. **GPU is RTX 5080 Laptop (sm_120).** Only the `allegro` env (torch 2.11+cu130) runs CUDA. Old RTX 3070
   numbers are stale.
4. User runs full NAMD sweeps themselves; single-cell smoke tests only. Don't git commit.

## Shared infrastructure (done)
- `scripts/opt/AGENT_BRIEF.md` — the brief every per-model agent gets (rules, facts, deliverables,
  RAM + disk rules at the bottom).
- `scripts/opt/bench_common.py` — the ONLY benchmark: interleaved timing, GPU flock
  (`scripts/opt/.gpu_bench.lock`), parity vs first model, `--jitter`, JIT-knob flags. Mimics the NAMD shim.
- `scripts/opt/COORD.md` — cross-agent findings log.
- `scripts/opt/INFRA_DONE` + `scripts/opt/infra/REPORT.md` — infra agent COMPLETE:
  NAMD shim + namd3 relinked on libtorch 2.11.0+cu130 (`~/compile_NAMD_MACE/libtorch-2.11.0+cu130`,
  backups `*.libtorch2.6.bak`), `NAMD_MLFF_EXTRA_LIBS` custom-op hook, opt-in `NAMD_MLFF_JIT_*` knobs
  (none a clear win; defaults kept). cuEquivariance loads in namd3 only via the pip-torch shim variant
  (`libnamd_mlff_pytorch.so`, recipe in INFRA_DONE).
- Scripted wrappers on GPU in allegro need
  `export LD_LIBRARY_PATH=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`
  (NNC fuser needs libnvrtc-builtins on the 3rd call); TorchANI also `TORCHANI_NO_WARN_EXTENSIONS=1`.

## Known open issue (not ours to fix unprompted)
- NAMD multi-walker (walk>=2) segfaults on client connect — regression in the Jul-31 `ComputeMLFF.C`
  (PBC / client-server protocol); Jun-5 `namd3.prefix_bak` works. Reported to the user, not patched.

## Agent queue — run sequentially, in this order
| # | agent | status | partial work on disk |
|---|---|---|---|
| 0 | infra | DONE | see above |
| 1 | schnet-nl (SchNet + src/edges.py neighbor list + CUDA graphs) | DONE (REPORT.md saved; graph shim not installed — user decision) | scripts/opt/schnet/{nl,cudagraph,bench_baseline_initial.json}, models/opt/schnet_baseline.pt |
| 2 | mace (MACE-OFF23, cuEquivariance) | DONE 2026-09-23 (REPORT.md saved): fast_cueqf 15x in namd3 via native uniform_1d op, no Python | scripts/opt/mace/*, models/opt/mace_* |
| 3 | nequip (NequIP-OAM-L) | DONE 2026-09-23 (REPORT.md saved): fast_oeq 5.3x in namd3 via native OEQ lib (liboeq_native.so), no Python | scripts/opt/nequip/*, models/opt/nequip_* |
| 4 | ani (ANI-2x) | DONE 2026-09-23 (REPORT.md saved): ani_fast 6.7x in namd3 @300 via native cuAEV lib; found element_list default bug (Cl->F net) — user decision | scripts/opt/ani/*, models/opt/ani_* |
| 5 | xmace (X-MACE fulvene) | SKIPPED by user 2026-09-23 ("basically MACE"; MACE cuEq/FastMACE route applies) | — |
| 6 | fennix (FeNNiX JAX/PJRT) | DONE 2026-09-23 (REPORT.md saved): runs on sm_120 as-is; 3-5x in namd3 via NAMD-side patches (namd_fennix_fxopt copy) + QMNoPntChrg on; shipped artifacts are TF32 (fp32 extra in models/opt/fennix_fp32) | scripts/opt/fennix/*, models/opt/fennix_* |

The original per-agent prompts are summarised by their rows above plus AGENT_BRIEF.md; each agent owns only its
wrapper file(s) + `scripts/opt/<name>/` (schnet-nl also owns `src/edges.py`, `src/nl_vesin.py`, and
CUDA-graph shim work behind `NAMD_MLFF_CUDA_GRAPH=1`, default off). Each must deliver
`scripts/opt/<name>/build.sh`, `models/opt/<name>_*.pt`, `scripts/opt/results/<name>_*.json` (baseline +
optimised in ONE bench_common invocation), and `scripts/opt/<name>/REPORT.md` — subagents cannot write
REPORT.md themselves (harness refuses), so the main session saves the report from the hand-back.

## Resuming
- Same Claude session: resume the stopped agents via SendMessage one at a time (their transcripts persist),
  telling them to check disk state and continue rather than redo.
- New session: launch a fresh agent per queue row with AGENT_BRIEF.md + that row, telling it to inspect its
  partial work first.
- After all agents: run `/home/rat/miniconda3/envs/allegro/bin/python -m pytest tests/ -q`, then do one
  final combined bench_common run of all optimised artifacts and write the cross-model comparison
  (speed + memory, 30/300/900/3000/6000 atoms, walkers 1 and 4).

## 2026-09-21 late: queue PAUSED after schnet-nl
- C: free dropped 19 GB -> 1.2 GB during the schnet-nl run (WSL / unchanged at 595G, so vhdx growth from
  written-then-deleted temp files, or Windows-side). Do not launch agent #2 (mace) until the user frees C:.
- Shared src/edges.py now has 2 default-on fixes (zero-cell det short-circuit, per-molecule batched NL);
  baselines re-exported from current source include them — compare against pre-edit backups if needed.
- Pre-existing test failures (2, PBC-related, forward() without cell): TestE2E_SchNetPack, TestE2E_TorchANI::test_train_and_wrap.
- 2026-09-22 ~00:45: mace agent stopped on disk rule (C: 19 -> 6.9 GB again). ext4.vhdx = 644.7 GiB vs 595 GiB
  used inside (discard mount is on). Suspect: full pytest suite (e2e tests train models / write temp files)
  grows the vhdx; it was run by both agents. Next resume: avoid full pytest unless needed (use -k mace + interface tests).

## RESUME HERE (after reboot, 2026-09-22)
- State: #0 infra DONE, #1 schnet-nl DONE, #2 mace PAUSED (interim 2), #3–#6 not started. No agent/processes running.
- Wait for the user's explicit go before launching anything.
- To resume mace: launch a fresh agent with AGENT_BRIEF.md + scripts/opt/mace/REPORT.md ("Resume steps");
  tell it: tests = `-k "mace or MACE"` + tests/test_interface_compliance.py only (NOT the full suite); hard-stop if /mnt/c < 3 GB.
- Then nequip, ani, xmace, fennix in order, one agent at a time; then pytest + final combined bench + cross-model comparison.
- User-decision items: install the CUDA-graph shim (schnet); src/cli.py lacks new schnet flags; cuEq in NAMD needs
  embedded Python (pyinit lib) unless Route B native op lands.

## 2026-09-23: RESUMED by user ("resume the performance optimizations, disk space is no issue anymore")
- C: now 143 GB free; disk rule in AGENT_BRIEF.md relaxed. Queue resumes at #2 mace, still one agent at a time.
- User requirement: no Python inside NAMD; MACE cuEq pyinit route rejected -> native uniform_1d op (Route B) mandatory, e3nn FastMACE as Python-free fallback.
- ANI element-order bug FIXED 2026-09-23 (Cl was scored by F net): source defaults + baked Z tables patched (backups *.pre_elemfix.bak), tests/test_ani_element_order.py.

## 2026-09-23: per-model queue COMPLETE (schnet, mace, nequip, ani, fennix; xmace skipped)
Pending user decisions: adopt fennix NAMD patches (namd_patch/*.patch, patched copy at ~/compile_NAMD_MACE/namd_fennix_fxopt),
QMNoPntChrg on in bench.conf.tmpl, same charge-restore hash fix in ComputeMLFF.C:2795, FeNNiX TF32 vs fp32 for comparison,
repoint namd_benchmarks/models symlinks + NAMD_MLFF_EXTRA_LIBS per model, schnet CUDA-graph shim.
Next (per plan): pytest, final combined bench of all optimised artifacts, cross-model comparison.
- 2026-09-23: final full pytest 121 passed / 16 skipped / 2 known failures; combined bench + scripts/opt/COMPARISON.md DONE.
- NAMD patch set for the user: scripts/opt/namd_patches/ (01 shim EXTRA_LIBS+knobs, 02 fennix backend, 03 ComputeQM fast index + README).
- L40S bundle (NAMD sweep only, optimised models, MAX_ATOMS knob) being built in scripts/opt/l40s_bundle/.
- L40S bundle DONE: scripts/opt/l40s_bundle/namd_ml_bench_2026-09-23.zip (273 MB), smoke-validated all 5 models.

## 2026-09-28: optimised builds generalised (commit 5b82b36)
- src/cli.py now has the SchNet fast flags (--fast etc.), TorchANI --lean and --extra-libs (resolves the "src/cli.py lacks new schnet flags" item above).
- mace/build_fast.py --state (any MACE), ani/build_fast.py --model ani1x|ani1ccx|ani2x, parity checks cycle all species; build scripts read PY/SP/TORCH/MACE_PY from env.
- Rebuilt MACE-OFF23, NequIP-OAM-L, ANI-2x, SchNet via the new path: identical to models/opt artifacts. Recipe documented in README "Guide: optimised builds".

## 2026-09-30: SevenNet optimised (user request, this session ran it directly, no subagent)
- scripts/opt/sevennet/{build.sh,parity.py,prof.py,run_grid.sh,namd_smoke/,REPORT.md}; rewrites in src/sevennet_fast.py, entry point `src.compile_sevennet --fast` (any checkpoint, multi-fidelity included); artifacts models/opt/sevennet_{baseline,oeq,fast}{,_d3}.pt.
- OEQ via SevenNet's own deploy(use_oeq=True) (new `src.compile_sevennet --oeq`) + the NequIP liboeq_native.so, unchanged; FastSevenNet exact rewrites (dense linears, gate, fused radial MLP, fused intro/si1).
- 7net-0: 5.7x @30, 7.7x @300, 12.4x @3000 (bench_common); namd3 300 atoms 40.5 -> 8.6 ms/step. 7net-l3i5 also works (9.5x/14.3x); multi-fidelity: kernels only.
- Wrapper: read_sevennet_metadata reads _extra_files from the zip; clear error if an OEQ deployment is loaded without the op lib. Scripted path unchanged.
- 2026-09-30 (user): wrapper now sums SevenNet energies in fp64; src.cli auto-loads liboeq_native.so for OEQ deployments. Pending: D3 is the next bottleneck at >=3000 atoms.
