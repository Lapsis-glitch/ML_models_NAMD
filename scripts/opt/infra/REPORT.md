# INFRA report (2026-09-21) — saved by the main session from the infra agent's hand-back

## Summary
- NAMD ML inference works on the RTX 5080 again: all 5 artifacts pass mlff_shim_test on GPU; NAMD walk0 cells ok for schnet, mace, ani2x, nequip_oam (step-0 energies match the old libtorch 2.6 runs).
- New libtorch: /home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130 (old libtorch/ untouched). Missing CUDA libs (cufft, cusparse, curand, cusolver, nvJitLink) are SYMLINKS into allegro's nvidia/cu13/lib (break if allegro is rebuilt; see lib/README.extra_cuda_libs). Added soname links libcublas.so.13, libcublasLt.so.13, libnvrtc-builtins.so.13.0.
- namd3 had to be RELINKED against 2.11 (it links libtorch via colvars torchann; one libtorch per process). Make.config TORCHDIR -> 2.11, colvarcomp_torchann.o rebuilt. Backups: namd3.libtorch2.6.bak, Make.config.libtorch2.6.bak, obj/colvarcomp_torchann.o.libtorch2.6.bak. Colvars torch component not exercised.
- Shim changes (mlff_shim.cpp, build_and_test.sh; backups *.pre_extralibs.bak, *.pre_knobs.bak, *.pre_libtorch211.bak; BUILD.local.md updated): NAMD_MLFF_EXTRA_LIBS hook (dlopen RTLD_NOW|RTLD_GLOBAL before torch::jit::load, promotes libtorch to global first), exit-time leak fix for the "Unrecognized stream" abort, opt-in runtime knobs.
- namd_benchmarks/lib: libnamd_mlff.so (C++ libtorch zip, default), libnamd_mlff_pytorch.so (against allegro pip torch/lib), mlff_shim_test, backups *.libtorch2.6.bak / *.pre_knobs.bak. env.sh backup env.sh.libtorch2.6.bak; LIBTORCH_ROOT and NAMD_MLFF_LIB paired (use a fresh shell or unset NAMD_MLFF_LIB when switching).
- cuEquivariance loads inside namd3 ONLY with the pip-torch shim variant (needs libtorch_python.so): LIBTORCH_ROOT=<allegro site-packages/torch>, nvidia/cu13/lib on LD_LIBRARY_PATH, NAMD_MLFF_EXTRA_LIBS=libpython3.12.so.1.0:libtorch_python.so:libcue_ops.so:cuequivariance_ops_torch_ext...so (exact paths in INFRA_DONE). No cuEq artifact run through NAMD yet. TorchANI 2.7.9 ships no compiled cuAEV; vesin-torch / OpenEquivariance not installed.
- WALK>=2 IS BROKEN: server partition segfaults when a client connects. Reproduces with the old 2.6 stack on CPU; the Jun-5 namd3.prefix_bak works -> regression in the Jul-31 ComputeMLFF.C changes (PBC / client-server protocol). Not patched.
- Overwritten benchmark run cells backed up to scripts/opt/infra/runs_backup_pre_libtorch211/.

## Runtime knobs (all opt-in, default unchanged)
NAMD_MLFF_JIT_PROFILING=0, NAMD_MLFF_JIT_OPTIMIZE=0, NAMD_MLFF_JIT_TEXPR=0/1, NAMD_MLFF_JIT_FUSION=DYNAMIC:20, NAMD_MLFF_JIT_PROFILED_RUNS=N, NAMD_MLFF_FREEZE=1, NAMD_MLFF_TF32=1 (numerics change), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. bench_common.py got --jitter, --jit-profiling/--jit-optimize/--jit-texpr/--jit-fusion/--tf32. Tools: infra/jit_knobs.py, shim_bench(.cpp), shim_knobs.sh, alloc_mem.py; raw data infra/results/.

Steady-state speedup vs default (Python, 6 interleaved rounds, 0.02 A jitter):

| model/atoms | default ms | noopt | legacy | notexpr | dyn20 | static20 | freeze | tf32 |
|---|---|---|---|---|---|---|---|---|
| schnet 30 | 3.87 | 0.94 | 1.11 | 0.91 | 1.10 | 1.11 | 1.11 | 1.07 |
| schnet 300 | 3.87 | 0.93 | 0.95 | 0.80 | 0.99 | 0.88 | 0.95 | 1.07 |
| schnet 900 | 4.38 | 0.94 | 0.99 | 0.82 | 1.01 | 0.93 | 0.91 | 1.12 |
| mace 30 (fp64) | 38.1 | 0.96 | 1.00 | 0.95 | 1.00 | 1.00 | 1.04 | 1.00 |
| mace 300 | 310 | 0.90 | 0.90 | 0.90 | 0.91 | 0.83 | 0.91 | 0.90 |
| ani2x 30 | 28.3 | 0.97 | 1.09 | 0.97 | 0.96 | 0.97 | 1.01 | 0.98 |
| ani2x 300 | 32.6 | 0.97 | 0.94 | 0.78 | 0.90 | 0.84 | 0.93 | 1.07 |
| ani2x 900 | 29.4 | 0.94 | 1.04 | 0.89 | 0.93 | 0.92 | 0.98 | 1.05 |
| nequip 30 | 121 | 0.98 | 0.98 | 0.95 | 1.03 | 1.04 | 1.04 | 0.95 |
| nequip 300 | 145 | 0.99 | 0.99 | 0.98 | 1.01 | 1.00 | 1.02 | 0.99 |
| nequip 900 | 533 | 1.00 | 0.99 | 0.99 | 0.99 | 1.01 | 1.01 | 0.99 |
| xmace 12 | 26.7 | 0.90 | 0.98 | 0.88 | 0.99 | 1.03 | 1.11 | 1.01 |
| xmace 120 | 31.1 | 0.91 | 1.03 | 0.90 | 1.01 | 0.99 | 1.06 | 1.02 |

- TF32 moves energies (schnet |dE| 0.19/1.6/5.2 kcal/mol at 30/300/900; nequip up to 21; ani |dF| ~0.1) — not for production.
- Warm-up (first 25 calls) is the one consistent difference: NAMD_MLFF_JIT_OPTIMIZE=0 cuts JIT warm-up 3-20x (e.g. nequip 19-21 s -> 3-6 s, mace30 5.5 s -> 1.0 s). Only matters for short runs; benchmarks should use steady state.
- Shim-level: schnet 30 atoms default 3.1-3.2 ms (0.4-0.8 ms faster than Python); pinned async copies not a bottleneck. Other models swung 2-3x at shim level because host load avg was ~27 (host/launch-bound at small sizes; ani2x flat ~28 ms from 30-900 atoms).
- Memory peak reserved default -> expandable_segments: schnet900 322->310 MiB, ani900 166->166, nequip300 2104->2074, mace300 2032->1778 (-12.5%).
- Conclusion: no knob is a clear numerically-neutral steady-state win; defaults kept. Real wins are model-side (MACE-OFF fp64, ANI/X-MACE launch-bound, NequIP-OAM-L heavy).

## Housekeeping by the agent
Ran `conda clean --tarballs`, deleted its scratch gdb env; no stray processes. To revert to 2.6: restore the *.libtorch2.6.bak files and run with LIBTORCH_ROOT=$COMPILE_ROOT/libtorch (no GPU on the 5080).
