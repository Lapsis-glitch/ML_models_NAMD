#!/usr/bin/env bash
# C-ABI (NAMD-like) timing of the CUDA-graph shim path, via infra's shim_bench.
# Holds the GPU lock for the whole run; ROUNDS interleaved rounds per system.
#   bash scripts/opt/schnet/cudagraph/shim_bench_graph.sh [systems="30 300 900 1800"] [rounds=2]
set -uo pipefail
REPO=/home/rat/PycharmProjects/ML_models_NAMD
cd "$REPO"
source namd_benchmarks/env.sh >/dev/null 2>&1
SYS="${1:-30 300 900 1800}"; ROUNDS="${2:-2}"
INST=$NAMD_MLFF_LIB                                     # installed shim (no graph code)
PATCHED=$REPO/scripts/opt/schnet/cudagraph/shim/libnamd_mlff.so
BENCH=$REPO/scripts/opt/infra/shim_bench
exec 9>"$REPO/scripts/opt/.gpu_bench.lock"; flock 9
for s in $SYS; do
  g=$(ls -d namd_benchmarks/systems/w*_${s}atoms)/qm.pdb
  for r in $(seq "$ROUNDS"); do
    for cfg in "base|$INST|models/opt/schnet_baseline.pt|0" \
               "fast|$INST|models/opt/schnet_fast.pt|0" \
               "fast+graph|$PATCHED|models/opt/schnet_fast.pt|1"; do
      IFS='|' read -r lab lib mod gr <<<"$cfg"
      res=$(NAMD_MLFF_CUDA_GRAPH=$gr "$BENCH" "$lib" "$mod" "$g" 1 25 200 0.02 0 2>/dev/null | grep RESULT)
      echo "water$s round$r $lab $res"
    done
  done
done
