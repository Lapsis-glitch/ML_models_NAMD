#!/usr/bin/env bash
# Phase-2 (infra): time the NAMD shim (C ABI, like NAMD) under different
# NAMD_MLFF_* runtime knobs.  Knobs are process-global, so every config is a
# separate shim_bench process; configs are alternated over ROUNDS rounds so
# laptop clock drift hits them equally.  Holds the shared GPU lock throughout.
#
#   scripts/opt/infra/shim_knobs.sh <model.pt> <geom.pdb|xyz> [W=1] [ROUNDS=3] [ITERS=200]
# env: SHIM (default namd_benchmarks/lib/libnamd_mlff.so), CONFIGS (space-separated names below)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
MODEL="$1"; GEOM="$2"; W="${3:-1}"; ROUNDS="${4:-3}"; ITERS="${5:-200}"
SHIM="${SHIM:-$REPO/namd_benchmarks/lib/libnamd_mlff.so}"
CONFIGS="${CONFIGS:-default legacy noopt notexpr dyn20 freeze tf32 expseg}"
export LD_LIBRARY_PATH="/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130/lib:${LD_LIBRARY_PATH:-}"

envfor() {
  case "$1" in
    default)  echo "" ;;
    legacy)   echo "NAMD_MLFF_JIT_PROFILING=0" ;;
    noopt)    echo "NAMD_MLFF_JIT_OPTIMIZE=0" ;;
    notexpr)  echo "NAMD_MLFF_JIT_TEXPR=0" ;;
    dyn20)    echo "NAMD_MLFF_JIT_FUSION=DYNAMIC:20" ;;
    static20) echo "NAMD_MLFF_JIT_FUSION=STATIC:20" ;;
    prof3)    echo "NAMD_MLFF_JIT_PROFILED_RUNS=3" ;;
    freeze)   echo "NAMD_MLFF_FREEZE=1" ;;
    tf32)     echo "NAMD_MLFF_TF32=1" ;;
    expseg)   echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" ;;
  esac
}

exec 9>"$REPO/scripts/opt/.gpu_bench.lock"
flock 9
echo "# model=$(basename "$MODEL") geom=$(basename "$GEOM") W=$W rounds=$ROUNDS iters=$ITERS shim=$(basename "$SHIM")"
for r in $(seq 1 "$ROUNDS"); do
  for c in $CONFIGS; do
    out=$(env $(envfor "$c") "$HERE/shim_bench" "$SHIM" "$MODEL" "$GEOM" "$W" 25 "$ITERS" 0.02 0 2>/dev/null | grep RESULT)
    echo "round=$r config=$c $out"
  done
done
