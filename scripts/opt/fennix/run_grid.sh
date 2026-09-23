#!/usr/bin/env bash
# FeNNiX PJRT bench grid: one invocation per system size (interleaved models, same jittered geometries).
#   base   = shipped artifact (namd_benchmarks/models/fennix/w<K>) through the ORIGINAL NAMD execute path
#   fast   = same artifact/executable through the patched execute path (namd_patch)       <- optimised
#   fp32   = EXTRA: true-fp32 export (matmul precision HIGHEST instead of TF32), fast path
#   w4     = EXTRA: 4-walker vmap export [4,N,3], fast path (compare with 4 x fast)
set -uo pipefail
cd "$(dirname "$0")"; source ./env_fennix.sh
OUT=$REPO/scripts/opt/results; mkdir -p $OUT
ITERS=${ITERS:-100}
for K in ${SIZES:-10 100 300 1000 2000}; do
  A=$((3*K))
  args=(base=$REPO/namd_benchmarks/models/fennix/w$K/manifest.json@namd
        fast=$REPO/namd_benchmarks/models/fennix/w$K/manifest.json@fast
        fp32=$REPO/models/opt/fennix_fp32/w$K/manifest.json@fast)
  [ -f $REPO/models/opt/fennix_w4/w$K/manifest.json ] && args+=(w4=$REPO/models/opt/fennix_w4/w$K/manifest.json@fast)
  echo "== $A atoms"
  flock $LOCK $BENCH --iters $ITERS --warmup 10 --jitter 0.02 --mem-fraction 0.9 --out $OUT/fennix_final_w$A.json "${args[@]}" 2>&1 \
    | grep -v -E "ptxas|^$|subprocess_compilation|slow_operation|Trying algorithm|operation took"
done
