#!/usr/bin/env bash
# Final NequIP-OAM-L bench grid: ONE (system, walkers) per bench_common invocation (profiling-executor
# gotcha, COORD.md), 15 GiB VRAM cap (WSL2 spills instead of OOM), OEQ through the NATIVE op lib only
# (same code path as NAMD's default shim; openequivariance python never imported).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
PY=/home/rat/miniconda3/envs/allegro/bin/python
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NAT="--extra-lib scripts/opt/nequip/oeq_native/liboeq_native.so"
M=models/opt
E3NN="--model base=$M/nequip_baseline.pt --model recomp=$M/nequip_recompiled_cuda.pt"
OEQ="--model oeq=$M/nequip_oeq.pt --model fast_oeq=$M/nequip_fast_oeq.pt"
TAG=${TAG:-final}
for N in ${SYSTEMS:-30 300 900 3000 6000}; do
  case $N in 30|300) IT="--warmup 15 --iters 30 --rounds 3" ;; 900) IT="--warmup 8 --iters 15 --rounds 2" ;; *) IT="--warmup 4 --iters 8 --rounds 2" ;; esac
  for W in ${WALKERS:-1 4}; do
    echo "=== N=$N W=$W"
    # the e3nn models need ~1.8 GiB per 300 atoms -> only where N*W <= 1200 (parity/speedup rows then refer
    # to the first model present)
    if [ $((N*W)) -le 1200 ]; then MODELS="$E3NN $OEQ"; else MODELS="$OEQ"; fi
    $PY scripts/opt/nequip/bench_nq.py $NAT $MODELS --systems $N --walkers $W --jitter 0.02 $IT ${EXTRA_ARGS:-} \
      --out scripts/opt/results/nequip_${TAG}_w${N}_W${W}.json 2>&1 | grep -v -i "warn\|^ *warnings\|^  main()"
  done
done
