#!/usr/bin/env bash
# SevenNet-0 bench grid: ONE (system, walkers) per bench_common invocation (profiling-executor
# gotcha, COORD.md), 15 GiB VRAM cap (WSL2 spills instead of OOM), OEQ through the NATIVE op lib only
# (same code path as NAMD's default shim; openequivariance python never imported).
#   SYSTEMS="30 300" WALKERS=1 TAG=x MODELS_EXTRA="--model f=models/opt/..." bash scripts/opt/sevennet/run_grid.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
PY=${PY:-/home/rat/miniconda3/envs/allegro/bin/python}
SP=${SP:-$("$PY" -c 'import site; print(site.getsitepackages()[0])')}
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
NAT="--extra-lib scripts/opt/nequip/oeq_native/liboeq_native.so"
M=models/opt
# e3nn baseline: ~4 GiB per 1000 atoms -> only where N*W <= 3000 (speedup/parity columns then refer
# to the first model present)
BASE="--model base=$M/sevennet_baseline.pt --model base_d3=$M/sevennet_baseline_d3.pt"
OPT="--model oeq=$M/sevennet_oeq.pt --model oeq_d3=$M/sevennet_oeq_d3.pt --model fast=$M/sevennet_fast.pt --model fast_d3=$M/sevennet_fast_d3.pt ${MODELS_EXTRA:-}"
TAG=${TAG:-final}
mkdir -p scripts/opt/results
for N in ${SYSTEMS:-30 300 900 3000 6000}; do
  case $N in 30|300) IT="--warmup 15 --iters 30 --rounds 3" ;; 900) IT="--warmup 8 --iters 15 --rounds 2" ;; *) IT="--warmup 4 --iters 8 --rounds 2" ;; esac
  for W in ${WALKERS:-1 4}; do
    echo "=== N=$N W=$W"
    if [ $((N*W)) -le 3000 ]; then MODELS="$BASE $OPT"; else MODELS="$OPT"; fi
    $PY scripts/opt/nequip/bench_nq.py $NAT $MODELS --systems $N --walkers $W --jitter 0.02 $IT ${EXTRA_ARGS:-} \
      --out scripts/opt/results/sevennet_${TAG}_w${N}_W${W}.json 2>&1 | grep -v -i "warn\|^ *warnings\|^  main()\|^\[bench\]"
  done
done
