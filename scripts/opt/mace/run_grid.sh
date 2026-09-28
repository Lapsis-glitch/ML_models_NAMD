#!/usr/bin/env bash
# Final MACE bench grid: ONE (system, walkers) per bench_common invocation (TorchScript
# profiling-executor gotcha, see COORD.md). cuEq artifacts run through the NATIVE op library
# (no cuequivariance python imported) -- same code path as NAMD's default shim.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
PY=/home/rat/miniconda3/envs/allegro/bin/python
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
NAT="--extra-lib $SP/cuequivariance_ops/lib/libcue_ops.so --extra-lib scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so"
M=models/opt
E3NN="--model base=$M/mace_baseline.pt --model fast_e3nn=$M/mace_fast_e3nn.pt"
CUEQ="--model cueqf=$M/mace_cueqf.pt --model fast_cueqf=$M/mace_fast_cueqf.pt --model fast_cueqf_f32_EXTRA=$M/mace_fast_cueqf_f32.pt"
for N in ${SYSTEMS:-30 300 900 3000 6000}; do
  case $N in 30|300) IT="--warmup 15 --iters 30 --rounds 3" ;; 900) IT="--warmup 8 --iters 15 --rounds 2" ;; *) IT="--warmup 4 --iters 6 --rounds 2" ;; esac
  for W in ${WALKERS:-1 4}; do
    echo "=== N=$N W=$W"
    # the e3nn models (baseline + fast_e3nn) need ~5 GiB per 900 atoms; above ~1200 total atoms they exceed the
    # 16 GiB card (WSL spills to host RAM and crawls) -> run them only where they fit; parity/speedup rows then
    # refer to the first model present (cueqf, itself at dF ~1e-13 vs baseline at every size that fits both).
    if [ $((N*W)) -le 1200 ]; then MODELS="$E3NN $CUEQ"; else MODELS="$CUEQ"; fi
    $PY scripts/opt/mace/bench_capped.py $NAT $MODELS --systems $N --walkers $W --jitter 0.02 $IT \
      --out scripts/opt/results/mace_final_w${N}_W${W}.json 2>&1 | grep -v -i "warn\|pynvml\|GPU information\|^ *warnings\|^  main()"
  done
done
