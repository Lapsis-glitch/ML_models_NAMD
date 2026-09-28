#!/usr/bin/env bash
# Final ANI-2x bench grid: ONE (system, walkers) per bench_common invocation (profiling-executor gotcha,
# COORD.md), 15 GiB VRAM cap (WSL2 spills instead of OOM), cuAEV through the NATIVE op lib only (same code
# path as NAMD's default shim; torchani python never imported). Then protein geometries (5 elements) and the
# periodic path (bench_pbc.py; bench_common only sends a zero cell).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
PY=/home/rat/miniconda3/envs/allegro/bin/python
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} TORCHANI_NO_WARN_EXTENSIONS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
A=scripts/opt/ani; M=models/opt; R=scripts/opt/results
NAT="--extra-lib $A/cuaev_native/libcuaev_native_precise.so"
MODELS="--model base=$M/ani_baseline.pt --model fast=$M/ani_fast.pt --model fast_nocache=$M/ani_fast_nocache.pt"
TAG=${TAG:-final}
for N in ${SYSTEMS:-30 300 900 3000 6000}; do
  case $N in 30|300|900) IT="--warmup 15 --iters 30 --rounds 3" ;; *) IT="--warmup 6 --iters 12 --rounds 2" ;; esac
  for W in ${WALKERS:-1 4}; do
    echo "=== N=$N W=$W"
    $PY scripts/opt/nequip/bench_nq.py $NAT $MODELS --systems $N --walkers $W --jitter 0.02 $IT \
      --out $R/ani_${TAG}_w${N}_W${W}.json 2>&1 | grep -E "^ *(water|system)|rror"
  done
done
if [ -z "${NO_PROT:-}" ]; then
  for G in prot_res1-2 prot_res1-20 prot_mono_dry; do
    echo "=== $G W=1"
    $PY scripts/opt/nequip/bench_nq.py $NAT $MODELS --systems= --geom $A/geoms_prot/$G.pdb --walkers 1 --jitter 0.02 \
      --warmup 10 --iters 20 --rounds 3 --out $R/ani_${TAG}_${G}_W1.json 2>&1 | grep -E "prot|rror"
  done
fi
if [ -z "${NO_PBC:-}" ]; then
  for N in 30 300 900; do
    $PY $A/bench_pbc.py $NAT $N base=$M/ani_baseline.pt fast=$M/ani_fast.pt fast_nocache=$M/ani_fast_nocache.pt 2>&1 | grep water
  done | tee $R/ani_${TAG}_pbc.txt
  $PY $A/bench_pbc.py $NAT --iters 4 --rounds 1 3000 base=$M/ani_baseline.pt fast=$M/ani_fast.pt 2>&1 | grep water | tee -a $R/ani_${TAG}_pbc.txt
  $PY $A/bench_pbc.py $NAT --iters 10 --rounds 2 6000 fast=$M/ani_fast.pt 2>&1 | grep water | tee -a $R/ani_${TAG}_pbc.txt
fi
