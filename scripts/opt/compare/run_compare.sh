#!/usr/bin/env bash
# Cross-model bench: recommended NAMD-loadable artifacts vs their baselines, one (N, W) per invocation,
# all native op libs in one process (verified to coexist), 15 GiB VRAM cap, expandable_segments.
# usage: run_compare.sh [N W]...   (default: full grid)
set -u
REPO=/home/rat/PycharmProjects/ML_models_NAMD; cd "$REPO"
PY=/home/rat/miniconda3/envs/allegro/bin/python
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-} TORCHANI_NO_WARN_EXTENSIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LIBS=(--extra-lib $SP/nvidia/cu13/lib/libnvrtc.so.13 --extra-lib $SP/cuequivariance_ops/lib/libcue_ops.so
      --extra-lib scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so
      --extra-lib scripts/opt/nequip/oeq_native/liboeq_native.so
      --extra-lib scripts/opt/ani/cuaev_native/libcuaev_native_precise.so)
M=models/opt
cells=("$@"); [ ${#cells[@]} -eq 0 ] && cells=(30 1 300 1 900 1 3000 1 6000 1 30 4 300 4 900 4 3000 4 6000 4)
for ((i=0; i<${#cells[@]}; i+=2)); do
  N=${cells[i]}; W=${cells[i+1]}; NW=$((N*W))
  models=()
  # schnet baseline: block-diagonal NL peaks 15.4 GB at 6000x4 -> reported alone in schnet REPORT
  [ $NW -lt 24000 ] && models+=(--model schnet_base=$M/schnet_baseline.pt)
  models+=(--model schnet_fast=$M/schnet_fast.pt)
  [ $NW -le 1200 ] && models+=(--model mace_base=$M/mace_baseline.pt)
  models+=(--model mace_fast=$M/mace_fast_cueqf.pt --model mace_fast_f32=$M/mace_fast_cueqf_f32.pt)
  [ $NW -le 1200 ] && models+=(--model nequip_base=$M/nequip_baseline.pt)
  models+=(--model nequip_fast=$M/nequip_fast_oeq.pt)
  models+=(--model ani_base=$M/ani_baseline.pt --model ani_fast=$M/ani_fast.pt)
  it=30; [ $NW -ge 3000 ] && it=10
  echo "=== N=$N W=$W ($(date +%T))"
  $PY scripts/opt/mace/bench_capped.py "${LIBS[@]}" "${models[@]}" --systems $N --walkers $W \
      --jitter 0.02 --warmup 10 --iters $it --rounds 3 --out scripts/opt/results/compare_w${N}_W${W}.json 2>&1 \
      | grep -vE "Warning|warnings.warn|^\s*$"
done
