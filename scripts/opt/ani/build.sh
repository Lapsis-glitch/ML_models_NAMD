#!/usr/bin/env bash
# Reproducibly build the ANI-2x optimisation artifacts (models/opt/ani_*.pt) + parity + NAMD-loadability.
#   bash scripts/opt/ani/build.sh            (~5 min; GPU steps take scripts/opt/.gpu_bench.lock)
# Needs: allegro env (torch 2.11+cu130, torchani 2.7.9), the libtorch 2.11 zip, and the CUDA 13.0 nvcc wheels in
# scripts/opt/ani/toolchain (installed once with:
#   pip install --no-cache-dir --no-deps --target scripts/opt/ani/toolchain nvidia-cuda-nvcc==13.0.88 \
#       nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88   -- allegro itself untouched)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
PY=/home/rat/miniconda3/envs/allegro/bin/python
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export TORCHANI_NO_WARN_EXTENSIONS=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
A=scripts/opt/ani; M=models/opt
LIB=$REPO/$A/cuaev_native/libcuaev_native_precise.so
EL="element_list=[1,6,7,8,16,9,17]"      # torchani ANI-2x species order H C N O S F Cl (see REPORT: wrapper default is wrong for F/Cl)
LOCK="flock scripts/opt/.gpu_bench.lock"

[ -x $A/toolchain/nvidia/cu13/bin/nvcc ] || $PY -m pip install --no-cache-dir --no-deps --target $A/toolchain \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88

echo "== 1. native cuAEV op library (torchani csrc, libtorch zip, no Python)"
VARIANT=precise bash $A/cuaev_native/build.sh

echo "== 2. baseline: current wrapper (defaults) around the deployed inner"
$PY $A/wrap.py models/compiled_ani2x.pt $M/ani_baseline.pt
$PY $A/wrap.py models/compiled_ani2x.pt $M/ani_baseline_lean.pt lean=True

echo "== 3. FastANI inners (torchani 2.7.9 pretrained ANI-2x; weights asserted equal to models/compiled_ani2x.pt)"
$PY $A/build_fast.py --cache-species --group-max-atoms 1024 $M/ani_inner_fast.pt           # recommended
$PY $A/build_fast.py $M/ani_inner_fast_nocache.pt                                         # no species cache
$PY $A/build_fast.py --cache-species --group-max-atoms 1024 --acc32 $M/ani_inner_fast_acc32.pt  # stock fp32 energy sum
for v in fast fast_nocache fast_acc32; do
  $PY $A/wrap.py --lib $LIB $M/ani_inner_$v.pt $M/ani_$v.pt "$EL" lean=True
done

echo "== 4. parity: fp64 torchani reference, then artifacts in a fresh process without torchani"
$LOCK $PY $A/ref_fp64.py $A/ref_fp64.pt
$LOCK $PY $A/parity_native.py --lib $LIB --ref $A/ref_fp64.pt base=$M/ani_baseline.pt \
  fast=$M/ani_fast.pt fast_nocache=$M/ani_fast_nocache.pt fast_acc32=$M/ani_fast_acc32.pt | tee $A/parity.txt

echo "== 5. NAMD loadability: default zip shim + NAMD_MLFF_EXTRA_LIBS=<cuAEV lib>"
( unset LD_LIBRARY_PATH NAMD_MLFF_LIB LIBTORCH_ROOT; source namd_benchmarks/env.sh >/dev/null
  for v in baseline fast fast_nocache fast_acc32; do
    if [ $v = baseline ]; then unset NAMD_MLFF_EXTRA_LIBS; else export NAMD_MLFF_EXTRA_LIBS=$LIB; fi
    echo "-- $v (extra libs: ${NAMD_MLFF_EXTRA_LIBS:-none})"
    $LOCK namd_benchmarks/lib/mlff_shim_test $NAMD_MLFF_LIB $M/ani_$v.pt 0 | tail -4
  done ) | tee $A/shim_test.txt
echo "done"
