#!/usr/bin/env bash
# Reproducibly build every MACE-OFF23 (medium) optimisation artifact in models/opt/.
#
#   mace_baseline.pt     current wrapper + models/compiled_mace_off23_medium.pt   (reference)
#   mace_fast_e3nn.pt    FastMACE exact rewrites on the e3nn inner   -> NAMD default shim, no custom ops
#   mace_cueqf.pt        cuEquivariance + fused conv TP              -> default shim + native op libs
#   mace_fast_cueqf.pt   FastMACE on cuEq (recommended)              -> default shim + native op libs
#   mace_fast_cueqf_f32.pt  fp32 variant of the above (EXTRA, changes numerics; see REPORT.md)
#
# plus the Python-free custom-op library scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so.
# NAMD needs, for the cuEq artifacts (default zip shim, nothing else changed):
#   SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
#   export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:<repo>/scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so
#
# usage: bash scripts/opt/mace/build.sh [--skip-check]      (GPU needed; takes ~10 min)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
# Override for another machine: PY (allegro python), SP (its site-packages), MACE_PY (MACE_312 python), TORCH (libtorch).
PY=${PY:-/home/rat/miniconda3/envs/allegro/bin/python}
SP=${SP:-$("${PY:-/home/rat/miniconda3/envs/allegro/bin/python}" -c 'import site; print(site.getsitepackages()[0])')}
# REQUIRED even at build time: without cu13 nvrtc on the path cuEq silently falls back to its naive path.
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
LOCK="flock $REPO/scripts/opt/.gpu_bench.lock"
M=models/opt

# 0. version-neutral weights dump (once, in the MACE_312 env; e3nn 0.4.4 pickles can't be read under e3nn 0.6)
if [ ! -f $M/mace_off23_medium_state.pt ]; then
  MACE_OFF_MODEL="${MACE_OFF_MODEL:-$HOME/.cache/mace/MACE-OFF23_medium.model}"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${MACE_PY:-/home/rat/miniconda3/envs/MACE_312/bin/python}" \
    scripts/opt/mace/extract_state.py "$MACE_OFF_MODEL" $M/mace_off23_medium_state.pt
fi

# 1. native op library (links libtorch zip + libcue_ops only; no Python)
bash scripts/opt/mace/cueq_native/build.sh

# 2. inner models (scripted, fp64 = native MACE-OFF precision)
$LOCK $PY scripts/opt/mace/compile_inner.py cueqf $M/mace_inner_cueqf_f64.pt
$LOCK $PY scripts/opt/mace/build_fast.py e3nn  $M/mace_inner_fast_e3nn_f64.pt
$LOCK $PY scripts/opt/mace/build_fast.py cueqf $M/mace_inner_fast_cueqf_f64.pt --no-plain-linear
$LOCK $PY scripts/opt/mace/build_fast.py cueqf $M/mace_inner_fast_cueqf_f32.pt --no-plain-linear --dtype float32

# 3. wrap with the CURRENT src/wrappers/wrap_compiled_mace.py (default settings)
$PY scripts/opt/mace/wrap.py models/compiled_mace_off23_medium.pt $M/mace_baseline.pt
$PY scripts/opt/mace/wrap.py $M/mace_inner_fast_e3nn_f64.pt $M/mace_fast_e3nn.pt
$PY scripts/opt/mace/wrap.py --cueq $M/mace_inner_cueqf_f64.pt $M/mace_cueqf.pt
$PY scripts/opt/mace/wrap.py --cueq $M/mace_inner_fast_cueqf_f64.pt $M/mace_fast_cueqf.pt
$PY scripts/opt/mace/wrap.py --cueq $M/mace_inner_fast_cueqf_f32.pt $M/mace_fast_cueqf_f32.pt

[ "${1:-}" = "--skip-check" ] && exit 0

# 4. acceptance: parity through the native op in a process that never imports cuEq python,
#    then the NAMD shim loadability test on the DEFAULT (libtorch zip) shim.
$LOCK $PY scripts/opt/mace/cueq_native/parity_native.py $M/mace_baseline.pt \
  $M/mace_fast_e3nn.pt $M/mace_cueqf.pt $M/mace_fast_cueqf.pt
(
  unset NAMD_MLFF_LIB LIBTORCH_ROOT
  export LD_LIBRARY_PATH=   # namd_benchmarks/env.sh sets its own
  source namd_benchmarks/env.sh >/dev/null
  $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $M/mace_fast_e3nn.pt 0
  export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:$REPO/scripts/opt/mace/cueq_native/libcueq_uniform1d_native.so
  $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $M/mace_fast_cueqf.pt 0
)
