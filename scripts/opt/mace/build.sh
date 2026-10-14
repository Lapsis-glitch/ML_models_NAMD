#!/usr/bin/env bash
# Reproducibly build every MACE-OFF23 (medium) optimisation artifact in models/opt/.
#
#   mace_baseline.pt     current wrapper + models/compiled_mace_off23_medium.pt   (reference)
#   mace_fast_e3nn.pt    FastMACE exact rewrites on the e3nn inner   -> NAMD default shim, no custom ops
#   mace_cueqf.pt        cuEquivariance + fused conv TP              -> default shim + native op libs
#   mace_fast_cueqf.pt   FastMACE on cuEq (recommended)              -> default shim + native op libs
#   mace_fast_cueqf_f32.pt  fp32 variant of the above (EXTRA, changes numerics; see REPORT.md)
#
# plus the Python-free custom-op library src/fast/native/cueq/libcueq_uniform1d_native.so.
# NAMD needs, for the cuEq artifacts (default zip shim, nothing else changed):
#   SP=<allegro env>/lib/python3.12/site-packages
#   export NAMD_MLFF_EXTRA_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:<repo>/src/fast/native/cueq/libcueq_uniform1d_native.so
#
# usage: bash scripts/opt/mace/build.sh [--skip-check]      (GPU needed; takes ~10 min)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
# Override for another machine: PY (allegro python), SP (its site-packages), MACE_PY (MACE_312 python),
# TORCH (libtorch the native lib links against; default: the torch of $PY).
export PY=${PY:-python}
SP=${SP:-$("$PY" -c 'import site; print(site.getsitepackages()[0])')}
export SP
# REQUIRED even at build time: without cu13 nvrtc on the path cuEq silently falls back to its naive path.
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
LOCK="flock $REPO/scripts/opt/.gpu_bench.lock"
M=models/opt
CUEQ_LIBS=$SP/nvidia/cu13/lib/libnvrtc.so.13:$SP/cuequivariance_ops/lib/libcue_ops.so:$REPO/src/fast/native/cueq/libcueq_uniform1d_native.so

# 0. version-neutral weights dump (once, in the MACE_312 env; e3nn 0.4.4 pickles can't be read under e3nn 0.6)
if [ ! -f $M/mace_off23_medium_state.pt ]; then
  MACE_OFF_MODEL="${MACE_OFF_MODEL:-$HOME/.cache/mace/MACE-OFF23_medium.model}"
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "${MACE_PY:?set MACE_PY to the MACE_312 env python}" \
    -m src.fast.mace.extract_state "$MACE_OFF_MODEL" $M/mace_off23_medium_state.pt
fi

# 1. native op library (links libtorch + libcue_ops only; no Python). PY/TORCH pass through the env.
bash src/fast/native/cueq/build.sh

# 2. inner models (scripted, fp64 = native MACE-OFF precision)
#    cuEq + fused conv, kernels only (no FastMACE rewrites)
$LOCK $PY - <<'PYEOF'
import torch
from e3nn.util import jit as ej
from src.fast.mace import _stock
m = _stock("models/opt/mace_off23_medium_state.pt", torch.float64, "cueqf", "cuda")
ej.compile(m).to("cpu").save("models/opt/mace_inner_cueqf_f64.pt")
print("saved models/opt/mace_inner_cueqf_f64.pt")
PYEOF
#    FastMACE (plain_linear on for e3nn, off for cuEq: the historic build settings)
$LOCK $PY -m src.fast.mace $M/mace_off23_medium_state.pt $M/mace_inner_fast_e3nn_f64.pt --kind e3nn --plain-linear
$LOCK $PY -m src.fast.mace $M/mace_off23_medium_state.pt $M/mace_inner_fast_cueqf_f64.pt --kind cueqf
$LOCK $PY -m src.fast.mace $M/mace_off23_medium_state.pt $M/mace_inner_fast_cueqf_f32.pt --kind cueqf --dtype float32

# 3. wrap with the CURRENT src/wrappers/wrap_compiled_mace.py (default settings)
$PY -m src.cli --model-type mace --compiled models/compiled_mace_off23_medium.pt --out $M/mace_baseline.pt
$PY -m src.cli --model-type mace --compiled $M/mace_inner_fast_e3nn_f64.pt --out $M/mace_fast_e3nn.pt
$PY -m src.cli --model-type mace --compiled $M/mace_inner_cueqf_f64.pt --out $M/mace_cueqf.pt --extra-libs "$CUEQ_LIBS"
$PY -m src.cli --model-type mace --compiled $M/mace_inner_fast_cueqf_f64.pt --out $M/mace_fast_cueqf.pt --extra-libs "$CUEQ_LIBS"
$PY -m src.cli --model-type mace --compiled $M/mace_inner_fast_cueqf_f32.pt --out $M/mace_fast_cueqf_f32.pt --extra-libs "$CUEQ_LIBS"

[ "${1:-}" = "--skip-check" ] && exit 0

# 4. acceptance: parity through the native op in a process that never imports cuEq python,
#    then the NAMD shim loadability test on the DEFAULT (libtorch zip) shim.
$LOCK $PY scripts/opt/mace/parity_native.py $M/mace_baseline.pt \
  $M/mace_fast_e3nn.pt $M/mace_cueqf.pt $M/mace_fast_cueqf.pt
(
  unset NAMD_MLFF_LIB LIBTORCH_ROOT
  export LD_LIBRARY_PATH=   # namd_benchmarks/env.sh sets its own
  source namd_benchmarks/env.sh >/dev/null
  $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $M/mace_fast_e3nn.pt 0
  export NAMD_MLFF_EXTRA_LIBS=$CUEQ_LIBS
  $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $M/mace_fast_cueqf.pt 0
)
