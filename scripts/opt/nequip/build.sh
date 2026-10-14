#!/usr/bin/env bash
# Reproducibly build every NequIP-OAM-L artifact of the nequip optimisation agent (allegro env).
#   1. native OEQ op library (no Python)             src/fast/native/oeq/liboeq_native.so
#   2. inner compiles from the NequIP-OAM-L package  models/opt/nequip_inner_{oeq,cueq}.nequip.pth, nequip_oam_l_cuda.nequip.pth
#   3. wraps with the CURRENT src/wrappers/wrap_compiled_nequip.py
#        nequip_baseline.pt        deployed inner models/compiled_nequip_oam_l.nequip.pth (BASELINE)
#        nequip_recompiled_cuda.pt same e3nn model recompiled in torch 2.11 (fp32 noise-floor reference)
#        nequip_oeq.pt             OpenEquivariance TP-conv kernels (needs liboeq_native.so)
#        nequip_fast_oeq.pt        OEQ + exact FastNequIP rewrites (half-edge MLP/conv, per-type self-connection)
#                                  (RECOMMENDED; needs liboeq_native.so)
#        nequip_cueq.pt            cuEquivariance (python-only ops, benchmark reference, NOT NAMD-loadable)
#   4. parity (fresh process, no openequivariance import) + mlff_shim_test on the default NAMD shim
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
# Override for another machine: PY (allegro python), SP (its site-packages), TORCH (libtorch the native
# lib links against; default: the torch of $PY), PKG (model package).
export PY=${PY:-python}
SP=${SP:-$("$PY" -c 'import site; print(site.getsitepackages()[0])')}
export SP
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
LOCK=scripts/opt/.gpu_bench.lock
NL=src/fast/native/oeq/liboeq_native.so
M=models/opt
# NequIP-OAM-L 0.1 package (nequip.net id mir-group/NequIP-OAM-L:0.1, cached by nequip)
PKG=${PKG:-$(ls ~/.nequip/model_cache/039e5620e6c89fd53bdd3b6bd33574e5b29704bb4df5bc339869baca3d5eead6.nequip.zip)}

bash src/fast/native/oeq/build.sh

flock $LOCK $PY -m src.compile_nequip --package "$PKG" --out $M/nequip_oam_l_cuda.nequip.pth --device cuda
flock $LOCK $PY -m src.compile_nequip --package "$PKG" --out $M/nequip_inner_oeq.nequip.pth --device cuda --modifiers enable_OpenEquivariance
flock $LOCK $PY -m src.compile_nequip --package "$PKG" --out $M/nequip_inner_cueq.nequip.pth --device cuda --modifiers enable_CuEquivariance
flock $LOCK $PY -m src.fast.nequip "$PKG" $M/nequip_inner_fast_oeq.nequip.pth

$PY -m src.cli --model-type nequip --compiled models/compiled_nequip_oam_l.nequip.pth --out $M/nequip_baseline.pt --r-max 6.0
$PY -m src.cli --model-type nequip --compiled $M/nequip_oam_l_cuda.nequip.pth --out $M/nequip_recompiled_cuda.pt --r-max 6.0
$PY -m src.cli --model-type nequip --compiled $M/nequip_inner_oeq.nequip.pth --out $M/nequip_oeq.pt --r-max 6.0 --extra-libs $NL
$PY -m src.cli --model-type nequip --compiled $M/nequip_inner_fast_oeq.nequip.pth --out $M/nequip_fast_oeq.pt --r-max 6.0 --extra-libs $NL
# cuEq ops are python-registered: import cuequivariance_torch before loading the inner (benchmark only)
$PY -c "import cuequivariance_torch, sys; from src.cli import main; main(sys.argv[1:])" \
  --model-type nequip --compiled $M/nequip_inner_cueq.nequip.pth --out $M/nequip_cueq.pt --r-max 6.0

# parity + NAMD loadability (native op lib only; this process never imports openequivariance)
flock $LOCK $PY scripts/opt/nequip/parity_native.py
flock $LOCK $PY scripts/opt/nequip/parity_fast.py
( unset NAMD_MLFF_LIB LIBTORCH_ROOT; source namd_benchmarks/env.sh >/dev/null
  export NAMD_MLFF_EXTRA_LIBS=$REPO/$NL
  for m in $M/nequip_baseline.pt $M/nequip_oeq.pt $M/nequip_fast_oeq.pt; do
    echo "== mlff_shim_test $m"; flock $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $m 0 | tail -4
  done )
