#!/usr/bin/env bash
# Reproducibly build every NequIP-OAM-L artifact of the nequip optimisation agent (allegro env).
#   1. native OEQ op library (no Python)             scripts/opt/nequip/oeq_native/liboeq_native.so
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
# Override for another machine: PY (allegro python), SP (its site-packages), TORCH (libtorch), PKG (model package).
PY=${PY:-/home/rat/miniconda3/envs/allegro/bin/python}
SP=${SP:-$("${PY:-/home/rat/miniconda3/envs/allegro/bin/python}" -c 'import site; print(site.getsitepackages()[0])')}
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
LOCK=scripts/opt/.gpu_bench.lock
NL=scripts/opt/nequip/oeq_native/liboeq_native.so
M=models/opt
# NequIP-OAM-L 0.1 package (nequip.net id mir-group/NequIP-OAM-L:0.1, cached by nequip)
PKG=${PKG:-$(ls ~/.nequip/model_cache/039e5620e6c89fd53bdd3b6bd33574e5b29704bb4df5bc339869baca3d5eead6.nequip.zip)}

bash scripts/opt/nequip/oeq_native/build.sh

flock $LOCK $PY scripts/opt/nequip/compile_ts.py "$PKG" $M/nequip_oam_l_cuda.nequip.pth --device cuda
flock $LOCK $PY scripts/opt/nequip/compile_ts.py "$PKG" $M/nequip_inner_oeq.nequip.pth --device cuda --modifiers enable_OpenEquivariance
flock $LOCK $PY scripts/opt/nequip/compile_ts.py "$PKG" $M/nequip_inner_cueq.nequip.pth --device cuda --modifiers enable_CuEquivariance
flock $LOCK $PY scripts/opt/nequip/build_fast.py "$PKG" $M/nequip_inner_fast_oeq.nequip.pth

$PY scripts/opt/nequip/wrap.py models/compiled_nequip_oam_l.nequip.pth $M/nequip_baseline.pt r_max=6.0
$PY scripts/opt/nequip/wrap.py $M/nequip_oam_l_cuda.nequip.pth $M/nequip_recompiled_cuda.pt r_max=6.0
$PY scripts/opt/nequip/wrap.py --lib $NL $M/nequip_inner_oeq.nequip.pth $M/nequip_oeq.pt r_max=6.0
$PY scripts/opt/nequip/wrap.py --lib $NL $M/nequip_inner_fast_oeq.nequip.pth $M/nequip_fast_oeq.pt r_max=6.0
$PY scripts/opt/nequip/wrap.py --import cuequivariance_torch $M/nequip_inner_cueq.nequip.pth $M/nequip_cueq.pt r_max=6.0

# parity + NAMD loadability (native op lib only; this process never imports openequivariance)
flock $LOCK $PY scripts/opt/nequip/parity_native.py
flock $LOCK $PY scripts/opt/nequip/parity_fast.py
( unset NAMD_MLFF_LIB LIBTORCH_ROOT; source namd_benchmarks/env.sh >/dev/null
  export NAMD_MLFF_EXTRA_LIBS=$REPO/$NL
  for m in $M/nequip_baseline.pt $M/nequip_oeq.pt $M/nequip_fast_oeq.pt; do
    echo "== mlff_shim_test $m"; flock $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $m 0 | tail -4
  done )
