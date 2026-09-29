#!/usr/bin/env bash
# Reproducibly build the SevenNet-0 (7net-0) artifacts of the SevenNet optimisation (allegro env, GPU).
#   1. native OEQ op library, shared with NequIP (no Python)   scripts/opt/nequip/oeq_native/liboeq_native.so
#   2. inner deployments (SevenNet's own LAMMPS serial deploy)
#        models/compiled_sevennet_0.pt        e3nn tensor products (BASELINE inner)
#        models/opt/sevennet_inner_oeq.pt     OpenEquivariance fused TP-conv kernels, same weights
#        models/opt/sevennet_inner_fast.pt    OEQ + exact FastSevenNet rewrites (src/sevennet_fast.py)
#   3. wraps with the CURRENT src/wrappers/wrap_sevennet.py (src/cli.py)
#        sevennet_baseline.pt / sevennet_baseline_d3.pt   e3nn inner (+ D3(BJ), PBE)
#        sevennet_oeq.pt / sevennet_oeq_d3.pt             OEQ inner (+ D3)  (need liboeq_native.so)
#        sevennet_fast.pt / sevennet_fast_d3.pt           FastSevenNet inner (+ D3)
#                                                         (RECOMMENDED; need liboeq_native.so)
#   4. parity (fresh process, native lib only, no openequivariance import) + mlff_shim_test on the default NAMD shim
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$REPO"
# Override for another machine: PY (allegro python), SP (its site-packages), CKPT (SevenNet checkpoint).
PY=${PY:-/home/rat/miniconda3/envs/allegro/bin/python}
SP=${SP:-$("$PY" -c 'import site; print(site.getsitepackages()[0])')}
export LD_LIBRARY_PATH=$SP/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
CKPT=${CKPT:-7net-0}
LOCK=scripts/opt/.gpu_bench.lock
NL=scripts/opt/nequip/oeq_native/liboeq_native.so
M=models/opt
mkdir -p $M

[ -f $NL ] || bash scripts/opt/nequip/oeq_native/build.sh

[ -f models/compiled_sevennet_0.pt ] || $PY -m src.compile_sevennet --checkpoint "$CKPT" --out models/compiled_sevennet_0.pt
flock $LOCK $PY -m src.compile_sevennet --checkpoint "$CKPT" --oeq --out $M/sevennet_inner_oeq.pt
flock $LOCK $PY -m src.compile_sevennet --checkpoint "$CKPT" --fast --out $M/sevennet_inner_fast.pt

$PY -m src.cli --model-type sevennet --compiled models/compiled_sevennet_0.pt --out $M/sevennet_baseline.pt
$PY -m src.cli --model-type sevennet --compiled models/compiled_sevennet_0.pt --d3 --out $M/sevennet_baseline_d3.pt
$PY -m src.cli --model-type sevennet --compiled $M/sevennet_inner_oeq.pt --extra-libs $NL --out $M/sevennet_oeq.pt
$PY -m src.cli --model-type sevennet --compiled $M/sevennet_inner_oeq.pt --extra-libs $NL --d3 --out $M/sevennet_oeq_d3.pt
$PY -m src.cli --model-type sevennet --compiled $M/sevennet_inner_fast.pt --extra-libs $NL --out $M/sevennet_fast.pt
$PY -m src.cli --model-type sevennet --compiled $M/sevennet_inner_fast.pt --extra-libs $NL --d3 --out $M/sevennet_fast_d3.pt

# parity + NAMD loadability (native op lib only; this process never imports openequivariance)
flock $LOCK $PY scripts/opt/sevennet/parity.py
( unset NAMD_MLFF_LIB LIBTORCH_ROOT; source namd_benchmarks/env.sh >/dev/null
  export NAMD_MLFF_EXTRA_LIBS=$REPO/$NL
  for m in $M/sevennet_baseline.pt $M/sevennet_oeq.pt $M/sevennet_fast.pt $M/sevennet_fast_d3.pt; do
    echo "== mlff_shim_test $m"; flock $LOCK namd_benchmarks/lib/mlff_shim_test "$NAMD_MLFF_LIB" $m 0 | tail -4
  done )
