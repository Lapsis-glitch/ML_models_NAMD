#!/usr/bin/env bash
# Reproducibly build everything for the FeNNiX (FENNIX-BIO1, JAX/PJRT) optimisation.
#
#   1. bench/bench_pjrt           C++ PJRT bench on the SAME runtime sources NAMD compiles in
#   2. models/opt/fennix_fp32/w<K>  EXTRA: true-fp32 exports (matmul precision HIGHEST, not TF32)
#      models/opt/fennix_w4/w<K>    EXTRA: 4-walker vmap exports [4,N,3]
#   3. $NAMD_OPT_ROOT (default ~/compile_NAMD_MACE/namd_fennix_fxopt): a COPY of the namd_fennix tree
#      with namd_patch/fennix_backend.patch (+ optional computeqm_fast_index.patch) applied and namd3
#      rebuilt.  The original namd_fennix tree/binary is never modified.
#
# The shipped single-walker artifacts (namd_benchmarks/models/fennix/w<K>) are used unchanged: the
# recommended optimisation is entirely on the NAMD side (patched backend + config), same executable.
#
# usage: bash scripts/opt/fennix/build.sh [--skip-exports] [--skip-namd]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
FENNIX_PY=${FENNIX_PY:-/home/rat/miniconda3/envs/fennix/bin/python}
NAMD_SRC_ROOT=${NAMD_SRC_ROOT:-/home/rat/compile_NAMD_MACE/namd_fennix}
NAMD_OPT_ROOT=${NAMD_OPT_ROOT:-/home/rat/compile_NAMD_MACE/namd_fennix_fxopt}
LOCK=$REPO/scripts/opt/.gpu_bench.lock
SKIP_EXPORTS=0; SKIP_NAMD=0
for a in "$@"; do case $a in --skip-exports) SKIP_EXPORTS=1;; --skip-namd) SKIP_NAMD=1;; esac; done

echo "== 1. bench tool"
FENNIX_PJRT_SRC=$NAMD_SRC_ROOT/src/fennix_pjrt bash "$HERE/bench/build.sh"

if [ $SKIP_EXPORTS -eq 0 ]; then
  echo "== 2. extra exports (fennix env, GPU)"
  mkdir -p "$REPO/models/opt/fennix_fp32" "$REPO/models/opt/fennix_w4" "$HERE/logs"
  export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_CPP_MIN_LOG_LEVEL=2
  for K in 10 100 300 1000 2000; do
    A=$((3*K)); PDB=$REPO/namd_benchmarks/systems/w${K}_${A}atoms/qm.pdb
    # true fp32: JAX_DEFAULT_MATMUL_PRECISION=highest bakes precision=HIGHEST into every dot_general
    JAX_DEFAULT_MATMUL_PRECISION=highest flock "$LOCK" "$FENNIX_PY" "$REPO/scripts/export_fennix_bio1_stablehlo.py" \
      --model "$REPO/models/fennix-bio1S.fnx" --pdb "$PDB" --out-dir "$REPO/models/opt/fennix_fp32/w$K" \
      > "$HERE/logs/export_fp32_w$K.log" 2>&1
    flock "$LOCK" "$FENNIX_PY" "$REPO/scripts/export_fennix_bio1_stablehlo.py" \
      --model "$REPO/models/fennix-bio1S.fnx" --pdb "$PDB" --n-walkers 4 --out-dir "$REPO/models/opt/fennix_w4/w$K" \
      > "$HERE/logs/export_w4_w$K.log" 2>&1
    echo "  w$K: fp32 HIGHEST dots=$(grep -c HIGHEST "$REPO/models/opt/fennix_fp32/w$K/fennix_bio1_eval.stablehlo.mlir")  w4 ok"
  done
fi

if [ $SKIP_NAMD -eq 0 ]; then
  echo "== 3. patched namd3 in $NAMD_OPT_ROOT"
  if [ ! -d "$NAMD_OPT_ROOT" ]; then
    cp -a "$NAMD_SRC_ROOT" "$NAMD_OPT_ROOT"
    (cd "$NAMD_OPT_ROOT" && patch -p1 < "$HERE/namd_patch/fennix_backend.patch" \
                         && patch -p1 < "$HERE/namd_patch/computeqm_fast_index.patch")
  else
    echo "  (tree exists; assuming patches already applied: $(grep -c fxopt "$NAMD_OPT_ROOT/src/ComputeFennix.C") fxopt markers)"
  fi
  # ComputeFennix.C / fennix_pjrt/*.cpp are #included by ComputeQM.C and not in Make.depends
  touch "$NAMD_OPT_ROOT/src/ComputeQM.C"
  (cd "$NAMD_OPT_ROOT/Linux-x86_64-g++" && make -j4 > "$HERE/logs/namd_build.log" 2>&1)
  n=$(strings "$NAMD_OPT_ROOT/Linux-x86_64-g++/namd3" | grep -c FENNIX_LEGACY_EXEC || true)
  echo "  namd3 rebuilt: $NAMD_OPT_ROOT/Linux-x86_64-g++/namd3 (patch markers in binary: $n)"
fi
echo "done. bench grid: bash $HERE/run_grid.sh ; NAMD smoke: $HERE/namd_smoke/run_smoke.sh"
