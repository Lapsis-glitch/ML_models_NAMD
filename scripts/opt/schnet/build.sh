#!/usr/bin/env bash
# schnet-nl: rebuild every SchNet artifact of the optimisation study.
#
#   bash scripts/opt/schnet/build.sh            # artifacts only
#   BUILD_SHIM=1 bash scripts/opt/schnet/build.sh   # + the CUDA-graph shim variant
#
# Inputs : models/compiled_schnet_default.pt (inner SchNetPack model, r_max 5.0 A)
# Outputs: models/opt/schnet_baseline.pt     wrapper source BEFORE schnet-nl (from the .pre_schnet_nl.bak copies)
#          models/opt/schnet_default_new.pt  current source, default flags (shared edges.py improvements only)
#          models/opt/schnet_fast.pt         current source, --fast (fast path + half-list filter + cell-list NL + CUDA-graph API)
#          models/opt/schnet_fast_fullfilter.pt  --fast --no-half-filter (ablation: fast path without the half-list filter)
#          scripts/opt/schnet/cudagraph/shim/libnamd_mlff.so   (BUILD_SHIM=1) shim + opt-in NAMD_MLFF_CUDA_GRAPH=1
set -euo pipefail
REPO=/home/rat/PycharmProjects/ML_models_NAMD
PY=/home/rat/miniconda3/envs/allegro/bin/python
HERE=$REPO/scripts/opt/schnet
INNER=$REPO/models/compiled_schnet_default.pt
OUT=$REPO/models/opt
mkdir -p "$OUT"
cd "$REPO"

# 1) baseline = the pre-schnet-nl wrapper + edges.py, exported from a scratch copy of src/
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cp -r "$REPO/src" "$TMP/src"
find "$TMP/src" -name __pycache__ -prune -exec rm -rf {} +
cp "$HERE/nl/edges.py.pre_schnet_nl.bak" "$TMP/src/edges.py"
cp "$HERE/wrap_schnetpack.py.pre_schnet_nl.bak" "$TMP/src/wrappers/wrap_schnetpack.py"
(cd "$TMP" && "$PY" -m src.wrappers.wrap_schnetpack --model "$INNER" --r-max 5.0 --out "$OUT/schnet_baseline.pt")

# 2) current source, default flags
"$PY" -m src.wrappers.wrap_schnetpack --model "$INNER" --r-max 5.0 --out "$OUT/schnet_default_new.pt"

# 3) current source, opt-in fast path
"$PY" -m src.wrappers.wrap_schnetpack --model "$INNER" --r-max 5.0 --fast \
      --graph-max-atoms 2048 --nl-cell-min-pairs 16000000 --half-min-atoms 1500 --out "$OUT/schnet_fast.pt"
"$PY" -m src.wrappers.wrap_schnetpack --model "$INNER" --r-max 5.0 --fast --no-half-filter \
      --graph-max-atoms 2048 --nl-cell-min-pairs 16000000 --out "$OUT/schnet_fast_fullfilter.pt"

# 4) optional: shim variant with the opt-in CUDA-graph path (does NOT touch the installed shim)
if [ "${BUILD_SHIM:-0}" = 1 ]; then
  S=/home/rat/compile_NAMD_MACE/namd_fennix/src/mlff_shim
  D=$HERE/cudagraph/shim
  cp "$S/mlff_shim.cpp" "$D/mlff_shim.cpp.orig"
  cp "$S/mlff_shim.h" "$S/mlff_shim_test.cpp" "$S/build_and_test.sh" "$D/"
  python3 "$D/apply_cuda_graph_patch.py" "$D/mlff_shim.cpp.orig" "$D/mlff_shim.cpp"
  OUT=$D bash "$D/build_and_test.sh"
fi
ls -la "$OUT"/schnet_*.pt
