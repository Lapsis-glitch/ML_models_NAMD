#!/usr/bin/env bash
#
# Build + validate the NAMD MLFF libtorch shim (Phase 1, standalone).
#
# Produces:
#   $OUT/libnamd_mlff.so   the shim; the ONLY thing that links libtorch
#   $OUT/mlff_shim_test     validator; links only -ldl and dlopen's the shim
#                           exactly the way NAMD will (RTLD_NOW | RTLD_LOCAL)
#
# With STUB_ENV set (a conda env whose torch version is <= the libtorch
# version) it also mints the deterministic autograd stub and runs the
# analytic + real-model checks.
#
# Env overrides:
#   TORCH     libtorch prefix   (default /home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130;
#             the old 2.6.0+cu126 tree is still at .../libtorch)
#   CUDA      cuda prefix whose include/ matches libtorch's CUDA major
#             (default: the allegro env's nvidia/cu13 pip package for cu130)
#   OUT       output dir        (default /tmp)
#   STUB_ENV  conda env w/ torch (optional; enables the test run)
#   MODEL     extra real .pt to run through the validator (optional)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH="${TORCH:-/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130}"
CUDA="${CUDA:-/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages/nvidia/cu13}"
OUT="${OUT:-/tmp}"

# libtorch's C++ ABI must match ours or std::string-bearing symbols won't link.
# Use grep -c (consumes all input) not grep -q: under `set -o pipefail`, grep -q
# closes the pipe early and nm dies on SIGPIPE, which would falsely report 0.
CXX11_COUNT=$(nm -D "$TORCH/lib/libtorch_cpu.so" 2>/dev/null | grep -c cxx11 || true)
ABI=1; [ "${CXX11_COUNT:-0}" -eq 0 ] && ABI=0
echo "libtorch=$TORCH  cuda=$CUDA  cxx11_symbols=$CXX11_COUNT  cxx11_abi=$ABI"

echo "[1/2] building libnamd_mlff.so"
g++ -shared -fPIC -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=$ABI \
  "$HERE/mlff_shim.cpp" -o "$OUT/libnamd_mlff.so" \
  -I"$HERE" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" \
  -I"$CUDA/include" \
  -L"$TORCH/lib" -Wl,-rpath,"$TORCH/lib" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -Wl,--as-needed \
  -ldl
# (no -lcudart: the shim makes no direct CUDA runtime calls; libtorch_cuda
#  pulls in its own bundled cudart.)

echo "[2/2] building mlff_shim_test (no libtorch link)"
g++ -O2 -std=c++17 "$HERE/mlff_shim_test.cpp" -o "$OUT/mlff_shim_test" -I"$HERE" -ldl

if [ -n "${STUB_ENV:-}" ]; then
  conda run -n "$STUB_ENV" python "$HERE/make_test_stub.py"
  echo "== stub CPU (analytic) =="; "$OUT/mlff_shim_test" "$OUT/libnamd_mlff.so" /tmp/mlff_stub.pt -1 analytic
  echo "== stub GPU (analytic) =="; "$OUT/mlff_shim_test" "$OUT/libnamd_mlff.so" /tmp/mlff_stub.pt 0 analytic
fi
if [ -n "${MODEL:-}" ]; then
  echo "== real model GPU =="; "$OUT/mlff_shim_test" "$OUT/libnamd_mlff.so" "$MODEL" 0
fi

echo "done: $OUT/libnamd_mlff.so  $OUT/mlff_shim_test"
