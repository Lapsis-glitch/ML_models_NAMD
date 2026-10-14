#!/usr/bin/env bash
# liboeq_native.so: OpenEquivariance's libtorch_tp_jit ops (upstream torch_core.hpp, unchanged) plus
# C++ autograd kernels (NequIP and SevenNet OEQ builds). Links libtorch + cudart/nvrtc/cublas, no Python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
OEQ="$SP/openequivariance/extension"
OUT="${OUTLIB:-$HERE/liboeq_native.so}"
# expose cudart/nvrtc under their SONAMEs so the loader resolves the same files libtorch uses
mkdir -p "$HERE/lib"; ln -sf "$CUDART" "$HERE/lib/libcudart.so.13"; ln -sf "$NVRTC" "$HERE/lib/libnvrtc.so.13"
WSL=""; [ -d /usr/lib/wsl/lib ] && WSL="-L/usr/lib/wsl/lib"
g++ -shared -fPIC -O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -DCUDA_BACKEND \
  "$HERE/oeq_native.cpp" "$OEQ/json11/json11.cpp" -o "$OUT" \
  -I"$OEQ" -I"$OEQ/backend" $TORCH_INC -L"$TL" -L"$CUDA/lib" $WSL -Wl,-rpath,"$HERE/lib:$TL:$CUDA/lib" \
  -Wl,--no-as-needed $TORCH_LIBS "$CUDART" "$NVRTC" -l:libcublas.so.13 -lcuda -Wl,--as-needed
echo "built $OUT"
