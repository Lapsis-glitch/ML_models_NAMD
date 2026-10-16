#!/usr/bin/env bash
# liboeq_native.so: OpenEquivariance's libtorch_tp_jit ops (upstream torch_core.hpp, unchanged) plus
# C++ autograd kernels (NequIP and SevenNet OEQ builds). Links libtorch + cudart/nvrtc/cublas, no Python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
OEQ="$SP/openequivariance/extension"
[ -d "$OEQ" ] || die "openequivariance is not installed: pip install openequivariance"
[ -n "$NVRTC" ] && [ -n "$CUBLAS" ] || die "libnvrtc / libcublas not found in $TL or $CUDA_LIB"
OUT="${OUTLIB:-$HERE/liboeq_native.so}"
# expose cudart/nvrtc under their SONAMEs so the loader resolves the same files libtorch uses
mkdir -p "$HERE/lib"
ln -sf "$CUDART" "$HERE/lib/libcudart.so.$CUDA_MAJOR"; ln -sf "$NVRTC" "$HERE/lib/libnvrtc.so.$CUDA_MAJOR"
WSL=""; [ -d /usr/lib/wsl/lib ] && WSL="-L/usr/lib/wsl/lib"
g++ -shared -fPIC -O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -DCUDA_BACKEND \
  "$HERE/oeq_native.cpp" "$OEQ/json11/json11.cpp" -o "$OUT" \
  -I"$OEQ" -I"$OEQ/backend" $TORCH_INC -L"$TL" -L"$CUDA_LIB" $WSL -Wl,-rpath,"$HERE/lib:$TL:$CUDA_LIB" \
  -Wl,--no-as-needed $TORCH_LIBS "$CUDART" "$NVRTC" "$CUBLAS" -lcuda -Wl,--as-needed
echo "built $OUT"
