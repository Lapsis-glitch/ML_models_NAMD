#!/usr/bin/env bash
# Build liboeq_native.so: OpenEquivariance's libtorch_tp_jit ops (upstream torch_core.hpp, unchanged)
# + C++ Autograd kernels, linked against the libtorch 2.11 zip used by NAMD's default shim.
# NEEDs only libtorch/c10 + the zip's own cudart/nvrtc/cublas + the driver: no Python, no libtorch_python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP=${SP:-$("${PY:-/home/rat/miniconda3/envs/allegro/bin/python}" -c 'import site; print(site.getsitepackages()[0])')}
TORCH="${TORCH:-/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130}"
CUDA="${CUDA:-$SP/nvidia/cu13}"
OEQ="$SP/openequivariance/extension"
TL="$TORCH/lib"
CUDART=$(ls $TL/libcudart-*.so.13); NVRTC=$(ls $TL/libnvrtc-*.so.13)
# the zip ships cudart/nvrtc only under hashed file names (SONAME libcudart.so.13 / libnvrtc.so.13):
# expose them under their SONAMEs in ./lib so the runtime loader resolves to the SAME files libtorch uses.
mkdir -p "$HERE/lib"; ln -sf "$CUDART" "$HERE/lib/libcudart.so.13"; ln -sf "$NVRTC" "$HERE/lib/libnvrtc.so.13"
g++ -shared -fPIC -O3 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -DCUDA_BACKEND \
  "$HERE/oeq_native.cpp" "$OEQ/json11/json11.cpp" -o "${OUTLIB:-$HERE/liboeq_native.so}" \
  -I"$OEQ" -I"$OEQ/backend" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$CUDA/include" \
  -L"$TL" -L/usr/lib/wsl/lib -Wl,-rpath,"$HERE/lib:$TL" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda "$CUDART" "$NVRTC" -l:libcublas.so.13 -lcuda \
  -Wl,--as-needed
echo "built ${OUTLIB:-$HERE/liboeq_native.so}"
