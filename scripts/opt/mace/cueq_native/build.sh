#!/usr/bin/env bash
# Build libcueq_uniform1d_native.so: native TORCH_LIBRARY registration of cuequivariance::uniform_1d.
# Links only libtorch (the C++ zip used by NAMD's default shim) + libcue_ops.so -- no Python, no libtorch_python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
TORCH="${TORCH:-/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130}"
CUDA="${CUDA:-$SP/nvidia/cu13}"
CUEOPS="$SP/cuequivariance_ops"
g++ -shared -fPIC -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 \
  "$HERE/uniform1d_op.cpp" -o "${OUTLIB:-$HERE/libcueq_uniform1d_native.so}" \
  -I"$SP" -I"$TORCH/include" -I"$TORCH/include/torch/csrc/api/include" -I"$CUDA/include" \
  -L"$TORCH/lib" -L"$CUEOPS/lib" -Wl,-rpath,"$CUEOPS/lib" -Wl,-rpath,"$CUDA/lib" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda -lcue_ops -Wl,--as-needed
echo "built ${OUTLIB:-$HERE/libcueq_uniform1d_native.so}"
