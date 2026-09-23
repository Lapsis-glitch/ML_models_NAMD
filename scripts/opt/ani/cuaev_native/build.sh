#!/usr/bin/env bash
# Build libcuaev_native.so: TorchANI's cuAEV extension (torchani 2.7.9 csrc: aev.cu, cuaev.cpp, unchanged
# except <torch/extension.h> -> <torch/torch.h>, i.e. no pybind11/Python) linked against the libtorch 2.11 zip
# used by NAMD's default shim. It registers TORCH_LIBRARY(cuaev) (custom class CuaevComputer + ops run*,
# CUDA + C++ Autograd kernels upstream) -> loadable via NAMD_MLFF_EXTRA_LIBS, no Python.
#
# nvcc: CUDA 13.0 (sm_120 needs >= 12.8; /usr/bin/nvcc is 12.0, /usr/local/cuda 12.6) from pip wheels
# installed with `pip install --target scripts/opt/ani/toolchain nvidia-cuda-nvcc==13.0.88
# nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 --no-deps` (allegro untouched).
#
# VARIANT=precise (default): plain nvcc math (expf/cosf/...), no TORCHANI_OPT, no -use_fast_math.
# VARIANT=upstream        : torchani's own CMake flags (-use_fast_math + TORCHANI_OPT intrinsics).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
TORCH="${TORCH:-/home/rat/compile_NAMD_MACE/libtorch-2.11.0+cu130}"
CUDA_RT="$SP/nvidia/cu13"                             # runtime headers (cuda_runtime.h, ...)
NVCC_HOME="${NVCC_HOME:-$HERE/../toolchain/nvidia/cu13}"  # nvcc + cccl (cub) + crt
NVCC="$NVCC_HOME/bin/nvcc"
SRC="$SP/torchani/csrc"
VARIANT="${VARIANT:-precise}"
OUT="${OUTLIB:-$HERE/libcuaev_native_${VARIANT}.so}"
TL="$TORCH/lib"
CUDART=$(ls $TL/libcudart-*.so.13)
mkdir -p "$HERE/src" "$HERE/lib" "$HERE/obj"
ln -sf "$CUDART" "$HERE/lib/libcudart.so.13"
for f in aev.cu aev.h cuaev.cpp cuaev_cub.cuh; do
  sed 's#<torch/extension.h>#<torch/torch.h>#' "$SRC/$f" > "$HERE/src/$f"
done
DEFS="-D_GLIBCXX_USE_CXX11_ABI=1"
INC="-I$HERE/src -I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$CUDA_RT/include"
if [ "$VARIANT" = upstream ]; then CUFLAGS="-use_fast_math -DTORCHANI_OPT"; else CUFLAGS=""; fi
"$NVCC" -c -O3 -std=c++17 -Xcompiler -fPIC $DEFS $INC $CUFLAGS -DCUB_WRAPPED_NAMESPACE=cuaev \
  -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 \
  --expt-extended-lambda --expt-relaxed-constexpr \
  "$HERE/src/aev.cu" -o "$HERE/obj/aev_${VARIANT}.o"
g++ -c -O3 -std=c++17 -fPIC $DEFS $INC "$HERE/src/cuaev.cpp" -o "$HERE/obj/cuaev.o"
g++ -shared -o "$OUT" "$HERE/obj/cuaev.o" "$HERE/obj/aev_${VARIANT}.o" \
  -L"$TL" -Wl,-rpath,"$HERE/lib:$TL" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda "$CUDART" -Wl,--as-needed
echo "built $OUT"
