#!/usr/bin/env bash
# libcuaev_native_<VARIANT>.so: TorchANI's cuAEV extension (the installed torchani's csrc, with
# <torch/extension.h> -> <torch/torch.h>, i.e. no pybind11) linked against libtorch, no Python.
#
# Needs nvcc >= 12.8 for sm_120. NVCC_HOME picks one; otherwise the CUDA 13.0 nvcc wheels are
# installed once into ./toolchain (the build env itself is not touched).
#
# VARIANT=precise (default): plain nvcc math, no -use_fast_math.
# VARIANT=upstream         : torchani's own flags (-use_fast_math + TORCHANI_OPT intrinsics).
# ARCH (default 120): CUDA compute capability to build for.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
VARIANT="${VARIANT:-precise}"
ARCH="${ARCH:-120}"
OUT="${OUTLIB:-$HERE/libcuaev_native_${VARIANT}.so}"
if [ -z "${NVCC_HOME:-}" ]; then
  NVCC_HOME="$HERE/toolchain/nvidia/cu13"
  [ -x "$NVCC_HOME/bin/nvcc" ] || "$PY" -m pip install --no-cache-dir --no-deps --target "$HERE/toolchain" \
      nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
fi
NVCC="$NVCC_HOME/bin/nvcc"
mkdir -p "$HERE/src" "$HERE/lib" "$HERE/obj"
ln -sf "$CUDART" "$HERE/lib/libcudart.so.13"
for f in aev.cu aev.h cuaev.cpp cuaev_cub.cuh; do
  sed 's#<torch/extension.h>#<torch/torch.h>#' "$SP/torchani/csrc/$f" > "$HERE/src/$f"
done
DEFS="-D_GLIBCXX_USE_CXX11_ABI=1"
INC="-I$HERE/src $TORCH_INC"
if [ "$VARIANT" = upstream ]; then CUFLAGS="-use_fast_math -DTORCHANI_OPT"; else CUFLAGS=""; fi
"$NVCC" -c -O3 -std=c++17 -Xcompiler -fPIC $DEFS $INC $CUFLAGS -DCUB_WRAPPED_NAMESPACE=cuaev \
  -gencode arch=compute_$ARCH,code=sm_$ARCH -gencode arch=compute_$ARCH,code=compute_$ARCH \
  --expt-extended-lambda --expt-relaxed-constexpr \
  "$HERE/src/aev.cu" -o "$HERE/obj/aev_${VARIANT}.o"
g++ -c -O3 -std=c++17 -fPIC $DEFS $INC "$HERE/src/cuaev.cpp" -o "$HERE/obj/cuaev.o"
g++ -shared -o "$OUT" "$HERE/obj/cuaev.o" "$HERE/obj/aev_${VARIANT}.o" \
  -L"$TL" -Wl,-rpath,"$HERE/lib:$TL" \
  -Wl,--no-as-needed $TORCH_LIBS "$CUDART" -Wl,--as-needed
echo "built $OUT"
