#!/usr/bin/env bash
# libcuaev_native_<VARIANT>.so: TorchANI's cuAEV extension (the installed torchani's csrc, with
# <torch/extension.h> -> <torch/torch.h>, i.e. no pybind11) linked against libtorch, no Python.
#
# NVCC_HOME: CUDA toolkit whose bin/nvcc supports the target GPU. Default: the first of $CUDA,
#   nvcc on PATH, ./toolchain whose nvcc supports ARCH; else (CUDA 13 only) the nvcc wheels are
#   pip-installed into ./toolchain.
# ARCH: compute capability to build for, e.g. 89, 90, 120 (default: the GPU of $PY's torch).
# VARIANT=precise (default): plain nvcc math, no -use_fast_math.
# VARIANT=upstream         : torchani's own flags (-use_fast_math + TORCHANI_OPT intrinsics).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
VARIANT="${VARIANT:-precise}"
ARCH="${ARCH:-$("$PY" -c 'import torch; a, b = torch.cuda.get_device_capability(); print(f"{a}{b}")')}"
OUT="${OUTLIB:-$HERE/libcuaev_native_${VARIANT}.so}"
supports() { [ -x "$1/bin/nvcc" ] && "$1/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_$ARCH"; }
if [ -z "${NVCC_HOME:-}" ]; then
  PATH_CUDA=$(command -v nvcc >/dev/null && dirname "$(dirname "$(command -v nvcc)")" || true)
  for c in "$CUDA" "$PATH_CUDA" "$HERE/toolchain/nvidia/cu13"; do
    if [ -n "$c" ] && supports "$c"; then NVCC_HOME=$c; break; fi
  done
fi
if [ -z "${NVCC_HOME:-}" ] && [ "$CUDA_MAJOR" = 13 ]; then
  "$PY" -m pip install --no-cache-dir --no-deps --target "$HERE/toolchain" \
      nvidia-cuda-nvcc==13.0.88 nvidia-cuda-cccl==13.0.85 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
  NVCC_HOME="$HERE/toolchain/nvidia/cu13"
fi
: "${NVCC_HOME:?no nvcc supporting sm_$ARCH found: set NVCC_HOME to a CUDA $CUDA_MAJOR toolkit that does}"
NVCC="$NVCC_HOME/bin/nvcc"
mkdir -p "$HERE/src" "$HERE/lib" "$HERE/obj"
ln -sf "$CUDART" "$HERE/lib/libcudart.so.$CUDA_MAJOR"
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
