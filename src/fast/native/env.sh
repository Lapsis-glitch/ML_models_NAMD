# Shared settings for the native op library builds (sourced by */build.sh).
#   PY     python of the build env (default: python on PATH)
#   TORCH  libtorch to link against; must be the one NAMD's shim uses
#          (default: the torch package of $PY)
#   CUDA   CUDA root with include/ and lib/ (default: the nvidia CUDA wheel
#          of $PY, else $CUDA_HOME, else /usr/local/cuda)
PY=${PY:-python}
SP=${SP:-$("$PY" -c 'import site; print(site.getsitepackages()[0])')}
TORCH=${TORCH:-$("$PY" -c 'import os, torch; print(os.path.dirname(torch.__file__))')}
if [ -z "${CUDA:-}" ]; then
  for c in "$SP"/nvidia/cu[0-9]* "${CUDA_HOME:-}" /usr/local/cuda; do
    [ -n "$c" ] && [ -d "$c/include" ] && { CUDA=$c; break; }
  done
fi
: "${CUDA:?no CUDA headers found; set CUDA to a CUDA root with include/ and lib/}"
TL="$TORCH/lib"
CUDA_LIB=$(ls -d "$CUDA"/lib64 "$CUDA"/lib 2>/dev/null | head -1)
# a libtorch zip ships cudart/nvrtc under hashed names; otherwise use the CUDA root's
first() { ls $@ 2>/dev/null | head -1 || true; }
CUDART=$(first "$TL"/libcudart-*.so.[0-9]*); CUDART=${CUDART:-$(first "$CUDA_LIB"/libcudart.so.[0-9]*)}
NVRTC=$(first "$TL"/libnvrtc-*.so.[0-9]*);   NVRTC=${NVRTC:-$(first "$CUDA_LIB"/libnvrtc.so.[0-9]*)}
CUBLAS=$(first "$TL"/libcublas.so.[0-9]*);   CUBLAS=${CUBLAS:-$(first "$CUDA_LIB"/libcublas.so.[0-9]*)}
CUDA_MAJOR=${CUDART##*.so.}; CUDA_MAJOR=${CUDA_MAJOR%%.*}
TORCH_INC="-I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$CUDA/include"
TORCH_LIBS="-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda"
