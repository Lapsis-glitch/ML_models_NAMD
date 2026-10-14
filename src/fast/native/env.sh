# Shared settings for the native op library builds (sourced by */build.sh).
#   PY     python of the build env (default: python on PATH)
#   TORCH  libtorch to link against; must be the one NAMD's shim uses
#          (default: the torch package of $PY)
#   CUDA   CUDA headers (default: the nvidia cu13 wheel of $PY)
PY=${PY:-python}
SP=${SP:-$("$PY" -c 'import site; print(site.getsitepackages()[0])')}
TORCH=${TORCH:-$("$PY" -c 'import os, torch; print(os.path.dirname(torch.__file__))')}
CUDA=${CUDA:-$SP/nvidia/cu13}
TL="$TORCH/lib"
# the libtorch zip ships cudart/nvrtc under hashed names; a pip torch uses the nvidia wheels
CUDART=$(ls "$TL"/libcudart-*.so.13 2>/dev/null | head -1 || true); CUDART=${CUDART:-$CUDA/lib/libcudart.so.13}
NVRTC=$(ls "$TL"/libnvrtc-*.so.13 2>/dev/null | head -1 || true); NVRTC=${NVRTC:-$CUDA/lib/libnvrtc.so.13}
TORCH_INC="-I$TORCH/include -I$TORCH/include/torch/csrc/api/include -I$CUDA/include"
TORCH_LIBS="-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda"
