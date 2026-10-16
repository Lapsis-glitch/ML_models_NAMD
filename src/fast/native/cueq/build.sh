#!/usr/bin/env bash
# libcueq_uniform1d_native.so: native TORCH_LIBRARY registration of cuequivariance::uniform_1d
# (MACE cuEquivariance builds). Links libtorch + libcue_ops.so only, no Python.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh"
CUEOPS="$SP/cuequivariance_ops"
[ -f "$CUEOPS/lib/libcue_ops.so" ] || die "cuequivariance-ops-torch is not installed: pip install cuequivariance-ops-torch-cu12 (or -cu13, matching torch's CUDA)"
OUT="${OUTLIB:-$HERE/libcueq_uniform1d_native.so}"
g++ -shared -fPIC -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 "$HERE/uniform1d_op.cpp" -o "$OUT" \
  -I"$SP" $TORCH_INC -L"$TL" -L"$CUEOPS/lib" -Wl,-rpath,"$CUEOPS/lib:$CUDA_LIB:$TL" \
  -Wl,--no-as-needed $TORCH_LIBS -lcue_ops -Wl,--as-needed
echo "built $OUT"
