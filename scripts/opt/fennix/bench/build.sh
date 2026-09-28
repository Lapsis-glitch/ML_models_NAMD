#!/usr/bin/env bash
# Builds bench_pjrt against the SAME PJRT runtime sources NAMD compiles in (namd_fennix/src/fennix_pjrt).
set -euo pipefail
cd "$(dirname "$0")"
SRC=${FENNIX_PJRT_SRC:-/home/rat/compile_NAMD_MACE/namd_fennix/src/fennix_pjrt}
g++ -O2 -std=c++17 -fpermissive -w -I"$SRC" -I"$SRC/third_party" bench_pjrt.cpp "$SRC/pjrt_plugin.cpp" "$SRC/artifact_bundle.cpp" -ldl -o bench_pjrt
echo built $(pwd)/bench_pjrt
