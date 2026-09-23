#!/usr/bin/env bash
# ===========================================================================
# Export per-size FeNNol (FENNIX-BIO1) StableHLO artifacts for the benchmark.
#
# The FENNIX/PJRT backend is bound to a fixed atom count + composition, so each
# water system needs its own artifact, derived from that system's qm.pdb (atom
# order O,H,H... -> z_list, neutral total charge).  Output goes to
#   models/fennix/w<K>/manifest.json   (+ .stablehlo.mlir and sidecars)
# which is exactly where run_benchmark.sh looks.
#
# Heavy + fragile: each export lowers and PJRT-compiles a fixed-shape graph;
# the largest sizes may exceed the 8 GB GPU during compile.  Export only the
# sizes you intend to benchmark.  Runs in the `fennix` conda env.
#
# Usage:
#   ./export_fennix.sh 10 20 60          # export these water counts
#   ./export_fennix.sh                   # default: all benchmark sizes
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
MODEL="$REPO/models/fennix-bio1S.fnx"
EXPORTER="$REPO/scripts/export_fennix_bio1_stablehlo.py"

COUNTS="${*:-10 20 60 100 300 600 1000 2000}"
for K in $COUNTS; do
  ATOMS=$((3*K))
  PDB="$HERE/systems/w${K}_${ATOMS}atoms/qm.pdb"
  OUT="$HERE/models/fennix/w${K}"
  if [ ! -f "$PDB" ]; then echo "skip w$K: missing $PDB"; continue; fi
  if [ -f "$OUT/manifest.json" ]; then echo "skip w$K: already exported"; continue; fi
  echo "=== exporting FeNNol artifact for w$K ($ATOMS atoms) ==="
  mkdir -p "$OUT"
  XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python "$EXPORTER" \
    --model "$MODEL" --pdb "$PDB" --out-dir "$OUT" \
    && echo "ok: $OUT/manifest.json" \
    || echo "FAILED w$K (likely GPU OOM at compile for large N) — see output above"
done
