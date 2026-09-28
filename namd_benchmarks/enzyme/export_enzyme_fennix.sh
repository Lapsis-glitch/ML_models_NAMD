#!/usr/bin/env bash
# ===========================================================================
# Export per-system FeNNol (FENNIX-BIO1) StableHLO artifacts for the enzyme
# benchmark.  The FENNIX/PJRT backend is bound to a fixed atom count + atom
# composition + total charge, so each built droplet (build_enzyme.py) needs its
# own artifact, derived from that system's qm.pdb and its meta.json net charge.
# Output: models/fennix_enzyme/<name>/manifest.json  — where run_enzyme.sh looks.
#
# Heavy + fragile: each export lowers and PJRT-compiles a fixed-shape graph.
# mono_dry (~3.3k atoms) compiles on an 8 GB GPU; the solvated dimer (tens of
# thousands of atoms) needs the 48 GB remote box.  Export only what you'll run.
# Runs in the `fennix` conda env.
#
# Usage:
#   ./export_enzyme_fennix.sh mono_dry
#   ./export_enzyme_fennix.sh dimer_shell10           # full solvated (remote)
#   ./export_enzyme_fennix.sh mono_dry dimer_dry ...  # several
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MODEL="$REPO/models/fennix-bio1S.fnx"
EXPORTER="$REPO/scripts/export_fennix_bio1_stablehlo.py"

# The standalone JAX export must use jax's OWN bundled CUDA wheels (nvidia-*-cu12),
# NOT the libtorch / system-CUDA libs that the NAMD run needs.  Those on the
# inherited LD_LIBRARY_PATH shadow jax's cuDNN/cuFFT/cuSOLVER and cause
# "XlaRuntimeError: INTERNAL: the library was not initialized".  So SCRUB
# LD_LIBRARY_PATH for the export (opposite of run_enzyme.sh, which *prepends*
# FENNIX_CUDA_LIBS for the NAMD-linked path).  If a box needs a newer libstdc++,
# override e.g.  EXPORT_LD=/softs/gcc/14.2/lib64 ./export_enzyme_fennix.sh ...
EXPORT_LD="${EXPORT_LD:-}"

NAMES="${*:-mono_dry}"
for S in $NAMES; do
  PDB="$HERE/systems/$S/qm.pdb"
  META="$HERE/systems/$S/meta.json"
  OUT="$HERE/models/fennix_enzyme/$S"
  if [ ! -f "$PDB" ]; then echo "skip $S: missing $PDB (build_enzyme.py $S)"; continue; fi
  if [ -f "$OUT/manifest.json" ]; then echo "skip $S: already exported"; continue; fi
  CH="$(python3 -c "import json;print(json.load(open('$META'))['net_charge'])" 2>/dev/null || echo 0)"
  NAT="$(python3 -c "import json;print(json.load(open('$META'))['n_atoms'])" 2>/dev/null || echo '?')"
  echo "=== exporting FeNNol artifact for $S ($NAT atoms, total charge $CH) ==="
  mkdir -p "$OUT"
  LD_LIBRARY_PATH="$EXPORT_LD" XLA_PYTHON_CLIENT_PREALLOCATE=false conda run -n fennix python "$EXPORTER" \
    --model "$MODEL" --pdb "$PDB" --out-dir "$OUT" --total-charge "$CH" \
    > "$OUT/export.log" 2>&1 \
    && echo "ok: $OUT/manifest.json" \
    || { echo "FAILED $S (see $OUT/export.log; check the traceback — CUDA-lib init, GPU OOM, etc.)"; tail -6 "$OUT/export.log"; }
done
