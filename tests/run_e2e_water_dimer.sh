#!/usr/bin/env bash
# =======================================================================
#  End-to-end water dimer test — shell wrapper
#
#  Loads the ORCA module, sets up the environment, and runs the full
#  pipeline: generate 20 water dimers → ORCA DFT → train all models
#  → wrap for NAMD.
#
#  Usage:
#      bash tests/run_e2e_water_dimer.sh                  # full run
#      bash tests/run_e2e_water_dimer.sh --skip-orca      # reuse data
#      bash tests/run_e2e_water_dimer.sh --workdir /tmp/e2e
#
#  Environment variables (optional):
#      ORCA_NPROCS    — MPI procs per ORCA job        (default: 1)
#      ORCA_WORKERS   — parallel ORCA workers          (default: 1)
#      ORCA_METHOD    — ORCA method line               (default: B3LYP def2-SVP EnGrad)
#      E2E_WORKDIR    — working directory              (default: auto tmpdir)
# =======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Load ORCA module ----
echo "============================================================"
echo "  Loading ORCA module..."
echo "============================================================"

# Source the modules init script first (needed on many HPC setups where
# 'module' is not automatically available in non-login shells).
for init in /etc/profile.d/modules.sh /usr/share/modules/init/bash \
            /opt/modules/init/bash /usr/share/Modules/init/bash; do
    if [[ -f "$init" ]]; then
        source "$init" 2>/dev/null
        break
    fi
done

if type module &>/dev/null; then
    module load orca 2>&1 || echo "WARNING: 'module load orca' failed"
else
    echo "WARNING: 'module' command not found — cannot load orca module"
fi

# Verify ORCA is available
if command -v orca &>/dev/null; then
    ORCA_BIN="$(command -v orca)"
    echo "  ORCA found: $ORCA_BIN"
    # ASE 3.23+ needs the bare command; set it so Python inherits it.
    export ASE_ORCA_COMMAND="${ASE_ORCA_COMMAND:-$ORCA_BIN}"
    echo "  ASE_ORCA_COMMAND=$ASE_ORCA_COMMAND"
    echo "  PATH includes: $(dirname "$ORCA_BIN")"
else
    echo "  WARNING: 'orca' not found on PATH after module load"
    echo "  The script will fall back to synthetic mock data."
fi

echo ""

# ---- Configuration ----
ORCA_NPROCS="${ORCA_NPROCS:-1}"
ORCA_WORKERS="${ORCA_WORKERS:-1}"
ORCA_METHOD="${ORCA_METHOD:-B3LYP def2-SVP EnGrad}"

# ---- Build CLI args ----
ARGS=()
ARGS+=("--orca-nprocs" "$ORCA_NPROCS")
ARGS+=("--n-workers" "$ORCA_WORKERS")
ARGS+=("--orca-method" "$ORCA_METHOD")
ARGS+=("--verbose")

if [[ -n "${E2E_WORKDIR:-}" ]]; then
    ARGS+=("--workdir" "$E2E_WORKDIR")
fi

# Pass through any extra CLI args (e.g. --skip-orca, --workdir)
ARGS+=("$@")

# ---- Run ----
echo "============================================================"
echo "  Running end-to-end water dimer pipeline"
echo "  Project: $PROJECT_ROOT"
echo "  ORCA:    nprocs=$ORCA_NPROCS  workers=$ORCA_WORKERS"
echo "  Method:  $ORCA_METHOD"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"
python -u tests/test_e2e_water_dimer.py "${ARGS[@]}"

