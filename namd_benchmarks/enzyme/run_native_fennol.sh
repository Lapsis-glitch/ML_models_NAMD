#!/usr/bin/env bash
# ===========================================================================
# Run a built enzyme droplet through FeNNol's OWN recommended MD driver
# (fennol_md) — the "standard FeNNiX run".  Unlike the NAMD path (run_enzyme.sh),
# this uses FeNNol's native JAX integrator at full precision, which is
# energy-conserving and STABLE on these systems (the NAMD shim's forces drift /
# explode; see README "Two engines / the drift").
#
# Settings follow FeNNol's protein example (examples/md/dhfr): dt=0.5 fs, Langevin
# thermostat, matmul_prec highest, non-periodic finite droplet (no cell).
#
# fennol_md JITs the full MD-step graph at runtime, which needs more GPU memory
# than the NAMD StableHLO path: the ~4.5k-atom droplet OOM/segfaults on an 8 GB
# GPU but runs on the 48 GB remote box.  Locally, use DEVICE=cpu (slow but stable).
#
# Usage:
#   DEVICE=cpu     ./run_native_fennol.sh mono_shell5      # local 8 GB: CPU
#   DEVICE=cuda:0  ./run_native_fennol.sh dimer_shell6     # remote 48 GB: GPU
#   NSTEPS=2000 DT=0.5 TEMP=300 ./run_native_fennol.sh mono_shell5
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh" >/dev/null

REPO="$(cd "$HERE/../.." && pwd)"
MODEL="$REPO/models/fennix-bio1S.fnx"
FENNOL="$FENNIX_ENV/bin/fennol_md"
CPPTRAJ_PY="$(dirname "$(command -v conda)")/../envs/cpptraj/bin/python"
[ -x "$CPPTRAJ_PY" ] || CPPTRAJ_PY="$HOME/miniconda3/envs/cpptraj/bin/python"

S="${1:-mono_shell5}"
DEVICE="${DEVICE:-cpu}"        # cuda:0 on the 48 GB box; cpu locally (8 GB OOMs)
NSTEPS="${NSTEPS:-200}"
DT="${DT:-0.5}"                # fs (FeNNol recommends <=0.5 for proteins)
TEMP="${TEMP:-300}"
GAMMA="${GAMMA:-10}"           # Langevin friction [THz]
NPRINT="${NPRINT:-10}"

SYS="$HERE/systems/$S"
[ -f "$SYS/system.prmtop" ] || { echo "system $S not built (build_enzyme.py $S)"; exit 1; }
OUT="$HERE/native_fennol/$S"; mkdir -p "$OUT"
QCH="$(python3 -c "import json;print(json.load(open('$SYS/meta.json'))['net_charge'])" 2>/dev/null || echo 0)"

# 1) coordinates -> Tinker-indexed xyz (FeNNol xyz_input{indexed yes})
"$CPPTRAJ_PY" "$HERE/native_fennol/pdb_to_tinker_xyz.py" "$SYS" "$OUT/system.xyz"

# 2) input.fnl (non-periodic; total_charge from the system's net charge)
{
  echo "device $DEVICE"
  echo "matmul_prec highest"
  echo "model_file $MODEL"
  [ "$QCH" -ne 0 ] 2>/dev/null && echo "total_charge $QCH"
  cat <<EOF
xyz_input{
  file system.xyz
  indexed yes
  has_comment_line no
}
nsteps = $NSTEPS
dt[fs] = $DT
traj_format xyz
nblist_skin 2.
tdump[ps] = 1.
nprint = $NPRINT
nsummary = 100
thermostat LGV
temperature = $TEMP
gamma[THz] = $GAMMA
EOF
} > "$OUT/input.fnl"

# 3) run.  GPU needs JAX's bundled CUDA libs ahead on LD_LIBRARY_PATH; CPU forces JAX_PLATFORMS=cpu.
echo "=== native fennol_md: $S ($(sed -n 1p "$OUT/system.xyz") atoms, q=$QCH) device=$DEVICE steps=$NSTEPS dt=$DT fs ==="
cd "$OUT"
if [ "$DEVICE" = "cpu" ]; then
  JAX_PLATFORMS=cpu "$FENNOL" input.fnl 2>&1 | tee run.log
else
  LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$FENNOL" input.fnl 2>&1 | tee run.log
fi
