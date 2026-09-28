#!/usr/bin/env bash
# ===========================================================================
# Pre-minimize each enzyme droplet ONCE with its classical ff14SB MM force field
# (QMForces OFF), so the full-ML timed run starts from a clash-free structure
# instead of raw crystal coords + freshly added hydrogens.
#
# Why MM (not ML): system.prmtop is a complete AMBER/ff14SB topology, so a cheap
# CG minimization robustly removes the H/clash strain that otherwise NaNs the
# very first FeNNiX dynamics steps (RATTLE failures). MM-relax -> ML-MD is the
# standard workflow.  Writes systems/<name>/system_min.coor (NAMD PDB-format
# coords), which run_enzyme.sh's render_conf picks up automatically as @COORDS@.
#
# Usage:
#   ./pre_minimize_enzyme.sh mono_dry
#   ./pre_minimize_enzyme.sh mono_dry dimer_shell10 ...
#   MINSTEPS=1000 ./pre_minimize_enzyme.sh mono_dry        # deeper
#   ./pre_minimize_enzyme.sh --force mono_dry              # re-minimize
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh" >/dev/null

MINSTEPS="${MINSTEPS:-1000}"
TIMEOUT="${TIMEOUT:-3600}"
FORCE=0; NAMES=()
for a in "$@"; do case "$a" in --force) FORCE=1 ;; *) NAMES+=("$a") ;; esac; done
[ "${#NAMES[@]}" -gt 0 ] || NAMES=(mono_dry)

mkdir -p /tmp/namd_enzyme_qmmm
echo "pre-minimize (MM/ff14SB): steps=$MINSTEPS systems=[${NAMES[*]}]"; echo
for S in "${NAMES[@]}"; do
  SYS="$HERE/systems/$S"; out="$SYS/system_min.coor"
  if [ ! -f "$SYS/system.prmtop" ]; then echo "SKIP $S (not built)"; continue; fi
  if [ "$FORCE" -eq 0 ] && [ -f "$out" ]; then echo "SKIP $S (have system_min.coor)"; continue; fi
  work="$(mktemp -d "/tmp/premin_${S}.XXXXXX")"
  # Pure-MM minimization config: no QM block at all, non-periodic, cutoff electrostatics.
  cat > "$work/min.conf" <<EOF
amber              yes
parmfile           $SYS/system.prmtop
coordinates        $SYS/system.pdb
temperature        0
outputname         relaxed
binaryoutput       no
exclude            scaled1-4
1-4scaling         0.833333
cutoff             12.0
pairlistdist       14.0
switching          on
switchdist         10.0
stepspercycle      20
minimize           $MINSTEPS
EOF
  echo -n "MIN   $S ... "
  t0=$(date +%s)
  ( cd "$work" && timeout "$TIMEOUT" "$NAMD3" min.conf > min.log 2>&1 )
  rc=$?; t1=$(date +%s)
  if [ "$rc" -eq 0 ] && [ -f "$work/relaxed.coor" ]; then
    cp "$work/relaxed.coor" "$out"
    echo "ok ($((t1-t0))s) -> $(basename "$out")"; rm -rf "$work"
  else
    echo "FAILED (rc=$rc, $((t1-t0))s) — log: $work/min.log"; tail -5 "$work/min.log" 2>/dev/null
  fi
done
