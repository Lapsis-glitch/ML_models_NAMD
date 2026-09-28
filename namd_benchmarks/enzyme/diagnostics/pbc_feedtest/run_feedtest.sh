#!/usr/bin/env bash
# Decisive check: does NAMD feed FeNNiX the SAME geometry with PBC as without?
# The FeNNiX graph ignores the cell, so step-0 POT must equal the no-PBC value
# (-25837.30 kcal/mol) for any cell that contains the cluster.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENZ="$(cd "$HERE/../.." && pwd)"
source "$ENZ/../env.sh" >/dev/null 2>&1
MAN="$ENZ/models/fennix_enzyme/mono_shell5/manifest.json"
SYS="$ENZ/systems/mono_shell5"

run_one() {  # $1 = L (cubic cell side); 0 = no PBC
  local L="$1" tag conf xsc extra=""
  if [ "$L" = "0" ]; then tag="nopbc"; else tag="pbc$L"
    xsc="$HERE/cell_$L.xsc"
    printf '# NAMD extended system\n#$LABELS step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z\n0 %s 0 0 0 %s 0 0 0 %s 0 0 0\n' "$L" "$L" "$L" > "$xsc"
    extra=$'extendedSystem '"$xsc"$'\nPME on\nPMEGridSpacing 1.0\nwrapAll on'
  fi
  conf="$HERE/$tag.conf"
  cat > "$conf" <<EOF
amber yes
parmfile $SYS/system.prmtop
coordinates $SYS/system.pdb
temperature 300
$extra
binaryoutput no
outputname $HERE/out_$tag
outputenergies 1
cutoff 12.0
pairlistdist 14.0
switching on
switchdist 10.0
exclude scaled1-4
1-4scaling 0.833333
rigidbonds all
langevin on
langevintemp 300
langevinHydrogen on
langevindamping 5
timestep 0.25
stepspercycle 1
fullElectFrequency 1
nonbondedfreq 1
QMForces on
QMSoftware fennol
QMExecPath $MAN
QMColumn beta
qmParamPDB $SYS/qm.pdb
QMBaseDir /tmp/namd_feedtest_$tag
QMChargeMode none
qmBondColumn occ
qmReplaceAll on
QMVdWParams off
qmElecEmbed off
QMSwitching off
QMPointChargeScheme none
QMMult 1 1
QMCharge 1 0
run 4
EOF
  pkill -9 -f '/namd3 ' 2>/dev/null || true
  rm -f /tmp/mlff_namd_*.sock 2>/dev/null || true
  mkdir -p "/tmp/namd_feedtest_$tag"
  LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH" timeout 300 "$NAMD3" "$conf" > "$HERE/out_$tag.log" 2>&1
  local rc=$?
  local pot0="$(grep -E '^ENERGY: +0 ' "$HERE/out_$tag.log" | awk '{print $14}')"
  local ok="$(grep -qE 'End of program|WallClock' "$HERE/out_$tag.log" && echo ok || echo FAIL)"
  printf '%-7s rc=%s %-4s  step0_POT=%s\n' "$tag" "$rc" "$ok" "${pot0:-NONE}"
  grep -iE 'FATAL|ERROR:|abort|terminate|RESOURCE_EXHAUSTED|replaceForces' "$HERE/out_$tag.log" | head -2 || true
}

for L in 0 64 60; do run_one "$L"; done
