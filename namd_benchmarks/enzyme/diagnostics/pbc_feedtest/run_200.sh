#!/usr/bin/env bash
# 200-step (50 fs) FeNNiX PBC run to confirm stability PAST the old ~11 fs crash point.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENZ="$(cd "$HERE/../.." && pwd)"
source "$ENZ/../env.sh" >/dev/null 2>&1
sed -e 's|^run 4|run 200|' -e 's|^outputenergies 1|outputenergies 10|' \
    -e "s|out_pbc64|out_pbc64_200|g" "$HERE/pbc64.conf" > "$HERE/pbc64_200.conf"
pkill -9 -f '/namd3 ' 2>/dev/null || true
rm -f /tmp/mlff_namd_*.sock 2>/dev/null || true
mkdir -p /tmp/namd_feedtest_pbc64
LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH" timeout 400 "$NAMD3" "$HERE/pbc64_200.conf" > "$HERE/out_pbc64_200.log" 2>&1
echo "rc=$?  end=$(grep -qE 'End of program|WallClock' "$HERE/out_pbc64_200.log" && echo ok || echo FAIL)"
grep -E '^ENERGY:' "$HERE/out_pbc64_200.log" | awk '{printf "step %4s  TEMP=%7.1f  POT=%12.1f  TOTAL=%12.1f\n",$2,$13,$14,$12}'
grep -iE 'FATAL|RATTLE|Constraint failure|abort|terminate' "$HERE/out_pbc64_200.log" | head -3 || true
