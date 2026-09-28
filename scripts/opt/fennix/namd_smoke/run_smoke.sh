#!/usr/bin/env bash
# Single-cell namd3 smoke test of a FeNNiX PJRT artifact (qmSoftware fennol), run in its OWN
# dir (does not touch namd_benchmarks/runs/).
# usage: run_smoke.sh LABEL MANIFEST.json [WATER=100] [STEPS=300] [NAMD3=<default build>]
#   EXTRA_CONF="<namd keywords>" is appended to the config (e.g. "QMNoPntChrg on")
#   env passthrough: FENNIX_MEM_FRACTION, FENNIX_ALLOC, XLA_FLAGS, NAMD_FENNIX_PLUGIN, FENNIX_ENV
set -uo pipefail
LABEL=$1; MANIFEST=$(readlink -f "$2"); K=${3:-100}; STEPS=${4:-300}
REPO=/home/rat/PycharmProjects/ML_models_NAMD
HERE=$REPO/scripts/opt/fennix/namd_smoke
unset NAMD_MLFF_LIB NAMD_MLFF_EXTRA_LIBS LIBTORCH_ROOT
source $REPO/namd_benchmarks/env.sh >/dev/null
[ -n "${5:-}" ] && NAMD3=$5
[ -n "$FENNIX_CUDA_LIBS" ] || { echo "FENNIX_CUDA_LIBS empty (FENNIX_ENV=$FENNIX_ENV)"; exit 2; }
export LD_LIBRARY_PATH="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"
ATOMS=$((3*K)); SYS=$REPO/namd_benchmarks/systems/w${K}_${ATOMS}atoms
coords=$SYS/water.pdb; [ -f $SYS/water_min.coor ] && coords=$SYS/water_min.coor
CELL=$HERE/runs/$LABEL/w${K}; rm -rf $CELL; mkdir -p $CELL /tmp/namd_bench_qmmm; cd $CELL
sed -e "s|@PRMTOP@|$SYS/water.prmtop|" -e "s|@COORDS@|$coords|" -e "s|@QMPDB@|$SYS/qm.pdb|" \
    -e "s|@QMSOFTWARE@|fennol|" -e "s|@QMEXEC@|$MANIFEST|" -e "s|@QMCHARGEMODE@|none|" -e "s|@QMCHARGE@|0.00|" \
    -e "s|@QMREPLACEALL@|off|" -e "s|@QMBASEDIR@|/tmp/namd_bench_qmmm|" -e "s|@STEPS@|$STEPS|" \
    -e "s|@STEPSPERCYCLE@|50|" -e "s|@OUTPUTFREQ@|50|" -e "s|@OUTPUTNAME@|bench|" -e "s|@EXTRA@|seed 4242\n${EXTRA_CONF:-}|" \
    $REPO/namd_benchmarks/templates/bench.conf.tmpl > bench.conf
echo "[smoke] $LABEL extra_conf='${EXTRA_CONF:-}' namd3=$NAMD3 manifest=$MANIFEST plugin=$NAMD_FENNIX_PLUGIN XLA_FLAGS=${XLA_FLAGS:-} memfrac=${FENNIX_MEM_FRACTION:-default}"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > gpu_mem.txt 2>/dev/null & smp=$!
t0=$(date +%s)
timeout 1800 $NAMD3 bench.conf > out.log 2>&1; rc=$?
kill $smp 2>/dev/null
echo "[smoke] rc=$rc wall=$(( $(date +%s)-t0 ))s gpu_peak=$(sort -n gpu_mem.txt | tail -1)MiB"
grep -E "^QMENERGY: +(0|1|2|10|50|100|200|300) " out.log
grep -E "^TIMING:" out.log | tail -3
grep -iE "error|fatal|segfault|abort|FENNIX" out.log | grep -v -E "ptxas|slow_operation" | head -5
