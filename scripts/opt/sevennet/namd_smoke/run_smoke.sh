#!/usr/bin/env bash
# Single-cell namd3 smoke test of a SevenNet artifact, run in its OWN dir (does not touch namd_benchmarks/runs/).
# usage: run_smoke.sh LABEL MODEL.pt {zip|native} [WATER=100] [STEPS=300]
#   zip    -> default C++ libtorch shim (libnamd_mlff.so), no extra libs (stock e3nn deployment)
#   native -> default zip shim + NAMD_MLFF_EXTRA_LIBS=scripts/opt/nequip/oeq_native/liboeq_native.so (OEQ ops)
set -uo pipefail
LABEL=$1; MODEL=$(readlink -f "$2"); MODE=$3; K=${4:-100}; STEPS=${5:-300}
REPO=/home/rat/PycharmProjects/ML_models_NAMD
HERE=$REPO/scripts/opt/sevennet/namd_smoke
SP=/home/rat/miniconda3/envs/allegro/lib/python3.12/site-packages
unset NAMD_MLFF_LIB NAMD_MLFF_EXTRA_LIBS
case "$MODE" in
  zip|native) unset LIBTORCH_ROOT ;;
  *) echo "bad mode $MODE"; exit 2 ;;
esac
source $REPO/namd_benchmarks/env.sh >/dev/null
case "$MODE" in
  native) export NAMD_MLFF_EXTRA_LIBS=$REPO/scripts/opt/nequip/oeq_native/liboeq_native.so ;;
esac
ATOMS=$((3*K)); SYS=$REPO/namd_benchmarks/systems/w${K}_${ATOMS}atoms
coords=$SYS/water.pdb; [ -f $SYS/water_min.coor ] && coords=$SYS/water_min.coor
CELL=$HERE/runs/$LABEL/w${K}; rm -rf $CELL; mkdir -p $CELL /tmp/namd_bench_qmmm; cd $CELL
sed -e "s|@PRMTOP@|$SYS/water.prmtop|" -e "s|@COORDS@|$coords|" -e "s|@QMPDB@|$SYS/qm.pdb|" \
    -e "s|@QMSOFTWARE@|mlff|" -e "s|@QMEXEC@|$MODEL|" -e "s|@QMCHARGEMODE@|none|" -e "s|@QMCHARGE@|0.00|" \
    -e "s|@QMREPLACEALL@|off|" -e "s|@QMBASEDIR@|/tmp/namd_bench_qmmm|" -e "s|@STEPS@|$STEPS|" \
    -e "s|@STEPSPERCYCLE@|50|" -e "s|@OUTPUTFREQ@|50|" -e "s|@OUTPUTNAME@|bench|" -e "s|@EXTRA@|seed 4242|" \
    $REPO/namd_benchmarks/templates/bench.conf.tmpl > bench.conf
echo "[smoke] $LABEL mode=$MODE shim=$NAMD_MLFF_LIB extra=${NAMD_MLFF_EXTRA_LIBS:-none}"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > gpu_mem.txt 2>/dev/null & smp=$!
t0=$(date +%s)
MLFF_TIMING=1 MLFF_TIMING_EVERY=50 timeout 900 $NAMD3 bench.conf > out.log 2>&1; rc=$?
kill $smp 2>/dev/null
echo "[smoke] rc=$rc wall=$(( $(date +%s)-t0 ))s gpu_peak=$(sort -n gpu_mem.txt | tail -1)MiB"
grep -E "^QMENERGY: +(0|1|2|10|50|100) " out.log
grep -E "worker batch|^TIMING:" out.log | tail -4
grep -iE "error|fatal|segfault|abort" out.log | head -5
