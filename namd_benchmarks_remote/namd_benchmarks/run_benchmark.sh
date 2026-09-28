#!/usr/bin/env bash
# ===========================================================================
# NAMD3 ML-FF / FeNNol / xTB QM benchmark driver.
#
# Sweeps  model x system-size x walker-count, runs each cell for $STEPS steps
# with NAMD timing every $OUTPUTFREQ steps and the native MLFF per-eval timing
# enabled, and records a status + the raw logs under runs/.
#
# Design:
#   * cheap -> expensive: sizes ascending, then walkers ascending, so useful
#     data lands before the slow/large tail (which may OOM on an 8 GB GPU).
#   * resumable: a cell with a completed runs/.../status.txt is skipped. Delete
#     that file (or pass --force) to re-run a cell.
#   * failure-tolerant: per-cell `timeout`; OOM / error / timeout are recorded
#     as the cell status and the sweep continues.
#   * walker 0 = a plain single `namd3` run (no replica framework) — the genuine
#     no-replica baseline.  walkers >=1 = `charmrun ++local +pN ... +replicas N`.
#
# Usage:
#   source env.sh            # (the driver also sources it itself)
#   ./run_benchmark.sh                 # run the whole grid
#   ./run_benchmark.sh --dry-run       # print the cell plan, run nothing
#   ./run_benchmark.sh --force         # re-run even completed cells
#   MODELS="mace xtb" ./run_benchmark.sh          # subset of models
#   WATER="10 20 60" WALKERS="0 1 2 4" ./run_benchmark.sh   # subset of grid
# ===========================================================================
set -uo pipefail
export OMP_NUM_THREADS=8
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" >/dev/null

# ----------------------------- configuration -------------------------------
# water-molecule counts -> atom counts are 3x these (30 60 180 300 900 1800 3000 6000)
WATER="${WATER:-10 20 60 100 300 600 1000 2000}"
WALKERS="${WALKERS:-0 1 2 3 4 6 8}"
MODELS="${MODELS:-mace ani2x nequip_oam schnet xtb fennol}"
STEPS="${STEPS:-200}"
OUTPUTFREQ="${OUTPUTFREQ:-50}"          # NAMD timing/energy + MLFF timing cadence + stepspercycle
TIMEOUT="${TIMEOUT:-1800}"               # seconds, per cell (then recorded as 'timeout')
GPU_SAMPLE="${GPU_SAMPLE:-1}"            # 1 = sample nvidia-smi peak memory per cell
MINIMIZE="${MINIMIZE:-0}"               # inline CG-min steps before the timed run (0 = off;
                                        # the default path is pre_minimize.sh, see below)
MINTEMP="${MINTEMP:-300}"               # re-seed velocities to this T (K) after inline minimization

# Preferred relaxation path is pre_minimize.sh, which minimizes each system ONCE
# with a robust model and writes systems/.../water_min.coor; render_conf below picks
# that up as the starting structure automatically. The MINIMIZE knob here is a
# secondary, opt-in INLINE minimization (each cell minimizes with its own model
# before the timed run). NB: inline `minimize` advances NAMD's step counter, so the
# timed run's PERFORMANCE step labels are offset by +$MINIMIZE — the steady-state
# parser must subtract that. (pre_minimize.sh has no such offset.)
if [ "${MINIMIZE}" -gt 0 ] 2>/dev/null; then
  EXTRA_BLOCK="minimize                ${MINIMIZE}\\nreinitvels              ${MINTEMP}"
else
  EXTRA_BLOCK=""
fi

# model -> (backend, exec).  exec for fennol is resolved per-size below.
declare -A BACKEND EXEC
BACKEND[mace]=mlff;        EXEC[mace]="$HERE/models/mace_off23.pt"
BACKEND[ani2x]=mlff;       EXEC[ani2x]="$HERE/models/ani2x.pt"
BACKEND[nequip_oam]=mlff;  EXEC[nequip_oam]="$HERE/models/nequip_oam.pt"
BACKEND[schnet]=mlff;      EXEC[schnet]="$HERE/models/schnet.pt"
BACKEND[xtb]=xtb;          EXEC[xtb]="xtb"
BACKEND[fennol]=fennol;    EXEC[fennol]="PER_SIZE"   # models/fennix/w<K>/manifest.json

TMPL="$HERE/templates/bench.conf.tmpl"
RUNS="$HERE/runs"
FORCE=0; DRYRUN=0
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    --dry-run) DRYRUN=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# ----------------------------- helpers -------------------------------------
fennol_manifest() { echo "$HERE/models/fennix/w$1/manifest.json"; }   # $1 = water count

render_conf() {  # $1=dir $2=sys $3=qmsoft $4=qmexec $5=outputname
  local coords="$2/water.pdb"
  # start the timed run from the pre-minimized structure if pre_minimize.sh made one
  [ -f "$2/water_min.coor" ] && coords="$2/water_min.coor"
  sed -e "s|@PRMTOP@|$2/water.prmtop|" -e "s|@COORDS@|$coords|" \
      -e "s|@QMPDB@|$2/qm.pdb|" -e "s|@QMSOFTWARE@|$3|" -e "s|@QMEXEC@|$4|" \
      -e "s|@QMCHARGEMODE@|none|" -e "s|@QMCHARGE@|0.00|" \
      -e "s|@QMBASEDIR@|/tmp/namd_bench_qmmm|" -e "s|@STEPS@|$STEPS|" \
      -e "s|@STEPSPERCYCLE@|$OUTPUTFREQ|" -e "s|@OUTPUTFREQ@|$OUTPUTFREQ|" \
      -e "s|@OUTPUTNAME@|$5|" -e "s|@EXTRA@|${EXTRA_BLOCK}|" \
      "$TMPL" > "$1/bench.conf"
}

classify() {  # $1=exit_code $2=log_glob  -> echo status
  local rc="$1" log="$2"
  local progressed=0
  grep -qE '^TIMING:' $log 2>/dev/null && progressed=1
  if grep -qiE 'out of memory|CUDA error: out of memory|bad_alloc|RESOURCE_EXHAUSTED|cudaErrorMemoryAllocation' $log 2>/dev/null; then
    echo oom
  elif [ "$rc" -eq 124 ]; then
    # distinguish a slow-but-working cell (made progress) from a true hang
    [ "$progressed" -eq 1 ] && echo timeout_partial || echo timeout_hang
  elif [ "$rc" -eq 0 ] && grep -qE 'End of program|WallClock:' $log 2>/dev/null; then echo ok
  else echo "error_rc${rc}"
  fi
}

# ----------------------------- main sweep ----------------------------------
mkdir -p /tmp/namd_bench_qmmm "$RUNS"
total=0; ran=0; skipped=0
echo "grid: models=[$MODELS] water=[$WATER] walkers=[$WALKERS] steps=$STEPS freq=$OUTPUTFREQ timeout=${TIMEOUT}s"
echo

for K in $WATER; do                          # ascending size  (cheap -> expensive)
  ATOMS=$((3*K)); SYS="$HERE/systems/w${K}_${ATOMS}atoms"
  for M in $MODELS; do
    qms="${BACKEND[$M]}"
    qme="${EXEC[$M]}"
    [ "$qme" = "PER_SIZE" ] && qme="$(fennol_manifest "$K")"
    for W in $WALKERS; do
      total=$((total+1))
      CELL="$RUNS/$M/w${K}_${ATOMS}atoms/walk${W}"
      tag="$M  ${ATOMS}atoms  walk${W}"
      # resume / skip
      if [ "$FORCE" -eq 0 ] && [ -f "$CELL/status.txt" ]; then
        skipped=$((skipped+1)); [ "$DRYRUN" -eq 1 ] && echo "SKIP  $tag (done)"; continue
      fi
      # fennol: skip cells whose per-size artifact has not been exported
      if [ "$qms" = "fennol" ] && [ ! -f "$qme" ]; then
        mkdir -p "$CELL"; echo "no_artifact" > "$CELL/status.txt"
        echo "SKIP  $tag (no fennol artifact: $qme)"; continue
      fi
      if [ "$DRYRUN" -eq 1 ]; then echo "RUN   $tag"; continue; fi

      rm -rf "$CELL"; mkdir -p "$CELL"; cd "$CELL"
      # hygiene: clear any stale MLFF server sockets / stray procs from a prior
      # (possibly killed) cell so the replica server-election starts clean.
      # NB: NAMD renames its Charm++ PE threads (comm becomes "NAMD masterPe"),
      # so `pkill namd3` matches NOTHING and stragglers leak (accumulating GPU
      # memory + contention that corrupts later cells). Match the binary path in
      # the full command line instead — this catches both the namd3 workers and
      # their charmrun parent. ('/namd3 ' is a plain substring, no regex traps
      # like the '++' in the Linux-x86_64-g++ path component.)
      pkill -9 -f '/namd3 ' 2>/dev/null; rm -f /tmp/mlff_namd_*.sock
      # backend-specific LD_LIBRARY_PATH: fennol needs JAX's cuDNN ahead of libtorch
      local_ld="$LD_LIBRARY_PATH"
      [ "$qms" = "fennol" ] && local_ld="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"

      if [ "$W" -eq 0 ]; then
        render_conf "$CELL" "$SYS" "$qms" "$qme" "bench"
        cmd=("$NAMD3" bench.conf)
        outglob="$CELL/out.0.log"
      else
        render_conf "$CELL" "$SYS" "$qms" "$qme" "bench.[myReplica]"
        cmd=("$CHARMRUN" ++local +p"$W" "$NAMD3" +replicas "$W" bench.conf +stdout out.%01d.log)
        outglob="$CELL/out.*.log"
      fi

      # optional GPU memory sampler
      smp=""
      if [ "$GPU_SAMPLE" -eq 1 ] && command -v nvidia-smi >/dev/null 2>&1; then
        ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) \
          > "$CELL/gpu_mem.txt" 2>/dev/null & smp=$!
      fi

      echo -n "RUN   $tag ... "
      t0=$(date +%s)
      if [ "$W" -eq 0 ]; then
        LD_LIBRARY_PATH="$local_ld" MLFF_TIMING=1 MLFF_TIMING_EVERY="$OUTPUTFREQ" \
          timeout "$TIMEOUT" "${cmd[@]}" > out.0.log 2>&1
      else
        LD_LIBRARY_PATH="$local_ld" MLFF_TIMING=1 MLFF_TIMING_EVERY="$OUTPUTFREQ" \
          timeout "$TIMEOUT" "${cmd[@]}" > namd.log 2>&1
      fi
      rc=$?; t1=$(date +%s)
      [ -n "$smp" ] && kill "$smp" 2>/dev/null
      st="$(classify "$rc" "$outglob")"
      peak=""
      [ -f "$CELL/gpu_mem.txt" ] && peak="$(sort -n "$CELL/gpu_mem.txt" 2>/dev/null | tail -1)"
      {
        echo "model=$M"; echo "backend=$qms"; echo "n_waters=$K"; echo "n_atoms=$ATOMS"
        echo "walkers=$W"; echo "steps=$STEPS"; echo "status=$st"; echo "exit_code=$rc"
        echo "wall_seconds=$((t1-t0))"; echo "gpu_peak_mib=${peak:-NA}"
      } > "$CELL/status.txt"
      ran=$((ran+1))
      echo "$st  (${rc}, $((t1-t0))s${peak:+, ${peak}MiB})"
      cd "$HERE"
    done
  done
done

echo
echo "done: $ran ran, $skipped skipped, $total total cells."
echo "gather:  python gather_results.py"
