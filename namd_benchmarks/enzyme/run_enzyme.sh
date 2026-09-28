#!/usr/bin/env bash
# ===========================================================================
# NAMD3 full-ML ENZYME benchmark driver (OMP decarboxylase, PDB 1X1Z).
#
# Sweeps  model x named-system x walker-count.  Each cell treats the WHOLE
# finite solvated enzyme droplet as the ML region (qmReplaceAll off by default;
# see QMREPLACEALL knob below — off avoids the multi-patch force-scatter crash) and runs
# $STEPS steps, with NAMD timing every $OUTPUTFREQ steps and the native MLFF
# per-eval timing enabled.  Mirrors ../run_benchmark.sh (same env, lib, NAMD
# build, resume/failure-tolerance, replica server-election hygiene) but the
# systems are named droplets built by build_enzyme.py, not water sizes.
#
# Usage:
#   ./run_enzyme.sh                          # default systems x models x walkers
#   ./run_enzyme.sh --dry-run                # print the cell plan, run nothing
#   ./run_enzyme.sh --force                  # re-run completed cells
#   SYSTEMS="mono_dry" MODELS="fennol" WALKERS="0" ./run_enzyme.sh
#   MODELS="fennol mace ani2x" SYSTEMS="dimer_shell10" ./run_enzyme.sh   # remote 48 GB
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh" >/dev/null          # parent env: NAMD3, CHARMRUN, libs, FENNIX_CUDA_LIBS

# ----------------------------- configuration -------------------------------
SYSTEMS="${SYSTEMS:-mono_dry}"                # named dirs under systems/ (build_enzyme.py)
WALKERS="${WALKERS:-0 1 2 4}"
MODELS="${MODELS:-fennol}"                    # fennol (per-system artifact); mace/ani2x also work (HCNOS)
STEPS="${STEPS:-200}"
OUTPUTFREQ="${OUTPUTFREQ:-50}"
TIMESTEP="${TIMESTEP:-1.0}"             # fs; FeNNiX's stiff surface may need <1 fs for stable MD
QMREPLACEALL="${QMREPLACEALL:-off}"     # off = bypass NAMD's buggy multi-patch replaceForces
                                        # scatter (ROOTCAUSE_NAMD_FORCE_BUG.md) so full-ML runs
                                        # stay stable and produce complete timings; set 'on' to
                                        # reproduce the historical crash / once the C++ fix lands.
TIMEOUT="${TIMEOUT:-3600}"
GPU_SAMPLE="${GPU_SAMPLE:-1}"
MINIMIZE="${MINIMIZE:-0}"               # FeNNiX CG-min steps before the timed run (0 = off).
MINTEMP="${MINTEMP:-300}"               # re-seed velocities to this T (K) after minimization.
# Minimizing with the SAME ML potential used for dynamics relaxes the as-built geometry to a
# FeNNiX local minimum, so the timed run starts with small forces (MM-min is counterproductive
# here: a non-periodic T=0 MM relax distorts the droplet into a FeNNiX-high-energy geometry).
if [ "${MINIMIZE}" -gt 0 ] 2>/dev/null; then
  EXTRA_BLOCK="minimize                ${MINIMIZE}\\nreinitvels              ${MINTEMP}"
else
  EXTRA_BLOCK=""
fi

# model -> backend, exec.  fennol exec is the per-system manifest (resolved below).
declare -A BACKEND EXEC
BACKEND[mace]=mlff;        EXEC[mace]="$HERE/../models/mace_off23.pt"
BACKEND[ani2x]=mlff;       EXEC[ani2x]="$HERE/../models/ani2x.pt"
BACKEND[fennol]=fennol;    EXEC[fennol]="PER_SYSTEM"

TMPL="$HERE/templates/enzyme.conf.tmpl"
RUNS="$HERE/runs"
FORCE=0; DRYRUN=0
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    --dry-run) DRYRUN=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

fennol_manifest() { echo "$HERE/models/fennix_enzyme/$1/manifest.json"; }   # $1 = system name
sys_charge() {  # $1 = system name -> net charge from meta.json (default 0)
  python3 -c "import json,sys;print(json.load(open('$HERE/systems/$1/meta.json')).get('net_charge',0))" 2>/dev/null || echo 0
}

render_conf() {  # $1=celldir $2=sysdir $3=qmsoft $4=qmexec $5=outputname $6=qmcharge
  local coords="$2/system.pdb"
  [ -f "$2/system_min.coor" ] && coords="$2/system_min.coor"   # pre-minimized start if present
  sed -e "s|@PRMTOP@|$2/system.prmtop|" -e "s|@COORDS@|$coords|" \
      -e "s|@QMPDB@|$2/qm.pdb|" -e "s|@QMSOFTWARE@|$3|" -e "s|@QMEXEC@|$4|" \
      -e "s|@QMCHARGE@|$6|" -e "s|@TIMESTEP@|$TIMESTEP|" -e "s|@QMREPLACEALL@|$QMREPLACEALL|" \
      -e "s|@QMBASEDIR@|/tmp/namd_enzyme_qmmm|" -e "s|@STEPS@|$STEPS|" \
      -e "s|@STEPSPERCYCLE@|$OUTPUTFREQ|" -e "s|@OUTPUTFREQ@|$OUTPUTFREQ|" \
      -e "s|@OUTPUTNAME@|$5|" -e "s|@EXTRA@|${EXTRA_BLOCK}|" \
      "$TMPL" > "$1/enzyme.conf"
}

classify() {  # $1=exit_code $2=log_glob -> status
  local rc="$1" log="$2" progressed=0
  grep -qE '^TIMING:' $log 2>/dev/null && progressed=1
  if grep -qiE 'out of memory|CUDA error: out of memory|bad_alloc|RESOURCE_EXHAUSTED|cudaErrorMemoryAllocation' $log 2>/dev/null; then
    echo oom
  elif [ "$rc" -eq 124 ]; then
    [ "$progressed" -eq 1 ] && echo timeout_partial || echo timeout_hang
  elif [ "$rc" -eq 0 ] && grep -qE 'End of program|WallClock:' $log 2>/dev/null; then echo ok
  else echo "error_rc${rc}"
  fi
}

# ----------------------------- main sweep ----------------------------------
mkdir -p /tmp/namd_enzyme_qmmm "$RUNS"
total=0; ran=0; skipped=0
echo "grid: models=[$MODELS] systems=[$SYSTEMS] walkers=[$WALKERS] steps=$STEPS freq=$OUTPUTFREQ timeout=${TIMEOUT}s"
echo

for S in $SYSTEMS; do
  SYS="$HERE/systems/$S"
  if [ ! -f "$SYS/system.prmtop" ]; then
    echo "SKIP system $S (not built: run build_enzyme.py $S)"; continue
  fi
  NAT="$(python3 -c "import json;print(json.load(open('$SYS/meta.json'))['n_atoms'])" 2>/dev/null || echo '?')"
  QCH="$(sys_charge "$S")"
  for M in $MODELS; do
    qms="${BACKEND[$M]}"; qme="${EXEC[$M]}"
    [ "$qme" = "PER_SYSTEM" ] && qme="$(fennol_manifest "$S")"
    for W in $WALKERS; do
      total=$((total+1))
      CELL="$RUNS/$M/$S/walk${W}"
      tag="$M  $S(${NAT}at,q${QCH})  walk${W}"
      if [ "$FORCE" -eq 0 ] && [ -f "$CELL/status.txt" ]; then
        skipped=$((skipped+1)); [ "$DRYRUN" -eq 1 ] && echo "SKIP  $tag (done)"; continue
      fi
      if [ "$qms" = "fennol" ] && [ ! -f "$qme" ]; then
        mkdir -p "$CELL"; echo "no_artifact" > "$CELL/status.txt"
        echo "SKIP  $tag (no fennol artifact: export_enzyme_fennix.sh $S)"; continue
      fi
      if [ "$DRYRUN" -eq 1 ]; then echo "RUN   $tag"; continue; fi

      rm -rf "$CELL"; mkdir -p "$CELL"; cd "$CELL"
      pkill -9 -f '/namd3 ' 2>/dev/null; rm -f /tmp/mlff_namd_*.sock
      local_ld="$LD_LIBRARY_PATH"
      [ "$qms" = "fennol" ] && local_ld="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"

      if [ "$W" -eq 0 ]; then
        render_conf "$CELL" "$SYS" "$qms" "$qme" "bench" "$QCH"
        cmd=("$NAMD3" enzyme.conf); outglob="$CELL/out.0.log"
      else
        render_conf "$CELL" "$SYS" "$qms" "$qme" "bench.[myReplica]" "$QCH"
        cmd=("$CHARMRUN" ++local +p"$W" "$NAMD3" +replicas "$W" enzyme.conf +stdout out.%01d.log)
        outglob="$CELL/out.*.log"
      fi

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
      peak=""; [ -f "$CELL/gpu_mem.txt" ] && peak="$(sort -n "$CELL/gpu_mem.txt" 2>/dev/null | tail -1)"
      {
        echo "model=$M"; echo "backend=$qms"; echo "system=$S"; echo "n_atoms=$NAT"
        echo "net_charge=$QCH"; echo "walkers=$W"; echo "steps=$STEPS"
        echo "status=$st"; echo "exit_code=$rc"
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
