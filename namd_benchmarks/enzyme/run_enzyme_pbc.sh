#!/usr/bin/env bash
# ===========================================================================
# NAMD3 full-ML ENZYME benchmark driver — PERIODIC (PBC + PME) variant.
#
# Companion to run_enzyme.sh (finite droplet).  Drives the periodic solvated
# boxes built by build_enzyme_pbc.py (systems/pbc_*) through the periodic
# template (templates/enzyme_pbc.conf.tmpl): CHARMM psf + .xsc cell + PME +
# wrapAll, whole system = ML region.  Same env / libs / timing / walker /
# resume logic as run_enzyme.sh.
#
# Smoke-test (no ML export needed — pure MM/CHARMM+PME, validates box/psf/xsc):
#   SMOKE_MM=1 SYSTEMS=pbc_mono STEPS=20 OUTPUTFREQ=5 ./run_enzyme_pbc.sh
#
# Full-ML run (needs the per-system FeNNiX artifact; 39k atoms -> 48 GB box):
#   ./export_enzyme_fennix.sh pbc_mono
#   SYSTEMS=pbc_mono MODELS=fennol WALKERS=0 STEPS=60 OUTPUTFREQ=5 TIMESTEP=0.5 ./run_enzyme_pbc.sh
#
# Usage / knobs mirror run_enzyme.sh:
#   ./run_enzyme_pbc.sh --dry-run | --force
#   SYSTEMS MODELS WALKERS STEPS OUTPUTFREQ TIMESTEP QMREPLACEALL TIMEOUT MINIMIZE
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../env.sh" >/dev/null

# ----------------------------- configuration -------------------------------
SYSTEMS="${SYSTEMS:-pbc_mono}"          # named dirs under systems/ (build_enzyme_pbc.py)
WALKERS="${WALKERS:-0}"
MODELS="${MODELS:-fennol}"              # fennol (per-system artifact); SMOKE_MM=1 -> pure MM
STEPS="${STEPS:-60}"
OUTPUTFREQ="${OUTPUTFREQ:-5}"
TIMESTEP="${TIMESTEP:-0.5}"             # fs; FeNNiX's stiff surface may want <=0.5
QMREPLACEALL="${QMREPLACEALL:-on}"     # full-ML: engine supplies ALL forces. 'off' was the
                                       # pre-fix multi-patch workaround (see ROOTCAUSE doc).
TIMEOUT="${TIMEOUT:-3600}"
GPU_SAMPLE="${GPU_SAMPLE:-1}"
MINIMIZE="${MINIMIZE:-0}"              # FeNNiX CG-min steps before the timed run (0 = off)
MINTEMP="${MINTEMP:-300}"
SMOKE_MM="${SMOKE_MM:-0}"              # 1 = MM/CHARMM+PME equilibration (QMForces off); promotes
                                       #     the relaxed restart to systems/<S>/system_equil.{coor,xsc}
                                       #     which the full-ML run then starts from.
MM_MIN="${MM_MIN:-500}"               # CG-minimisation steps in the MM equilibration
RIGIDBONDS="${RIGIDBONDS:-all}"       # 'all' (RATTLE H bonds) or 'none' (native FeNNol uses none)
START="${START:-equil}"               # 'equil' = start full-ML from the MM-relaxed restart (avoids
                                       #   fresh-build clashes; for a PERIODIC box this is bulk water,
                                       #   in-distribution — unlike the droplet case in README gotchas).
                                       # 'asbuilt' = start from the freshly-solvated system.pdb.
                                       # Verify on the remote: compare step-0 FeNNiX POT for both.

EXTRA_BLOCK=""
if [ "${MINIMIZE}" -gt 0 ] 2>/dev/null; then
  EXTRA_BLOCK="minimize                ${MINIMIZE}\\nreinitvels              ${MINTEMP}"
fi

declare -A BACKEND EXEC
BACKEND[mace]=mlff;        EXEC[mace]="$HERE/../models/mace_off23.pt"
BACKEND[ani2x]=mlff;       EXEC[ani2x]="$HERE/../models/ani2x.pt"
BACKEND[fennol]=fennol;    EXEC[fennol]="PER_SYSTEM"

TMPL="$HERE/templates/enzyme_pbc.conf.tmpl"
TOPPAR="$HERE/toppar"
RUNS="$HERE/runs_pbc"
FORCE=0; DRYRUN=0
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    --dry-run) DRYRUN=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

fennol_manifest() { echo "$HERE/models/fennix_enzyme/$1/manifest.json"; }
sys_charge() {
  python3 -c "import json;print(json.load(open('$HERE/systems/$1/meta.json')).get('net_charge',0))" 2>/dev/null || echo 0
}

render_conf() {  # $1=celldir $2=sysdir $3=qmsoft $4=qmexec $5=outputname $6=qmcharge
  local extra="$EXTRA_BLOCK"
  # Start coordinates/cell: prefer an MM-equilibrated restart if the box has one
  # (relaxed water network — FeNNiX drifts when started from clashing geometry).
  local coords="$2/system.pdb" xsc="$2/system.xsc"
  if [ "$SMOKE_MM" -ne 1 ] && [ "$START" = "equil" ] \
     && [ -f "$2/system_equil.coor" ] && [ -f "$2/system_equil.xsc" ]; then
    coords="$2/system_equil.coor"; xsc="$2/system_equil.xsc"
  fi
  if [ "$SMOKE_MM" -eq 1 ]; then
    # MM equilibration: minimize out fresh-build clashes, then heat (QM engine off)
    extra="minimize                ${MM_MIN}\\nreinitvels              300"
  fi
  sed -e "s|@PSF@|$2/system.psf|" -e "s|@COORDS@|$coords|" -e "s|@XSC@|$xsc|" \
      -e "s|@TOPPAR@|$TOPPAR|" -e "s|@QMPDB@|$2/qm.pdb|" \
      -e "s|@QMSOFTWARE@|$3|" -e "s|@QMEXEC@|$4|" -e "s|@QMCHARGE@|$6|" \
      -e "s|@TIMESTEP@|$TIMESTEP|" -e "s|@QMREPLACEALL@|$QMREPLACEALL|" -e "s|@RIGIDBONDS@|$RIGIDBONDS|" \
      -e "s|@QMBASEDIR@|/tmp/namd_enzyme_pbc_qmmm|" -e "s|@STEPS@|$STEPS|" \
      -e "s|@STEPSPERCYCLE@|$OUTPUTFREQ|" -e "s|@OUTPUTFREQ@|$OUTPUTFREQ|" \
      -e "s|@OUTPUTNAME@|$5|" -e "s|@EXTRA@|${extra}|" \
      "$TMPL" > "$1/enzyme_pbc.conf"
  if [ "$SMOKE_MM" -eq 1 ]; then
    # pure MM/CHARMM+PME (QM off), 2 fs ok with rigidbonds, and SAVE the restart
    sed -i -e 's|^QMForces .*|QMForces                off|' \
           -e 's|^timestep .*|timestep                2.0|' \
           -e 's|^restartfreq .*|restartfreq             1000000|' "$1/enzyme_pbc.conf"
  fi
}

classify() {
  local rc="$1" log="$2" progressed=0
  grep -qE '^TIMING:|^ENERGY:' $log 2>/dev/null && progressed=1
  if grep -qiE 'out of memory|CUDA error: out of memory|bad_alloc|RESOURCE_EXHAUSTED|cudaErrorMemoryAllocation' $log 2>/dev/null; then echo oom
  elif [ "$rc" -eq 124 ]; then [ "$progressed" -eq 1 ] && echo timeout_partial || echo timeout_hang
  elif [ "$rc" -eq 0 ] && grep -qE 'End of program|WallClock:' $log 2>/dev/null; then echo ok
  else echo "error_rc${rc}"; fi
}

# ----------------------------- main sweep ----------------------------------
mkdir -p /tmp/namd_enzyme_pbc_qmmm "$RUNS"
total=0; ran=0; skipped=0
echo "grid: models=[$MODELS] systems=[$SYSTEMS] walkers=[$WALKERS] steps=$STEPS freq=$OUTPUTFREQ smoke_mm=$SMOKE_MM"
echo

for S in $SYSTEMS; do
  SYS="$HERE/systems/$S"
  if [ ! -f "$SYS/system.psf" ] || [ ! -f "$SYS/system.xsc" ]; then
    echo "SKIP system $S (not built periodic: run build_enzyme_pbc.py)"; continue
  fi
  NAT="$(python3 -c "import json;print(json.load(open('$SYS/meta.json'))['n_atoms'])" 2>/dev/null || echo '?')"
  QCH="$(sys_charge "$S")"
  MLIST="$MODELS"; [ "$SMOKE_MM" -eq 1 ] && MLIST="mm"
  for M in $MLIST; do
    if [ "$SMOKE_MM" -eq 1 ]; then qms="none"; qme="none"
    else
      qms="${BACKEND[$M]}"; qme="${EXEC[$M]}"
      [ "$qme" = "PER_SYSTEM" ] && qme="$(fennol_manifest "$S")"
    fi
    for W in $WALKERS; do
      total=$((total+1))
      CELL="$RUNS/$M/$S/walk${W}"
      tag="$M  $S(${NAT}at,q${QCH})  walk${W}"
      if [ "$FORCE" -eq 0 ] && [ -f "$CELL/status.txt" ]; then
        skipped=$((skipped+1)); [ "$DRYRUN" -eq 1 ] && echo "SKIP  $tag (done)"; continue
      fi
      if [ "$SMOKE_MM" -ne 1 ] && [ "$qms" = "fennol" ] && [ ! -f "$qme" ]; then
        mkdir -p "$CELL"; echo "no_artifact" > "$CELL/status.txt"
        echo "SKIP  $tag (no fennol artifact: export_enzyme_fennix.sh $S)"; continue
      fi
      if [ "$DRYRUN" -eq 1 ]; then echo "RUN   $tag"; continue; fi

      rm -rf "$CELL"; mkdir -p "$CELL"; cd "$CELL"
      pkill -9 -f '/namd3 ' 2>/dev/null; rm -f /tmp/mlff_namd_*.sock
      local_ld="$LD_LIBRARY_PATH"; fennol_env=()
      if [ "$qms" = "fennol" ]; then
        local_ld="$FENNIX_CUDA_LIBS:$LD_LIBRARY_PATH"
        # (a) NAMD's XLA plugin was built against cuDNN >=9.8, but libtorch (linked into
        #     namd3 with an RPATH) bundles cuDNN 9.5.1, which wins over LD_LIBRARY_PATH.
        #     LD_PRELOAD the fennix env's newer cuDNN (>=9.8, also fine for libtorch).
        _fcudnn="$(ls "$FENNIX_ENV"/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn.so.9* 2>/dev/null | sort -V | tail -1)"
        [ -n "$_fcudnn" ] && fennol_env+=(LD_PRELOAD="$_fcudnn")
        # (b) default XLA preallocation gives the compute (BFC) pool only ~7 GB while a
        #     CollectiveBFCAllocator hogs ~40 GB -> RESOURCE_EXHAUSTED. Disable it (same as
        #     the export) so the compute allocator can grow into the full GPU.
        fennol_env+=(XLA_PYTHON_CLIENT_PREALLOCATE=false)
      fi

      if [ "$W" -eq 0 ]; then
        render_conf "$CELL" "$SYS" "$qms" "$qme" "bench" "$QCH"
        cmd=("$NAMD3" enzyme_pbc.conf); outglob="$CELL/out.0.log"
      else
        render_conf "$CELL" "$SYS" "$qms" "$qme" "bench.[myReplica]" "$QCH"
        cmd=("$CHARMRUN" ++local +p"$W" "$NAMD3" +replicas "$W" enzyme_pbc.conf +stdout out.%01d.log)
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
        env LD_LIBRARY_PATH="$local_ld" "${fennol_env[@]}" MLFF_TIMING=1 MLFF_TIMING_EVERY="$OUTPUTFREQ" \
          timeout "$TIMEOUT" "${cmd[@]}" > out.0.log 2>&1
      else
        env LD_LIBRARY_PATH="$local_ld" "${fennol_env[@]}" MLFF_TIMING=1 MLFF_TIMING_EVERY="$OUTPUTFREQ" \
          timeout "$TIMEOUT" "${cmd[@]}" > namd.log 2>&1
      fi
      rc=$?; t1=$(date +%s)
      [ -n "$smp" ] && kill "$smp" 2>/dev/null
      st="$(classify "$rc" "$outglob")"
      peak=""; [ -f "$CELL/gpu_mem.txt" ] && peak="$(sort -n "$CELL/gpu_mem.txt" 2>/dev/null | tail -1)"
      {
        echo "model=$M"; echo "backend=$qms"; echo "system=$S"; echo "n_atoms=$NAT"
        echo "net_charge=$QCH"; echo "walkers=$W"; echo "steps=$STEPS"; echo "smoke_mm=$SMOKE_MM"
        echo "status=$st"; echo "exit_code=$rc"
        echo "wall_seconds=$((t1-t0))"; echo "gpu_peak_mib=${peak:-NA}"
      } > "$CELL/status.txt"
      ran=$((ran+1))
      echo "$st  (${rc}, $((t1-t0))s${peak:+, ${peak}MiB})"
      # promote the MM-equilibrated restart so the full-ML run starts relaxed
      if [ "$SMOKE_MM" -eq 1 ] && [ "$st" = "ok" ] && [ -f "$CELL/bench.coor" ]; then
        cp -f "$CELL/bench.coor" "$SYS/system_equil.coor"
        cp -f "$CELL/bench.xsc"  "$SYS/system_equil.xsc"
        echo "      -> equilibrated restart: $SYS/system_equil.{coor,xsc}"
      fi
      cd "$HERE"
    done
  done
done

echo
echo "done: $ran ran, $skipped skipped, $total total cells."
