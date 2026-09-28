#!/usr/bin/env bash
# ===========================================================================
# Pre-minimize each benchmark system ONCE with a robust ML model, so every
# benchmark cell can start its timed MD from a relaxed (clash-free) structure
# instead of the raw packed box.
#
# Writes  systems/w<K>_<A>atoms/water_min.coor  (NAMD text output = PDB format),
# which run_benchmark.sh's render_conf picks up automatically as @COORDS@.
#
# Why: gives every model the SAME equilibrated starting structure (fair) instead
# of a raw packed box, costs 8 minimizations not 336 inline ones, and keeps the
# timed runs pure dynamics (clean PERFORMANCE parsing). Good practice regardless.
#
# IMPORTANT - this is NOT a crash fix. Direct testing (mace @180a, packed box vs.
# schnet-minimized) showed the observed benchmark crashes are not structural:
#   * mace@180a goes catastrophically unstable in DYNAMICS, judged within its own
#     energy scale (controlled same-seed test, packed vs schnet-minimized start):
#     energy spikes to ~1e8 (packed) / ~1e10 (relaxed) vs its ~2.88e6 baseline.
#     Minimization DELAYS the first blowup (~step 46 -> ~106) but does NOT prevent
#     it. (NB: ~-2.88e6 is mace's legit total energy incl. atomic self-energies,
#     NOT garbage -- absolute energies are not comparable across models.)
#   * fennol @300a starts fine (~-1.2e3) and drifts during dynamics, then hits the
#     universal walk0 teardown segfault (rc139).
#   * xtb can't dlopen libxtb (glibc), the largest sizes hit GPU OOM, and some
#     multi-walker cells die on charmrun "Socket closed before recv".
# Those need their own fixes (retrain/swap mace, fix the teardown crash, rebuild
# libxtb, smaller systems / more GPU mem). Use this for a clean, fair start.
#
# Usage:
#   ./pre_minimize.sh                         # all sizes, schnet, 500 steps
#   MINSTEPS=1000 ./pre_minimize.sh           # deeper minimization
#   MINMODEL=schnet WATER="10 20 60" ./pre_minimize.sh
#   ./pre_minimize.sh --force                 # re-minimize even if water_min.coor exists
# ===========================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh" >/dev/null

WATER="${WATER:-10 20 60 100 300 600 1000 2000}"   # water counts; atoms = 3x
MINSTEPS="${MINSTEPS:-500}"                          # CG minimization steps
MINMODEL="${MINMODEL:-schnet}"                       # robust model to relax with
TIMEOUT="${TIMEOUT:-3600}"                           # seconds per system
TMPL="$HERE/templates/bench.conf.tmpl"

declare -A EXECP
EXECP[mace]="$HERE/models/mace_off23.pt"
EXECP[ani2x]="$HERE/models/ani2x.pt"
EXECP[nequip_oam]="$HERE/models/nequip_oam.pt"
EXECP[schnet]="$HERE/models/schnet.pt"
qme="${EXECP[$MINMODEL]:-}"
[ -n "$qme" ] || { echo "unknown MINMODEL=$MINMODEL (use mace|ani2x|nequip_oam|schnet)" >&2; exit 2; }
[ -f "$qme" ] || { echo "model artifact missing: $qme" >&2; exit 2; }

FORCE=0
for a in "$@"; do case "$a" in --force) FORCE=1 ;; *) echo "unknown arg: $a" >&2; exit 2 ;; esac; done

mkdir -p /tmp/namd_bench_qmmm
echo "pre-minimize: model=$MINMODEL steps=$MINSTEPS sizes=[$WATER]"
echo
done_n=0; skip_n=0; fail_n=0
for K in $WATER; do
  ATOMS=$((3*K)); SYS="$HERE/systems/w${K}_${ATOMS}atoms"
  out="$SYS/water_min.coor"
  tag="w${K} (${ATOMS} atoms)"
  if [ ! -f "$SYS/water.pdb" ]; then echo "SKIP  $tag (no water.pdb)"; skip_n=$((skip_n+1)); continue; fi
  if [ "$FORCE" -eq 0 ] && [ -f "$out" ]; then echo "SKIP  $tag (have water_min.coor)"; skip_n=$((skip_n+1)); continue; fi

  work="$(mktemp -d "/tmp/premin_w${K}.XXXXXX")"
  # minimize-only config: minimize $MINSTEPS, then `run 0` (no dynamics). outputname
  # 'relaxed' -> relaxed.coor (PDB-format) written at end of program.
  sed -e "s|@PRMTOP@|$SYS/water.prmtop|" -e "s|@COORDS@|$SYS/water.pdb|" \
      -e "s|@QMPDB@|$SYS/qm.pdb|" -e "s|@QMSOFTWARE@|mlff|" -e "s|@QMEXEC@|$qme|" \
      -e "s|@QMCHARGEMODE@|none|" -e "s|@QMCHARGE@|0.00|" \
      -e "s|@QMBASEDIR@|/tmp/namd_bench_qmmm|" -e "s|@STEPS@|0|" \
      -e "s|@STEPSPERCYCLE@|50|" -e "s|@OUTPUTFREQ@|50|" \
      -e "s|@OUTPUTNAME@|relaxed|" -e "s|@EXTRA@|minimize                ${MINSTEPS}|" \
      "$TMPL" > "$work/min.conf"

  echo -n "MIN   $tag ... "
  # NOTE: deliberately NO host-wide `pkill namd3` / socket rm here — that would
  # kill unrelated NAMD jobs on a shared machine. `timeout` reaps our own child.
  t0=$(date +%s)
  ( cd "$work" && timeout "$TIMEOUT" "$NAMD3" min.conf > min.log 2>&1 )
  rc=$?; t1=$(date +%s)
  if [ "$rc" -eq 0 ] && [ -f "$work/relaxed.coor" ]; then
    cp "$work/relaxed.coor" "$out"
    echo "ok ($((t1-t0))s) -> $(basename "$out")"
    rm -rf "$work"; done_n=$((done_n+1))
  else
    echo "FAILED (rc=$rc, $((t1-t0))s) — log kept: $work/min.log"
    fail_n=$((fail_n+1))
  fi
done
echo
echo "done: $done_n minimized, $skip_n skipped, $fail_n failed."
echo "now run the sweep — run_benchmark.sh will start every cell from water_min.coor."
