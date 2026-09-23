#!/usr/bin/env python
"""Gather NAMD ML-FF / FeNNol / xTB benchmark results into a CSV + summary.

Walks runs/<model>/w<K>_<atoms>atoms/walk<W>/ and, per cell, extracts:

  * status / exit / wall / GPU-peak   (from status.txt)
  * NAMD `TIMING:` per-interval wall-s/step  (out.0.log, stdout) — the universal
    metric across all backends.  The first interval includes model load /
    (FeNNol) StableHLO compile / GPU warmup, so the headline `s_per_step` is the
    mean of the *later* intervals; the first is reported separately.
  * MLFF per-eval timing (mlff backends only; stderr -> namd.log or out.0.log):
    `[calcMLFF] infer/total` and `[worker batch] forward` — cumulative averages,
    so we also derive the windowed steady-state from the last two prints.

ns/day is computed from the steady s/step (timestep is 1 fs).

Usage:  python gather_results.py            # writes results/summary.csv, prints table
"""
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "results", "summary.csv")
BASELINE_FILE = os.path.join(HERE, "results", "gpu_baseline.txt")


def read_baseline():
    """Idle GPU board memory (MiB), recorded with no namd3 running, so the
    per-cell board peak can be reduced to the run's own GPU footprint. WSL2
    can't attribute GPU memory per process, so board-peak-minus-baseline is the
    footprint measure — and for batched (replica) cells that whole-board delta
    (server model + N replica CUDA contexts + batch working set) is exactly the
    'what does batched mode cost on the GPU' number."""
    try:
        return int(open(BASELINE_FILE).read().strip())
    except (OSError, ValueError):
        return None

# NAMD: "TIMING: 100  CPU: 12.3, 0.45/step  Wall: 13.1, 0.50/step, ... MB ..."
# MULTILINE so ^ matches each line, not just the start of the file.
RE_NAMD = re.compile(
    r"^TIMING:\s+(\d+)\s+CPU:\s+[\d.]+,\s+([\d.]+)/step\s+Wall:\s+[\d.]+,\s+([\d.]+)/step",
    re.MULTILINE)
# MLFF: "MLFF TIMING [calcMLFF] n=100 avg_ms: pre=.. infer=X post=.. send=.. total=T"
RE_CALC = re.compile(r"MLFF TIMING \[calcMLFF\] n=(\d+) avg_ms: .*infer=([\d.]+).*total=([\d.]+)")
RE_WORK = re.compile(r"MLFF TIMING \[worker batch\] n=(\d+) avg_ms: .*forward=([\d.]+)")


def read_status(path):
    d = {}
    if os.path.isfile(path):
        for line in open(path):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


def collect(globs, regex):
    """Return list of regex-match groups (as floats) across the given log files."""
    rows = []
    for g in globs:
        for fn in sorted(glob.glob(g)):
            try:
                txt = open(fn, errors="ignore").read()
            except OSError:
                continue
            for m in regex.finditer(txt):
                rows.append(tuple(float(x) for x in m.groups()))
            if rows:                       # first file that has data wins (replica 0)
                return rows
    return rows


def windowed(cum_rows, idx_n=0, idx_v=1):
    """Steady-state value from the last two cumulative-average prints:
       inst = (avg2*n2 - avg1*n1) / (n2 - n1).  Falls back to last cum avg."""
    if not cum_rows:
        return None
    if len(cum_rows) == 1:
        return cum_rows[-1][idx_v]
    n1, v1 = cum_rows[-2][idx_n], cum_rows[-2][idx_v]
    n2, v2 = cum_rows[-1][idx_n], cum_rows[-1][idx_v]
    if n2 <= n1:
        return cum_rows[-1][idx_v]
    return (v2 * n2 - v1 * n1) / (n2 - n1)


def main():
    cells = sorted(glob.glob(os.path.join(RUNS, "*", "w*atoms", "walk*")))
    baseline = read_baseline()
    fields = ["model", "backend", "n_atoms", "n_waters", "walkers", "status",
              "s_per_step", "ns_per_day", "first_interval_s_per_step", "n_intervals",
              "mlff_infer_ms", "mlff_forward_ms_batch", "wall_seconds",
              "gpu_peak_mib", "gpu_mem_over_baseline_mib", "exit_code"]
    out_rows = []
    for cell in cells:
        st = read_status(os.path.join(cell, "status.txt"))
        if not st:
            continue
        # NAMD per-interval wall/step from replica-0 stdout
        namd = collect([os.path.join(cell, "out.0.log"),
                        os.path.join(cell, "out.*.log")], RE_NAMD)
        steps = [r[0] for r in namd]
        walls = [r[2] for r in namd]
        first = walls[0] if walls else None
        steady = (sum(walls[1:]) / len(walls[1:])) if len(walls) > 1 else first
        nsday = (0.0864 / steady) if steady else None     # 1 fs step -> ns/day
        # MLFF per-eval timing (stderr -> namd.log for replicas, out.0.log for W=0)
        logs = [os.path.join(cell, "namd.log"), os.path.join(cell, "out.0.log")]
        infer = windowed(collect(logs, RE_CALC), 0, 1)
        fwd = windowed(collect(logs, RE_WORK), 0, 1)
        # model's own GPU footprint = board peak - idle baseline (see read_baseline)
        peak_raw = st.get("gpu_peak_mib", "")
        over_base = ""
        if baseline is not None and peak_raw not in ("", "NA"):
            try:
                over_base = str(max(0, int(peak_raw) - baseline))
            except ValueError:
                over_base = ""
        out_rows.append({
            "model": st.get("model", ""), "backend": st.get("backend", ""),
            "n_atoms": st.get("n_atoms", ""), "n_waters": st.get("n_waters", ""),
            "walkers": st.get("walkers", ""), "status": st.get("status", ""),
            "s_per_step": f"{steady:.4f}" if steady else "",
            "ns_per_day": f"{nsday:.4f}" if nsday else "",
            "first_interval_s_per_step": f"{first:.4f}" if first else "",
            "n_intervals": len(walls),
            "mlff_infer_ms": f"{infer:.2f}" if infer is not None else "",
            "mlff_forward_ms_batch": f"{fwd:.2f}" if fwd is not None else "",
            "wall_seconds": st.get("wall_seconds", ""),
            "gpu_peak_mib": st.get("gpu_peak_mib", ""),
            "gpu_mem_over_baseline_mib": over_base,
            "exit_code": st.get("exit_code", ""),
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda r: (
            r["model"], int(r["n_atoms"] or 0), int(r["walkers"] or 0))))
    print(f"wrote {OUT}  ({len(out_rows)} cells)")
    if baseline is not None:
        print(f"GPU idle baseline = {baseline} MiB  "
              f"(GPU_own = board peak - baseline; walk0=single, walk>=1=batched replicas)\n")
    else:
        print("GPU idle baseline = (results/gpu_baseline.txt missing) -> GPU_own blank\n")

    # compact console table
    hdr = (f"{'model':<11} {'atoms':>6} {'walk':>4} {'status':>10} {'s/step':>9} "
           f"{'ns/day':>8} {'infer_ms':>8} {'fwd_ms':>8} {'GPU_peak':>8} {'GPU_own':>8}")
    print(hdr); print("-" * len(hdr))
    for r in sorted(out_rows, key=lambda r: (r["model"], int(r["n_atoms"] or 0), int(r["walkers"] or 0))):
        print(f"{r['model']:<11} {r['n_atoms']:>6} {r['walkers']:>4} {r['status']:>10} "
              f"{r['s_per_step']:>9} {r['ns_per_day']:>8} {r['mlff_infer_ms']:>8} "
              f"{r['mlff_forward_ms_batch']:>8} {r['gpu_peak_mib']:>8} {r['gpu_mem_over_baseline_mib']:>8}")


if __name__ == "__main__":
    sys.exit(main())
