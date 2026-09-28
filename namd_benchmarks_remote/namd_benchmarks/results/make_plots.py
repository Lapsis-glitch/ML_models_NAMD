#!/usr/bin/env python3
"""
Build aggregate-throughput and GPU-memory plots vs number of walkers from the
*current* remote sweep in runs/ (the published full_results.csv is stale).

Per-walker steady-state s/step is recovered by differencing the last two
cumulative NAMD `PERFORMANCE:` lines (warmup-free), then converted to ns/day
with the run's timestep. Aggregate ns/day = sum over all walker logs.
walk0 is dropped (redundant single-walker point + contaminated gpu_peak_mib);
walk1 is the n_walkers=1 point.
"""
import os, re, glob, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "runs"))

PERF = re.compile(r'PERFORMANCE:\s+(\d+)\s+averaging\s+([\d.eE+-]+)\s+ns/day,\s+([\d.eE+-]+)\s+sec/step')
TS   = re.compile(r'^\s*timestep\s+([\d.]+)', re.M)

MODELS = ["schnet", "ani2x", "mace", "nequip_oam", "fennol"]
COLORS = {"schnet":"#1f77b4","ani2x":"#2ca02c","mace":"#d62728",
          "nequip_oam":"#9467bd","fennol":"#ff7f0e"}
PANEL_SIZES = [30, 60, 180, 900]          # atoms
WALK_DIRS = ["walk1","walk2","walk3","walk4","walk6","walk8"]  # drop walk0


def read_status(path):
    d = {}
    with open(path) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                d[k] = v
    return d


def steady_sstep(logpath):
    """Warmup-free s/step from the last two cumulative PERFORMANCE lines."""
    pts = []
    try:
        with open(logpath, errors="ignore") as fh:
            for line in fh:
                m = PERF.search(line)
                if m:
                    pts.append((int(m.group(1)), float(m.group(3))))  # (step, cum s/step)
    except FileNotFoundError:
        return None
    if not pts:
        return None
    if len(pts) == 1:
        return pts[-1][1]                          # only cumulative available
    (s1, a1), (s2, a2) = pts[-2], pts[-1]
    if s2 <= s1:
        return a2
    iv = (s2 * a2 - s1 * a1) / (s2 - s1)            # interval (steady) s/step
    return iv if iv > 0 else a2


def timestep_fs(rundir):
    conf = os.path.join(rundir, "bench.conf")
    try:
        m = TS.search(open(conf).read())
        if m:
            return float(m.group(1))
    except FileNotFoundError:
        pass
    return 1.0


rows = []
for st_path in glob.glob(os.path.join(ROOT, "*", "*", "walk*", "status.txt")):
    rundir = os.path.dirname(st_path)
    walkname = os.path.basename(rundir)
    if walkname == "walk0":
        continue
    st = read_status(st_path)
    model = st.get("model")
    atoms = int(st.get("n_atoms", 0))
    status = st.get("status", "?")
    gpu = st.get("gpu_peak_mib", "")
    gpu = int(gpu) if gpu.isdigit() else None
    ts = timestep_fs(rundir)

    logs = sorted(glob.glob(os.path.join(rundir, "out.*.log")))
    nwalk = len(logs) or 1
    per = []
    for lg in logs:
        ss = steady_sstep(lg)
        if ss and ss > 0:
            per.append(0.0864 * ts / ss)           # ns/day for this walker
    agg = sum(per) if per else None
    rows.append(dict(model=model, n_atoms=atoms, n_walkers=nwalk, status=status,
                     n_logs=len(logs), n_perf=len(per),
                     per_walker_nsday=(sum(per)/len(per) if per else None),
                     agg_nsday=agg, gpu_peak_mib=gpu))

# tidy CSV (regenerated aggregate from runs/)
csv_path = os.path.join(HERE, "runs_aggregated.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model","n_atoms","n_walkers","status",
        "n_logs","n_perf","per_walker_nsday","agg_nsday","gpu_peak_mib"])
    w.writeheader()
    for r in sorted(rows, key=lambda r:(r["model"], r["n_atoms"], r["n_walkers"])):
        w.writerow(r)
print("wrote", csv_path, "(", len(rows), "runs )")


def series(metric, model, atoms):
    pts = [(r["n_walkers"], r[metric], r["status"]) for r in rows
           if r["model"] == model and r["n_atoms"] == atoms and r[metric] is not None]
    pts.sort()
    return pts


def make_panel_fig(metric, ylabel, title, fname, logy):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, atoms in zip(axes.flat, PANEL_SIZES):
        for model in MODELS:
            pts = series(metric, model, atoms)
            if not pts:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; ssts = [p[2] for p in pts]
            ax.plot(xs, ys, "-", color=COLORS[model], lw=1.6, alpha=.85, zorder=2)
            ok  = [(x,y) for x,y,s in zip(xs,ys,ssts) if s == "ok"]
            bad = [(x,y) for x,y,s in zip(xs,ys,ssts) if s != "ok"]
            if ok:
                ax.scatter(*zip(*ok), color=COLORS[model], s=46, zorder=3,
                           edgecolor="white", linewidth=.6, label=model)
            if bad:  # truncated/crashed-at-teardown runs: hollow markers
                ax.scatter(*zip(*bad), facecolor="white", edgecolor=COLORS[model],
                           s=46, zorder=3, linewidth=1.4,
                           label=None if ok else model)
        ax.set_title(f"{atoms} atoms  ({atoms//3} waters)", fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=.4)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks([1,2,3,4,6,8])
    for ax in axes[-1]:
        ax.set_xlabel("number of concurrent walkers")
    for ax in axes[:,0]:
        ax.set_ylabel(ylabel)
    # single combined legend
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(.5, .965))
    fig.suptitle(title, fontsize=14, y=1.0, fontweight="bold")
    fig.text(.5, .005, "Solid = clean 'ok' run · hollow = run crashed/truncated but "
             "produced steady-state timing · walk0 excluded (contaminated memory)",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .02, 1, .93])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def make_norm_fig(fname):
    """Aggregate ns/day normalized to each model's own 1-walker value (scaling efficiency)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for ax, atoms in zip(axes.flat, PANEL_SIZES):
        ax.plot([1, 8], [1, 8], "k--", lw=1.2, alpha=.45, zorder=1,
                label="ideal (linear)")
        for model in MODELS:
            pts = series("agg_nsday", model, atoms)
            base = next((y for x, y, s in pts if x == 1 and y), None)
            if not base:
                continue  # no single-walker baseline -> can't normalize
            xs = [p[0] for p in pts]; ys = [p[1] / base for p in pts]
            ssts = [p[2] for p in pts]
            ax.plot(xs, ys, "-", color=COLORS[model], lw=1.6, alpha=.85, zorder=2)
            ok  = [(x, y) for x, y, s in zip(xs, ys, ssts) if s == "ok"]
            bad = [(x, y) for x, y, s in zip(xs, ys, ssts) if s != "ok"]
            if ok:
                ax.scatter(*zip(*ok), color=COLORS[model], s=46, zorder=3,
                           edgecolor="white", linewidth=.6, label=model)
            if bad:
                ax.scatter(*zip(*bad), facecolor="white", edgecolor=COLORS[model],
                           s=46, zorder=3, linewidth=1.4, label=None if ok else model)
        ax.set_title(f"{atoms} atoms  ({atoms//3} waters)", fontsize=11)
        ax.grid(True, ls=":", alpha=.4)
        ax.set_xticks([1, 2, 3, 4, 6, 8])
        ax.axhline(1.0, color="#999", lw=.8, alpha=.5, zorder=1)
    for ax in axes[-1]:
        ax.set_xlabel("number of concurrent walkers")
    for ax in axes[:, 0]:
        ax.set_ylabel("aggregate ns/day  ÷  1-walker value")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(.5, .965))
    fig.suptitle("NAMD ML-FF: walker-scaling efficiency (aggregate ns/day normalized "
                 "to 1 walker)", fontsize=14, y=1.0, fontweight="bold")
    fig.text(.5, .005, "Curve at the dashed diagonal = perfect linear scaling · "
             "flat at 1 = no benefit from extra walkers · hollow = truncated run",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, .02, 1, .93])
    out = os.path.join(HERE, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


make_panel_fig("agg_nsday", "aggregate throughput (ns/day, log)",
               "NAMD ML-FF: aggregate sampling throughput vs walkers (remote sweep, runs/)",
               "agg_speed_vs_walkers.png", logy=True)
make_panel_fig("gpu_peak_mib", "GPU peak memory (MiB)",
               "NAMD ML-FF: GPU peak memory vs walkers (remote sweep, runs/)",
               "gpu_mem_vs_walkers.png", logy=False)
make_norm_fig("agg_speed_normalized_vs_walkers.png")
