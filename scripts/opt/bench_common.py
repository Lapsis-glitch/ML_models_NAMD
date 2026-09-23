#!/usr/bin/env python
"""
Shared inference benchmark for NAMD-ready TorchScript artifacts.

Every optimisation agent uses THIS script so numbers are comparable.  It feeds
each artifact exactly what the NAMD shim feeds it (float64 coords on the GPU
with requires_grad=True, int64 Z, empty point-charge tensors, a zero [1,3,3]
cell when the forward signature takes one) and times the full call including
the output copy back to the host, like the shim does.

Fairness rules baked in:
  * All artifacts given in one invocation are timed INTERLEAVED (round-robin
    over several rounds) in the same process, so laptop-GPU clock drift hits
    them equally.  Only compare numbers from the same invocation.
  * The whole run holds an exclusive flock on scripts/opt/.gpu_bench.lock, so
    concurrent agents never time on a shared GPU.  Waiting is expected.
  * Parity (max |dE|, max |dF|) is reported against the FIRST artifact given,
    so pass the baseline first.

Examples
--------
  python scripts/opt/bench_common.py \
      --model base=models/mace_off23_wrapped_identical.pt \
      --model cueq=scripts/opt/mace/mace_cueq.pt \
      --systems 30,300,900,3000 --walkers 1 --out scripts/opt/results/mace.json

  # batched path (forward_batch) with 4 replicas of each system
  python scripts/opt/bench_common.py --model base=... --systems 300 --walkers 4

  # artifact needs a custom-op library loaded first (cuEquivariance, vesin, ...)
  python scripts/opt/bench_common.py --extra-lib /path/libfoo.so --model ...

  # arbitrary geometry instead of the water boxes (xyz or pdb)
  python scripts/opt/bench_common.py --geom models/fulvene.xyz --model ...

  # MD-like inputs: fresh coordinates every call (base + N(0, 0.02 A) noise), so
  # neighbour-list sizes change per call like in NAMD (exposes JIT re-specialisation)
  python scripts/opt/bench_common.py --jitter 0.02 --model ...

  # TorchScript executor knobs (process-global; apply to every --model; same
  # knobs as the shim's NAMD_MLFF_JIT_* env vars -- see scripts/opt/INFRA_DONE)
  python scripts/opt/bench_common.py --jit-profiling 0 --jit-fusion DYNAMIC:20 --model ...
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
SYSTEMS_DIR = REPO / "namd_benchmarks" / "systems"
LOCK_PATH = Path(__file__).resolve().parent / ".gpu_bench.lock"

_SYM2Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
          "CL": 17, "Br": 35, "BR": 35, "I": 53}


# --------------------------------------------------------------------------
#  geometry
# --------------------------------------------------------------------------
def _read_pdb(path: Path):
    xyz, Z = [], []
    for line in path.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            el = line[76:78].strip() or line[12:16].strip().lstrip("0123456789")[:1]
            Z.append(_SYM2Z[el.capitalize() if len(el) > 1 else el])
    return torch.tensor(xyz, dtype=torch.float64), torch.tensor(Z, dtype=torch.int64)


def _read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    xyz, Z = [], []
    for line in lines[2:2 + n]:
        p = line.split()
        Z.append(_SYM2Z[p[0]] if not p[0].isdigit() else int(p[0]))
        xyz.append([float(v) for v in p[1:4]])
    return torch.tensor(xyz, dtype=torch.float64), torch.tensor(Z, dtype=torch.int64)


def load_geom(path: Path):
    return _read_pdb(path) if path.suffix == ".pdb" else _read_xyz(path)


def water_system(n_atoms: int):
    matches = sorted(SYSTEMS_DIR.glob(f"w*_{n_atoms}atoms"))
    if not matches:
        avail = sorted(p.name for p in SYSTEMS_DIR.glob("w*_*atoms"))
        raise SystemExit(f"no water system with {n_atoms} atoms; available: {avail}")
    return _read_pdb(matches[0] / "qm.pdb")


# --------------------------------------------------------------------------
#  calling an artifact the way the NAMD shim does
# --------------------------------------------------------------------------
def _arity(mod, name):
    try:
        return len(mod._c._get_method(name).schema.arguments)  # includes self
    except Exception:
        return -1


class Runner:
    def __init__(self, label, path, dev):
        self.label, self.path = label, path
        self.mod = torch.jit.load(str(path), map_location=dev)
        self.mod.eval()
        self.dev = dev
        self.fwd_cell = _arity(self.mod, "forward") >= 6
        self.batch_cell = _arity(self.mod, "forward_batch") >= 8
        self.has_batch = _arity(self.mod, "forward_batch") > 0
        self.pc_x = torch.zeros(0, 3, dtype=torch.float64, device=dev)
        self.pc_q = torch.zeros(0, dtype=torch.float64, device=dev)

    def prepare(self, xyz, Z, walkers):
        """Upload inputs once (the shim keeps resident device buffers)."""
        d = self.dev
        self.walkers = walkers
        self.h_forces = torch.empty(xyz.shape[0] * walkers, 3, dtype=torch.float64).pin_memory() \
            if d.type == "cuda" else torch.empty(xyz.shape[0] * walkers, 3, dtype=torch.float64)
        if walkers == 1:
            self.coords = xyz.to(d)
            self.Z = Z.to(d)
        else:
            n = xyz.shape[0]
            # small deterministic jitter so replicas are not bit-identical
            g = torch.Generator().manual_seed(0)
            reps = [xyz + 0.01 * torch.randn(xyz.shape, generator=g, dtype=xyz.dtype) * (i > 0)
                    for i in range(walkers)]
            self.coords = torch.cat(reps).to(d)
            self.Z = Z.repeat(walkers).to(d)
            self.batch = torch.arange(walkers).repeat_interleave(n).to(d)
            self.ptr = (torch.arange(walkers + 1) * n).to(d)
        self.jitter_pool = None
        if JITTER > 0:
            g = torch.Generator(device=d).manual_seed(1234)
            self.jitter_pool = [self.coords + JITTER * torch.randn(self.coords.shape, generator=g, device=d,
                                                                   dtype=self.coords.dtype) for _ in range(64)]
            self._k = -1

    def call(self):
        src = self.coords
        pool = getattr(self, "jitter_pool", None)
        if pool:
            self._k = getattr(self, "_k", -1) + 1
            src = pool[self._k % len(pool)]
        c = src.detach().clone().requires_grad_(True)
        if self.walkers == 1:
            args = [c, self.Z, self.pc_x, self.pc_q]
            if self.fwd_cell:
                args.append(torch.zeros(1, 3, 3, dtype=torch.float64, device=self.dev))
            out = self.mod(*args)
        else:
            args = [c, self.Z, self.batch, self.ptr, self.pc_x, self.pc_q]
            if self.batch_cell:
                args.append(torch.zeros(self.walkers, 3, 3, dtype=torch.float64, device=self.dev))
            out = self.mod.forward_batch(*args)
        e, f = out[0], out[1]
        # host copy of forces + energy, like the shim (this is the sync point)
        self.h_forces.copy_(f.detach().to(torch.float64), non_blocking=True)
        e_host = e.detach().to(torch.float64).cpu()
        return e_host, self.h_forces


JITTER = 0.0


def apply_jit_knobs(args):
    """Process-global TorchScript knobs; None = leave torch's default."""
    if args.jit_profiling is not None:
        torch._C._jit_set_profiling_executor(bool(args.jit_profiling))
        torch._C._jit_set_profiling_mode(bool(args.jit_profiling))
    if args.jit_optimize is not None:
        torch._C._set_graph_executor_optimize(bool(args.jit_optimize))
    if args.jit_texpr is not None:
        torch._C._jit_set_texpr_fuser_enabled(bool(args.jit_texpr))
    if args.jit_fusion:
        strat = [(k.upper(), int(v)) for k, v in (x.split(":") for x in args.jit_fusion.split(","))]
        torch._C._jit_set_fusion_strategy(strat)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


# --------------------------------------------------------------------------
#  main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                    help="artifact to time; first one is the parity reference")
    ap.add_argument("--systems", default="30,300,900,3000",
                    help="comma list of water-box atom counts from namd_benchmarks/systems")
    ap.add_argument("--geom", action="append", default=[],
                    help="extra .xyz/.pdb geometry to benchmark (repeatable)")
    ap.add_argument("--walkers", default="1", help="comma list; >1 uses forward_batch")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--iters", type=int, default=30, help="timed calls per round")
    ap.add_argument("--rounds", type=int, default=3, help="interleaved rounds")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--extra-lib", action="append", default=[],
                    help="custom-op .so to torch.ops.load_library before loading models")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="per-call coordinate noise sigma in A (0 = fixed geometry, the old behaviour)")
    ap.add_argument("--jit-profiling", type=int, choices=[0, 1], default=None,
                    help="0 = legacy TorchScript executor (NAMD_MLFF_JIT_PROFILING)")
    ap.add_argument("--jit-optimize", type=int, choices=[0, 1], default=None,
                    help="0 = graph executor optimisations off (NAMD_MLFF_JIT_OPTIMIZE)")
    ap.add_argument("--jit-texpr", type=int, choices=[0, 1], default=None,
                    help="TensorExpr (NNC) fuser on/off (NAMD_MLFF_JIT_TEXPR)")
    ap.add_argument("--jit-fusion", default=None,
                    help='fusion strategy, e.g. "DYNAMIC:20" or "STATIC:2,DYNAMIC:10" (NAMD_MLFF_JIT_FUSION)')
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (changes fp32 numerics)")
    ap.add_argument("--out", help="write JSON results here")
    ap.add_argument("--no-lock", action="store_true", help="skip the GPU flock (CPU runs only!)")
    args = ap.parse_args()

    for lib in args.extra_lib:
        torch.ops.load_library(lib)
    global JITTER
    JITTER = args.jitter
    apply_jit_knobs(args)

    dev = torch.device(args.device)
    lock_cm = contextlib.nullcontext()
    if not args.no_lock:
        LOCK_PATH.touch(exist_ok=True)
        fh = open(LOCK_PATH, "w")
        t0 = time.time()
        print(f"[bench] waiting for GPU lock {LOCK_PATH} ...", flush=True)
        fcntl.flock(fh, fcntl.LOCK_EX)
        print(f"[bench] got lock after {time.time() - t0:.0f}s", flush=True)

    runners = []
    for spec in args.model:
        label, _, path = spec.partition("=")
        runners.append(Runner(label, Path(path), dev))

    geoms = []
    for s in [x for x in args.systems.split(",") if x.strip()]:
        geoms.append((f"water{s}", *water_system(int(s))))
    for g in args.geom:
        geoms.append((Path(g).stem, *load_geom(Path(g))))

    meta = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
    }
    results = []
    for gname, xyz, Z in geoms:
        for W in [int(w) for w in args.walkers.split(",")]:
            ref = None
            rows = {}
            active = []
            for r in runners:
                if W > 1 and not r.has_batch:
                    continue
                try:
                    r.prepare(xyz, Z, W)
                    torch.cuda.empty_cache() if dev.type == "cuda" else None
                    if dev.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(dev)
                    base_mem = torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0
                    for _ in range(args.warmup):
                        e, f = r.call()
                    _sync(dev)
                    peak = (torch.cuda.max_memory_allocated(dev) - base_mem) / 2**20 if dev.type == "cuda" else 0
                    e = e.clone(); f = f.clone()
                    if ref is None:
                        ref = (e, f)
                    rows[r.label] = {
                        "peak_alloc_mib": round(peak, 1),
                        "dE_max_kcal": float((e - ref[0]).abs().max()),
                        "dF_max_kcal_A": float((f - ref[1]).abs().max()),
                        "times_ms": [],
                    }
                    active.append(r)
                except Exception as ex:  # OOM etc. -> record and move on
                    msg = str(ex).splitlines()[0][:200]
                    rows[r.label] = {"error": msg}
                    print(f"[bench] {r.label} {gname} W={W}: ERROR {msg}", flush=True)
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
            for _ in range(args.rounds):
                for r in active:
                    try:
                        r.prepare(xyz, Z, W)
                        r.call(); _sync(dev)
                        ts = rows[r.label]["times_ms"]
                        for _ in range(args.iters):
                            t = time.perf_counter()
                            r.call()
                            _sync(dev)
                            ts.append((time.perf_counter() - t) * 1e3)
                    except Exception as ex:
                        rows[r.label]["error"] = str(ex).splitlines()[0][:200]
            for label, row in rows.items():
                ts = row.pop("times_ms", [])
                if ts:
                    ts.sort()
                    row.update(median_ms=round(statistics.median(ts), 3),
                               p10_ms=round(ts[len(ts) // 10], 3),
                               p90_ms=round(ts[(9 * len(ts)) // 10], 3),
                               n=len(ts))
                results.append({"system": gname, "n_atoms": int(xyz.shape[0]),
                                "walkers": W, "model": label, **row})

    # ---- print table ----
    hdr = f"{'system':>12} {'W':>2} {'model':>22} {'median_ms':>10} {'p10':>8} {'p90':>8} {'peakMiB':>8} {'speedup':>7} {'dE_max':>9} {'dF_max':>9}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    base = {}
    for row in results:
        key = (row["system"], row["walkers"])
        if "error" in row and "median_ms" not in row:
            print(f"{row['system']:>12} {row['walkers']:>2} {row['model']:>22}  ERROR: {row['error'][:60]}")
            continue
        base.setdefault(key, row["median_ms"])
        print(f"{row['system']:>12} {row['walkers']:>2} {row['model']:>22} {row['median_ms']:>10.3f} "
              f"{row['p10_ms']:>8.3f} {row['p90_ms']:>8.3f} {row['peak_alloc_mib']:>8.1f} "
              f"{base[key] / row['median_ms']:>6.2f}x {row['dE_max_kcal']:>9.2e} {row['dF_max_kcal_A']:>9.2e}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"meta": meta, "results": results}, indent=1))
        print(f"\n[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
