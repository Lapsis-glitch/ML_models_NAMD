#!/usr/bin/env python
"""
CUDA-graph ceiling test for SchNet (schnet-nl agent).

Question: if the network + backward (NL excluded, edges padded to a fixed
capacity) were replayed as one CUDA graph, how fast would a step be, compared
with the scripted wrapper call?

  flock scripts/opt/.gpu_bench.lock python scripts/opt/schnet/cudagraph/graph_ceiling.py

Graph body = inner SchNet submodules (pairwise -> representation -> Atomwise
outnet) + a fixed-size index_add for the per-molecule sum (the stock Atomwise
does int(idx_m[-1]), a host sync that cannot be captured) + autograd.grad.
Padded edges are self-edges spread over all atoms (k % N) with an offset of 2*r_max, so their
cosine cutoff is exactly 0 and they add exactly 0 to energy and forces.
"""
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/opt"))
from bench_common import water_system  # noqa: E402
from src.edges import build_edges  # noqa: E402
from src.constants import EV_TO_KCAL  # noqa: E402

dev = torch.device("cuda:0")
inner = torch.jit.load(str(REPO / "models/compiled_schnet_default.pt"), map_location=dev).eval()
wrapped = torch.jit.load(str(REPO / "models/opt/schnet_baseline.pt"), map_location=dev).eval()
pair = getattr(inner.input_modules, "0")
rep = inner.representation
outnet = getattr(inner.output_modules, "0").outnet
R_MAX = 5.0


def med(fn, n=60, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def energy_forces(pos32, Z, idx_i, idx_j, offsets, idx_m, cell):
    pos = pos32.detach().requires_grad_(True)
    d = {"_positions": pos, "_atomic_numbers": Z, "_idx_i": idx_i, "_idx_j": idx_j,
         "_offsets": offsets, "_cell": cell, "_idx_m": idx_m}
    d = pair(d)
    d = rep(d)
    y = outnet(d["scalar_representation"]).squeeze(-1)
    e = torch.zeros(1, dtype=y.dtype, device=y.device).index_add(0, idx_m, y)
    (g,) = torch.autograd.grad([e.sum()], [pos])
    return e, -g


for s in (30, 300, 900, 3000):
    xyz, Z = water_system(s)
    c64 = xyz.to(dev); Zd = Z.to(dev)
    ei, _, _ = build_edges(c64.float(), R_MAX)
    E = ei.shape[1]
    cap = int(E * 1.25) + 64
    idx_i = torch.zeros(cap, dtype=torch.long, device=dev); idx_j = torch.zeros_like(idx_i)
    offs = torch.zeros(cap, 3, dtype=torch.float32, device=dev); offs[E:, 0] = 2 * R_MAX
    # spread padding self-edges over all atoms (all-on-atom-0 serialises the scatter atomics)
    spread = torch.arange(cap, device=dev) % s
    idx_i.copy_(spread); idx_j.copy_(spread)
    idx_i[:E] = ei[0]; idx_j[:E] = ei[1]
    idx_m = torch.zeros(s, dtype=torch.long, device=dev)
    cell = torch.zeros(3, 3, device=dev)
    static_pos = c64.float().clone()

    # reference (unpadded) vs padded eager
    e_ref, f_ref = energy_forces(c64.float(), Zd, ei[0], ei[1], torch.zeros(E, 3, device=dev), idx_m, cell)
    e_pad, f_pad = energy_forces(static_pos, Zd, idx_i, idx_j, offs, idx_m, cell)

    # capture
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            energy_forces(static_pos, Zd, idx_i, idx_j, offs, idx_m, cell)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        e_st, f_st = energy_forces(static_pos, Zd, idx_i, idx_j, offs, idx_m, cell)
    h_f = torch.empty(s, 3, dtype=torch.float64).pin_memory()

    def replay():
        static_pos.copy_(c64)          # NAMD coords in (fp64 -> fp32)
        g.replay()
        h_f.copy_(f_st.to(torch.float64) * EV_TO_KCAL, non_blocking=True)
        return (e_st.double() * EV_TO_KCAL).cpu()

    def replay_with_nl():             # + NL outside the graph (sync on nonzero) + edge copy
        e2, _, _ = build_edges(c64.float(), R_MAX)
        n = e2.shape[1]
        idx_i[:n].copy_(e2[0]); idx_j[:n].copy_(e2[1])
        return replay()

    pcx = torch.zeros(0, 3, dtype=torch.float64, device=dev); pcq = torch.zeros(0, dtype=torch.float64, device=dev)
    zc = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)
    hw = torch.empty(s, 3, dtype=torch.float64).pin_memory()

    def wrapper_call():
        e, f, _, _ = wrapped(c64.detach().clone().requires_grad_(True), Zd, pcx, pcq, zc)
        hw.copy_(f, non_blocking=True)
        return e.cpu()

    def eager_fn():
        return energy_forces(c64.float(), Zd, ei[0], ei[1], torch.zeros(E, 3, device=dev), idx_m, cell)

    replay(); torch.cuda.synchronize()
    print(f"water{s}: E={E} cap={cap} | pad-vs-ref dE={float((e_pad - e_ref).abs().max()):.2e} eV "
          f"dF={float((f_pad - f_ref).abs().max()):.2e} | graph-vs-ref dF={float((f_st - f_ref).abs().max()):.2e}")
    print(f"   wrapper {med(wrapper_call):7.3f} ms | eager body {med(eager_fn):7.3f} ms | "
          f"graph replay {med(replay):7.3f} ms | graph+NL {med(replay_with_nl):7.3f} ms")
    del g
    torch.cuda.empty_cache()
