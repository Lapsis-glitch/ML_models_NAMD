"""Cell-list NL vs dense NL: identical edge_index / vecs / lengths (eager + scripted, GPU + CPU)."""
import sys, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/opt"))
from bench_common import water_system
from src import edges as E
cell_s = torch.jit.script(E.build_edges_cell); cellb_s = torch.jit.script(E.build_edges_cell_batched)
dense_s = torch.jit.script(E.build_edges)
ok = True
def cmp(tag, a, b):
    global ok
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    ok &= same
    print(f"{tag:40s} E={a[0].shape[1]:8d} identical={same}")
for dev in ("cuda:0", "cpu"):
    g = torch.Generator().manual_seed(7)
    for s in (3, 30, 300, 900, 3000, 6000):
        if dev == "cpu" and s > 900: continue
        x, _ = water_system(s)
        for jit in (0.0, 0.05):
            c = (x + jit * torch.randn(x.shape, generator=g, dtype=x.dtype)).float().to(dev)
            cmp(f"{dev} water{s} jitter={jit} eager", E.build_edges_cell(c, 5.0), E.build_edges(c, 5.0))
            for _ in range(3):   # 3rd call = NNC-fused versions of both
                a_s, d_s = cell_s(c, 5.0), dense_s(c, 5.0)
            cmp(f"{dev} water{s} jitter={jit} script", a_s, E.build_edges(c, 5.0))
            cmp(f"{dev} water{s} jitter={jit} script-vs-script", a_s, d_s)
        if s in (30, 300, 3000):
            for W in (2, 4):
                c = torch.cat([x + 0.01 * torch.randn(x.shape, generator=g, dtype=x.dtype) * (i > 0) for i in range(W)]).float().to(dev)
                ptr = (torch.arange(W + 1) * s).to(dev)
                cmp(f"{dev} water{s} W={W} batched", cellb_s(c, ptr, 5.0), E.build_edges_batched(c, ptr, 5.0))
    x, _ = water_system(300); c = x.float().to(dev)
    for p in ([0, 100, 300], [0, 150, 151, 300], [0, 1, 300]):
        ptr = torch.tensor(p, device=dev)
        cmp(f"{dev} unequal ptr {p}", cellb_s(c, ptr, 5.0), E.build_edges_batched(c, ptr, 5.0))
    one = torch.zeros(1, 3, device=dev)
    cmp(f"{dev} single atom", cell_s(one, 5.0), E.build_edges(one, 5.0))
    far = torch.tensor([[0., 0, 0], [100., 0, 0], [0, 4.9999, 0], [0, 5.0, 0.0001]], device=dev)
    cmp(f"{dev} sparse/edge-of-cutoff", cell_s(far, 5.0), E.build_edges(far, 5.0))
print("ALL IDENTICAL" if ok else "MISMATCH")
