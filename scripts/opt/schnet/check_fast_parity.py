"""Parity of schnet_fast.pt (fast path, cell-list NL, graph_step) vs schnet_baseline.pt, CPU + GPU.
Run under the GPU flock.  Prints max |dE| (kcal/mol) and max |dF| (kcal/mol/A) per case."""
import sys, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/opt"))
from bench_common import water_system
worst = 0.0
for dev in ("cuda:0", "cpu"):
    d = torch.device(dev)
    base = torch.jit.load(str(REPO / "models/opt/schnet_baseline.pt"), map_location=d).eval()
    fast = torch.jit.load(str(REPO / "models/opt/schnet_fast.pt"), map_location=d).eval()
    newd = torch.jit.load(str(REPO / "models/opt/schnet_default_new.pt"), map_location=d).eval()
    pcx = torch.zeros(0, 3, dtype=torch.float64, device=d); pcq = torch.zeros(0, dtype=torch.float64, device=d)
    def run1(m, c, Z, cell):
        e, f, q, v = m(c.clone().requires_grad_(True), Z, pcx, pcq, cell)
        return e.detach(), f.detach(), v.detach()
    def runB(m, c, Z, batch, ptr, cells):
        e, f, q, v = m.forward_batch(c.clone().requires_grad_(True), Z, batch, ptr, pcx, pcq, cells)
        return e.detach(), f.detach(), v.detach()
    def rep(tag, a, b):
        global worst
        dE = float((a[0] - b[0]).abs().max()); dF = float((a[1] - b[1]).abs().max())
        dV = float((a[2] - b[2]).abs().max()) if len(a) > 2 else 0.0
        print(f"{dev:6s} {tag:44s} shapes E{tuple(a[0].shape)}=={tuple(b[0].shape)} F{tuple(a[1].shape)} "
              f"dE={dE:.2e} dF={dF:.2e} dV={dV:.2e}")
        assert a[0].shape == b[0].shape and a[1].shape == b[1].shape
        worst = max(worst, dF)
    sizes = (30, 300, 6000) if dev != "cpu" else (30, 300)
    for s in sizes:
        x, Z = water_system(s); c = x.to(d); Zd = Z.to(d)
        zc = torch.zeros(1, 3, 3, dtype=torch.float64, device=d)
        ref = run1(base, c, Zd, zc)
        rep(f"water{s} forward fast", run1(fast, c, Zd, zc), ref)
        rep(f"water{s} forward default_new", run1(newd, c, Zd, zc), ref)
        if s <= 1024:
            cap = int(fast.graph_capacity(c))
            e, f, n = fast.graph_step(c, Zd, cap)
            rep(f"water{s} graph_step (cap {cap}, n_edges {int(n)})", (e.reshape(()), f), ref[:2])
            e, f, n = fast.graph_step(c, Zd, 16)   # overflow must be reported
            assert int(n) > 16, "overflow not reported"
        for W in (4,):
            if s == 6000 and dev == "cuda:0":
                pass
            g = torch.Generator().manual_seed(0)
            cb = torch.cat([x + 0.01 * torch.randn(x.shape, generator=g, dtype=x.dtype) * (i > 0) for i in range(W)]).to(d)
            Zb = Z.repeat(W).to(d); batch = torch.arange(W).repeat_interleave(s).to(d); ptr = (torch.arange(W + 1) * s).to(d)
            cells = torch.zeros(W, 3, 3, dtype=torch.float64, device=d)
            if s == 6000:   # the baseline's block-diagonal NL needs ~15 GB here; compare against default_new instead
                refb = runB(newd, cb, Zb, batch, ptr, cells)
                rep(f"water{s} W={W} forward_batch fast vs default_new", runB(fast, cb, Zb, batch, ptr, cells), refb)
            else:
                refb = runB(base, cb, Zb, batch, ptr, cells)
                rep(f"water{s} W={W} forward_batch fast", runB(fast, cb, Zb, batch, ptr, cells), refb)
                rep(f"water{s} W={W} forward_batch default_new", runB(newd, cb, Zb, batch, ptr, cells), refb)
    # unequal molecule sizes through forward_batch
    x, Z = water_system(300); c = x.to(d); Zd = Z.to(d)
    ptr = torch.tensor([0, 90, 300], device=d); batch = torch.repeat_interleave(torch.arange(2, device=d), ptr[1:] - ptr[:-1])
    cells = torch.zeros(2, 3, 3, dtype=torch.float64, device=d)
    rep("water300 unequal ptr [0,90,300] fast", runB(fast, c, Zd, batch, ptr, cells), runB(base, c, Zd, batch, ptr, cells))
    # periodic path must be untouched
    cell = (torch.eye(3, dtype=torch.float64, device=d) * 20.0).reshape(1, 3, 3)
    x, Z = water_system(300); c = x.to(d); Zd = Z.to(d)
    rep("water300 PERIODIC 20A box fast", run1(fast, c, Zd, cell), run1(base, c, Zd, cell))
print(f"worst dF {worst:.2e}")
