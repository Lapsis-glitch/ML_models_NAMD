"""Time dense vs cell-list NL (scripted, GPU), single and 4-walker; peak memory. Run under the GPU flock."""
import sys, time, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/opt"))
from bench_common import water_system
from src import edges as E
fns = {"dense": torch.jit.script(E.build_edges), "cell": torch.jit.script(E.build_edges_cell),
       "denseB": torch.jit.script(E.build_edges_batched), "cellB": torch.jit.script(E.build_edges_cell_batched)}
dev = torch.device("cuda:0")
def med(f, n=30):
    for _ in range(5): f()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1e3)
    return sorted(ts)[n // 2]
def peak(f):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m0 = torch.cuda.memory_allocated(); f(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - m0) / 2**20
for s in (30, 300, 900, 1800, 3000, 6000):
    x, _ = water_system(s); c = x.float().to(dev)
    W = 4; cb = torch.cat([c] * W); ptr = (torch.arange(W + 1) * s).to(dev)
    r = {k: (med(lambda: f(c, 5.0)), peak(lambda: f(c, 5.0))) for k, f in (("dense", fns["dense"]), ("cell", fns["cell"]))}
    r.update({k: (med(lambda: f(cb, ptr, 5.0)), peak(lambda: f(cb, ptr, 5.0))) for k, f in (("denseB", fns["denseB"]), ("cellB", fns["cellB"]))})
    print(f"water{s:<5} " + "  ".join(f"{k} {t:6.3f}ms {m:7.1f}MiB" for k, (t, m) in r.items()), flush=True)
