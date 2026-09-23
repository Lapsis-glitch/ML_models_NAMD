"""Quick parity + rough timing of wrapped inner variants (eager wrapper, not scripted).
usage: flock scripts/opt/.gpu_bench.lock python scripts/opt/mace/parity_inner.py SYSTEM inner1 inner2 ..."""
import sys, time, torch
sys.path.insert(0, "scripts/opt"); sys.path.insert(0, ".")
import cuequivariance_torch  # noqa: F401  registers cuEq torch ops
from bench_common import water_system
from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper
n = int(sys.argv[1]); xyz, Z = water_system(n)
dev = "cuda"
xyz, Z = xyz.to(dev), Z.to(dev)
pc = torch.zeros(0, 3, dtype=torch.float64, device=dev); pq = torch.zeros(0, dtype=torch.float64, device=dev)
cell = torch.zeros(1, 3, 3, dtype=torch.float64, device=dev)
ref = None
for p in sys.argv[2:]:
    w = MACE_TS_Wrapper(p, device=dev)
    for _ in range(5):
        c = xyz.clone().requires_grad_(True)
        e, f, q, v = w(c, Z, pc, pq, cell)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        c = xyz.clone().requires_grad_(True)
        e, f, q, v = w(c, Z, pc, pq, cell)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / 20 * 1e3
    if ref is None: ref = (e.detach(), f.detach())
    print(f"{p:45s} E={e.item():.6f} dE={abs(e.item()-ref[0].item()):.2e} dF={(f-ref[1]).abs().max().item():.2e} {dt:.2f} ms")
