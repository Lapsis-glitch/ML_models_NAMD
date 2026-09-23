"""Bit-identity check: new build_edges_batched vs the pre-schnet-nl version (backup copy)."""
import importlib.util, sys, torch
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/opt"))
from bench_common import water_system
from src import edges as new
import importlib.machinery
spec = importlib.util.spec_from_loader("old_edges", importlib.machinery.SourceFileLoader("old_edges", str(Path(__file__).with_name("edges.py.pre_schnet_nl.bak"))))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
torch.jit.script(new.build_edges_batched); torch.jit.script(new.cell_is_periodic)
ok = True
def cmp(tag, a, b):
    global ok
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    ok &= same
    print(f"{tag:28s} E={a[0].shape[1]:8d} identical={same}")
for s, Ws in ((30, (1, 2, 4)), (300, (1, 4)), (900, (4,)), (3000, (4,)), (6000, (2,))):
    x, Z = water_system(s)
    for W in Ws:
        g = torch.Generator().manual_seed(0)
        c = torch.cat([x + 0.01 * torch.randn(x.shape, generator=g, dtype=x.dtype) * (i > 0) for i in range(W)]).float().to(dev)
        ptr = (torch.arange(W + 1) * s).to(dev)
        cmp(f"water{s} W={W}", new.build_edges_batched(c, ptr, 5.0), old.build_edges_batched(c, ptr, 5.0))
        del c; torch.cuda.empty_cache()
# unequal ptr (general path) and an N divisible by B but unequal
x, _ = water_system(300); c = x.float().to(dev)
for p in ([0, 100, 300], [0, 150, 151, 300], [0, 90, 150, 300]):
    ptr = torch.tensor(p, device=dev)
    cmp(f"unequal ptr {p}", new.build_edges_batched(c, ptr, 5.0), old.build_edges_batched(c, ptr, 5.0))
z = torch.zeros(3, 3, device=dev); cell = torch.eye(3, device=dev) * 20
flat = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 0]], device=dev)
nan = torch.full((3, 3), float("nan"), device=dev)
for t, cc in (("zero", z), ("box", cell), ("flat", flat), ("nan", nan)):
    a, b = new.cell_is_periodic(cc), old.cell_is_periodic(cc); ok &= (a == b)
    print(f"cell_is_periodic {t}: new={a} old={b}")
print("ALL IDENTICAL" if ok else "MISMATCH")
