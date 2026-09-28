"""Correctness checks for the periodic edge builders."""
import sys, itertools
import torch

sys.path.insert(0, "/home/rat/PycharmProjects/ML_models_NAMD")
from src.edges import (build_edges, build_edges_pbc, build_edges_batched_pbc,
                       cell_is_periodic, min_perp_width)

torch.manual_seed(0)
fails = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- scripting
try:
    s_pbc = torch.jit.script(build_edges_pbc)
    s_batched = torch.jit.script(build_edges_batched_pbc)
    check("torch.jit.script compiles both periodic builders", True)
except Exception as e:
    check("torch.jit.script compiles both periodic builders", False, repr(e))
    raise SystemExit(1)


# ------------------------------------------------------------- test system
L = 12.0
rmax = 3.5
N = 40
cell = torch.eye(3, dtype=torch.float64) * L
coords = torch.rand(N, 3, dtype=torch.float64) * L


def brute_force(coords, cell, rmax):
    """Explicit 3x3x3 image enumeration, the definition we want to match."""
    out = []
    N = coords.shape[0]
    for i in range(N):
        for j in range(N):
            for n in itertools.product((-1, 0, 1), repeat=3):
                nv = torch.tensor(n, dtype=coords.dtype)
                d = coords[j] + nv @ cell - coords[i]
                r = float(d.norm())
                if 1e-12 < r < rmax:
                    out.append((i, j, r))
    return out


ref = brute_force(coords, cell, rmax)
ei, ev, el, us = s_pbc(coords, cell, rmax)

check("edge count matches brute-force image enumeration",
      ei.shape[1] == len(ref), f"got {ei.shape[1]}, expected {len(ref)}")

ref_d = torch.tensor(sorted(r for _, _, r in ref), dtype=torch.float64)
got_d = torch.sort(el)[0]
if ref_d.shape == got_d.shape:
    check("edge lengths match brute force",
          bool(torch.allclose(ref_d, got_d, atol=1e-10)),
          f"max diff {float((ref_d - got_d).abs().max()):.3e}")
else:
    check("edge lengths match brute force", False, "shape mismatch")

# The identity every consuming model relies on.
recon = coords[ei[1]] - coords[ei[0]] + us @ cell
check("edge_vecs == pos[e1] - pos[e0] + unit_shifts @ cell",
      bool(torch.allclose(recon, ev, atol=1e-12)),
      f"max diff {float((recon - ev).abs().max()):.3e}")

check("unit_shifts are integers",
      bool(torch.allclose(us, torch.round(us), atol=1e-12)))

check("all edges inside the cutoff", bool((el < rmax).all()))


# --------------------------------------------- shift sign (the classic bug)
# Two atoms straddling a face, 0.5 A apart through the boundary.
pair = torch.tensor([[0.25, 5.0, 5.0], [L - 0.25, 5.0, 5.0]], dtype=torch.float64)
pei, pev, pel, pus = s_pbc(pair, cell, rmax)
check("pair across a periodic face sees 0.5 A, not L-0.5",
      pei.shape[1] == 2 and bool(torch.allclose(pel, torch.full((2,), 0.5, dtype=torch.float64), atol=1e-12)),
      f"lengths {pel.tolist()}")


# ------------------------------------------------------ translation invariance
for label, shift in (("lattice vector", torch.tensor([L, 0.0, 0.0], dtype=torch.float64)),
                     ("arbitrary offset", torch.tensor([1.234, -7.5, 0.9], dtype=torch.float64))):
    _, _, el2, _ = s_pbc(coords + shift, cell, rmax)
    check(f"edge lengths invariant under translation by a {label}",
          bool(torch.allclose(torch.sort(el)[0], torch.sort(el2)[0], atol=1e-10)))


# ------------------------------------------------------------ blocking is inert
_, _, el_b, _ = s_pbc(coords, cell, rmax, 7)   # awkward block size
check("row blocking does not change the result",
      bool(torch.allclose(torch.sort(el)[0], torch.sort(el_b)[0], atol=1e-12)))


# --------------------------------------------------------------- triclinic
tri = torch.tensor([[L, 0.0, 0.0], [2.0, L, 0.0], [1.0, 1.5, L]], dtype=torch.float64)
tc = torch.rand(N, 3, dtype=torch.float64) @ tri
ref_t = brute_force(tc, tri, rmax)
_, _, el_t, _ = s_pbc(tc, tri, rmax)
check("triclinic cell matches brute force",
      el_t.shape[0] == len(ref_t) and bool(torch.allclose(
          torch.sort(el_t)[0],
          torch.tensor(sorted(r for _, _, r in ref_t), dtype=torch.float64), atol=1e-10)),
      f"got {el_t.shape[0]}, expected {len(ref_t)}")


# ------------------------------------------------- box-too-small must be loud
try:
    s_pbc(coords, torch.eye(3, dtype=torch.float64) * (2 * rmax - 0.1), rmax)
    check("too-small box raises", False, "no exception")
except Exception as e:
    check("too-small box raises", "too small" in str(e).lower(), str(e)[:70])


# ------------------------------------------------------------------ batched
ptr = torch.tensor([0, N, 2 * N], dtype=torch.long)
cells2 = torch.stack([cell, cell * 1.1])
cat = torch.cat([coords, coords * 1.1], dim=0)
bei, bev, bel, bus = s_batched(cat, ptr, cells2, rmax)
check("batched: no edges cross molecule boundaries",
      bool((((bei[0] < N) & (bei[1] < N)) | ((bei[0] >= N) & (bei[1] >= N))).all()))
check("batched: first molecule reproduces the single-molecule result",
      int((bei[0] < N).sum()) == ei.shape[1],
      f"got {int((bei[0] < N).sum())}, expected {ei.shape[1]}")
check("batched: identity holds with per-molecule cells",
      bool(torch.allclose(bev,
           cat[bei[1]] - cat[bei[0]] + torch.einsum(
               'ei,eij->ej', bus,
               torch.where((bei[0] < N).unsqueeze(-1).unsqueeze(-1),
                           cells2[0].unsqueeze(0), cells2[1].unsqueeze(0))),
           atol=1e-12)))


# ----------------------------------------------------- non-periodic agreement
zero = torch.zeros(3, 3, dtype=torch.float64)
check("cell_is_periodic rejects an all-zero cell", not cell_is_periodic(zero))
check("cell_is_periodic accepts a real box", cell_is_periodic(cell))

# With a box far larger than the cutoff, periodic and non-periodic must agree.
big = torch.eye(3, dtype=torch.float64) * 1000.0
_, _, el_np, = build_edges(coords, rmax)[1:3] + (None,) if False else (None, None, None)
nei, nev, nel = build_edges(coords, rmax)
_, _, el_big, us_big = s_pbc(coords, big, rmax)
check("huge box reproduces the non-periodic edge list",
      nel.shape == el_big.shape and bool(torch.allclose(
          torch.sort(nel)[0], torch.sort(el_big)[0], atol=1e-12)))
check("huge box produces only zero shifts", bool((us_big.abs() < 1e-12).all()))

print()
print(("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S): " + ", ".join(fails)))
if __name__ == "__main__":
    sys.exit(1 if fails else 0)
else:
    def test_no_failures():
        """Collected by pytest; the checks above run at import."""
        assert not fails, fails
