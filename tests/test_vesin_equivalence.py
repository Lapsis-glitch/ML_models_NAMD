"""
The vesin edge builder has to agree with the pure one.

The pure minimum-image builder in ``src/edges.py`` is the reference here.  It
is slow but it is simple enough to check by eye and it is already tested
against explicit image enumeration in ``test_pbc_edges.py``.  So the job of
this file is narrow: show that swapping in vesin does not change the answer.

Agreement is checked in float64.  vesin does its search in double no matter
what it is handed, while the pure builder rounds in whatever dtype it is given,
so in float32 the two can disagree about a pair sitting a hair either side of
the cutoff.  That is a real difference and it gets its own test, which pins
down how large the disagreement is allowed to be rather than papering over it.

Run it either way:

    python -m pytest tests/test_vesin_equivalence.py -v
    python tests/test_vesin_equivalence.py
    python tests/test_vesin_equivalence.py --bench
"""

import sys
import time

import torch

sys.path.insert(0, "/home/rat/PycharmProjects/ML_models_NAMD")

from src.edges import build_edges_pbc, make_pbc_edge_builder
from src.nl_vesin import (VESIN_AVAILABLE, PurePBCEdges, VesinPBCEdges,
                          build_edges_pbc_vesin, vesin_version)

try:
    import pytest
except ImportError:
    pytest = None


def _need_vesin():
    """Skip under pytest, fail loudly when run as a script."""
    if VESIN_AVAILABLE:
        return
    if pytest is not None:
        pytest.skip("vesin is not installed in this environment")
    raise RuntimeError("vesin is not installed in this environment")


# -----------------------------------------------------------------------
#  Test systems
# -----------------------------------------------------------------------

CUBIC = torch.eye(3, dtype=torch.float64) * 12.0

# Sheared enough that a wrong shift convention shows up as a wrong distance
# rather than cancelling out the way it can in a cube.
TRICLINIC = torch.tensor([[12.0, 0.0, 0.0],
                          [2.0, 12.0, 0.0],
                          [1.0, 1.5, 12.0]], dtype=torch.float64)

R_MAX = 3.5


def _system(cell, n=200, seed=0, device=None):
    g = torch.Generator().manual_seed(seed)
    frac = torch.rand(n, 3, generator=g, dtype=torch.float64)
    out = frac @ cell.cpu()
    return out if device is None else out.to(device)


def _usable_cuda():
    """
    torch.cuda.is_available() is not enough here.

    One of the environments in this repo has a torch build with no kernels for
    the card that is actually installed, and it only says so when something
    runs, so run something.
    """
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(8, device="cuda").sum().item()
        return True
    except Exception:
        return False


def _edge_set(edge_index, unit_shifts):
    """
    An edge as a hashable identity: who, to whom, through which image.

    Comparing these directly is stricter than comparing lengths, and it does
    not care about the order the edges came out in, which is the one thing the
    two backends are allowed to differ on.
    """
    ei = edge_index.cpu()
    us = torch.round(unit_shifts).to(torch.int64).cpu()
    return {
        (int(ei[0, k]), int(ei[1, k]),
         int(us[k, 0]), int(us[k, 1]), int(us[k, 2]))
        for k in range(ei.size(1))
    }


# -----------------------------------------------------------------------
#  Equivalence
# -----------------------------------------------------------------------

def _check_equivalence(cell, label, device=None):
    cell = cell if device is None else cell.to(device)
    coords = _system(cell, device=device)

    pei, pev, pel, pus = build_edges_pbc(coords, cell, R_MAX)
    vei, vev, vel, vus = build_edges_pbc_vesin(coords, cell, R_MAX)

    assert vei.device == coords.device, (
        f"{label}: edges came back on {vei.device}, positions are on {coords.device}"
    )

    assert vei.size(1) == pei.size(1), (
        f"{label}: edge count {vei.size(1)} from vesin, {pei.size(1)} from pure"
    )

    ps = torch.sort(pel)[0]
    vs = torch.sort(vel)[0]
    dmax = float((ps - vs).abs().max())
    assert dmax < 1e-10, f"{label}: sorted edge lengths differ by {dmax:.3e}"

    assert _edge_set(vei, vus) == _edge_set(pei, pus), (
        f"{label}: the two builders disagree about which pairs exist"
    )

    # The identity every consuming model relies on, checked on the vesin output.
    recon = coords[vei[1]] - coords[vei[0]] + vus @ cell
    idmax = float((recon - vev).abs().max())
    assert idmax < 1e-12, f"{label}: edge_vecs identity off by {idmax:.3e}"

    assert bool(torch.allclose(vus, torch.round(vus), atol=1e-12)), \
        f"{label}: unit_shifts are not integers"
    assert bool((vel < R_MAX).all()), f"{label}: an edge escaped the cutoff"

    # Contract types, not just values.
    assert vei.dtype == torch.long
    assert vev.dtype == coords.dtype and vel.dtype == coords.dtype
    assert vus.dtype == coords.dtype
    assert vev.shape == (vei.size(1), 3) and vus.shape == (vei.size(1), 3)

    return vei.size(1)


def test_cubic_matches_pure_builder():
    _need_vesin()
    n = _check_equivalence(CUBIC, "cubic")
    print(f"    cubic: {n} edges, identical to the pure builder")


def test_triclinic_matches_pure_builder():
    _need_vesin()
    n = _check_equivalence(TRICLINIC, "triclinic")
    print(f"    triclinic: {n} edges, identical to the pure builder")


def test_cuda_matches_pure_builder():
    """
    The CUDA path is genuinely different code, so it gets its own check.

    vesin has no GPU kernels.  Given CUDA positions it copies them down, walks
    its cell list on the host, and hands indices back up, and those indices are
    then used to index the live CUDA positions.  A round trip like that is
    exactly where indices and shifts could quietly stop lining up, and this is
    the path a GPU-resident MD run would take.
    """
    _need_vesin()
    if not _usable_cuda():
        msg = "no usable CUDA device in this environment"
        if pytest is not None:
            pytest.skip(msg)
        print(f"    SKIP cuda: {msg}")
        return
    dev = torch.device("cuda")
    nc = _check_equivalence(CUBIC, "cubic/cuda", dev)
    nt = _check_equivalence(TRICLINIC, "triclinic/cuda", dev)
    print(f"    cuda: {nc} cubic and {nt} triclinic edges, identical to the "
          "pure builder on the same device")


def test_float32_only_disagrees_at_the_cutoff():
    """
    In float32 the two backends can split on a borderline pair.

    vesin searches in double whatever it is handed, the pure builder rounds in
    the dtype it is given, so a pair almost exactly at r_max can land on
    different sides.  Anything they disagree about has to be one of those, and
    nothing else.
    """
    _need_vesin()
    base = _system(CUBIC, n=400, seed=1)

    # A random system almost never puts a pair close enough to the cutoff for
    # this to bite, so plant some.  These sit within a part in ten million of
    # r_max, which is about where float32 stops being able to tell.
    g = torch.Generator().manual_seed(21)
    d = torch.randn(60, 3, generator=g, dtype=torch.float64)
    d = d / d.norm(dim=1, keepdim=True)
    eps = torch.linspace(-2e-7, 2e-7, 60, dtype=torch.float64).unsqueeze(1)
    anchors = _system(CUBIC, n=60, seed=22)
    planted = anchors + d * R_MAX * (1.0 + eps)

    coords64 = torch.cat([base, anchors, planted], dim=0)
    coords = coords64.to(torch.float32)
    cell = CUBIC.to(torch.float32)

    pei, _, _, pus = build_edges_pbc(coords, cell, R_MAX)
    vei, _, _, vus = build_edges_pbc_vesin(coords, cell, R_MAX)

    pure = _edge_set(pei, pus)
    ves = _edge_set(vei, vus)
    only = (pure - ves) | (ves - pure)

    # Judge the disputed pairs in double, but from the float32 positions that
    # were actually handed in, since those rounded coordinates are what both
    # backends were asked about.
    ref = coords.to(torch.float64)
    worst = 0.0
    for (i, j, sx, sy, sz) in only:
        shift = torch.tensor([sx, sy, sz], dtype=torch.float64)
        d = float((ref[j] - ref[i] + shift @ CUBIC).norm())
        gap = abs(d - R_MAX)
        worst = max(worst, gap)

    print(f"    float32: {len(pure)} pure vs {len(ves)} vesin edges, "
          f"{len(only)} disputed, worst is {worst:.2e} A from the cutoff")
    assert worst < 1e-4, (
        f"a disputed edge sits {worst:.3e} A from the cutoff, which is too far "
        "to blame on float32"
    )


def test_empty_edge_list_matches_the_pure_shapes():
    """A cutoff nothing can reach still has to return the right empty tensors."""
    _need_vesin()
    coords = _system(CUBIC, n=50, seed=2)
    tiny = 0.01

    pei, pev, pel, pus = build_edges_pbc(coords, CUBIC, tiny)
    vei, vev, vel, vus = build_edges_pbc_vesin(coords, CUBIC, tiny)

    assert pei.size(1) == 0 and vei.size(1) == 0
    for p, v, name in ((pei, vei, "edge_index"), (pev, vev, "edge_vecs"),
                       (pel, vel, "edge_len"), (pus, vus, "unit_shifts")):
        assert p.shape == v.shape, f"{name}: {p.shape} vs {v.shape}"
        assert p.dtype == v.dtype, f"{name}: {p.dtype} vs {v.dtype}"
        assert p.device == v.device, f"{name}: {p.device} vs {v.device}"
    print("    empty: shapes, dtypes and device all match the pure builder")


def test_min_image_guard_is_kept():
    """
    vesin could handle a box smaller than twice the cutoff, but we do not let
    it, so that both backends accept exactly the same systems.
    """
    _need_vesin()
    small = torch.eye(3, dtype=torch.float64) * (2 * R_MAX - 0.5)
    coords = _system(small, n=30, seed=3)

    for label, fn in (("pure", lambda: build_edges_pbc(coords, small, R_MAX)),
                      ("vesin", lambda: build_edges_pbc_vesin(coords, small, R_MAX))):
        raised = ""
        try:
            fn()
        except Exception as e:
            raised = str(e)
        assert "too small" in raised.lower(), f"{label} did not refuse a small box"

    # And it can be turned off deliberately, in which case vesin reports the
    # extra images the pure builder cannot see.
    off = VesinPBCEdges(R_MAX, enforce_min_image=False)
    ei, _, _, _ = off(coords, small, R_MAX)
    assert ei.size(1) > 0
    print(f"    guard: both refuse a small box; with it off vesin finds "
          f"{ei.size(1)} multi-image edges")


def test_cutoff_mismatch_is_refused():
    """A builder made for one cutoff must not quietly serve another."""
    _need_vesin()
    b = VesinPBCEdges(R_MAX)
    coords = _system(CUBIC, n=30, seed=4)
    raised = ""
    try:
        b(coords, CUBIC, R_MAX + 1.0)
    except Exception as e:
        raised = str(e)
    assert "cutoff" in raised.lower(), "a mismatched cutoff went through"
    print("    mismatch: a wrong r_max at the call site is refused")


def test_gradients_flow_through_the_vesin_edges():
    """Forces come from these vectors, so they have to stay on the graph."""
    _need_vesin()
    coords = _system(CUBIC, n=80, seed=5).requires_grad_(True)
    _, vev, vel, _ = build_edges_pbc_vesin(coords, CUBIC, R_MAX)
    vel.sum().backward()
    assert coords.grad is not None
    assert bool(torch.isfinite(coords.grad).all())
    assert float(coords.grad.abs().max()) > 0.0
    print("    autograd: gradients reach the positions")


# -----------------------------------------------------------------------
#  Backend selection
# -----------------------------------------------------------------------

def test_factory_default_stays_pure():
    b = make_pbc_edge_builder(R_MAX)
    assert isinstance(b, PurePBCEdges), "the default backend is no longer pure"

    coords = _system(CUBIC, n=60, seed=6)
    ei, ev, el, us = b(coords, CUBIC, R_MAX)
    pei, pev, pel, pus = build_edges_pbc(coords, CUBIC, R_MAX)
    assert bool(torch.equal(ei, pei)) and bool(torch.equal(ev, pev))
    print("    factory: the default is still the pure builder, byte for byte")


def test_factory_auto_falls_back_without_vesin():
    b = make_pbc_edge_builder(R_MAX, backend="auto")
    if VESIN_AVAILABLE:
        assert isinstance(b, VesinPBCEdges)
    else:
        assert isinstance(b, PurePBCEdges)
    print(f"    factory: auto picked {type(b).__name__}")


def test_unknown_backend_is_refused():
    raised = ""
    try:
        make_pbc_edge_builder(R_MAX, backend="nonsense")
    except ValueError as e:
        raised = str(e)
    assert "nonsense" in raised
    print("    factory: an unknown backend name is refused")


# -----------------------------------------------------------------------
#  TorchScript, which is the whole point of the repo
# -----------------------------------------------------------------------

class _Tiny(torch.nn.Module):
    """The smallest thing that exercises a held edge builder end to end."""

    def __init__(self, builder, r_max: float):
        super().__init__()
        self.edges = builder
        self.r_max = float(r_max)

    def forward(self, coords: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        _, _, edge_len, _ = self.edges(coords, cell, self.r_max)
        return edge_len.sum().reshape(1)


def _script_save_reload(builder, tmp_name):
    import os
    import tempfile

    coords = _system(CUBIC, n=120, seed=7)
    m = _Tiny(builder, R_MAX)
    eager = float(m(coords, CUBIC))

    s = torch.jit.script(m)
    scripted = float(s(coords, CUBIC))

    path = os.path.join(tempfile.gettempdir(), tmp_name)
    torch.jit.save(s, path)
    r = torch.jit.load(path)
    reloaded = float(r(coords, CUBIC))
    os.remove(path)

    assert abs(eager - scripted) < 1e-9, f"scripting changed the answer: {eager} vs {scripted}"
    assert abs(eager - reloaded) < 1e-9, f"reload changed the answer: {eager} vs {reloaded}"
    return eager


def test_pure_builder_scripts_saves_and_reloads():
    v = _script_save_reload(PurePBCEdges(R_MAX), "vesin_eq_pure.pt")
    print(f"    torchscript: pure builder survives save and reload, sum {v:.6f}")


def test_vesin_builder_scripts_saves_and_reloads():
    _need_vesin()
    v = _script_save_reload(VesinPBCEdges(R_MAX), "vesin_eq_vesin.pt")
    print(f"    torchscript: vesin builder survives save and reload, sum {v:.6f}")


def test_scripted_backends_agree():
    """Both scripted builders inside the same module have to give one answer."""
    _need_vesin()
    coords = _system(CUBIC, n=120, seed=7)
    p = torch.jit.script(_Tiny(PurePBCEdges(R_MAX), R_MAX))
    v = torch.jit.script(_Tiny(VesinPBCEdges(R_MAX), R_MAX))
    a, b = float(p(coords, CUBIC)), float(v(coords, CUBIC))
    assert abs(a - b) < 1e-9, f"scripted backends disagree: {a} vs {b}"
    print(f"    torchscript: both backends give {a:.6f}")


# -----------------------------------------------------------------------
#  Benchmark, not part of the test run
# -----------------------------------------------------------------------

def _time_builder(fn, coords, cell, r_max, device, repeats=20):
    for _ in range(3):
        fn(coords, cell, r_max)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(coords, cell, r_max)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e3


def bench(n=3800, L=34.0, r_max=5.5, dtype=torch.float32):
    print(f"\nN={n}  cubic L={L}  r_max={r_max}  dtype={dtype}")
    print(f"torch {torch.__version__}   vesin {vesin_version() or 'absent'}")

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        try:
            torch.zeros(8, device="cuda").sum().item()
            devices.append(torch.device("cuda"))
        except Exception as e:
            print(f"  (no GPU run: {type(e).__name__}: {str(e).splitlines()[0]})")

    g = torch.Generator().manual_seed(11)
    base = torch.rand(n, 3, generator=g, dtype=torch.float64) * L
    base_cell = torch.eye(3, dtype=torch.float64) * L

    for dev in devices:
        coords = base.to(dtype).to(dev)
        cell = base_cell.to(dtype).to(dev)

        pure = PurePBCEdges(r_max)
        ei, _, _, _ = pure(coords, cell, r_max)
        t_pure = _time_builder(pure, coords, cell, r_max, dev)
        line = f"  {str(dev):5s} pure  {t_pure:8.2f} ms/step   {ei.size(1)} edges"

        if VESIN_AVAILABLE:
            ves = VesinPBCEdges(r_max)
            vi, _, _, _ = ves(coords, cell, r_max)
            t_ves = _time_builder(ves, coords, cell, r_max, dev)
            line += (f"\n  {str(dev):5s} vesin {t_ves:8.2f} ms/step   "
                     f"{vi.size(1)} edges   {t_pure / t_ves:.1f}x")
        print(line)


# -----------------------------------------------------------------------
#  Standalone runner
# -----------------------------------------------------------------------

def _main():
    if "--bench" in sys.argv:
        bench()
        return 0

    print(f"torch {torch.__version__}   vesin "
          f"{vesin_version() if VESIN_AVAILABLE else 'NOT INSTALLED'}\n")

    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"{len(failed)} FAILURE(S): " + ", ".join(failed))
        return 1
    print(f"ALL {len(tests)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
