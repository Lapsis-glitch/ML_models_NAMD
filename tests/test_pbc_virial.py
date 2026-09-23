"""
Settle the sign, transpose and normalisation of the periodic virial that
``MACE_TS_Wrapper`` now returns, by finite difference against the energy.

The wrapper asks MACE for ``virials``, which MACE produces as the derivative of
the energy with respect to a strain of the box, negated on the way out.  Written
out, with D the strain that MACE applies internally as
``r -> r (I + D)`` and ``cell -> cell (I + D)``:

    virials_returned = -dE/dD

so the number this test measures by finite difference, dE/dD, should come out as
MINUS the returned virial.  Separately, the quantity NAMD wants is
``W = sum_i f_i (x) r_i``, and the derivation says ``W = +virials_returned``
once both are symmetrised.  Both claims are checked below; the finite difference
is the arbiter, and if it disagrees with either, the printout says so.

Run directly (``python tests/test_pbc_virial.py``) for the full report, or under
pytest for pass/fail.
"""

import os
import subprocess
import sys
import tempfile
import types

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.edges import build_edges_pbc  # noqa: E402  (read-only, for assertions)
from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper  # noqa: E402

MODEL = os.path.join(_REPO, "models", "compiled_mace_off23_medium.pt")

# r_max is 5 A, so minimum imaging needs at least 10 A across.  The finite
# difference compresses the box, and build_edges_pbc raises rather than
# degrading quietly, so leave enough headroom that no leg of the sweep trips it.
BOX = 14.0

EMPTY_PC = torch.zeros((0, 3), dtype=torch.float64)
EMPTY_PC_Q = torch.zeros((0,), dtype=torch.float64)


# -------------------------------------------------------------------
#  Test system: waters spread through the box, some pairs bonded only
#  through a face so that the imaging actually matters.
# -------------------------------------------------------------------

def _water_box():
    """Six waters in a 14 A cube, two of them straddling the x face."""
    coords = torch.tensor([
        [0.40, 6.00, 6.00], [1.36, 6.00, 6.00], [0.16, 6.93, 6.00],
        [BOX - 1.40, 6.00, 6.00], [BOX - 0.44, 6.00, 6.00], [BOX - 1.64, 6.93, 6.00],
        [5.00, 2.00, 9.00], [5.96, 2.00, 9.00], [4.76, 2.93, 9.00],
        [7.00, 3.20, 9.60], [7.96, 3.20, 9.60], [6.76, 4.13, 9.60],
        [2.00, 11.00, 3.00], [2.96, 11.00, 3.00], [1.76, 11.93, 3.00],
        [9.00, 9.00, 12.50], [9.96, 9.00, 12.50], [8.76, 9.93, 12.50],
    ], dtype=torch.float64)
    Z = torch.tensor([8, 1, 1] * 6, dtype=torch.int64)
    cell = torch.eye(3, dtype=torch.float64) * BOX
    return coords, Z, cell


def _load():
    return torch.jit.script(MACE_TS_Wrapper(MODEL, device="cpu").eval())


def _call(model, coords, Z, cell):
    """One evaluation on a fresh leaf copy of the coordinates.

    MACE sets requires_grad on whatever tensor it is handed, and float64 in
    means .to(float64) hands back the very same object.  Reusing a tensor
    across calls would therefore drag autograd state from one evaluation into
    the next, and a strained copy of an already-grad-enabled tensor is not a
    leaf, which raises.  Detaching per call keeps every evaluation independent.
    """
    c = coords.detach().clone()
    k = cell.detach().clone().reshape(1, 3, 3)
    return model(c, Z, EMPTY_PC, EMPTY_PC_Q, k)


# -------------------------------------------------------------------
#  Finite difference
# -------------------------------------------------------------------

def _strain_direction(i, j):
    """Symmetric direction P such that contracting dE/dD with it gives dE/dD_ij.

    dE/dh for D = h*P is sum_ab (dE/dD_ab) P_ab.  Half on each off-diagonal
    entry picks out the ij component once dE/dD is symmetric, which it is,
    because MACE only ever uses 0.5*(D + D^T).
    """
    P = torch.zeros((3, 3), dtype=torch.float64)
    if i == j:
        P[i, i] = 1.0
    else:
        P[i, j] = 0.5
        P[j, i] = 0.5
    return P


def _fd_dE_dstrain(model, coords, Z, cell, h):
    """Central-difference dE/dD, all nine components, in kcal/mol."""
    fd = torch.zeros((3, 3), dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            P = _strain_direction(i, j)
            plus = eye + h * P
            minus = eye - h * P
            # Positions and box are strained together, and the edges are then
            # rebuilt from the strained box, which is what makes this the same
            # deformation MACE differentiates against.
            e_p = _call(model, coords @ plus, Z, cell @ plus)[0]
            e_m = _call(model, coords @ minus, Z, cell @ minus)[0]
            fd[i, j] = (float(e_p) - float(e_m)) / (2.0 * h)
    return fd


def _fmt(m):
    return "\n".join(
        "    [" + "  ".join("%14.6f" % float(x) for x in row) + "]" for row in m
    )


def _rel(a, b):
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-30)
    return float((a - b).abs().max()) / scale


# -------------------------------------------------------------------
#  Tests
# -------------------------------------------------------------------

def test_virial_matches_finite_difference():
    """dE/d(strain) by central difference against the returned virial."""
    model = _load()
    coords, Z, cell = _water_box()

    # If nothing is imaged the strain virial degenerates to the cluster sum and
    # the test stops distinguishing the two, so confirm the box is doing work.
    _, _, _, unit_shifts = build_edges_pbc(
        coords.to(torch.float32), cell.to(torch.float32), model.r_max
    )
    n_imaged = int((unit_shifts != 0).any(dim=1).sum())
    print("\nEdges crossing a face: %d" % n_imaged)
    assert n_imaged > 0, "no pair interacts through a face; nothing to test"

    e, f, q, virial = _call(model, coords, Z, cell)

    print("Energy                 : %.8f kcal/mol" % float(e))
    print("Returned virial (kcal/mol):\n" + _fmt(virial))

    # Two step sizes: agreement at one h alone cannot tell a correct
    # derivative from a coincidence, and drift between them would expose
    # truncation error or noise from the float32 edge build.
    results = {}
    for h in (1e-4, 1e-5):
        fd = _fd_dE_dstrain(model, coords, Z, cell, h)
        results[h] = fd
        print("\nFinite difference dE/dD at h = %g (kcal/mol):\n" % h + _fmt(fd))
        print("  rel. diff vs -virial : %.3e" % _rel(fd, -virial))
        print("  rel. diff vs +virial : %.3e" % _rel(fd, virial))

    h_big, h_small = 1e-4, 1e-5
    stability = _rel(results[h_big], results[h_small])
    print("\nStep-size stability (h=1e-4 vs h=1e-5): %.3e" % stability)
    assert stability < 1e-4, (
        "finite difference is not converged in h; the comparison below would "
        "be meaningless (rel. drift %.3e)" % stability
    )

    fd = results[h_small]
    err_neg = _rel(fd, -virial)
    err_pos = _rel(fd, virial)

    print("\nDerivation says dE/dD = -virials_returned.")
    if err_neg < err_pos:
        print("CONFIRMED: dE/dD matches -virial to %.3e relative." % err_neg)
    else:
        print("*** DERIVED SIGN IS WRONG ***")
        print("*** The finite difference matches +virial (%.3e), not -virial "
              "(%.3e).  Trust the finite difference. ***" % (err_pos, err_neg))

    assert err_neg < 1e-5, (
        "returned virial does not equal -dE/d(strain): rel. err %.3e against "
        "-virial, %.3e against +virial" % (err_neg, err_pos)
    )

    # What NAMD would have computed on its own, for scale.  The gap is the
    # reason for this whole change: every pair that interacts through a face
    # contributes the raw separation here instead of the imaged one.
    naive = torch.einsum("ai,aj->ij", f, coords)
    naive = 0.5 * (naive + naive.transpose(-1, -2))
    print("\nNAMD's own sum f (x) r on absolute positions, same system:\n"
          + _fmt(naive))
    print("  rel. difference from the strain virial: %.3e"
          % _rel(naive, virial))


def test_virial_matches_finite_difference_triclinic():
    """The same finite difference on a skewed box.

    A cubic box cannot catch a transposed cell convention, because L*I times
    (I + D) is the same matrix either way round.  Skewing the box makes the
    two orderings differ, so this is what actually pins down that the lattice
    vectors travel as rows all the way from the edge builder into MACE.
    """
    model = _load()
    coords, Z, _ = _water_box()
    # The first lattice vector is the tilted one, and it is also the one the
    # imaged pairs travel along, so the tilt is on the path being tested
    # rather than sitting unused in a corner of the matrix.
    cell = torch.tensor([
        [15.0, 2.5, 1.5],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0],
    ], dtype=torch.float64)

    _, _, _, unit_shifts = build_edges_pbc(
        coords.to(torch.float32), cell.to(torch.float32), model.r_max
    )
    n_imaged = int((unit_shifts != 0).any(dim=1).sum())
    print("\nEdges crossing a face in the tilted box: %d" % n_imaged)
    assert n_imaged > 0, "nothing is imaged here, so this proves nothing"

    _, _, _, virial = _call(model, coords, Z, cell)
    fd = _fd_dE_dstrain(model, coords, Z, cell, 1e-5)

    print("\nTriclinic returned virial (kcal/mol):\n" + _fmt(virial))
    print("Triclinic finite difference dE/dD (kcal/mol):\n" + _fmt(fd))
    print("  rel. diff vs -virial : %.3e" % _rel(fd, -virial))
    print("  rel. diff vs +virial : %.3e" % _rel(fd, virial))

    assert _rel(fd, -virial) < 1e-5


def test_virial_is_symmetric():
    model = _load()
    coords, Z, cell = _water_box()
    _, _, _, virial = _call(model, coords, Z, cell)
    asym = float((virial - virial.transpose(-1, -2)).abs().max())
    print("\nAsymmetry of returned virial: %.3e" % asym)
    assert asym == 0.0


def test_virial_reduces_to_cluster_sum_in_a_large_box():
    """With no interaction crossing a face, the strain virial is sum f (x) r.

    This is the cross-check that needs no finite differencing: it pins the
    sign, the transpose and the scale in one comparison, because with every
    unit shift zero the strain acts on the positions alone and the box drops
    out of the derivative.
    """
    model = _load()
    coords, Z, _ = _water_box()

    big = 40.0
    cell = torch.eye(3, dtype=torch.float64) * big

    # The claim only holds if nothing is actually imaged, so check it rather
    # than trusting the box size.
    _, _, _, unit_shifts = build_edges_pbc(
        coords.to(torch.float32), cell.to(torch.float32), model.r_max
    )
    n_imaged = int((unit_shifts != 0).any(dim=1).sum())
    print("\nEdges crossing a face in the %.0f A box: %d" % (big, n_imaged))
    assert n_imaged == 0, "box is not large enough for this comparison to hold"

    _, f, _, virial = _call(model, coords, Z, cell)

    w_namd = torch.einsum("ai,aj->ij", f, coords)          # sum f (x) r
    w_namd = 0.5 * (w_namd + w_namd.transpose(-1, -2))
    r_cross_f = torch.einsum("ai,aj->ij", coords, f)       # sum r (x) f
    r_cross_f = 0.5 * (r_cross_f + r_cross_f.transpose(-1, -2))

    print("Returned virial:\n" + _fmt(virial))
    print("sum f (x) r (the NAMD form):\n" + _fmt(w_namd))
    print("rel. diff virial vs sum f (x) r : %.3e" % _rel(virial, w_namd))
    print("rel. diff virial vs sum r (x) f : %.3e" % _rel(virial, r_cross_f))

    assert _rel(virial, w_namd) < 1e-8, (
        "returned virial is not the cluster sum f (x) r in a box with no "
        "imaging: rel. err %.3e" % _rel(virial, w_namd)
    )


def test_virial_is_translation_invariant():
    """Sliding the whole system through the box must not change the virial.

    This is the property that fails for a sum over absolute positions, and it
    is what makes the strain virial usable for pressure control: nothing
    physical depends on where the box origin happens to sit.
    """
    model = _load()
    coords, Z, cell = _water_box()

    _, f0, _, v0 = _call(model, coords, Z, cell)

    shift = torch.tensor([5.3, -2.7, 9.1], dtype=torch.float64)
    moved = torch.remainder(coords + shift, BOX)
    _, f1, _, v1 = _call(model, moved, Z, cell)

    print("\nVirial before the shift:\n" + _fmt(v0))
    print("Virial after wrapping the system by (5.3, -2.7, 9.1) A:\n" + _fmt(v1))
    print("  rel. change: %.3e" % _rel(v0, v1))

    naive0 = torch.einsum("ai,aj->ij", f0, coords)
    naive1 = torch.einsum("ai,aj->ij", f1, moved)
    print("  the same shift moves NAMD's sum f (x) r by %.3e relative"
          % _rel(naive0, naive1))

    assert _rel(v0, v1) < 1e-8


def test_batched_virial_matches_single():
    """Two identical replicas must each reproduce the single-molecule virial.

    forward_batch builds its shifts with a per-molecule cell lookup rather than
    one shared cell, so this is the cheapest thing that catches an indexing
    mistake there.
    """
    model = _load()
    coords, Z, cell = _water_box()

    _, _, _, v_single = _call(model, coords, Z, cell)

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    cells = torch.stack([cell, cell]).detach().clone()

    energies, forces, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells,
    )

    assert virials.shape == (2, 3, 3)
    print("\nBatched virial, replica 0:\n" + _fmt(virials[0]))
    print("Single-molecule virial:\n" + _fmt(v_single))
    print("rel. diff replica 0 vs single : %.3e" % _rel(virials[0], v_single))
    print("rel. diff replica 1 vs single : %.3e" % _rel(virials[1], v_single))
    assert _rel(virials[0], v_single) < 1e-10
    assert _rel(virials[1], v_single) < 1e-10


def test_non_periodic_virial_is_zero():
    model = _load()
    coords, Z, _ = _water_box()
    zero_cell = torch.zeros((3, 3), dtype=torch.float64)
    _, _, _, virial = _call(model, coords, Z, zero_cell)
    assert virial.shape == (3, 3)
    assert float(virial.abs().max()) == 0.0

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    cells = torch.zeros((2, 3, 3), dtype=torch.float64)
    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells,
    )
    assert virials.shape == (2, 3, 3)
    assert float(virials.abs().max()) == 0.0


# -------------------------------------------------------------------
#  Regressions against the two earlier versions of this file
# -------------------------------------------------------------------

def _load_variant(path, name):
    """Import an older copy of the wrapper without putting it in the package.

    The file uses relative imports, so those are rewritten to absolute ones and
    the source is executed into a throwaway module.  Nothing is written into
    src/, so the two versions can be held side by side in one process.
    """
    with open(path) as fh:
        src = fh.read()
    src = src.replace("from ..constants import", "from src.constants import")
    src = src.replace("from ..edges import", "from src.edges import")
    src = src.replace("from ..export import", "from src.export import")
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def _variant_path(env_key, default_name):
    scratch = os.environ.get("MLFF_VIRIAL_SCRATCH", "")
    if scratch:
        p = os.path.join(scratch, default_name)
        if os.path.exists(p):
            return p
    return os.environ.get(env_key, "")


def test_non_periodic_path_unchanged_vs_pre_virial():
    """The cluster path must be untouched by the virial work, bit for bit."""
    path = _variant_path("MLFF_PRE_VIRIAL_WRAPPER", "wrap_mace_pre_virial.py")
    if not path or not os.path.exists(path):
        print("\nskipped: no pre-virial snapshot available")
        return

    old = _load_variant(path, "wrap_mace_pre_virial")
    old_model = torch.jit.script(old.MACE_TS_Wrapper(MODEL, device="cpu").eval())
    new_model = _load()

    coords, Z, _ = _water_box()
    zero_cell = torch.zeros((1, 3, 3), dtype=torch.float64)

    e_old, f_old, q_old = old_model(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q, zero_cell.clone())
    e_new, f_new, q_new, v_new = new_model(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q, zero_cell.clone())

    print("\nNon-periodic, pre-virial vs now:")
    print("  |dE|  = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF|  = %.3e" % float((f_old - f_new).abs().max()))
    print("  |dQ|  = %.3e" % float((q_old - q_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)
    assert torch.equal(q_old, q_new)
    assert float(v_new.abs().max()) == 0.0

    # The periodic leg below is the one that matters most.  Asking for virials
    # makes MACE throw away the shifts it was handed and rebuild them from
    # unit_shifts and the cell, so if those two disagreed with each other the
    # geometry would quietly change and the energy would move.  An energy that
    # matches to the last bit is the evidence that they agree.
    cell = (torch.eye(3, dtype=torch.float64) * BOX).reshape(1, 3, 3)
    e_old, f_old, q_old = old_model(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q, cell.clone())
    e_new, f_new, q_new, _ = new_model(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q, cell.clone())
    print("Periodic, pre-virial vs now (shifts rebuilt from unit_shifts):")
    print("  |dE|  = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF|  = %.3e" % float((f_old - f_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)
    assert torch.equal(q_old, q_new)


def test_non_periodic_path_unchanged_vs_head():
    """Same check against the last committed version, which predates PBC.

    That copy has no cell argument at all, so this answers a wider question
    than the one above: whether the whole periodic conversion left a cluster
    calculation alone, not just the virial addition.
    """
    # Pulled straight out of git so this stays runnable later, unlike the
    # pre-virial copy above which was never committed.
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:src/wrappers/wrap_compiled_mace.py"],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        print("\nskipped: could not read the committed wrapper from git")
        return

    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src)
    tmp.close()
    path = tmp.name

    head = _load_variant(path, "wrap_mace_head")
    # The file has to outlive scripting: TorchScript reads the class source
    # back off disk rather than from the module object.
    head_model = torch.jit.script(head.MACE_TS_Wrapper(MODEL, device="cpu").eval())
    new_model = _load()

    coords, Z, _ = _water_box()

    e_old, f_old, q_old = head_model(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q)
    e_new, f_new, q_new, v_new = new_model(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q,
        torch.zeros((1, 3, 3), dtype=torch.float64))

    print("\nNon-periodic, HEAD vs now:")
    print("  |dE|  = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF|  = %.3e" % float((f_old - f_new).abs().max()))
    print("  |dQ|  = %.3e" % float((q_old - q_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)
    assert torch.equal(q_old, q_new)
    assert float(v_new.abs().max()) == 0.0

    os.unlink(path)


def test_torchscript_compiles():
    """Scripting has to survive the extra return value in both entry points."""
    scripted = _load()
    assert scripted is not None
    coords, Z, cell = _water_box()
    out = _call(scripted, coords, Z, cell)
    assert len(out) == 4


if __name__ == "__main__":
    fns = [
        test_torchscript_compiles,
        test_virial_matches_finite_difference,
        test_virial_matches_finite_difference_triclinic,
        test_virial_is_symmetric,
        test_virial_is_translation_invariant,
        test_virial_reduces_to_cluster_sum_in_a_large_box,
        test_batched_virial_matches_single,
        test_non_periodic_virial_is_zero,
        test_non_periodic_path_unchanged_vs_pre_virial,
        test_non_periodic_path_unchanged_vs_head,
    ]
    failures = 0
    for fn in fns:
        print("\n" + "=" * 68)
        print(fn.__name__)
        print("=" * 68)
        try:
            fn()
            print("PASS")
        except Exception as exc:            # noqa: BLE001
            failures += 1
            print("FAIL: %s: %s" % (type(exc).__name__, exc))
    print("\n%d of %d failed" % (failures, len(fns)))
    sys.exit(1 if failures else 0)
