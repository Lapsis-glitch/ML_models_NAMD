"""
Check the periodic virial that the X-MACE and SchNetPack wrappers now return.

Neither model hands one over.  X-MACE advertises ``compute_virials`` but its
compiled ``get_outputs`` ends in ``return (forces, None, None, hessian)``, so
the output is hardwired to None and the ``displacement`` it carries is a zero
tensor that never reaches the positions, the shifts or the cell.  SchNetPack has
stress code but bakes ``calc_stress=False`` into the artifact and takes no
runtime flags.  Both wrappers therefore derive the virial from a strain, using
the shared helpers in ``src/virial.py``, and this file is what says the result
is right rather than merely plausible.

The finite difference is the headline, but on its own it is weak evidence for
X-MACE: that model runs in float32 with a total energy near -143,700 kcal/mol,
so rounding alone moves the energy by about 0.01 kcal/mol and the difference
quotient inherits that.  Two checks here are exact and carry the argument
instead:

  * in a box large enough that nothing is imaged, the strain virial must equal
    ``sum_i f_i (x) r_i``, which pins sign, transpose and scale at once;
  * sliding the system through the box must not move the virial, which is
    exactly what fails for a sum over absolute positions.

The finite difference is then used to discriminate rather than to certify: it is
compared both against the virial and against the wrong answer you get by
straining the positions and leaving the periodic images where they were.  With
most of the edges crossing a face those two are far apart, so the comparison
still means something at float32 precision.

Run directly (``python tests/test_virial_xmace_schnet.py``) for the full report,
or under pytest for pass/fail.
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
from src.wrappers.wrap_schnetpack import SchNetPack_Wrapper  # noqa: E402
from src.wrappers.wrap_xmace import XMACE_TS_Wrapper  # noqa: E402

XMACE_MODEL = os.path.join(_REPO, "models", "fulvene_compiled.pt")
SCHNET_MODEL = os.path.join(_REPO, "models", "compiled_schnet_default.pt")

SCHNET_R_MAX = 5.0

# Both cutoffs are 5 A, so minimum imaging needs at least 10 A across.  The
# finite difference compresses the box and build_edges_pbc raises rather than
# degrading quietly, so leave headroom for every leg of the sweep.
BOX = 13.0

# Big enough that nothing is imaged, for the cluster-sum comparison.
BIG_BOX = 40.0

EMPTY_PC = torch.zeros((0, 3), dtype=torch.float64)
EMPTY_PC_Q = torch.zeros((0,), dtype=torch.float64)


# -------------------------------------------------------------------
#  Test systems
# -------------------------------------------------------------------

def _xmace_system():
    """Carbons and hydrogens spread so most pairs meet through the x face.

    The fulvene model only knows H and C, and it was trained on one molecule,
    so the energy of this arrangement means nothing physically.  That is fine:
    what is being checked is whether the wrapper's strain reaches the imaged
    pairs, and any smooth differentiable energy answers that question.
    """
    coords = torch.tensor([
        [0.30, 6.0, 6.0], [1.70, 6.0, 6.0], [0.60, 7.3, 6.0],
        [BOX - 0.90, 6.0, 6.0], [BOX - 2.1, 6.6, 6.0], [BOX - 1.3, 7.7, 6.2],
        [0.30, 6.0, 7.1], [1.70, 6.0, 7.2], [0.60, 8.3, 6.0],
        [BOX - 0.90, 6.0, 7.2], [BOX - 2.1, 7.6, 6.0], [BOX - 1.3, 7.7, 7.3],
    ], dtype=torch.float64)
    Z = torch.tensor([6] * 6 + [1] * 6, dtype=torch.int64)
    cell = torch.eye(3, dtype=torch.float64) * BOX
    return coords, Z, cell


def _schnet_system():
    """Four waters, two of them bonded to each other only through the x face."""
    coords = torch.tensor([
        [0.40, 6.00, 6.00], [1.36, 6.00, 6.00], [0.16, 6.93, 6.00],
        [BOX - 1.40, 6.00, 6.00], [BOX - 0.44, 6.00, 6.00], [BOX - 1.64, 6.93, 6.00],
        [5.00, 2.00, 9.00], [5.96, 2.00, 9.00], [4.76, 2.93, 9.00],
        [7.00, 3.20, 9.60], [7.96, 3.20, 9.60], [6.76, 4.13, 9.60],
    ], dtype=torch.float64)
    Z = torch.tensor([8, 1, 1] * 4, dtype=torch.int64)
    cell = torch.eye(3, dtype=torch.float64) * BOX
    return coords, Z, cell


def _load_xmace():
    return torch.jit.script(
        XMACE_TS_Wrapper(XMACE_MODEL, state_idx=0, device="cpu").eval()
    )


def _load_schnet():
    return torch.jit.script(
        SchNetPack_Wrapper(SCHNET_MODEL, r_max=SCHNET_R_MAX, device="cpu").eval()
    )


def _call(model, coords, Z, cell):
    """One evaluation on a fresh copy of the inputs.

    Both wrappers set requires_grad on tensors derived from what they are
    handed, so reusing a tensor across calls would drag autograd state from one
    evaluation into the next.  Copying per call keeps them independent.
    """
    c = coords.detach().clone()
    k = cell.detach().clone().reshape(1, 3, 3)
    return model(c, Z, EMPTY_PC, EMPTY_PC_Q, k)


# -------------------------------------------------------------------
#  Finite difference
# -------------------------------------------------------------------

def _strain_direction(i, j):
    """Symmetric direction P whose contraction with dE/dD gives dE/dD_ij.

    Half on each off-diagonal entry picks the ij component out once dE/dD is
    symmetric, which it is, because the helpers only ever use 0.5*(D + D^T).
    """
    P = torch.zeros((3, 3), dtype=torch.float64)
    if i == j:
        P[i, i] = 1.0
    else:
        P[i, j] = 0.5
        P[j, i] = 0.5
    return P


def _fd_dE_dstrain(model, coords, Z, cell, h, strain_cell=True):
    """Central-difference dE/dD, all nine components, in kcal/mol.

    With ``strain_cell`` the positions and the box are deformed together and
    the wrapper rebuilds the shifts from the deformed box, which is the same
    deformation the wrapper differentiates against.  With it off the box is
    left alone, which is the classic mistake: the imaged neighbours stay where
    they were while their home atoms move.  The wrong answer is worth
    measuring, because it is what a plausible but incorrect virial would match.
    """
    fd = torch.zeros((3, 3), dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)
    for i in range(3):
        for j in range(3):
            P = _strain_direction(i, j)
            plus = eye + h * P
            minus = eye - h * P
            cell_p = cell @ plus if strain_cell else cell
            cell_m = cell @ minus if strain_cell else cell
            e_p = _call(model, coords @ plus, Z, cell_p)[0]
            e_m = _call(model, coords @ minus, Z, cell_m)[0]
            fd[i, j] = (float(e_p) - float(e_m)) / (2.0 * h)
    return fd


def _fmt(m):
    return "\n".join(
        "    [" + "  ".join("%14.6f" % float(x) for x in row) + "]" for row in m
    )


def _rel(a, b):
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-30)
    return float((a - b).abs().max()) / scale


def _n_imaged(coords, cell, r_max):
    _, _, _, unit_shifts = build_edges_pbc(
        coords.to(torch.float32), cell.to(torch.float32), r_max
    )
    return int((unit_shifts != 0).any(dim=1).sum()), int(unit_shifts.size(0))


# -------------------------------------------------------------------
#  X-MACE has no native virial to take
# -------------------------------------------------------------------

def test_xmace_has_no_native_virial():
    """The reason the wrapper derives its own, stated as a check.

    X-MACE takes compute_virials and compute_stress and returns None for both,
    so the cheap fused route the MACE wrapper uses does not exist here.  If a
    later checkpoint ever grows one, this is the test that will say so.
    """
    inner = torch.jit.load(XMACE_MODEL, map_location="cpu").eval()
    coords, Z, cell = _xmace_system()
    coords32 = coords.to(torch.float32).detach().requires_grad_(True)
    cell32 = cell.to(torch.float32)
    r_max = float(inner.r_max)

    edge_index, _, _, unit_shifts = build_edges_pbc(coords32, cell32, r_max)
    numbers = inner.atomic_numbers
    node_attrs = (Z.unsqueeze(1) == numbers.unsqueeze(0)).to(torch.float32)
    N = coords.size(0)

    out = inner(
        {
            "positions": coords32,
            "atomic_numbers": Z,
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "shifts": unit_shifts @ cell32,
            "unit_shifts": unit_shifts,
            "cell": cell32,
            "batch": torch.zeros(N, dtype=torch.long),
            "ptr": torch.tensor([0, N]),
        },
        training=False,
        compute_force=True,
        compute_hessian=False,
        compute_virials=True,
        compute_stress=False,
    )
    print("\nAsked X-MACE for compute_virials=True:")
    print("  out['virials'] is", out.get("virials"))
    print("  out['stress']  is", out.get("stress"))
    assert out.get("virials") is None
    assert out.get("stress") is None


# -------------------------------------------------------------------
#  Finite difference, one per model
# -------------------------------------------------------------------

def _finite_difference_report(name, model, coords, Z, cell, r_max,
                              steps, tol_correct, tol_gap):
    imaged, total = _n_imaged(coords, cell, r_max)
    print("\n%s: %d of %d edges cross a face" % (name, imaged, total))
    assert imaged > 0, "no pair interacts through a face; nothing to test"

    e, f, q, virial = _call(model, coords, Z, cell)
    print("Energy                    : %.6f kcal/mol" % float(e))
    print("Returned virial (kcal/mol):\n" + _fmt(virial))

    results = {}
    for h in steps:
        fd = _fd_dE_dstrain(model, coords, Z, cell, h)
        results[h] = fd
        print("\nFinite difference dE/dD at h = %g (kcal/mol):\n" % h + _fmt(fd))
        print("  rel. diff vs -virial : %.3e" % _rel(fd, -virial))
        print("  rel. diff vs +virial : %.3e" % _rel(fd, virial))

    # Printed rather than asserted on.  Both models run in float32, so shrinking
    # the step makes the difference quotient WORSE, not better: below about
    # h = 3e-3 the rounding in the energy outweighs the truncation error.  The
    # drift here therefore measures the noise floor and not convergence, which
    # is why the exact checks further down carry the argument.
    drift = _rel(results[steps[0]], results[steps[1]])
    print("\nStep-size drift (h=%g vs h=%g): %.3e" % (steps[0], steps[1], drift))

    fd = results[steps[1]]
    err_neg = _rel(fd, -virial)
    err_pos = _rel(fd, virial)
    print("Derivation says dE/dD = -virial.")
    print("  against -virial: %.3e" % err_neg)
    print("  against +virial: %.3e" % err_pos)
    assert err_neg < err_pos, "the sign is backwards; trust the finite difference"

    # The discriminator.  Straining the positions and leaving the box alone is
    # the wrong deformation, and a virial that had missed the periodic images
    # would match THIS instead.  It has to be far away, or the agreement above
    # proves nothing about the imaging.
    fd_bad = _fd_dE_dstrain(model, coords, Z, cell, steps[1], strain_cell=False)
    err_bad = _rel(fd_bad, -virial)
    print("\nSame difference with the box held fixed (the wrong deformation):\n"
          + _fmt(fd_bad))
    print("  rel. diff vs -virial : %.3e" % err_bad)
    print("  correct / wrong      : %.3e vs %.3e" % (err_neg, err_bad))

    assert err_neg < tol_correct, (
        "virial does not equal -dE/d(strain): rel. err %.3e" % err_neg)
    assert err_bad > tol_gap, (
        "the position-only strain gives nearly the same answer (%.3e), so this "
        "comparison cannot tell a correct virial from one that ignores the "
        "periodic images" % err_bad)

    # What NAMD would have summed on its own, for scale.
    naive = torch.einsum("ai,aj->ij", f, coords)
    naive = 0.5 * (naive + naive.transpose(-1, -2))
    print("\nNAMD's own sum f (x) r over absolute positions:\n" + _fmt(naive))
    print("  rel. difference from the strain virial: %.3e" % _rel(naive, virial))


def test_xmace_virial_matches_finite_difference():
    model = _load_xmace()
    coords, Z, cell = _xmace_system()
    # float32 weights and an energy near -143,700 kcal/mol put roughly 0.01
    # kcal/mol of rounding into every evaluation, so the step has to be large
    # enough to lift the signal clear of it.  A sweep over h bottoms out around
    # 3e-3 at under a percent; anything smaller is worse, not better.
    _finite_difference_report(
        "X-MACE", model, coords, Z, cell, model.r_max,
        steps=(1e-2, 3e-3), tol_correct=2e-2, tol_gap=0.2,
    )


def test_schnet_virial_matches_finite_difference():
    model = _load_schnet()
    coords, Z, cell = _schnet_system()
    # Same float32 story, but the energy here is a few tens of kcal/mol rather
    # than a hundred thousand, so there is far less rounding to fight and the
    # agreement is two orders of magnitude better.
    _finite_difference_report(
        "SchNetPack", model, coords, Z, cell, SCHNET_R_MAX,
        steps=(1e-2, 3e-3), tol_correct=1e-3, tol_gap=0.2,
    )


def test_xmace_virial_matches_finite_difference_triclinic():
    """The same difference on a skewed box.

    A cubic box cannot catch a transposed cell convention, because L*I times
    (I + D) is the same matrix either way round.  Skewing it makes the two
    orderings differ, which is what pins down that the lattice vectors travel
    as rows from the edge builder into the model.
    """
    model = _load_xmace()
    coords, Z, _ = _xmace_system()
    cell = torch.tensor([
        [15.0, 2.5, 1.5],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0],
    ], dtype=torch.float64)

    imaged, total = _n_imaged(coords, cell, model.r_max)
    print("\nTilted box: %d of %d edges cross a face" % (imaged, total))
    assert imaged > 0

    _, _, _, virial = _call(model, coords, Z, cell)
    fd = _fd_dE_dstrain(model, coords, Z, cell, 3e-3)
    print("Returned virial:\n" + _fmt(virial))
    print("Finite difference dE/dD:\n" + _fmt(fd))
    print("  rel. diff vs -virial : %.3e" % _rel(fd, -virial))
    assert _rel(fd, -virial) < 2e-2


# -------------------------------------------------------------------
#  Exact checks that do not depend on finite-difference precision
# -------------------------------------------------------------------

def _cluster_sum_check(name, model, coords, Z, r_max, tol):
    """With nothing imaged, the strain virial is sum_i f_i (x) r_i.

    No differencing involved: with every unit shift zero the strain acts on the
    positions alone and the box drops out of the derivative.  Sign, transpose
    and scale are all pinned by this one comparison.
    """
    cell = torch.eye(3, dtype=torch.float64) * BIG_BOX
    imaged, total = _n_imaged(coords, cell, r_max)
    print("\n%s in a %.0f A box: %d of %d edges cross a face"
          % (name, BIG_BOX, imaged, total))
    assert imaged == 0, "box is not large enough for this comparison to hold"

    _, f, _, virial = _call(model, coords, Z, cell)

    w_namd = torch.einsum("ai,aj->ij", f, coords)
    w_namd = 0.5 * (w_namd + w_namd.transpose(-1, -2))

    print("Returned virial:\n" + _fmt(virial))
    print("sum f (x) r:\n" + _fmt(w_namd))
    print("  rel. diff: %.3e" % _rel(virial, w_namd))
    assert _rel(virial, w_namd) < tol


def test_xmace_virial_reduces_to_cluster_sum_in_a_large_box():
    model = _load_xmace()
    coords, Z, _ = _xmace_system()
    # float32 throughout, and the forces are not even reproducible to the last
    # bit run to run, so 1e-5 is as tight as this can honestly be.
    _cluster_sum_check("X-MACE", model, coords, Z, model.r_max, 1e-5)


def test_schnet_virial_reduces_to_cluster_sum_in_a_large_box():
    model = _load_schnet()
    coords, Z, _ = _schnet_system()
    _cluster_sum_check("SchNetPack", model, coords, Z, SCHNET_R_MAX, 1e-6)


def test_xmace_second_state_virial_belongs_to_the_same_state():
    """The virial and the forces must both come from the state that was asked for.

    Everywhere else here uses state 0.  X-MACE returns an energy per state and
    a set of forces per state, and the wrapper picks one index out of each; if
    those two indices ever drifted apart the cluster-sum comparison would fail
    at once, because it would be checking state 1 forces against a state 0
    virial.  The two states are far apart, so it discriminates.
    """
    model = torch.jit.script(
        XMACE_TS_Wrapper(XMACE_MODEL, state_idx=1, device="cpu").eval()
    )
    coords, Z, _ = _xmace_system()
    _cluster_sum_check("X-MACE state 1", model, coords, Z, model.r_max, 1e-5)


def _translation_check(name, model, coords, Z, cell, tol):
    """Sliding the system through the box must not move the virial.

    This is the property that fails for a sum over absolute positions, and it
    is what makes a strain virial usable for pressure control: nothing physical
    depends on where the box origin sits.
    """
    _, f0, _, v0 = _call(model, coords, Z, cell)
    shift = torch.tensor([5.3, -2.7, 9.1], dtype=torch.float64)
    moved = torch.remainder(coords + shift, BOX)
    _, f1, _, v1 = _call(model, moved, Z, cell)

    print("\n%s virial before the shift:\n" % name + _fmt(v0))
    print("after wrapping by (5.3, -2.7, 9.1) A:\n" + _fmt(v1))
    print("  rel. change: %.3e" % _rel(v0, v1))
    naive0 = torch.einsum("ai,aj->ij", f0, coords)
    naive1 = torch.einsum("ai,aj->ij", f1, moved)
    print("  the same shift moves NAMD's sum f (x) r by %.3e relative"
          % _rel(naive0, naive1))
    assert _rel(v0, v1) < tol


def test_xmace_virial_is_translation_invariant():
    model = _load_xmace()
    coords, Z, cell = _xmace_system()
    _translation_check("X-MACE", model, coords, Z, cell, 1e-5)


def test_schnet_virial_is_translation_invariant():
    model = _load_schnet()
    coords, Z, cell = _schnet_system()
    _translation_check("SchNetPack", model, coords, Z, cell, 1e-6)


# -------------------------------------------------------------------
#  Shape, symmetry, batching, and the non-periodic zero
# -------------------------------------------------------------------

def _symmetry_and_shape(name, model, coords, Z, cell):
    _, _, _, virial = _call(model, coords, Z, cell)
    assert virial.shape == (3, 3)
    assert virial.dtype == torch.float64
    asym = float((virial - virial.transpose(-1, -2)).abs().max())
    print("\n%s asymmetry of the returned virial: %.3e" % (name, asym))
    assert asym == 0.0


def test_xmace_virial_is_symmetric():
    model = _load_xmace()
    coords, Z, cell = _xmace_system()
    _symmetry_and_shape("X-MACE", model, coords, Z, cell)


def test_schnet_virial_is_symmetric():
    model = _load_schnet()
    coords, Z, cell = _schnet_system()
    _symmetry_and_shape("SchNetPack", model, coords, Z, cell)


def _batched_check(name, model, coords, Z, cell, tol):
    """Two identical replicas must each reproduce the single-molecule virial.

    forward_batch builds one strain per molecule and looks the cells up per
    edge, so this is the cheapest thing that catches an indexing mistake there.
    A shared strain would show up as both replicas reporting twice the answer.
    """
    _, _, _, v_single = _call(model, coords, Z, cell)

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    cells = torch.stack([cell, cell]).detach().clone()

    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells,
    )
    assert virials.shape == (2, 3, 3)
    print("\n%s batched virial, replica 0:\n" % name + _fmt(virials[0]))
    print("single-molecule virial:\n" + _fmt(v_single))
    print("  rel. diff replica 0: %.3e" % _rel(virials[0], v_single))
    print("  rel. diff replica 1: %.3e" % _rel(virials[1], v_single))
    assert _rel(virials[0], v_single) < tol
    assert _rel(virials[1], v_single) < tol


def test_xmace_batched_virial_matches_single():
    model = _load_xmace()
    coords, Z, cell = _xmace_system()
    _batched_check("X-MACE", model, coords, Z, cell, 1e-4)


def test_schnet_batched_virial_matches_single():
    model = _load_schnet()
    coords, Z, cell = _schnet_system()
    _batched_check("SchNetPack", model, coords, Z, cell, 1e-8)


def _batched_mixed_cells_check(name, model, coords, Z, r_max, tol):
    """Two replicas in DIFFERENT boxes, each matching its own single result.

    Two identical cubes hide two separate mistakes.  A per-molecule cell lookup
    that always reaches for entry zero passes when the entries are the same, and
    a transposed lattice convention passes when the matrix is diagonal, because
    L*I is its own transpose.  The batched strain is written out by hand in the
    wrapper rather than going through the shared helper, so it needs its own
    look: one tilted box, one cube, and each virial checked against the
    single-molecule answer for the box it was given.
    """
    tilted = torch.tensor([
        [15.0, 2.5, 1.5],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0],
    ], dtype=torch.float64)
    cubic = torch.eye(3, dtype=torch.float64) * BOX

    for label, k in (("tilted", tilted), ("cubic", cubic)):
        imaged, total = _n_imaged(coords, k, r_max)
        print("\n%s, %s box: %d of %d edges cross a face"
              % (name, label, imaged, total))
        assert imaged > 0

    _, _, _, v_tilted = _call(model, coords, Z, tilted)
    _, _, _, v_cubic = _call(model, coords, Z, cubic)
    # If the two boxes gave the same virial there would be nothing to tell
    # apart and the comparison below would pass on any wiring.
    print("  the two boxes differ by %.3e relative" % _rel(v_tilted, v_cubic))
    assert _rel(v_tilted, v_cubic) > 0.05

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    cells = torch.stack([tilted, cubic]).detach().clone()

    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells,
    )
    print("  replica 0 against the tilted single result: %.3e"
          % _rel(virials[0], v_tilted))
    print("  replica 1 against the cubic single result : %.3e"
          % _rel(virials[1], v_cubic))
    print("  replica 0 against the CUBIC result        : %.3e"
          % _rel(virials[0], v_cubic))
    assert _rel(virials[0], v_tilted) < tol, (
        "replica 0 does not reproduce its own tilted box; it is %.3e from the "
        "cubic answer instead" % _rel(virials[0], v_cubic))
    assert _rel(virials[1], v_cubic) < tol, "replica 1 does not match its own box"


def test_xmace_batched_virial_with_different_cells():
    model = _load_xmace()
    coords, Z, _ = _xmace_system()
    _batched_mixed_cells_check("X-MACE", model, coords, Z, model.r_max, 1e-4)


def test_schnet_batched_virial_with_different_cells():
    model = _load_schnet()
    coords, Z, _ = _schnet_system()
    # Looser than the identical-cells test above.  Two unlike boxes give the
    # molecules different edge counts, so the batched reductions run in a
    # different order from the single-molecule ones, and in float32 that alone
    # is worth about 1e-6.
    _batched_mixed_cells_check("SchNetPack", model, coords, Z, SCHNET_R_MAX, 1e-5)


def _non_periodic_zero(name, model, coords, Z):
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
    print("\n%s non-periodic virial is zero in both entry points" % name)


def test_xmace_non_periodic_virial_is_zero():
    model = _load_xmace()
    coords, Z, _ = _xmace_system()
    _non_periodic_zero("X-MACE", model, coords, Z)


def test_schnet_non_periodic_virial_is_zero():
    model = _load_schnet()
    coords, Z, _ = _schnet_system()
    _non_periodic_zero("SchNetPack", model, coords, Z)


# -------------------------------------------------------------------
#  The cluster path must not have moved
# -------------------------------------------------------------------

def _load_variant(path, name):
    """Import an older copy of a wrapper without putting it in the package.

    The file uses relative imports, so those are rewritten to absolute ones and
    the source is executed into a throwaway module.  Nothing is written into
    src/, so both versions can be held side by side in one process.
    """
    with open(path) as fh:
        src = fh.read()
    src = src.replace("from ..constants import", "from src.constants import")
    src = src.replace("from ..edges import", "from src.edges import")
    src = src.replace("from ..export import", "from src.export import")
    src = src.replace("from ..virial import", "from src.virial import")
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def _head_copy(relpath, name):
    """The last committed version of a wrapper, loaded as a module.

    That copy predates the periodic work entirely: its forward takes four
    arguments and returns three values.  Comparing against it therefore answers
    the wider question, whether the whole conversion left a cluster calculation
    alone, not just whether the virial addition did.
    """
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:" + relpath],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return None, None
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src)
    tmp.close()
    # The file has to outlive scripting: TorchScript reads the class source
    # back off disk rather than from the module object.
    return _load_variant(tmp.name, name), tmp.name


def test_schnet_non_periodic_path_unchanged_vs_head():
    """SchNetPack is reproducible to the last bit, so demand exactly that."""
    head, path = _head_copy("src/wrappers/wrap_schnetpack.py", "wrap_schnet_head")
    if head is None:
        print("\nskipped: could not read the committed wrapper from git")
        return

    old = torch.jit.script(
        head.SchNetPack_Wrapper(
            model_path=SCHNET_MODEL, r_max=SCHNET_R_MAX, device="cpu",
        ).eval()
    )
    new = _load_schnet()
    coords, Z, _ = _schnet_system()

    e_old, f_old, q_old = old(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q)
    e_new, f_new, q_new, v_new = new(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q,
        torch.zeros((1, 3, 3), dtype=torch.float64))

    print("\nSchNetPack non-periodic, HEAD vs now:")
    print("  |dE| = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF| = %.3e" % float((f_old - f_new).abs().max()))
    print("  |dQ| = %.3e" % float((q_old - q_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)
    assert torch.equal(q_old, q_new)
    assert float(v_new.abs().max()) == 0.0

    os.unlink(path)


def test_xmace_non_periodic_path_unchanged_vs_head():
    """X-MACE forces are not reproducible run to run, so measure that first.

    Evaluating the same geometry twice with the same code moves the forces by
    around 1e-5 kcal/mol/A, which is float32 rounding somewhere inside the
    model rather than anything the wrapper did.  torch.equal on forces can
    therefore never pass here, for any version of the file.  What CAN be
    demanded is that the old and new wrappers differ by no more than the model
    differs from itself, and that the energy, which is reproducible, is
    identical bit for bit.
    """
    head, path = _head_copy("src/wrappers/wrap_xmace.py", "wrap_xmace_head")
    if head is None:
        print("\nskipped: could not read the committed wrapper from git")
        return

    old = torch.jit.script(
        head.XMACE_TS_Wrapper(XMACE_MODEL, state_idx=0, device="cpu").eval()
    )
    new = _load_xmace()
    coords, Z, _ = _xmace_system()
    zero_cell = torch.zeros((1, 3, 3), dtype=torch.float64)

    # Several repeats, because the scatter is random and one pair of runs can
    # land on the same answer by luck, which would understate it.
    runs = [old(coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q)
            for _ in range(4)]
    self_scatter = 0.0
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            self_scatter = max(
                self_scatter, float((runs[a][1] - runs[b][1]).abs().max()))

    e_old, f_old, q_old = runs[0]
    e_new, f_new, q_new, v_new = new(
        coords.detach().clone(), Z, EMPTY_PC, EMPTY_PC_Q, zero_cell.clone())

    cross = float((f_old - f_new).abs().max())
    print("\nX-MACE non-periodic, HEAD vs now:")
    print("  |dE|                        = %.3e" % float((e_old - e_new).abs().max()))
    print("  HEAD against itself, |dF|   = %.3e" % self_scatter)
    print("  HEAD against now,    |dF|   = %.3e" % cross)
    print("  |dQ|                        = %.3e" % float((q_old - q_new).abs().max()))

    for r in runs[1:]:
        assert torch.equal(e_old, r[0]), "energy is not reproducible either"
    assert torch.equal(e_old, e_new)
    assert torch.equal(q_old, q_new)
    if self_scatter == 0.0:
        # The model behaved itself this time, so ask for the strong version.
        assert torch.equal(f_old, f_new)
    else:
        assert cross <= 4.0 * self_scatter, (
            "forces moved by %.3e, more than the model's own run-to-run "
            "scatter of %.3e, so the cluster path did change"
            % (cross, self_scatter))

    assert float(v_new.abs().max()) == 0.0

    os.unlink(path)


# -------------------------------------------------------------------
#  Scripting
# -------------------------------------------------------------------

def test_torchscript_compiles():
    """Both entry points have to survive the extra return value."""
    for name, load, system in (
        ("X-MACE", _load_xmace, _xmace_system),
        ("SchNetPack", _load_schnet, _schnet_system),
    ):
        model = load()
        coords, Z, cell = system()
        out = _call(model, coords, Z, cell)
        assert len(out) == 4
        print("%s scripts and returns 4 values" % name)


if __name__ == "__main__":
    fns = [
        test_torchscript_compiles,
        test_xmace_has_no_native_virial,
        test_xmace_virial_matches_finite_difference,
        test_xmace_virial_matches_finite_difference_triclinic,
        test_schnet_virial_matches_finite_difference,
        test_xmace_virial_reduces_to_cluster_sum_in_a_large_box,
        test_schnet_virial_reduces_to_cluster_sum_in_a_large_box,
        test_xmace_second_state_virial_belongs_to_the_same_state,
        test_xmace_virial_is_translation_invariant,
        test_schnet_virial_is_translation_invariant,
        test_xmace_virial_is_symmetric,
        test_schnet_virial_is_symmetric,
        test_xmace_batched_virial_matches_single,
        test_schnet_batched_virial_matches_single,
        test_xmace_batched_virial_with_different_cells,
        test_schnet_batched_virial_with_different_cells,
        test_xmace_non_periodic_virial_is_zero,
        test_schnet_non_periodic_virial_is_zero,
        test_schnet_non_periodic_path_unchanged_vs_head,
        test_xmace_non_periodic_path_unchanged_vs_head,
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
