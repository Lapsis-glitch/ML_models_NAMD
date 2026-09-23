"""
Finite-difference checks on the periodic virial now returned by the NequIP and
TorchANI wrappers.

The two arrived at their virial by different routes and this file is where that
is checked rather than assumed.

  NequIP already computes a strain derivative on every forward pass and writes
  it to its output dict as ``virial``, with no flag to ask for or suppress it.
  The wrapper reads that key, so what is under test is whether NequIP's sign
  and unit are the ones NAMD wants.

  TorchANI has no such machinery, so the wrapper deforms the positions and the
  box itself and differentiates.  What is under test there is whether the
  deformation actually reaches TorchANI's periodic images, which it builds
  internally from the cell rather than from anything we hand it.

Either way the claim is the same: the returned virial is -dE/d(strain).  The
finite difference measures dE/d(strain) directly, so it should come out as
minus the returned matrix.  Both models run their internals in float32, so the
difference will not close to machine precision; the sign is settled instead by
the gap between the two candidate matches, which is three orders of magnitude.

The large-box test is the stronger evidence and it needs no differencing at
all.  Put every atom far enough from a face and no interaction crosses one, at
which point the strain acts on the positions alone and the virial must equal
the plain sum f (x) r.  That pins sign, transpose and scale in one comparison.

Both models load under /home/rat/miniconda3/envs/nequip_env/bin/python.  Run
the file directly for the full report, or under pytest for pass/fail.
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

from src.wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper  # noqa: E402
from src.wrappers.wrap_torchani import TorchANI_Wrapper  # noqa: E402

NEQUIP_MODEL = os.path.join(_REPO, "models", "compiled_nequip_oam_l.nequip.pth")
ANI_MODEL = os.path.join(_REPO, "models", "compiled_ani2x.pt")

NEQUIP_R_MAX = 6.0
# TorchANI keeps its own cutoff; this is only the fallback if the artifact does
# not expose it, and it is ANI-2x's radial cutoff.
ANI_R_MAX_FALLBACK = 5.2

# NequIP's minimum-image builder refuses a box narrower than twice the cutoff,
# and the finite difference compresses the box, so leave headroom.
BOX = 14.0

EMPTY_PC = torch.zeros((0, 3), dtype=torch.float64)
EMPTY_PC_Q = torch.zeros((0,), dtype=torch.float64)


# -------------------------------------------------------------------
#  Test system
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


def _load_nequip():
    return torch.jit.script(
        NequIP_Allegro_Wrapper(NEQUIP_MODEL, device="cpu", r_max=NEQUIP_R_MAX).eval()
    )


def _load_ani():
    return torch.jit.script(TorchANI_Wrapper(ANI_MODEL, device="cpu").eval())


def _ani_cutoff(model):
    """TorchANI's own cutoff, so the imaging check below is not guessing."""
    try:
        return float(model.inner.ani.cutoff)
    except Exception:                                   # pragma: no cover
        return ANI_R_MAX_FALLBACK


def _call(model, coords, Z, cell):
    """One evaluation on a fresh leaf copy of the coordinates.

    Both wrappers set requires_grad on tensors derived from what they are
    handed, so reusing a tensor between calls would carry autograd state from
    one evaluation into the next.  Detaching per call keeps them independent.
    """
    c = coords.detach().clone()
    k = cell.detach().clone().reshape(1, 3, 3)
    return _detached(model(c, Z, EMPTY_PC, EMPTY_PC_Q, k))


def _detached(out):
    """Strip the graph off a model's outputs.

    NequIP hands back tensors that are still attached to the graph it built,
    and reading one as a Python float warns about it.  Nothing here needs a
    gradient from a wrapper, so cut them loose at the door.
    """
    return tuple(t.detach() for t in out)


def _settled(model, coords, Z, cell, n_args=5, warmup=3):
    """Evaluate after a few warm-up calls, and return the last result.

    Two things move between calls otherwise.  TorchScript re-optimises a module
    after its first runs, and the optimised kernels do not always produce the
    same last bit as the unoptimised ones, which the warm-up settles.  And the
    deployed NequIP model reduces its force accumulation across threads in
    whatever order they finish, so two calls on one instance with one input
    give forces differing in the last float32 bits, about 6e-05 on forces of
    91.  Its energies are unaffected.  Neither has anything to do with the
    virial, but together they make an exact old-versus-new comparison
    impossible unless the reduction is pinned, and one thread pins it.  The
    count is put back afterwards so importing this file does not quietly
    single-thread everything else in a pytest session.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = None
        for _ in range(warmup + 1):
            c = coords.detach().clone()
            if n_args == 4:
                out = model(c, Z, EMPTY_PC, EMPTY_PC_Q)
            else:
                k = cell.detach().clone().reshape(1, 3, 3)
                out = model(c, Z, EMPTY_PC, EMPTY_PC_Q, k)
    finally:
        torch.set_num_threads(prev_threads)
    return _detached(out)


# -------------------------------------------------------------------
#  Imaging check, done here rather than through src/edges
# -------------------------------------------------------------------

def _imaged_edges(coords, cell, cutoff):
    """Count directed pairs inside *cutoff* that are only neighbours through a
    face.

    Written out here instead of calling build_edges_pbc because TorchANI does
    not use that builder at all, and because the builder refuses a box narrower
    than twice the cutoff, which would stop this from being usable as a general
    check.
    """
    inv = torch.linalg.inv(cell)
    d = coords.unsqueeze(0) - coords.unsqueeze(1)
    n = torch.round(d @ inv)
    dmin = d - n @ cell
    r = dmin.norm(dim=2)
    near = (r < cutoff) & (r > 0.0)
    return int(((n.abs().sum(dim=2) > 0) & near).sum())


# -------------------------------------------------------------------
#  Finite difference
# -------------------------------------------------------------------

def _strain_direction(i, j):
    """Symmetric direction P such that contracting dE/dD with it gives dE/dD_ij.

    dE/dh for D = h*P is sum_ab (dE/dD_ab) P_ab.  Half on each off-diagonal
    entry picks out the ij component once dE/dD is symmetric, which it is,
    because every route here symmetrises the strain before using it.
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
            # Positions and box are strained together, and each model then
            # rebuilds its own neighbours from the strained box, which is what
            # makes this the same deformation the wrappers differentiate.
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


def _fd_report(label, model, coords, Z, cell, steps, tol):
    """Run the finite difference at several step sizes and check the sign.

    Reports the match against both candidate signs.  A correct virial gives a
    small number against -virial and a number near 2 against +virial, since the
    two differ by twice the matrix; asserting on both is what pins the sign,
    and it does not depend on choosing a tolerance that float32 noise happens
    to fit inside.
    """
    _, f, _, virial = _call(model, coords, Z, cell)
    print("\n%s returned virial (kcal/mol):\n%s" % (label, _fmt(virial)))

    best = None
    for h in steps:
        fd = _fd_dE_dstrain(model, coords, Z, cell, h)
        err_neg = _rel(fd, -virial)
        err_pos = _rel(fd, virial)
        print("  h = %-9g rel vs -virial: %.3e   rel vs +virial: %.3e"
              % (h, err_neg, err_pos))
        # The sign has to hold at every step size, not only the flattering one.
        assert err_neg < 0.25 * err_pos, (
            "%s: at h = %g the finite difference is no closer to -virial "
            "(%.3e) than to +virial (%.3e)" % (label, h, err_neg, err_pos))
        if best is None or err_neg < best[1]:
            best = (fd, err_neg, err_pos, h)

    fd, err_neg, err_pos, h = best
    print("  best at h = %g:\n%s" % (h, _fmt(fd)))
    print("  dE/dD should equal -virial; measured %.3e away from it and "
          "%.3e away from +virial." % (err_neg, err_pos))

    assert err_neg < tol, (
        "%s: returned virial is not -dE/d(strain): rel. err %.3e against "
        "-virial, %.3e against +virial" % (label, err_neg, err_pos))
    assert err_pos > 1.0, (
        "%s: the two candidate signs are not distinguishable, so this "
        "measurement settles nothing" % label)
    return virial, f


# -------------------------------------------------------------------
#  NequIP
# -------------------------------------------------------------------

def test_nequip_virial_matches_finite_difference():
    model = _load_nequip()
    coords, Z, cell = _water_box()

    n_imaged = _imaged_edges(coords, cell, NEQUIP_R_MAX)
    print("\nEdges crossing a face: %d" % n_imaged)
    assert n_imaged > 0, "no pair interacts through a face; nothing to test"

    virial, f = _fd_report("NequIP", model, coords, Z, cell,
                           steps=(1e-3, 3e-4), tol=2e-3)

    # What NAMD would have summed on its own, for scale.  The gap is the reason
    # for the whole change: pairs that interact through a face contribute their
    # raw separation here instead of the imaged one.
    naive = torch.einsum("ai,aj->ij", f, coords)
    naive = 0.5 * (naive + naive.transpose(-1, -2))
    print("\nNAMD's own sum f (x) r on absolute positions:\n" + _fmt(naive))
    print("  rel. difference from the strain virial: %.3e" % _rel(naive, virial))


def test_nequip_virial_matches_finite_difference_triclinic():
    """The same check on a skewed box.

    A cubic box cannot catch a transposed cell convention, because L*I times
    (I + D) is the same matrix either way round.  Skewing it makes the two
    orderings differ, so this is what pins down that the lattice vectors travel
    as rows from the edge builder into NequIP.
    """
    model = _load_nequip()
    coords, Z, _ = _water_box()
    cell = torch.tensor([
        [15.0, 2.5, 1.5],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0],
    ], dtype=torch.float64)

    n_imaged = _imaged_edges(coords, cell, NEQUIP_R_MAX)
    print("\nEdges crossing a face in the tilted box: %d" % n_imaged)
    assert n_imaged > 0, "nothing is imaged here, so this proves nothing"

    _fd_report("NequIP (triclinic)", model, coords, Z, cell,
               steps=(1e-3, 3e-4), tol=3e-3)


def test_nequip_virial_reduces_to_cluster_sum_in_a_large_box():
    """With nothing crossing a face the virial must be the plain sum f (x) r.

    No finite differencing, so no float32 noise to argue with: this pins the
    sign, the transpose and the scale in one comparison.
    """
    model = _load_nequip()
    coords, Z, _ = _water_box()

    big = 40.0
    cell = torch.eye(3, dtype=torch.float64) * big
    n_imaged = _imaged_edges(coords, cell, NEQUIP_R_MAX)
    print("\nEdges crossing a face in the %.0f A box: %d" % (big, n_imaged))
    assert n_imaged == 0, "box is not large enough for this comparison to hold"

    _, f, _, virial = _call(model, coords, Z, cell)

    w_namd = torch.einsum("ai,aj->ij", f, coords)
    w_namd = 0.5 * (w_namd + w_namd.transpose(-1, -2))
    print("Returned virial:\n" + _fmt(virial))
    print("sum f (x) r (the NAMD form):\n" + _fmt(w_namd))
    print("rel. diff: %.3e" % _rel(virial, w_namd))
    assert _rel(virial, w_namd) < 1e-5


def test_nequip_virial_is_symmetric_and_batched_matches_single():
    model = _load_nequip()
    coords, Z, cell = _water_box()

    _, _, _, v_single = _call(model, coords, Z, cell)
    asym = float((v_single - v_single.transpose(-1, -2)).abs().max())
    print("\nAsymmetry of returned virial: %.3e" % asym)
    assert asym == 0.0

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    cells = torch.stack([cell, cell]).detach().clone()

    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells)

    assert virials.shape == (2, 3, 3)
    print("Batched replica 0:\n" + _fmt(virials[0]))
    print("rel. diff replica 0 vs single : %.3e" % _rel(virials[0], v_single))
    print("rel. diff replica 1 vs single : %.3e" % _rel(virials[1], v_single))
    # Not exact, and it cannot be: the batch is one graph of thirty-six atoms
    # where the single call is one of eighteen, so float32 sums land in a
    # different order.  What this catches is an indexing mistake in the
    # per-molecule split, which would show up as a difference of order one
    # rather than of order the last bit.
    assert _rel(virials[0], v_single) < 1e-4
    assert _rel(virials[1], v_single) < 1e-4


def test_nequip_non_periodic_virial_is_zero():
    model = _load_nequip()
    coords, Z, _ = _water_box()
    zero_cell = torch.zeros((3, 3), dtype=torch.float64)
    _, _, _, virial = _call(model, coords, Z, zero_cell)
    assert virial.shape == (3, 3)
    assert virial.dtype == torch.float64
    assert float(virial.abs().max()) == 0.0

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q,
        torch.zeros((2, 3, 3), dtype=torch.float64))
    assert virials.shape == (2, 3, 3)
    assert float(virials.abs().max()) == 0.0


# -------------------------------------------------------------------
#  TorchANI
# -------------------------------------------------------------------

def test_ani_virial_matches_finite_difference():
    model = _load_ani()
    coords, Z, cell = _water_box()
    cutoff = _ani_cutoff(model)

    n_imaged = _imaged_edges(coords, cell, cutoff)
    print("\nTorchANI cutoff %.2f A, pairs crossing a face: %d"
          % (cutoff, n_imaged))
    assert n_imaged > 0, "no pair interacts through a face; nothing to test"

    # Larger steps than NequIP needs.  ANI-2x runs in float32 and its total
    # energy is dominated by per-atom reference terms that the strain does not
    # move, so the energy difference has to be pushed well clear of the last
    # float32 bit before it says anything.
    virial, f = _fd_report("TorchANI", model, coords, Z, cell,
                           steps=(1e-2, 3e-3), tol=1e-2)

    naive = torch.einsum("ai,aj->ij", f, coords)
    naive = 0.5 * (naive + naive.transpose(-1, -2))
    print("\nNAMD's own sum f (x) r on absolute positions:\n" + _fmt(naive))
    print("  rel. difference from the strain virial: %.3e" % _rel(naive, virial))


def test_ani_virial_matches_finite_difference_triclinic():
    model = _load_ani()
    coords, Z, _ = _water_box()
    cell = torch.tensor([
        [15.0, 2.5, 1.5],
        [0.0, 15.0, 0.0],
        [0.0, 0.0, 15.0],
    ], dtype=torch.float64)
    cutoff = _ani_cutoff(model)

    n_imaged = _imaged_edges(coords, cell, cutoff)
    print("\nPairs crossing a face in the tilted box: %d" % n_imaged)
    assert n_imaged > 0, "nothing is imaged here, so this proves nothing"

    # The tilted box gives a virial roughly thirty times smaller than the cubic
    # one while the float32 noise on the energy stays where it was, so the
    # steps here are larger still.
    _fd_report("TorchANI (triclinic)", model, coords, Z, cell,
               steps=(3e-2, 1e-2, 3e-3), tol=2e-2)


def test_ani_virial_reduces_to_cluster_sum_in_a_large_box():
    """The noise-free half of the TorchANI evidence.

    TorchANI wraps coordinates into the box before doing anything else, so this
    also confirms the wrapping does not move an atom here: every coordinate is
    already inside the larger box, and a wrap would show up as a virial that no
    longer matches the sum taken over the positions we passed in.
    """
    model = _load_ani()
    coords, Z, _ = _water_box()
    cutoff = _ani_cutoff(model)

    big = 40.0
    cell = torch.eye(3, dtype=torch.float64) * big
    n_imaged = _imaged_edges(coords, cell, cutoff)
    print("\nPairs crossing a face in the %.0f A box: %d" % (big, n_imaged))
    assert n_imaged == 0, "box is not large enough for this comparison to hold"

    _, f, _, virial = _call(model, coords, Z, cell)

    w_namd = torch.einsum("ai,aj->ij", f, coords)
    w_namd = 0.5 * (w_namd + w_namd.transpose(-1, -2))
    r_cross_f = torch.einsum("ai,aj->ij", coords, f)
    r_cross_f = 0.5 * (r_cross_f + r_cross_f.transpose(-1, -2))

    print("Returned virial:\n" + _fmt(virial))
    print("sum f (x) r (the NAMD form):\n" + _fmt(w_namd))
    print("rel. diff virial vs sum f (x) r : %.3e" % _rel(virial, w_namd))
    print("rel. diff virial vs sum r (x) f : %.3e" % _rel(virial, r_cross_f))
    assert _rel(virial, w_namd) < 1e-4


def test_ani_virial_is_symmetric_and_batched_matches_single():
    model = _load_ani()
    coords, Z, cell = _water_box()

    _, _, _, v_single = _call(model, coords, Z, cell)
    asym = float((v_single - v_single.transpose(-1, -2)).abs().max())
    print("\nAsymmetry of returned virial: %.3e" % asym)
    assert asym == 0.0

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    cells = torch.stack([cell, cell]).detach().clone()

    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells)

    assert virials.shape == (2, 3, 3)
    print("Batched replica 0:\n" + _fmt(virials[0]))
    print("rel. diff replica 0 vs single : %.3e" % _rel(virials[0], v_single))
    print("rel. diff replica 1 vs single : %.3e" % _rel(virials[1], v_single))
    assert _rel(virials[0], v_single) < 1e-10
    assert _rel(virials[1], v_single) < 1e-10


def test_ani_empty_walker_still_gets_a_virial_slot():
    """A batch with an empty molecule must still report one virial per walker.

    The periodic batch loop skips the model for an empty walker, and if it
    skipped the virial as well the stack would come back short and every walker
    after it would be handed its neighbour's box.
    """
    model = _load_ani()
    coords, Z, cell = _water_box()
    n = coords.size(0)

    # Walker 0 is empty, walkers 1 and 2 hold a copy of the system each.
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, 0, n, 2 * n], dtype=torch.int64)
    cells = torch.stack([cell, cell, cell]).detach().clone()

    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, cells)

    print("\nvirials shape with an empty leading walker: %s"
          % (tuple(virials.shape),))
    assert virials.shape == (3, 3, 3)
    assert float(virials[0].abs().max()) == 0.0

    _, _, _, v_single = _call(model, coords, Z, cell)
    print("rel. diff walker 1 vs single : %.3e" % _rel(virials[1], v_single))
    assert _rel(virials[1], v_single) < 1e-10


def test_ani_non_periodic_virial_is_zero():
    model = _load_ani()
    coords, Z, _ = _water_box()
    zero_cell = torch.zeros((3, 3), dtype=torch.float64)
    _, _, _, virial = _call(model, coords, Z, zero_cell)
    assert virial.shape == (3, 3)
    assert virial.dtype == torch.float64
    assert float(virial.abs().max()) == 0.0

    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    _, _, _, virials = model.forward_batch(
        coords2, Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q,
        torch.zeros((2, 3, 3), dtype=torch.float64))
    assert virials.shape == (2, 3, 3)
    assert float(virials.abs().max()) == 0.0


def _translation_invariance(label, model, tol):
    """Slide the whole system out of the box and check the virial does not move.

    Nothing physical depends on where the box origin sits, and this is exactly
    the property that fails for a sum over absolute positions, so it is the
    property that makes a strain virial worth having in the first place.

    The shift is applied without wrapping, on purpose.  NAMD is free to send
    coordinates that lie outside the box, and both models deal with that
    themselves: TorchANI folds them back into the central cell before doing
    anything, and the shared periodic edge builder rounds pairwise
    displacements rather than positions.  Neither of those paths is touched by
    a test whose atoms all start inside the box, and if the folding step
    dropped the cell out of the graph the virial would go wrong for
    out-of-box atoms alone.
    """
    coords, Z, cell = _water_box()
    _, f0, _, v0 = _call(model, coords, Z, cell)

    shift = torch.tensor([5.3, -2.7, 9.1], dtype=torch.float64)
    moved = coords + shift
    outside = int(((moved < 0.0) | (moved >= BOX)).any(dim=1).sum())
    print("\n%s: atoms now sitting outside the box: %d of %d"
          % (label, outside, coords.size(0)))
    assert outside > 0, "the shift left everything inside; nothing new is tested"

    _, f1, _, v1 = _call(model, moved, Z, cell)

    print("Virial before:\n" + _fmt(v0))
    print("Virial after the unwrapped shift:\n" + _fmt(v1))
    print("  rel. change: %.3e" % _rel(v0, v1))
    assert _rel(v0, v1) < tol, (
        "%s: the virial changed by %.3e when the system was slid out of the "
        "box" % (label, _rel(v0, v1)))

    # The same shift folded back in.  Every atom now sits in a different image
    # from where it started, which is what moves NAMD's own sum: a rigid
    # translation on its own barely touches that sum, because it shifts it by
    # the net force times the displacement and the net force is near zero.
    wrapped = torch.remainder(moved, BOX)
    _, f2, _, v2 = _call(model, wrapped, Z, cell)
    print("Virial after wrapping the shifted system back in:\n" + _fmt(v2))
    print("  rel. change: %.3e" % _rel(v0, v2))

    naive0 = torch.einsum("ai,aj->ij", f0, coords)
    naive2 = torch.einsum("ai,aj->ij", f2, wrapped)
    print("  the same wrap moves NAMD's sum f (x) r by %.3e relative"
          % _rel(naive0, naive2))

    assert _rel(v0, v2) < tol, (
        "%s: the virial changed by %.3e when the system was wrapped through "
        "the box" % (label, _rel(v0, v2)))


def test_nequip_virial_is_translation_invariant():
    _translation_invariance("NequIP", _load_nequip(), tol=1e-4)


def test_ani_virial_is_translation_invariant():
    _translation_invariance("TorchANI", _load_ani(), tol=1e-4)


# -------------------------------------------------------------------
#  Regressions against the versions that predate the virial
# -------------------------------------------------------------------

def _load_variant(path, name):
    """Import an older copy of a wrapper without putting it in the package.

    The file uses relative imports, so those are rewritten to absolute ones and
    the source is executed into a throwaway module.  Nothing is written into
    src/, so two versions can be held side by side in one process.
    """
    with open(path) as fh:
        src = fh.read()
    for mod in ("constants", "edges", "export", "virial"):
        src = src.replace("from ..%s import" % mod, "from src.%s import" % mod)
    m = types.ModuleType(name)
    m.__file__ = path
    exec(compile(src, path, "exec"), m.__dict__)
    return m


def _variant_path(env_key, default_name):
    scratch = os.environ.get("MLFF_VIRIAL_SCRATCH", "")
    if scratch:
        p = os.path.join(scratch, default_name)
        if os.path.exists(p):
            return p
    return os.environ.get(env_key, "")


def _head_copy(rel_path):
    """The committed version of a wrapper, written to a file that outlives the
    call because TorchScript reads class source back off disk."""
    try:
        src = subprocess.check_output(
            ["git", "show", "HEAD:" + rel_path],
            cwd=_REPO, text=True, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError):
        return ""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src)
    tmp.close()
    return tmp.name


def test_nequip_non_periodic_path_unchanged_vs_pre_virial():
    """The cluster path must be untouched by the virial work, bit for bit."""
    path = _variant_path("MLFF_PRE_VIRIAL_NEQUIP", "wrap_nequip_pre_virial.py")
    if not path or not os.path.exists(path):
        print("\nskipped: no pre-virial NequIP snapshot available")
        return

    old = _load_variant(path, "wrap_nequip_pre_virial")
    old_model = torch.jit.script(old.NequIP_Allegro_Wrapper(
        NEQUIP_MODEL, device="cpu", r_max=NEQUIP_R_MAX).eval())
    new_model = _load_nequip()

    coords, Z, cell = _water_box()
    zero_cell = torch.zeros((3, 3), dtype=torch.float64)

    e_old, f_old, q_old = _settled(old_model, coords, Z, zero_cell)
    e_new, f_new, q_new, v_new = _settled(new_model, coords, Z, zero_cell)

    print("\nNequIP non-periodic, pre-virial vs now:")
    print("  |dE| = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF| = %.3e" % float((f_old - f_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)
    assert torch.equal(q_old, q_new)
    assert float(v_new.abs().max()) == 0.0

    # The periodic leg matters too: reading an extra key out of the output dict
    # must not have changed what went into the model.
    e_old, f_old, _ = _settled(old_model, coords, Z, cell)
    e_new, f_new, _, _ = _settled(new_model, coords, Z, cell)
    print("NequIP periodic, pre-virial vs now:")
    print("  |dE| = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF| = %.3e" % float((f_old - f_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)


def test_ani_non_periodic_path_unchanged_vs_pre_virial():
    path = _variant_path("MLFF_PRE_VIRIAL_ANI", "wrap_ani_pre_virial.py")
    if not path or not os.path.exists(path):
        print("\nskipped: no pre-virial TorchANI snapshot available")
        return

    old = _load_variant(path, "wrap_ani_pre_virial")
    old_model = torch.jit.script(old.TorchANI_Wrapper(ANI_MODEL, device="cpu").eval())
    new_model = _load_ani()

    coords, Z, cell = _water_box()
    zero_cell = torch.zeros((3, 3), dtype=torch.float64)

    e_old, f_old, q_old = _settled(old_model, coords, Z, zero_cell)
    e_new, f_new, q_new, v_new = _settled(new_model, coords, Z, zero_cell)

    print("\nTorchANI non-periodic, pre-virial vs now:")
    print("  |dE| = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF| = %.3e" % float((f_old - f_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)
    assert torch.equal(q_old, q_new)
    assert float(v_new.abs().max()) == 0.0

    # The periodic leg is worth checking as well: the strain is zero, so the
    # geometry TorchANI sees must be the geometry it saw before, and an energy
    # that matches to the last bit is the evidence that adding it changed
    # nothing about the calculation itself.
    e_old, f_old, _ = _settled(old_model, coords, Z, cell)
    e_new, f_new, _, _ = _settled(new_model, coords, Z, cell)
    print("TorchANI periodic, pre-virial vs now:")
    print("  |dE| = %.3e" % float((e_old - e_new).abs().max()))
    print("  |dF| = %.3e" % float((f_old - f_new).abs().max()))
    assert torch.equal(e_old, e_new)
    assert torch.equal(f_old, f_new)

    # And the non-periodic batch, which takes a different route through the
    # padded layout and so could drift on its own.
    n = coords.size(0)
    coords2 = torch.cat([coords, coords], dim=0).detach().clone()
    Z2 = torch.cat([Z, Z], dim=0)
    batch = torch.cat([torch.zeros(n, dtype=torch.int64),
                       torch.ones(n, dtype=torch.int64)])
    ptr = torch.tensor([0, n, 2 * n], dtype=torch.int64)
    zero_cells = torch.zeros((2, 3, 3), dtype=torch.float64)
    eb_old, fb_old, _ = _detached(old_model.forward_batch(
        coords2.clone(), Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, zero_cells.clone()))
    eb_new, fb_new, _, _ = _detached(new_model.forward_batch(
        coords2.clone(), Z2, batch, ptr, EMPTY_PC, EMPTY_PC_Q, zero_cells.clone()))
    print("TorchANI non-periodic batch, pre-virial vs now:")
    print("  |dE| = %.3e" % float((eb_old - eb_new).abs().max()))
    print("  |dF| = %.3e" % float((fb_old - fb_new).abs().max()))
    assert torch.equal(eb_old, eb_new)
    assert torch.equal(fb_old, fb_new)


def test_non_periodic_path_unchanged_vs_head():
    """Same comparison against the last committed versions, which predate PBC.

    Those copies have no cell argument at all, so this answers a wider question
    than the snapshots above: whether the whole periodic conversion, virial
    included, left a cluster calculation alone.
    """
    for rel, cls, load_new, kwargs in (
        ("src/wrappers/wrap_compiled_nequip.py", "NequIP_Allegro_Wrapper",
         _load_nequip, {"deployed_path": NEQUIP_MODEL, "device": "cpu",
                        "r_max": NEQUIP_R_MAX}),
        ("src/wrappers/wrap_torchani.py", "TorchANI_Wrapper",
         _load_ani, {"model_path": ANI_MODEL, "device": "cpu"}),
    ):
        path = _head_copy(rel)
        if not path:
            print("\nskipped: could not read %s from git" % rel)
            continue

        head = _load_variant(path, "head_" + cls)
        head_model = torch.jit.script(getattr(head, cls)(**kwargs).eval())
        new_model = load_new()

        coords, Z, _ = _water_box()
        zero_cell = torch.zeros((3, 3), dtype=torch.float64)
        e_old, f_old, q_old = _settled(head_model, coords, Z, zero_cell,
                                       n_args=4)
        e_new, f_new, q_new, v_new = _settled(new_model, coords, Z, zero_cell)

        print("\n%s non-periodic, HEAD vs now:" % cls)
        print("  |dE| = %.3e" % float((e_old - e_new).abs().max()))
        print("  |dF| = %.3e" % float((f_old - f_new).abs().max()))
        assert torch.equal(e_old, e_new), cls
        assert torch.equal(f_old, f_new), cls
        assert torch.equal(q_old, q_new), cls
        assert float(v_new.abs().max()) == 0.0

        os.unlink(path)


def test_torchscript_compiles():
    """Scripting has to survive the extra return value in both entry points."""
    coords, Z, cell = _water_box()
    for load in (_load_nequip, _load_ani):
        m = load()
        assert len(_call(m, coords, Z, cell)) == 4


if __name__ == "__main__":
    fns = [
        test_torchscript_compiles,
        test_nequip_virial_matches_finite_difference,
        test_nequip_virial_matches_finite_difference_triclinic,
        test_nequip_virial_reduces_to_cluster_sum_in_a_large_box,
        test_nequip_virial_is_symmetric_and_batched_matches_single,
        test_nequip_virial_is_translation_invariant,
        test_nequip_non_periodic_virial_is_zero,
        test_ani_virial_matches_finite_difference,
        test_ani_virial_matches_finite_difference_triclinic,
        test_ani_virial_reduces_to_cluster_sum_in_a_large_box,
        test_ani_virial_is_symmetric_and_batched_matches_single,
        test_ani_empty_walker_still_gets_a_virial_slot,
        test_ani_virial_is_translation_invariant,
        test_ani_non_periodic_virial_is_zero,
        test_nequip_non_periodic_path_unchanged_vs_pre_virial,
        test_ani_non_periodic_path_unchanged_vs_pre_virial,
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
