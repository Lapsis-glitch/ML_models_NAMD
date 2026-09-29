"""
Real-model checks for the SevenNet wrapper, against SevenNet itself.

The reference is SevenNet's own ASE calculator on the same checkpoint.  It
builds its own graph (matscipy neighbour list, its own type map) and takes
forces and stress from its own autograd, so agreeing with it pins the type map,
the edge convention, the image shifts and the virial sign/scale in one go.  The
non-periodic molecule mixes H C N O S Cl so a wrong type index would show.

On top of that, the usual contract checks: the virial against a strain finite
difference in a triclinic box, the large-box limit against sum r (x) f,
forward_batch against separate forward calls, and the scripted file against
the eager wrapper.

Needs ``sevenn`` (allegro env) and ``models/compiled_sevennet_0.pt``::

    python -m src.compile_sevennet --checkpoint 7net-0 --out models/compiled_sevennet_0.pt

SevenNet runs in float32, so tolerances are relative to the largest force.
"""

import os
import sys
import tempfile

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.constants import EV_TO_KCAL  # noqa: E402

MODEL = os.path.join(_REPO, "models", "compiled_sevennet_0.pt")
CHECKPOINT = "7net-0"

pytest.importorskip("sevenn")
if not os.path.exists(MODEL):
    pytest.skip(f"{MODEL} missing (python -m src.compile_sevennet)",
                allow_module_level=True)

from src.wrappers.wrap_sevennet import SevenNet_Wrapper  # noqa: E402

EMPTY_PC = torch.zeros((0, 3), dtype=torch.float64)
EMPTY_PC_Q = torch.zeros((0,), dtype=torch.float64)
ZERO_CELL = torch.zeros((1, 3, 3), dtype=torch.float64)

# Widths stay above 2 x 5 A cutoff with room for the finite-difference strain.
TRICLINIC = torch.tensor([[13.0, 0.0, 0.0],
                          [2.0, 12.5, 0.0],
                          [1.0, 1.5, 13.5]], dtype=torch.float64)


# -------------------------------------------------------------------
#  Systems
# -------------------------------------------------------------------

def _molecule():
    """CH3Cl, NH3 and H2S within cutoff of each other; no box."""
    coords = torch.tensor([
        [0.00, 0.00, 0.00], [1.78, 0.00, 0.00],                 # C Cl
        [-0.36, 1.03, 0.00], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
        [-1.2, 0.3, 3.2], [-0.4, 0.8, 3.6], [-1.2, -0.6, 3.6], [-2.0, 0.8, 3.5],  # N H H H
        [2.5, 2.6, 1.0], [3.8, 2.6, 1.2], [2.3, 3.9, 1.3],      # S H H
    ], dtype=torch.float64)
    Z = torch.tensor([6, 17, 1, 1, 1, 7, 1, 1, 1, 16, 1, 1])
    return coords, Z


def _water_box(seed=0):
    """Eight randomly oriented waters on a 2x2x2 fractional grid of TRICLINIC."""
    g = torch.Generator().manual_seed(seed)
    mono = torch.tensor([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
                        dtype=torch.float64)
    coords = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                frac = (torch.tensor([i, j, k], dtype=torch.float64) + 0.25
                        + 0.3 * torch.rand(3, generator=g, dtype=torch.float64)) / 2
                q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
                coords.append(frac @ TRICLINIC + mono @ q.T)
    return torch.cat(coords), torch.tensor([8, 1, 1] * 8)


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------

@pytest.fixture(scope="module")
def wrapper():
    return SevenNet_Wrapper(MODEL).eval()


def _run(w, coords, Z, cell):
    c = coords.clone().requires_grad_(True)
    e, f, q, v = w(c, Z, EMPTY_PC, EMPTY_PC_Q, cell.reshape(1, 3, 3))
    return e.detach(), f.detach(), v.detach()


def _reference(coords, Z, cell):
    """SevenNet's ASE calculator: (energy kcal/mol, forces, virial kcal/mol)."""
    from ase import Atoms
    from sevenn.calculator import SevenNetCalculator

    periodic = bool(cell.abs().sum() > 0)
    atoms = Atoms(numbers=Z.numpy(), positions=coords.numpy(),
                  cell=cell.numpy() if periodic else None, pbc=periodic)
    atoms.calc = SevenNetCalculator(model=CHECKPOINT, device="cpu")
    e = atoms.get_potential_energy() * EV_TO_KCAL
    f = torch.tensor(atoms.get_forces()) * EV_TO_KCAL
    v = torch.zeros(3, 3, dtype=torch.float64)
    if periodic:
        # ASE stress = (1/V) dE/d(strain) = -virial / V.
        v = -torch.tensor(atoms.get_stress(voigt=False)) * atoms.get_volume() * EV_TO_KCAL
    return e, f, v


def _close(a, b, scale, rtol):
    return (a - b).abs().max().item() <= rtol * max(scale, 1.0)


# -------------------------------------------------------------------
#  Against SevenNet's own calculator
# -------------------------------------------------------------------

def test_matches_ase_calculator_molecule(wrapper):
    coords, Z = _molecule()
    e, f, v = _run(wrapper, coords, Z, ZERO_CELL)
    e_ref, f_ref, _ = _reference(coords, Z, torch.zeros(3, 3))
    fmax = f_ref.abs().max().item()
    assert abs(e.item() - e_ref) < 1e-5 * abs(e_ref) + 1e-3
    assert _close(f, f_ref, fmax, 1e-4)
    assert v.abs().max().item() == 0.0


def test_matches_ase_calculator_periodic(wrapper):
    coords, Z = _water_box()
    e, f, v = _run(wrapper, coords, Z, TRICLINIC)
    e_ref, f_ref, v_ref = _reference(coords, Z, TRICLINIC)
    assert abs(e.item() - e_ref) < 1e-5 * abs(e_ref) + 1e-3
    assert _close(f, f_ref, f_ref.abs().max().item(), 1e-4)
    assert _close(v, v_ref, v_ref.abs().max().item(), 1e-4)


# -------------------------------------------------------------------
#  Contract checks
# -------------------------------------------------------------------

def test_virial_matches_strain_finite_difference(wrapper):
    coords, Z = _water_box(seed=1)
    _, _, v = _run(wrapper, coords, Z, TRICLINIC)
    h = 1e-3
    fd = torch.zeros(3, 3, dtype=torch.float64)
    for a in range(3):
        for b in range(3):
            eps = torch.zeros(3, 3, dtype=torch.float64)
            eps[a, b] += 0.5 * h
            eps[b, a] += 0.5 * h
            ep = _run(wrapper, coords @ (torch.eye(3) + eps), Z,
                      TRICLINIC @ (torch.eye(3) + eps))[0]
            em = _run(wrapper, coords @ (torch.eye(3) - eps), Z,
                      TRICLINIC @ (torch.eye(3) - eps))[0]
            fd[a, b] = -(ep - em).item() / (2 * h)
    # float32 energies of ~1e5 kcal/mol limit how well a difference can close.
    assert _close(v, fd, v.abs().max().item(), 2e-2), (v, fd)


def test_large_box_virial_is_sum_r_outer_f(wrapper):
    coords, Z = _molecule()
    box = torch.eye(3, dtype=torch.float64) * 40.0
    coords = coords + 20.0
    e0, f0, _ = _run(wrapper, coords, Z, ZERO_CELL)
    e, f, v = _run(wrapper, coords, Z, box)
    assert torch.allclose(e, e0, rtol=1e-6)
    expected = coords.T @ f
    expected = 0.5 * (expected + expected.T)
    assert _close(v, expected, v.abs().max().item(), 1e-5)


@pytest.mark.parametrize("periodic", [False, True])
def test_forward_batch_matches_forward(wrapper, periodic):
    ca, Za = _water_box(seed=2)
    cb, Zb = _water_box(seed=3)
    cell_a = TRICLINIC
    cell_b = TRICLINIC * 1.05
    if not periodic:
        cell_a = cell_b = torch.zeros(3, 3, dtype=torch.float64)
    ea, fa, va = _run(wrapper, ca, Za, cell_a)
    eb, fb, vb = _run(wrapper, cb, Zb, cell_b)

    coords = torch.cat([ca, cb]).requires_grad_(True)
    Z = torch.cat([Za, Zb])
    n = ca.size(0)
    batch = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    ptr = torch.tensor([0, n, 2 * n])
    E, F, Q, V = wrapper.forward_batch(coords, Z, batch, ptr, EMPTY_PC, EMPTY_PC_Q,
                                       torch.stack([cell_a, cell_b]))
    fmax = max(fa.abs().max().item(), fb.abs().max().item())
    assert torch.allclose(E, torch.stack([ea, eb]), rtol=1e-6)
    assert _close(F.detach(), torch.cat([fa, fb]), fmax, 1e-5)
    assert _close(V.detach(), torch.stack([va, vb]), va.abs().max().item(), 1e-5)


def test_scripted_matches_eager(wrapper):
    from src.export import export_wrapped

    path = os.path.join(tempfile.mkdtemp(), "sevennet_mlff.pt")
    export_wrapped(SevenNet_Wrapper(MODEL).eval(), path, model_type="SevenNet")
    scripted = torch.jit.load(path)
    coords, Z = _water_box(seed=4)
    # Not bit-equal: the inner model's profiling executor fuses kernels after
    # the first calls, which moves float32 rounding by ~1e-6 relative.
    for cell in (ZERO_CELL[0], TRICLINIC):
        e0, f0, v0 = _run(wrapper, coords, Z, cell)
        e1, f1, v1 = _run(scripted, coords, Z, cell)
        assert torch.allclose(e0, e1, rtol=1e-6)
        assert _close(f1, f0, f0.abs().max().item(), 1e-5)
        assert _close(v1, v0, v0.abs().max().item(), 1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_matches_cpu(wrapper):
    gpu = SevenNet_Wrapper(MODEL, device="cuda").eval()
    coords, Z = _water_box(seed=5)
    e0, f0, v0 = _run(wrapper, coords, Z, TRICLINIC)
    c = coords.cuda().requires_grad_(True)
    e1, f1, _, v1 = gpu(c, Z.cuda(), EMPTY_PC.cuda(), EMPTY_PC_Q.cuda(),
                        TRICLINIC.reshape(1, 3, 3).cuda())
    fmax = f0.abs().max().item()
    assert abs(e1.item() - e0.item()) < 1e-5 * abs(e0.item())
    assert _close(f1.detach().cpu(), f0, fmax, 1e-4)
    assert _close(v1.detach().cpu(), v0, v0.abs().max().item(), 1e-4)
