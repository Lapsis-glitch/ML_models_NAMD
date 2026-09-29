"""
Checks for the TorchScript D3(BJ) add-on (``src/d3.py``), against SevenNet.

The reference is SevenNet's own CUDA D3 (``sevenn.calculator.D3Calculator``),
the code ``SevenNetD3Calculator`` and LAMMPS use.  Agreement there covers the
parameter parsing, the coordination numbers, the C6 interpolation, the image
sum and the stress.  On top of that:

  * atoms moved by lattice vectors (NAMD does not wrap) give the same answer;
  * the virial against a strain finite difference and, in a box too big for
    any image to interact, against sum r (x) f;
  * forward_batch against forward, and the scripted file against eager;
  * with ``models/compiled_sevennet_0.pt``, SevenNet + D3 against
    ``SevenNetD3Calculator``.

D3 is isolated by wrapping a base model that returns zeros, so most of this
needs only ``sevenn`` (for the parameters), not model weights.  The reference
comparisons need CUDA, because SevenNet's D3 is CUDA-only.
"""

import os
import sys
import tempfile
from typing import Tuple

import pytest
import torch
from torch import nn

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

pytest.importorskip("sevenn")

from src.constants import EV_TO_KCAL  # noqa: E402
from src.d3 import D3_Wrapper  # noqa: E402

SEVENNET_MODEL = os.path.join(_REPO, "models", "compiled_sevennet_0.pt")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="SevenNet's D3 is CUDA-only")

EMPTY_PC = torch.zeros((0, 3), dtype=torch.float64)
EMPTY_PC_Q = torch.zeros((0,), dtype=torch.float64)
ZERO_CELL = torch.zeros((3, 3), dtype=torch.float64)

# Small enough that the 50 A cutoff reaches several images in every direction.
TRICLINIC = torch.tensor([[11.0, 0.0, 0.0],
                          [2.0, 10.5, 0.0],
                          [1.0, 1.5, 11.5]], dtype=torch.float64)


class _ZeroBase(nn.Module):
    """A wrapper that contributes nothing, so D3_Wrapper returns D3 alone."""

    def __init__(self):
        super().__init__()
        self.supports_batch: bool = True
        self.supports_pbc: bool = True

    def forward(self, coords: torch.Tensor, Z: torch.Tensor, pc_coords: torch.Tensor,
                pc_charges: torch.Tensor, cell: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = coords.size(0)
        z = torch.zeros((), dtype=torch.float64, device=coords.device)
        return (z, torch.zeros((N, 3), dtype=torch.float64, device=coords.device),
                torch.zeros(N, dtype=torch.float64, device=coords.device),
                torch.zeros((3, 3), dtype=torch.float64, device=coords.device))

    @torch.jit.export
    def forward_batch(self, coords: torch.Tensor, Z: torch.Tensor, batch: torch.Tensor,
                      ptr: torch.Tensor, pc_coords: torch.Tensor, pc_charges: torch.Tensor,
                      cells: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        N = coords.size(0)
        B = ptr.size(0) - 1
        return (torch.zeros(B, dtype=torch.float64, device=coords.device),
                torch.zeros((N, 3), dtype=torch.float64, device=coords.device),
                torch.zeros(N, dtype=torch.float64, device=coords.device),
                torch.zeros((B, 3, 3), dtype=torch.float64, device=coords.device))


@pytest.fixture(scope="module")
def d3():
    return D3_Wrapper(_ZeroBase()).eval()


# -------------------------------------------------------------------
#  Systems
# -------------------------------------------------------------------

def _molecule():
    """CH3Cl, NH3 and H2S; mixed elements so a wrong table index would show."""
    coords = torch.tensor([
        [0.00, 0.00, 0.00], [1.78, 0.00, 0.00],
        [-0.36, 1.03, 0.00], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
        [-1.2, 0.3, 3.2], [-0.4, 0.8, 3.6], [-1.2, -0.6, 3.6], [-2.0, 0.8, 3.5],
        [2.5, 2.6, 1.0], [3.8, 2.6, 1.2], [2.3, 3.9, 1.3],
    ], dtype=torch.float64)
    Z = torch.tensor([6, 17, 1, 1, 1, 7, 1, 1, 1, 16, 1, 1])
    return coords, Z


def _water_box(seed=0, cell=TRICLINIC):
    """Eight randomly oriented waters on a 2x2x2 fractional grid of the cell."""
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
                coords.append(frac @ cell + mono @ q.T)
    return torch.cat(coords), torch.tensor([8, 1, 1] * 8)


# -------------------------------------------------------------------
#  Helpers
# -------------------------------------------------------------------

def _run(w, coords, Z, cell):
    c = coords.clone().requires_grad_(True)
    e, f, q, v = w(c, Z, EMPTY_PC, EMPTY_PC_Q, cell.reshape(1, 3, 3))
    return e.detach(), f.detach(), v.detach()


def _ase_reference(calc, coords, Z, cell):
    """(energy kcal/mol, forces, virial kcal/mol) from an ASE calculator."""
    from ase import Atoms

    periodic = bool(cell.abs().sum() > 0)
    atoms = Atoms(numbers=Z.numpy(), positions=coords.numpy(),
                  cell=cell.numpy() if periodic else None, pbc=periodic)
    atoms.calc = calc
    e = atoms.get_potential_energy() * EV_TO_KCAL
    f = torch.tensor(atoms.get_forces()) * EV_TO_KCAL
    v = torch.zeros(3, 3, dtype=torch.float64)
    if periodic:
        # ASE stress = (1/V) dE/d(strain) = -virial / V.
        v = -torch.tensor(atoms.get_stress(voigt=False)) * atoms.get_volume() * EV_TO_KCAL
    return e, f, v


def _close(a, b, scale, rtol):
    return (a - b).abs().max().item() <= rtol * max(scale, 1e-12)


# -------------------------------------------------------------------
#  Against SevenNet's CUDA D3
# -------------------------------------------------------------------

@needs_cuda
def test_matches_sevennet_d3_molecule(d3):
    from sevenn.calculator import D3Calculator

    coords, Z = _molecule()
    e, f, v = _run(d3, coords, Z, ZERO_CELL)
    e_ref, f_ref, _ = _ase_reference(D3Calculator(), coords, Z, ZERO_CELL)
    assert abs(e.item() - e_ref) < 1e-5 * abs(e_ref)
    assert _close(f, f_ref, f_ref.abs().max().item(), 1e-4)
    assert v.abs().max().item() == 0.0


@needs_cuda
def test_matches_sevennet_d3_periodic(d3):
    from sevenn.calculator import D3Calculator

    coords, Z = _water_box()
    e, f, v = _run(d3, coords, Z, TRICLINIC)
    e_ref, f_ref, v_ref = _ase_reference(D3Calculator(), coords, Z, TRICLINIC)
    assert abs(e.item() - e_ref) < 1e-5 * abs(e_ref)
    assert _close(f, f_ref, f_ref.abs().max().item(), 1e-4)
    assert _close(v, v_ref, v_ref.abs().max().item(), 1e-4)


@needs_cuda
@pytest.mark.skipif(not os.path.exists(SEVENNET_MODEL),
                    reason="models/compiled_sevennet_0.pt missing")
def test_sevennet_plus_d3_matches_sevennet_d3_calculator():
    from sevenn.calculator import SevenNetD3Calculator
    from src.wrappers.wrap_sevennet import SevenNet_Wrapper

    w = D3_Wrapper(SevenNet_Wrapper(SEVENNET_MODEL)).eval()
    coords, Z = _water_box(seed=1)
    e, f, v = _run(w, coords, Z, TRICLINIC)
    e_ref, f_ref, v_ref = _ase_reference(
        SevenNetD3Calculator(model="7net-0", device="cpu"), coords, Z, TRICLINIC)
    assert abs(e.item() - e_ref) < 1e-5 * abs(e_ref)
    assert _close(f, f_ref, f_ref.abs().max().item(), 1e-4)
    assert _close(v, v_ref, v_ref.abs().max().item(), 1e-4)


# -------------------------------------------------------------------
#  Contract checks (no CUDA, no weights)
# -------------------------------------------------------------------

def test_unwrapped_coordinates_give_the_same_answer(d3):
    coords, Z = _water_box(seed=2)
    moved = coords.clone()
    moved[0:3] += TRICLINIC[0] * 2 - TRICLINIC[2]       # a whole water, two cells away
    moved[4] -= TRICLINIC[1]                            # one H of another water
    e0, f0, v0 = _run(d3, coords, Z, TRICLINIC)
    e1, f1, v1 = _run(d3, moved, Z, TRICLINIC)
    assert torch.allclose(e1, e0, rtol=1e-12)
    assert _close(f1, f0, f0.abs().max().item(), 1e-9)
    assert _close(v1, v0, v0.abs().max().item(), 1e-9)


def test_forces_match_finite_difference(d3):
    coords, Z = _molecule()
    _, f, _ = _run(d3, coords, Z, ZERO_CELL)
    h = 1e-4
    for a in (0, 1, 5, 9):
        for k in range(3):
            cp = coords.clone()
            cm = coords.clone()
            cp[a, k] += h
            cm[a, k] -= h
            fd = -(_run(d3, cp, Z, ZERO_CELL)[0] - _run(d3, cm, Z, ZERO_CELL)[0]) / (2 * h)
            assert abs(fd.item() - f[a, k].item()) < 1e-6 * f.abs().max().item() + 1e-9


def _strain_fd(w, coords, Z, cell, h):
    fd = torch.zeros(3, 3, dtype=torch.float64)
    for a in range(3):
        for b in range(3):
            eps = torch.zeros(3, 3, dtype=torch.float64)
            eps[a, b] += 0.5 * h
            eps[b, a] += 0.5 * h
            ep = _run(w, coords @ (torch.eye(3) + eps), Z, cell @ (torch.eye(3) + eps))[0]
            em = _run(w, coords @ (torch.eye(3) - eps), Z, cell @ (torch.eye(3) - eps))[0]
            fd[a, b] = -(ep - em).item() / (2 * h)
    return fd


def test_virial_matches_strain_finite_difference():
    """
    D3's cutoffs are hard (as in SevenNet), so under a strain some pair-images
    cross 50 A / 21 A and the energy jumps; the difference picks that up and
    the analytic virial (like SevenNet's stress) does not, ~1e-4 relative.
    Freezing the pair list and dropping the cutoff masks makes the energy
    smooth, and then the two must agree to rounding.
    """
    coords, Z = _water_box(seed=3)

    w = D3_Wrapper(_ZeroBase()).eval()
    _, _, v = _run(w, coords, Z, TRICLINIC)
    assert _close(v, _strain_fd(w, coords, Z, TRICLINIC, 1e-5), v.abs().max().item(), 2e-3)

    frozen = w.d3.pairs(coords, TRICLINIC, True)
    w.d3.pairs = lambda pos, cell, periodic: frozen
    w.d3.rthr = float("inf")
    w.d3.cnthr = float("inf")
    _, _, v = _run(w, coords, Z, TRICLINIC)
    fd = _strain_fd(w, coords, Z, TRICLINIC, 1e-5)
    assert _close(v, fd, v.abs().max().item(), 1e-8), (v, fd)


def test_large_box_virial_is_sum_r_outer_f(d3):
    coords, Z = _molecule()
    box = torch.eye(3, dtype=torch.float64) * 120.0      # images > 50 A away
    coords = coords + 60.0
    e0, f0, _ = _run(d3, coords, Z, ZERO_CELL)
    e, f, v = _run(d3, coords, Z, box)
    assert torch.allclose(e, e0, rtol=1e-12)
    assert _close(f, f0, f0.abs().max().item(), 1e-10)
    expected = coords.T @ f
    expected = 0.5 * (expected + expected.T)
    assert _close(v, expected, v.abs().max().item(), 1e-10)


@pytest.mark.parametrize("periodic", [False, True])
def test_forward_batch_matches_forward(d3, periodic):
    ca, Za = _water_box(seed=4)
    cb, Zb = _molecule()
    cell_a = TRICLINIC
    cell_b = TRICLINIC * 1.1
    if not periodic:
        cell_a = cell_b = ZERO_CELL
    ea, fa, va = _run(d3, ca, Za, cell_a)
    eb, fb, vb = _run(d3, cb, Zb, cell_b)

    coords = torch.cat([ca, cb]).requires_grad_(True)
    Z = torch.cat([Za, Zb])
    batch = torch.cat([torch.zeros(ca.size(0)), torch.ones(cb.size(0))]).long()
    ptr = torch.tensor([0, ca.size(0), ca.size(0) + cb.size(0)])
    E, F, Q, V = d3.forward_batch(coords, Z, batch, ptr, EMPTY_PC, EMPTY_PC_Q,
                                  torch.stack([cell_a, cell_b]))
    assert torch.allclose(E, torch.stack([ea, eb]), rtol=1e-12)
    assert torch.allclose(F.detach(), torch.cat([fa, fb]), rtol=1e-10, atol=1e-12)
    assert torch.allclose(V.detach(), torch.stack([va, vb]), rtol=1e-10, atol=1e-12)


def test_scripted_matches_eager(d3):
    from src.export import export_wrapped

    path = os.path.join(tempfile.mkdtemp(), "d3.pt")
    export_wrapped(D3_Wrapper(_ZeroBase()).eval(), path, model_type="D3")
    scripted = torch.jit.load(path)
    coords, Z = _water_box(seed=5)
    for cell in (ZERO_CELL, TRICLINIC):
        e0, f0, v0 = _run(d3, coords, Z, cell)
        e1, f1, v1 = _run(scripted, coords, Z, cell)
        assert torch.allclose(e0, e1, rtol=1e-12)
        assert torch.allclose(f0, f1, rtol=1e-10, atol=1e-12)
        assert torch.allclose(v0, v1, rtol=1e-10, atol=1e-12)


def test_unknown_functional_is_refused():
    with pytest.raises(ValueError, match="Unknown D3"):
        D3_Wrapper(_ZeroBase(), functional="not-a-functional")
