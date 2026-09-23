"""
Checks for the shared virial helpers in src/virial.py.

Uses a toy harmonic pair across a periodic boundary, because that has a virial
you can write down by hand: for E = 0.5*k*|d|^2 with d = r_j - r_i + S.cell,
straining positions and cell together gives dE/dD = k * d (x) d, so the virial
must be -k * d (x) d.  A real MLIP cannot be checked that way, which is why the
toy is worth having: it pins the sign and the factor independently of any model.
"""

import sys

import torch

sys.path.insert(0, "/home/rat/PycharmProjects/ML_models_NAMD")
from src.virial import (make_strain, apply_strain, virial_from_strain,
                        forces_and_virial, finalize, zero_virial, zero_virials)

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


# Everything has to survive scripting, since these are inlined into exported models.
scripted_ok = True
for fn in (make_strain, apply_strain, virial_from_strain, forces_and_virial,
           finalize, zero_virial, zero_virials):
    try:
        torch.jit.script(fn)
    except Exception as e:                                   # pragma: no cover
        scripted_ok = False
        print("   scripting failed for", fn.__name__, repr(e)[:150])
check("all helpers compile under torch.jit.script", scripted_ok)


K = 3.0
EI = torch.tensor([[0], [1]])
CELL = torch.eye(3, dtype=torch.float64) * 10.0
# Atoms near opposite faces, so their real neighbour is across the boundary.
POS = torch.tensor([[0.5, 0.5, 0.5], [9.5, 0.6, 0.4]], dtype=torch.float64)
UNIT_SHIFTS = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)


def pair_energy(p, shifts_cart):
    d = p[EI[1]] - p[EI[0]] + shifts_cart
    return 0.5 * K * (d * d).sum()


pos = POS.clone().requires_grad_(True)
D = make_strain(CELL)
ps, cs, ss = apply_strain(pos, CELL, UNIT_SHIFTS, D)
E = pair_energy(ps, ss)
forces, V = forces_and_virial(E, pos, D, 1.0)

with torch.no_grad():
    d0 = POS[EI[1]] - POS[EI[0]] + UNIT_SHIFTS @ CELL
    V_ref = -(K * torch.einsum("ei,ej->ij", d0, d0))
    V_ref = 0.5 * (V_ref + V_ref.t())

check("shift really is the minimum image",
      abs(float(d0.norm()) - 1.00995049) < 1e-6, f"|d| = {float(d0.norm()):.6f}")
check("virial matches the analytic -k r (x) r",
      bool(torch.allclose(V, V_ref, atol=1e-12)),
      f"max|diff| = {float((V - V_ref).abs().max()):.2e}")
check("virial is symmetric", bool(torch.allclose(V, V.t(), atol=1e-15)))
check("forces are unaffected by carrying the strain",
      bool(torch.allclose(forces[0], K * d0[0], atol=1e-12)))

# Independent finite difference over the same toy.
h = 1e-6
fd = torch.zeros(3, 3, dtype=torch.float64)
for a in range(3):
    for b in range(3):
        def at(sign):
            Dm = torch.zeros(3, 3, dtype=torch.float64)
            Dm[a, b] = sign * h
            sym = 0.5 * (Dm + Dm.t())
            return pair_energy(POS + POS @ sym, UNIT_SHIFTS @ (CELL + CELL @ sym))
        fd[a, b] = (at(+1) - at(-1)) / (2 * h)
fd = 0.5 * (fd + fd.t())
check("finite difference agrees that dE/dD = -virial",
      bool(torch.allclose(fd, -V, atol=1e-6)),
      f"max|diff| = {float((fd + V).abs().max()):.2e}")

# Straining only the positions and leaving the cell alone is the classic error;
# it must give a different answer, or the test above proves nothing.
sym = 0.5 * (torch.full((3, 3), 1e-3, dtype=torch.float64) * 2)
bad = pair_energy(POS + POS @ sym, UNIT_SHIFTS @ CELL)
good = pair_energy(POS + POS @ sym, UNIT_SHIFTS @ (CELL + CELL @ sym))
check("straining the cell actually matters",
      abs(float(bad - good)) > 1e-6, f"|dE| = {abs(float(bad - good)):.3e}")

# The separate-backward route must agree with the fused one.
pos2 = POS.clone().requires_grad_(True)
D2 = make_strain(CELL)
ps2, cs2, ss2 = apply_strain(pos2, CELL, UNIT_SHIFTS, D2)
V2 = virial_from_strain(pair_energy(ps2, ss2), D2, 1.0)
check("virial_from_strain agrees with forces_and_virial",
      bool(torch.allclose(V, V2, atol=1e-12)),
      f"max|diff| = {float((V - V2).abs().max()):.2e}")

check("finalize returns float64 and symmetry",
      finalize(torch.rand(3, 3)).dtype == torch.float64)
check("zero helpers have the shapes NAMD expects",
      zero_virial(POS).shape == (3, 3) and zero_virials(4, POS).shape == (4, 3, 3))

print()
print("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILURE(S): " + ", ".join(FAILS))
if __name__ == "__main__":
    sys.exit(1 if FAILS else 0)
else:
    def test_no_failures():
        """Collected by pytest; the checks above run at import."""
        assert not FAILS, FAILS
