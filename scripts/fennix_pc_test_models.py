"""Synthetic JAX models for testing point-charge embedding in the FeNNix
StableHLO path (export_fennix_bio1_stablehlo.py --test-model).

JAX twins of NAMD's src/mlff_shim/make_pc_stub.py (the TorchScript stubs used
to test MLFF forward_pc_forces), with the same functional forms and numbers in
kcal/mol, so NAMD should see the same forces from both backends:

    E       = sum_i |x_i|^2 / 10 + sum_iJ a Q_J exp(-|x_i - R_J|^2 / s)
    charges = 0.05 (Z - 6)                                   (pc-stub)
            = -0.834 (O), 0.417 (H)                          (pc-stub-tip3p)
            = 0.05 (Z_i - 6) + 0.1 tanh(sum_J Q_J exp(-|x_i - R_J|^2 / s))
              + 0.02 sum_{k != i} exp(-|x_i - x_k|^2 / 2)    (pc-stub-cr)

with a = 0.7, s = 4.  A model function takes (x [N,3], R [P,3], Q [P],
mask [P] bool) with the padded slots already made inert by the exporter
(far sentinel positions, zero charges) and returns (energy in eV, charges
[N] in e).  Every point-charge term here is charge-weighted, so the
sanitised Q alone removes padded slots; the mask is still applied, as a
model with position-only point-charge features would have to.

The NumPy float64 reference (`reference_*`) is independent of JAX and gives
the full NAMD-side potential U = E + sum_i q_i phi_i (phi NOT held fixed),
whose negative gradient the artifact forces plus NAMD's fixed-charge Coulomb
forces must reproduce.
"""

from __future__ import annotations

import numpy as np

EV_TO_KCAL = 23.0621
COULOMB_KCAL = 332.0636  # NAMD's Coulomb constant, kcal*A/(mol*e^2)
STUB_A = 0.7
STUB_S = 4.0

TEST_MODELS = ("pc-stub", "pc-stub-cr", "pc-stub-tip3p")


def _static_charges(name: str, Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    if name == "pc-stub-tip3p":
        if not np.all(np.isin(Z, (1, 8))):
            raise ValueError("pc-stub-tip3p takes only O and H atoms")
        return np.where(Z == 8, -0.834, 0.417)
    return 0.05 * (Z - 6.0)


def make_model_fn(name: str, Z, ev_to_kcal: float = EV_TO_KCAL):
    """Return model_fn(x, R, Q, mask) -> (energy_eV, charges) for `name`."""
    if name not in TEST_MODELS:
        raise ValueError(f"unknown test model {name!r}; choose from {TEST_MODELS}")
    import jax.numpy as jnp

    q0_np = _static_charges(name, Z)
    n_atoms = int(q0_np.shape[0])
    charge_response = name == "pc-stub-cr"

    def model_fn(x, R, Q, mask):
        dtype = x.dtype
        q0 = jnp.asarray(q0_np, dtype=dtype)
        e = jnp.sum(x * x) / 10.0
        Qm = jnp.where(mask, Q, 0.0)
        if R.shape[0] > 0:
            d2 = jnp.sum((x[:, None, :] - R[None, :, :]) ** 2, axis=-1)
            g = jnp.exp(-d2 / STUB_S)
            e = e + STUB_A * jnp.sum(Qm[None, :] * g)
        q = q0
        if charge_response:
            dqq = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
            off = 1.0 - jnp.eye(n_atoms, dtype=dtype)
            q = q + 0.02 * jnp.sum(jnp.exp(-dqq / 2.0) * off, axis=1)
            if R.shape[0] > 0:
                q = q + 0.1 * jnp.tanh(jnp.sum(Qm[None, :] * g, axis=1))
        return e / ev_to_kcal, q

    return model_fn


# ---------------------------------------------------------------- reference

def reference_energy_charges(name: str, Z, x, R, Q):
    """NumPy float64: (E kcal/mol, charges e) for real point charges only."""
    x = np.asarray(x, np.float64); R = np.asarray(R, np.float64).reshape(-1, 3)
    Q = np.asarray(Q, np.float64).reshape(-1)
    q = _static_charges(name, Z).copy()
    e = float(np.sum(x * x) / 10.0)
    if R.shape[0]:
        d2 = np.sum((x[:, None, :] - R[None, :, :]) ** 2, axis=-1)
        g = np.exp(-d2 / STUB_S)
        e += STUB_A * float(np.sum(Q[None, :] * g))
    if name == "pc-stub-cr":
        dqq = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        off = 1.0 - np.eye(x.shape[0])
        q = q + 0.02 * np.sum(np.exp(-dqq / 2.0) * off, axis=1)
        if R.shape[0]:
            q = q + 0.1 * np.tanh(np.sum(Q[None, :] * g, axis=1))
    return e, q


def reference_total_potential(name: str, Z, x, R, Q):
    """U = E + sum_i q_i phi_i in kcal/mol (phi not held fixed): the energy
    NAMD's total force derives from under the embedding contract."""
    e, q = reference_energy_charges(name, Z, x, R, Q)
    R = np.asarray(R, np.float64).reshape(-1, 3)
    Q = np.asarray(Q, np.float64).reshape(-1)
    if R.shape[0]:
        r = np.linalg.norm(np.asarray(x, np.float64)[:, None, :] - R[None, :, :], axis=-1)
        e += COULOMB_KCAL * float(np.sum(q[:, None] * Q[None, :] / r))
    return e


def reference_coulomb_forces(x, q, R, Q):
    """Fixed-charge QM-PC Coulomb forces (kcal/mol/A) on the QM atoms and the
    point charges: what NAMD's qmPntChrgCoulomb applies."""
    x = np.asarray(x, np.float64); R = np.asarray(R, np.float64).reshape(-1, 3)
    d = x[:, None, :] - R[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    f = COULOMB_KCAL * (np.asarray(q)[:, None] * np.asarray(Q)[None, :] / r**3)[..., None] * d
    return f.sum(axis=1), -f.sum(axis=0)


def reference_forces_fd(name: str, Z, x, R, Q, h: float = 1e-4):
    """Central finite differences of U: (F_qm [N,3], F_pc [P,3]), kcal/mol/A."""
    x = np.array(x, np.float64); R = np.array(R, np.float64).reshape(-1, 3)
    fx = np.zeros_like(x); fr = np.zeros_like(R)
    for arr, out in ((x, fx), (R, fr)):
        for idx in np.ndindex(arr.shape):
            v = arr[idx]
            arr[idx] = v + h; up = reference_total_potential(name, Z, x, R, Q)
            arr[idx] = v - h; um = reference_total_potential(name, Z, x, R, Q)
            arr[idx] = v
            out[idx] = -(up - um) / (2 * h)
    return fx, fr


def reference_point_charges(coords, n: int, seed: int = 7):
    """Deterministic point charges around a QM region: n charges on shells
    3.5 to 8 A from the QM centroid, alternating sign, magnitudes 0.3-0.9."""
    rng = np.random.default_rng(seed)
    c = np.asarray(coords, np.float64).reshape(-1, 3).mean(axis=0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.uniform(3.5, 8.0, size=(n, 1))
    R = c + v * r
    Q = rng.uniform(0.3, 0.9, size=n) * np.where(np.arange(n) % 2 == 0, -1.0, 1.0)
    return R.astype(np.float32), Q.astype(np.float32)
