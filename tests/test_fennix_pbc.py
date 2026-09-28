"""Periodic FeNNol export: virial sign/normalisation and image convention.

Exercises the exact function that scripts/export_fennix_bio1_stablehlo.py
lowers (build_eval_fn), with the real FENNIX-BIO1 weights, on CPU.  Needs the
`fennix` env:

    JAX_PLATFORMS=cpu conda run -n fennix python -m pytest tests/test_fennix_pbc.py -v

Checks, mirroring the TorchScript PBC tests (test_pbc_virial.py,
test_pbc_mace_image.py):
  - strain finite difference pins sign and normalisation of the virial,
    returned in NAMD's convention (W = sum_i f_i (x) r_i under PBC);
  - in a box much larger than the molecule, the periodic virial equals the
    cluster sum NAMD would compute itself;
  - two waters that touch only through the x face give the same energy as the
    explicitly imaged pair evaluated without PBC, and a different energy from
    the non-periodic evaluation of the wrapped coordinates;
  - the overflow output is 0 for a normal structure and 1 when the pair buffer
    is deliberately too small.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
fennol = pytest.importorskip("fennol")

_REPO = Path(__file__).resolve().parents[1]
_MODEL = _REPO / "models" / "fennix-bio1S.fnx"
if not _MODEL.exists():
    pytest.skip(f"model not found: {_MODEL}", allow_module_level=True)


def _load_exporter():
    path = _REPO / "scripts" / "export_fennix_bio1_stablehlo.py"
    spec = importlib.util.spec_from_file_location("fennix_export_pbc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx = _load_exporter()

WATER = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]], dtype=np.float32)


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a - b*b + c*c - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a - b*b - c*c + d*d],
    ])


def water_box(n_side, spacing, seed=0):
    rng = np.random.default_rng(seed)
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                centre = (np.array([i, j, k]) + 0.5) * spacing + rng.normal(scale=0.15, size=3)
                coords.append(WATER @ _rotation(rng).T + centre)
    Z = np.tile(np.array([8, 1, 1], dtype=np.int32), n_side ** 3)
    return Z, np.concatenate(coords).astype(np.float32)


def make_eval(Z, coords, cell, margin=1.25):
    model = fennol.FENNIX.load(str(_MODEL))
    raw = {
        "species": Z,
        "coordinates": coords,
        "natoms": np.array([len(Z)], dtype=np.int32),
        "batch_index": np.zeros(len(Z), dtype=np.int32),
        "total_charge": np.array([0], dtype=np.int32),
    }
    if cell is not None:
        raw["cells"] = cell[None]
        raw["flags"] = {"minimum_image": None}
    model.preprocess(**raw)
    state = fx.scale_nblist_capacity(model.preproc_state, margin)
    info = {}
    fn = fx.build_eval_fn(jnp, model, state, jnp.asarray(Z), 0,
                          periodic=cell is not None, charges_key="charges", info=info)
    return jax.jit(fn), info


@pytest.fixture(scope="module")
def box():
    Z, coords = water_box(5, 3.2)          # 125 waters, 16 A cube
    cell = np.diag([16.0, 16.0, 16.0]).astype(np.float32)
    f, info = make_eval(Z, coords, cell)
    return Z, coords, cell, f, info


def test_outputs_and_contract(box):
    Z, coords, cell, f, info = box
    e, forces, q, vir, ovf = f(coords, cell[None])
    assert e.shape == (1,)
    assert forces.shape == (len(Z), 3)
    assert q.shape == (len(Z),)
    assert vir.shape == (3, 3)
    assert ovf.shape == (1,) and float(ovf[0]) == 0.0
    # BIO1 has no per-atom `charges` output, so the contract gives zeros.
    assert info["charges_source"] == "zeros"
    assert np.all(np.asarray(q) == 0.0)
    assert np.allclose(np.asarray(vir), np.asarray(vir).T, atol=1e-3 * np.abs(vir).max())


def test_virial_matches_strain_finite_difference(box):
    Z, coords, cell, f, _ = box
    W = np.asarray(f(coords, cell[None])[3], dtype=np.float64)
    # W = -(dE/dS)^T, so dE/dS_ij = -W_ji.  Strain x -> x (I + eps D), cell likewise.
    eps = 2e-3
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]:
        D = np.zeros((3, 3))
        D[i, j] = 1.0
        D = 0.5 * (D + D.T)                     # symmetric strain (virial is symmetric)
        ep = []
        for s in (+1, -1):
            S = np.eye(3) + s * eps * D
            ep.append(float(f((coords @ S).astype(np.float32), (cell @ S).astype(np.float32)[None])[0][0]))
        dE = (ep[0] - ep[1]) / (2 * eps)
        predicted = -np.sum(W.T * D)
        wrong_sign = -predicted
        tol = 0.02 * max(1.0, np.abs(W).max())
        assert abs(dE - predicted) < tol, (i, j, dE, predicted)
        assert abs(dE - wrong_sign) > abs(dE - predicted)


def test_virial_equals_cluster_sum_in_huge_box():
    Z, coords = water_box(2, 3.2, seed=3)        # 8 waters, isolated in a 60 A box
    coords = coords + 25.0
    cell = np.diag([60.0, 60.0, 60.0]).astype(np.float32)
    f, _ = make_eval(Z, coords, cell)
    _, forces, _, W, _ = f(coords, cell[None])
    forces = np.asarray(forces, dtype=np.float64)
    cluster = np.einsum("ik,il->kl", forces, coords.astype(np.float64))   # NAMD's sum f (x) r
    assert np.allclose(np.asarray(W), cluster, atol=2e-3 * max(1.0, np.abs(cluster).max()))


def test_image_across_x_face_matches_explicit_cluster():
    L = 16.0
    cell = np.diag([L, L, L]).astype(np.float32)
    a = WATER + np.array([0.3, 8.0, 8.0])
    b = WATER + np.array([L - 2.6, 8.0, 8.0])     # 2.9 A from `a` through the x face
    Z = np.array([8, 1, 1, 8, 1, 1], dtype=np.int32)
    wrapped = np.concatenate([a, b]).astype(np.float32)
    imaged = np.concatenate([a, b - np.array([L, 0, 0])]).astype(np.float32)

    f_pbc, _ = make_eval(Z, wrapped, cell)
    f_iso, _ = make_eval(Z, imaged, None)
    f_iso_wrapped, _ = make_eval(Z, wrapped, None)

    e_pbc, F_pbc = (np.asarray(v) for v in f_pbc(wrapped, cell[None])[:2])
    e_img, F_img = (np.asarray(v) for v in f_iso(imaged)[:2])
    e_far = np.asarray(f_iso_wrapped(wrapped)[0])

    assert abs(float(e_pbc[0] - e_img[0])) < 1e-4
    assert np.max(np.abs(F_pbc - F_img)) < 1e-4
    assert abs(float(e_pbc[0] - e_far[0])) > 1e-3   # the boundary actually mattered


def test_overflow_flag_fires_when_pair_buffer_too_small(box):
    Z, coords, cell, _, _ = box
    model = fennol.FENNIX.load(str(_MODEL))
    raw = {"species": Z, "coordinates": coords,
           "natoms": np.array([len(Z)], dtype=np.int32),
           "batch_index": np.zeros(len(Z), dtype=np.int32),
           "total_charge": np.array([0], dtype=np.int32),
           "cells": cell[None], "flags": {"minimum_image": None}}
    model.preprocess(**raw)
    # Shrink the parent graph's pair capacity to a tenth.
    state = model.preproc_state
    layers = list(state["layers_state"])
    layers[0] = type(layers[0])({**layers[0], "npairs": int(layers[0]["npairs"]) // 10})
    state = type(state)({**state, "layers_state": tuple(layers)})
    fn = fx.build_eval_fn(jnp, model, state, jnp.asarray(Z), 0,
                          periodic=True, charges_key="charges", info={})
    ovf = np.asarray(jax.jit(fn)(coords, cell[None])[4])
    assert float(ovf[0]) == 1.0


def test_min_perp_width_and_cryst1(tmp_path):
    pdb = tmp_path / "x.pdb"
    pdb.write_text("CRYST1   20.000   30.000   40.000  90.00  90.00  90.00 P 1           1\n")
    cell = fx.load_pdb_cryst1(pdb)
    assert np.allclose(cell, np.diag([20.0, 30.0, 40.0]), atol=1e-5)
    assert abs(fx.cell_min_perp_width(cell) - 20.0) < 1e-5
    sheared = np.array([[20, 0, 0], [15, 5, 0], [0, 0, 40]], dtype=np.float32)
    assert fx.cell_min_perp_width(sheared) < 5.0 + 1e-5
