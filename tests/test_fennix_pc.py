"""Point-charge embedding in the FeNNix StableHLO export (NAMD QMElecEmbed).

Run in the fennix env: `JAX_PLATFORMS=cpu python -m pytest tests/test_fennix_pc.py`.

The contract (NAMD MLFF_BUILD_GUIDE.md 4.1): the artifact's energy E excludes
the QM-PC Coulomb sum, forces / pc_forces are -grad of L = E + sum_i q_i
stop_gradient(phi_i), and NAMD adds the fixed-charge Coulomb forces on both
sides.  So artifact forces + Coulomb forces must equal -grad U with
U = E + sum_i q_i phi_i (phi not fixed), which the NumPy float64 reference
in scripts/fennix_pc_test_models.py differentiates independently of JAX.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fennix_export = _load("fennix_export_pc", ROOT / "scripts" / "export_fennix_bio1_stablehlo.py")
tm = fennix_export._test_models()

EV = tm.EV_TO_KCAL
MODELS = ["pc-stub", "pc-stub-cr"]

# ACE-ALA-NME side chain + link atom, as NAMD sends it (C, 3 H, link H).
Z5 = np.array([6, 1, 1, 1, 1])
X5 = np.array([[5.661, 4.221, -1.232], [5.123, 4.521, -2.131], [6.630, 4.719, -1.206],
               [5.809, 3.141, -1.241], [5.083, 4.502, -0.351]])


def _pc_eval(name, capacity):
    return fennix_export.build_pc_eval_fn(
        jnp, tm.make_model_fn(name, Z5, EV), capacity=capacity,
        ev_to_kcal=EV, coulomb_kcal=tm.COULOMB_KCAL)


def _padded(r, q, capacity, garbage=False, seed=3):
    n = r.shape[0]
    rng = np.random.default_rng(seed)
    pr = np.zeros((capacity, 3)); pq = np.zeros(capacity); pm = np.zeros(capacity)
    pr[:n], pq[:n], pm[:n] = r, q, 1.0
    if garbage and capacity > n:
        # Masked slots with charges and positions the model must not see,
        # one of them exactly on a QM atom (a 1/r singularity if leaked).
        pr[n:] = X5.mean(0) + rng.normal(scale=3.0, size=(capacity - n, 3))
        pr[n] = X5[0]
        pq[n:] = rng.uniform(-1.0, 1.0, size=capacity - n)
    return pr, pq, pm


def _run(fn, x, pr, pq, pm, dtype):
    out = jax.jit(fn)(jnp.asarray(x, dtype), jnp.asarray(pr, dtype),
                      jnp.asarray(pq, dtype), jnp.asarray(pm, dtype))
    return [np.asarray(o, np.float64) for o in out]


@pytest.mark.parametrize("name", MODELS)
def test_forces_and_pc_forces_match_finite_differences_x64(name):
    from jax.experimental import enable_x64

    r, q = tm.reference_point_charges(X5, 11)
    r, q = r.astype(np.float64), q.astype(np.float64)
    fu_qm, fu_pc = tm.reference_forces_fd(name, Z5, X5, r, q, h=1e-5)
    with enable_x64():
        e, f, charges, pcf, ovf = _run(_pc_eval(name, 16), X5, *_padded(r, q, 16), jnp.float64)
    e_ref, q_ref = tm.reference_energy_charges(name, Z5, X5, r, q)
    fc_qm, fc_pc = tm.reference_coulomb_forces(X5, charges, r, q)
    assert abs(e[0] * EV - e_ref) < 1e-10
    np.testing.assert_allclose(charges, q_ref, atol=1e-12)
    # NAMD's total force = artifact force (kcal) + fixed-charge Coulomb force.
    np.testing.assert_allclose(f * EV + fc_qm, fu_qm, atol=2e-6)
    np.testing.assert_allclose(pcf[:11] * EV + fc_pc, fu_pc, atol=2e-6)
    assert np.all(pcf[11:] == 0.0)
    assert ovf[0] == 0.0
    if name == "pc-stub-cr":
        # Negative control: the charge response is a real part of F_qm, so
        # -dE/dx alone (what an artifact without it would return) is off.
        h = 1e-5
        fe = np.zeros_like(X5)
        for idx in np.ndindex(X5.shape):
            xp = X5.copy(); xp[idx] += h
            xm = X5.copy(); xm[idx] -= h
            fe[idx] = -(tm.reference_energy_charges(name, Z5, xp, r, q)[0]
                        - tm.reference_energy_charges(name, Z5, xm, r, q)[0]) / (2 * h)
        assert np.abs(fe + fc_qm - fu_qm).max() > 1e-2


@pytest.mark.parametrize("name", MODELS)
def test_float32_artifact_function_matches_reference(name):
    r, q = tm.reference_point_charges(X5, 11)
    fu_qm, fu_pc = tm.reference_forces_fd(name, Z5, X5, r.astype(np.float64), q.astype(np.float64))
    e, f, charges, pcf, _ = _run(_pc_eval(name, 16), X5, *_padded(r, q, 16), jnp.float32)
    fc_qm, fc_pc = tm.reference_coulomb_forces(X5, charges, r, q)
    scale = max(np.abs(fu_qm).max(), np.abs(fu_pc).max())
    np.testing.assert_allclose(f * EV + fc_qm, fu_qm, atol=1e-5 * scale)
    np.testing.assert_allclose(pcf[:11] * EV + fc_pc, fu_pc, atol=1e-5 * scale)


@pytest.mark.parametrize("name", MODELS)
def test_padding_invariance(name):
    """P = numPC (no padding) and P = capacity with garbage in the masked
    slots give the same result; padded pc_forces are exactly 0 and finite."""
    r, q = tm.reference_point_charges(X5, 9)
    tight = _run(_pc_eval(name, 9), X5, *_padded(r, q, 9), jnp.float32)
    padded = _run(_pc_eval(name, 40), X5, *_padded(r, q, 40, garbage=True), jnp.float32)
    for a, b in zip(tight[:3], padded[:3]):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(tight[3], padded[3][:9], rtol=1e-6, atol=1e-6)
    assert np.all(padded[3][9:] == 0.0)
    assert all(np.all(np.isfinite(o)) for o in padded)


def test_sentinel_lattice_is_far_and_spread():
    """Padded slots are parked on a lattice far from the QM atoms and from
    each other, so a model with point-charge to point-charge features or a
    cutoff-limited feature never sees them."""
    seen = {}

    def spy(x, R, Q, mask):
        jax.debug.callback(lambda r: seen.__setitem__("R", np.asarray(r)), R)
        return jnp.sum(x) * 0.0, jnp.zeros(x.shape[0])

    fn = fennix_export.build_pc_eval_fn(jnp, spy, capacity=30, ev_to_kcal=EV,
                                        coulomb_kcal=tm.COULOMB_KCAL)
    pr, pq, pm = _padded(np.zeros((2, 3)) + [[3.0, 0, 0], [0, 3.0, 0]], np.array([0.5, -0.5]), 30, garbage=True)
    fn(jnp.asarray(X5, jnp.float32), jnp.asarray(pr, jnp.float32), jnp.asarray(pq, jnp.float32),
       jnp.asarray(pm, jnp.float32))
    r = np.asarray(seen["R"], np.float64)
    np.testing.assert_allclose(r[:2], pr[:2], atol=1e-6)
    pad = r[2:]
    d_qm = np.linalg.norm(pad[:, None, :] - X5[None, :, :], axis=-1)
    d_pp = np.linalg.norm(pad[:, None, :] - pad[None, :, :], axis=-1) + np.eye(len(pad)) * 1e9
    assert d_qm.min() > 9000.0
    assert d_pp.min() > 99.0


def test_no_point_charges_all_masked():
    """All slots masked (NAMD's call when embedding is not active): the model
    sees no point charges."""
    r = np.zeros((0, 3)); q = np.zeros(0)
    e, f, charges, pcf, _ = _run(_pc_eval("pc-stub-cr", 8), X5, *_padded(r, q, 8, garbage=True), jnp.float32)
    e_ref, q_ref = tm.reference_energy_charges("pc-stub-cr", Z5, X5, r, q)
    fu_qm, _ = tm.reference_forces_fd("pc-stub-cr", Z5, X5, r, q)
    assert abs(e[0] * EV - e_ref) < 1e-4
    np.testing.assert_allclose(charges, q_ref, atol=1e-6)
    np.testing.assert_allclose(f * EV, fu_qm, atol=1e-4)
    assert np.all(pcf == 0.0)


# One water, the only composition pc-stub-tip3p takes (O, H only).
ZW = np.array([8, 1, 1])
XW = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])


def test_tip3p_stub_matches_finite_differences_x64():
    """Fixed TIP3P charges: no charge response, so forces are -dE/dx and
    pc_forces -dE/dR, plus NAMD's Coulomb force, against the float64 FD."""
    from jax.experimental import enable_x64

    r, q = tm.reference_point_charges(XW, 7)
    r, q = r.astype(np.float64), q.astype(np.float64)
    fu_qm, fu_pc = tm.reference_forces_fd("pc-stub-tip3p", ZW, XW, r, q, h=1e-5)
    fn = fennix_export.build_pc_eval_fn(
        jnp, tm.make_model_fn("pc-stub-tip3p", ZW, EV), capacity=10,
        ev_to_kcal=EV, coulomb_kcal=tm.COULOMB_KCAL)
    with enable_x64():
        e, f, charges, pcf, _ = _run(fn, XW, *_padded(r, q, 10), jnp.float64)
    e_ref, q_ref = tm.reference_energy_charges("pc-stub-tip3p", ZW, XW, r, q)
    np.testing.assert_allclose(charges, [-0.834, 0.417, 0.417], atol=1e-12)
    np.testing.assert_allclose(charges, q_ref, atol=1e-12)
    assert abs(e[0] * EV - e_ref) < 1e-10
    fc_qm, fc_pc = tm.reference_coulomb_forces(XW, charges, r, q)
    np.testing.assert_allclose(f * EV + fc_qm, fu_qm, atol=2e-6)
    np.testing.assert_allclose(pcf[:7] * EV + fc_pc, fu_pc, atol=2e-6)
    assert np.all(pcf[7:] == 0.0)
    with pytest.raises(ValueError, match="only O and H"):
        tm.make_model_fn("pc-stub-tip3p", Z5, EV)


def _write_pdb(path, z, x):
    sym = {1: "H", 6: "C", 8: "O"}
    lines = [f"ATOM  {i+1:5d} {sym[zz]:<4} QM  Q{i+1:4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {sym[zz]:>2}"
             for i, (zz, p) in enumerate(zip(z, x))]
    Path(path).write_text("\n".join(lines + ["END"]) + "\n")


def test_export_pc_artifact(tmp_path):
    pdb = tmp_path / "region.pdb"
    _write_pdb(pdb, Z5, X5)
    out = tmp_path / "art"
    fennix_export.main(["--test-model", "pc-stub-cr", "--pc-capacity", "40",
                        "--pdb", str(pdb), "--out-dir", str(out)])
    m = json.loads((out / "manifest.json").read_text())
    assert m["pc_embedding"] is True and m["pc_capacity"] == 40
    assert m["pc_coulomb_kcal"] == tm.COULOMB_KCAL
    assert abs(m["pc_coulomb_ev"] * m["conversion"]["ev_to_kcal"] - tm.COULOMB_KCAL) < 1e-9
    assert m["input_names"] == ["coordinates", "pc_coordinates", "pc_charges", "pc_mask"]
    assert m["output_names"] == ["energy", "forces", "charges", "pc_forces", "overflow"]
    shapes = {e["name"]: e["shape"] for e in m["input_signature"] + m["output_signature"]}
    assert shapes["pc_coordinates"] == [40, 3] and shapes["pc_forces"] == [40, 3]
    assert shapes["pc_charges"] == [40] and shapes["pc_mask"] == [40]
    assert m["matmul_precision"] == "highest"
    assert m["periodic"] is False
    for k in ("energy_max_abs_diff_ev", "forces_max_abs_diff_ev_a", "pc_forces_max_abs_diff_ev_a"):
        assert m["validation"][k] < 1e-6
    hlo = (out / m["artifacts"]["stablehlo_mlir"]).read_text()
    assert "custom_call" not in hlo


def test_export_plain_test_model_keeps_old_layout(tmp_path):
    pdb = tmp_path / "region.pdb"
    _write_pdb(pdb, Z5, X5)
    out = tmp_path / "plain"
    fennix_export.main(["--test-model", "pc-stub", "--pdb", str(pdb), "--out-dir", str(out)])
    m = json.loads((out / "manifest.json").read_text())
    assert m["pc_embedding"] is False and "pc_capacity" not in m
    assert m["input_names"] == ["coordinates"]
    assert m["output_names"] == ["energy", "forces", "charges", "overflow"]
    assert m["matmul_precision"] == "default"


@pytest.mark.parametrize("argv, msg", [
    (["--pc-capacity", "8", "--pbc", "--test-model", "pc-stub"], "--pbc"),
    (["--pc-capacity", "8"], "no FeNNol model"),
    (["--pc-capacity", "0", "--test-model", "pc-stub"], ">= 1"),
    (["--test-model", "pc-stub", "--pbc"], "periodic"),
])
def test_refusals(tmp_path, argv, msg):
    with pytest.raises(SystemExit, match=msg):
        fennix_export.main(argv + ["--z-list", "8,1,1", "--out-dir", str(tmp_path / "x")])
