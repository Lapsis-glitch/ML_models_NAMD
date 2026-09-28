import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest


def _load_fennix_export_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_fennix_bio1_stablehlo.py"
    )
    spec = importlib.util.spec_from_file_location("fennix_export", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


fennix_export = _load_fennix_export_module()


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    resname: str,
    resid: int,
    x: float,
    y: float,
    z: float,
    *,
    record: str = "ATOM",
    altloc: str = " ",
    chain: str = "A",
    element: str = "",
    charge: str = "",
) -> str:
    return (
        f"{record:<6}{serial:>5} {atom_name:<4}{altloc:1}{resname:>3} {chain:1}{resid:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{1.00:>6.2f}          {element:>2}{charge:>2}"
    )


class TestLoadPdbSystem:
    def test_default_out_dir_tracks_pdb_stem_and_atom_count(self):
        out_dir = fennix_export.default_out_dir(42, Path("/tmp/my system.pdb"))

        assert out_dir.name == "fennix_bio1_stablehlo_my_system_n42"

    def test_default_out_dir_tracks_walker_count(self):
        out_dir = fennix_export.default_out_dir(42, Path("/tmp/my system.pdb"), n_walkers=8)

        assert out_dir.name == "fennix_bio1_stablehlo_my_system_n42_w8"

    def test_default_walker_coords_batches_identical_systems(self):
        base = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        coords = fennix_export.default_walker_coords(base, n_walkers=3)

        assert coords.shape == (3, 2, 3)
        np.testing.assert_allclose(coords[0], base)
        np.testing.assert_allclose(coords[1, :, 0], base[:, 0] + 0.05)

    def test_load_walker_coords_npy_rejects_wrong_atom_count(self, tmp_path):
        coords_path = tmp_path / "walkers.npy"
        np.save(coords_path, np.zeros((4, 2, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="does not match system atom count"):
            fennix_export.load_walker_coords_npy(coords_path, n_atoms=3)

    def test_reads_atomic_numbers_and_coordinates_from_pdb(self, tmp_path):
        pdb_path = tmp_path / "water.pdb"
        pdb_path.write_text(
            "\n".join(
                [
                    _pdb_atom_line(1, "O", "WAT", 1, 0.000, 0.000, 0.000, element="O"),
                    _pdb_atom_line(2, "H1", "WAT", 1, 0.957, 0.000, 0.000, element="H"),
                    _pdb_atom_line(3, "H2", "WAT", 1, -0.240, 0.927, 0.000, element="H"),
                    "TER",
                    "END",
                ]
            )
            + "\n"
        )

        Z, coords, symbols, total_charge, explicit_charge_count = fennix_export.load_pdb_system(pdb_path)

        assert Z.tolist() == [8, 1, 1]
        np.testing.assert_allclose(
            coords,
            np.asarray(
                [
                    [0.000, 0.000, 0.000],
                    [0.957, 0.000, 0.000],
                    [-0.240, 0.927, 0.000],
                ],
                dtype=np.float32,
            ),
        )
        assert symbols == ["O", "H", "H"]
        assert total_charge == 0
        assert explicit_charge_count == 0

    def test_uses_first_model_skips_non_a_altloc_and_infers_elements(self, tmp_path):
        pdb_path = tmp_path / "altloc.pdb"
        pdb_path.write_text(
            "\n".join(
                [
                    "MODEL        1",
                    _pdb_atom_line(1, " CA ", "ALA", 1, 1.000, 2.000, 3.000, altloc="B"),
                    _pdb_atom_line(1, " CA ", "ALA", 1, 4.000, 5.000, 6.000, altloc="A"),
                    _pdb_atom_line(2, "FE", "HEM", 2, 7.000, 8.000, 9.000, record="HETATM", charge="2+"),
                    "ENDMDL",
                    "MODEL        2",
                    _pdb_atom_line(3, "O", "HOH", 3, 0.000, 0.000, 0.000, element="O"),
                    "ENDMDL",
                    "END",
                ]
            )
            + "\n"
        )

        Z, coords, symbols, total_charge, explicit_charge_count = fennix_export.load_pdb_system(pdb_path)

        assert Z.tolist() == [6, 26]
        np.testing.assert_allclose(
            coords,
            np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32),
        )
        assert symbols == ["C", "Fe"]
        assert total_charge == 2
        assert explicit_charge_count == 1

    def test_rejects_invalid_pdb_charge_field(self, tmp_path):
        pdb_path = tmp_path / "bad_charge.pdb"
        pdb_path.write_text(
            _pdb_atom_line(1, "ZN", "ZN", 1, 0.0, 0.0, 0.0, record="HETATM", element="ZN", charge="++")
            + "\n"
        )

        with pytest.raises(ValueError, match="Unsupported PDB charge field"):
            fennix_export.load_pdb_system(pdb_path)


def _fake_import_module_factory():
    class FakeLowered:
        def compiler_ir(self, dialect):
            if dialect == "stablehlo":
                return "module @stablehlo"
            if dialect == "hlo":
                return SimpleNamespace(as_hlo_text=lambda: "HloModule test")
            raise AssertionError(dialect)

    class FakeJit:
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, *args):
            return self._fn(*args)

        def lower(self, *args):
            return FakeLowered()

    class FakeJnp:
        int32 = np.int32
        float32 = np.float32

        @staticmethod
        def asarray(x, dtype=None):
            return np.asarray(x, dtype=dtype)

        @staticmethod
        def reshape(x, shape):
            return np.reshape(x, shape)

        @staticmethod
        def any(x):
            return np.any(x)

        @staticmethod
        def array(x, dtype=None):
            return np.array(x, dtype=dtype)

        @staticmethod
        def zeros(shape, dtype=None):
            return np.zeros(shape, dtype=dtype)

    class FakeModel:
        cutoff = 7.5
        energy_unit = "ev"
        energy_terms = ["nn_energies_mean", "ff_energy"]
        variables = {"weights": 1}
        preproc_state = {"capacity": "fixed"}

        def __init__(self):
            self.preprocessing = SimpleNamespace(process=lambda state, raw: raw)

        def preprocess(self, **raw):
            self.preproc_state = {"n_atoms": int(raw["natoms"][0])}
            return raw

        def _energy_and_forces(self, variables, pre):
            coords = np.asarray(pre["coordinates"], dtype=np.float32)
            energy = np.array([coords.sum()], dtype=np.float32)
            forces = -coords.astype(np.float32)
            return energy, forces, None

        def energy_and_forces(self, **raw):
            return self._energy_and_forces(self.variables, raw)

    class FakeFENNIX:
        @staticmethod
        def load(path):
            return FakeModel()

    class FakeCompileOptions:
        def SerializeAsString(self):
            return b"compile-options"

    class FakeJax:
        __version__ = "fake-jax"

        @staticmethod
        def devices():
            return ["cpu:0"]

        @staticmethod
        def jit(fn):
            return FakeJit(fn)

        @staticmethod
        def vmap(fn, in_axes=0, out_axes=0):
            assert in_axes == 0
            assert out_axes == 0

            def wrapped(coords_batch):
                per_walker = [fn(coords) for coords in np.asarray(coords_batch)]
                return tuple(
                    np.stack([np.asarray(out[k]) for out in per_walker], axis=0)
                    for k in range(len(per_walker[0]))
                )

            return wrapped

    def fake_import_module(name):
        if name == "jax":
            return FakeJax
        if name == "jax.numpy":
            return FakeJnp
        if name == "fennol":
            return SimpleNamespace(FENNIX=FakeFENNIX)
        if name == "jax._src.compiler":
            return SimpleNamespace(get_compile_options=lambda *args: FakeCompileOptions())
        raise ImportError(name)

    return fake_import_module


class TestExporterMain:
    def test_main_keeps_single_walker_shapes_unchanged(self, tmp_path):
        out_dir = tmp_path / "single"

        with mock.patch("importlib.import_module", side_effect=_fake_import_module_factory()):
            fennix_export.main([
                "--model", str(tmp_path / "fake.fnx"),
                "--out-dir", str(out_dir),
                "--z-list", "8,1,1",
            ])

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["n_walkers"] == 1
        assert manifest["input_signature"][0]["shape"] == [3, 3]
        assert manifest["output_signature"][0]["shape"] == [1]
        assert manifest["output_signature"][1]["shape"] == [3, 3]
        # Same contract as the TorchScript wrappers: charges always present
        # (zeros when the model has none), plus the overflow flag.
        assert manifest["output_names"] == ["energy", "forces", "charges", "overflow"]
        assert manifest["input_names"] == ["coordinates"]
        assert manifest["periodic"] is False
        assert manifest["output_signature"][2]["shape"] == [3]
        assert manifest["output_signature"][2]["source"] == "zeros"

        reference = np.load(out_dir / "reference.npz")
        assert reference["coordinates"].shape == (3, 3)

    def test_main_writes_batched_multi_walker_artifact(self, tmp_path):
        out_dir = tmp_path / "batched"
        walker_coords = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.2, 0.8, 0.0]],
                [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.1, 0.8, 0.0]],
                [[0.2, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 0.8, 0.0]],
            ],
            dtype=np.float32,
        )
        coords_path = tmp_path / "walkers.npy"
        np.save(coords_path, walker_coords)

        with mock.patch("importlib.import_module", side_effect=_fake_import_module_factory()):
            fennix_export.main([
                "--model", str(tmp_path / "fake.fnx"),
                "--out-dir", str(out_dir),
                "--z-list", "8,1,1",
                "--walker-coords-npy", str(coords_path),
            ])

        manifest = json.loads((out_dir / "manifest.json").read_text())
        assert manifest["n_walkers"] == 3
        assert manifest["input_signature"][0]["shape"] == [3, 3, 3]
        assert manifest["output_signature"][0]["shape"] == [3, 1]
        assert manifest["output_signature"][1]["shape"] == [3, 3, 3]
        assert manifest["source_system"]["walker_coords_npy"] == str(coords_path.resolve())

        reference = np.load(out_dir / "reference.npz")
        np.testing.assert_allclose(reference["coordinates"], walker_coords)
        assert reference["energy_jit_ev"].shape == (3, 1)
        assert reference["forces_jit_ev_a"].shape == (3, 3, 3)

        runtime_ref = (out_dir / "reference_runtime.txt").read_text()
        assert "input_dims=3,3,3" in runtime_ref

    def test_main_rejects_conflicting_walker_count(self, tmp_path):
        coords_path = tmp_path / "walkers.npy"
        np.save(coords_path, np.zeros((2, 3, 3), dtype=np.float32))

        with mock.patch("importlib.import_module", side_effect=_fake_import_module_factory()):
            with pytest.raises(SystemExit, match="does not match --walker-coords-npy batch size 2"):
                fennix_export.main([
                    "--model", str(tmp_path / "fake.fnx"),
                    "--z-list", "8,1,1",
                    "--n-walkers", "3",
                    "--walker-coords-npy", str(coords_path),
                ])


