import importlib.util
from pathlib import Path

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

