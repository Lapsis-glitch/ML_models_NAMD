"""
Integration tests for the full train → wrap → export pipeline.

These tests verify that:
  1. ``prepare_data`` produces valid output in all formats.
  2. Each training script can run (1–2 epochs) on tiny synthetic data.
  3. The trained artifact can be loaded and wrapped.
  4. The wrapped model obeys the NAMD output contract.

Tests that require external tools (mace, nequip-train, etc.) are
marked with ``@pytest.mark.slow`` and auto-skipped if the tool is
not available.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from typing import Dict


# ===================================================================
#  Markers
# ===================================================================

def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


skip_no_mace = pytest.mark.skipif(
    not _has_command("mace_run_train"),
    reason="mace_run_train not found",
)
skip_no_nequip = pytest.mark.skipif(
    not _has_command("nequip-train"),
    reason="nequip-train not found",
)

try:
    import schnetpack  # noqa: F401
    _has_schnetpack = True
except ImportError:
    _has_schnetpack = False

skip_no_schnetpack = pytest.mark.skipif(
    not _has_schnetpack,
    reason="schnetpack not installed",
)

try:
    import torchani  # noqa: F401
    _has_torchani = True
except ImportError:
    _has_torchani = False

skip_no_torchani = pytest.mark.skipif(
    not _has_torchani,
    reason="torchani not installed",
)


# ===================================================================
#  Test: prepare_data
# ===================================================================

class TestPrepareData:

    def test_xyz_splits_exist(self, prepared_data_dir):
        for name in ["train.xyz", "val.xyz", "test.xyz"]:
            p = Path(prepared_data_dir) / "xyz" / name
            assert p.exists(), f"{p} not found"

    def test_xyz_frame_counts(self, prepared_data_dir):
        """Total frames across splits should equal 10."""
        from ase.io import read as ase_read

        total = 0
        for name in ["train.xyz", "val.xyz", "test.xyz"]:
            p = Path(prepared_data_dir) / "xyz" / name
            frames = ase_read(str(p), index=":")
            total += len(frames)
        assert total == 10

    def test_xyz_has_energy_forces(self, prepared_data_dir):
        from ase.io import read as ase_read
        from src.training.prepare_data import get_energy, get_forces

        frames = ase_read(
            str(Path(prepared_data_dir) / "xyz" / "train.xyz"),
            index=":",
        )
        for f in frames:
            assert get_energy(f) is not None, "No energy found"
            assert get_forces(f) is not None, "No forces found"

    def test_torchani_h5_exists(self, prepared_data_dir):
        try:
            import h5py  # noqa: F401
        except ImportError:
            pytest.skip("h5py not installed")
        p = Path(prepared_data_dir) / "torchani" / "data.h5"
        assert p.exists()


# ===================================================================
#  Test: Training + Wrapping (real training, marked slow)
# ===================================================================

@pytest.mark.slow
class TestMACEPipeline:

    @skip_no_mace
    def test_train_and_wrap(self, prepared_data_dir, tmp_path):
        from src.training.train_mace import main as train_main

        out_dir = str(tmp_path / "mace_out")
        train_main([
            "--data-dir", prepared_data_dir,
            "--output-dir", out_dir,
            "--r-max", "5.0",
            "--max-epochs", "1",
            "--batch-size", "2",
            "--device", "cpu",
        ])

        # Check that some model file was produced
        out_path = Path(out_dir)
        model_files = list(out_path.glob("*.pt")) + list(out_path.glob("*.model"))
        assert len(model_files) > 0, "No model file produced by MACE training"


@pytest.mark.slow
class TestNequIPPipeline:

    @skip_no_nequip
    def test_train_and_wrap(self, prepared_data_dir, tmp_path):
        from src.training.train_nequip import main as train_main

        out_dir = str(tmp_path / "nequip_out")
        train_main([
            "--data-dir", prepared_data_dir,
            "--output-dir", out_dir,
            "--r-max", "5.0",
            "--max-epochs", "1",
            "--batch-size", "2",
        ])

        out_path = Path(out_dir)
        deployed = out_path / "nequip_deployed.pth"
        packaged = out_path / "nequip_packaged.nequip.zip"
        assert deployed.exists() or packaged.exists(), \
            "NequIP training did not produce deployed or packaged output"


@pytest.mark.slow
class TestAllegroPipeline:

    @skip_no_nequip
    def test_train_and_wrap(self, prepared_data_dir, tmp_path):
        from src.training.train_allegro import main as train_main

        out_dir = str(tmp_path / "allegro_out")
        try:
            import allegro  # noqa: F401
        except ImportError:
            pytest.skip("allegro not installed")

        train_main([
            "--data-dir", prepared_data_dir,
            "--output-dir", out_dir,
            "--r-max", "5.0",
            "--max-epochs", "1",
            "--batch-size", "2",
        ])

        out_path = Path(out_dir)
        deployed = out_path / "allegro_deployed.pth"
        packaged = out_path / "allegro_packaged.nequip.zip"
        assert deployed.exists() or packaged.exists(), \
            "Allegro training did not produce deployed or packaged output"


@pytest.mark.slow
class TestSchNetPackPipeline:

    @skip_no_schnetpack
    def test_train_and_wrap(self, prepared_data_dir, tmp_path):
        from src.training.train_schnetpack import main as train_main

        out_dir = str(tmp_path / "schnet_out")
        train_main([
            "--data-dir", prepared_data_dir,
            "--output-dir", out_dir,
            "--r-max", "5.0",
            "--max-epochs", "2",
            "--batch-size", "2",
            "--device", "cpu",
        ])

        scripted = Path(out_dir) / "schnet_scripted.pt"
        assert scripted.exists(), "SchNetPack training did not produce output"

        # Verify wrapping
        from src.wrappers.wrap_schnetpack import SchNetPack_Wrapper
        wrapper = SchNetPack_Wrapper(
            model_path=str(scripted), r_max=5.0, device="cpu",
        ).eval()

        coords = torch.randn(3, 3, dtype=torch.float64)
        Z = torch.tensor([8, 1, 1], dtype=torch.int64)
        pc_c = torch.zeros((0, 3), dtype=torch.float64)
        pc_q = torch.zeros(0, dtype=torch.float64)

        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        assert energy.dtype == torch.float64
        assert forces.shape == (3, 3)
        assert charges.shape == (3,)


@pytest.mark.slow
class TestTorchANIPipeline:

    @skip_no_torchani
    def test_train_and_wrap(self, prepared_data_dir, tmp_path):
        from src.training.train_torchani import main as train_main

        out_dir = str(tmp_path / "ani_out")
        train_main([
            "--data-dir", prepared_data_dir,
            "--output-dir", out_dir,
            "--elements", "H", "O",
            "--max-epochs", "2",
            "--batch-size", "2",
            "--device", "cpu",
        ])

        scripted = Path(out_dir) / "torchani_scripted.pt"
        assert scripted.exists(), "TorchANI training did not produce output"

        # Verify wrapping
        from src.wrappers.wrap_torchani import TorchANI_Wrapper
        wrapper = TorchANI_Wrapper(
            model_path=str(scripted), device="cpu",
            element_list=[1, 8],
        ).eval()

        coords = torch.tensor([
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ], dtype=torch.float64)
        Z = torch.tensor([8, 1, 1], dtype=torch.int64)
        pc_c = torch.zeros((0, 3), dtype=torch.float64)
        pc_q = torch.zeros(0, dtype=torch.float64)

        energy, forces, charges = wrapper(coords, Z, pc_c, pc_q)
        assert energy.dtype == torch.float64
        assert forces.shape == (3, 3)
        assert charges.shape == (3,)


# ===================================================================
#  Test: Wrapper-only (mock inner models, no training needed)
# ===================================================================

class _MockMACEInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_max = 5.0
        self.atomic_numbers = [1, 8]

    def forward(self, batch_dict: Dict[str, torch.Tensor],
                training: bool = False, compute_force: bool = True,
                compute_virials: bool = False, compute_stress: bool = False,
                compute_displacement: bool = False,
                compute_hessian: bool = False) -> Dict[str, torch.Tensor]:
        pos = batch_dict["positions"]
        N = pos.size(0)
        dev = pos.device
        ptr = batch_dict["ptr"]
        B = ptr.size(0) - 1
        return {
            "energy": torch.randn(B, dtype=torch.float64, device=dev),
            "forces": torch.randn(N, 3, dtype=torch.float64, device=dev),
        }


class _MockNequIPInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_max = 4.0
        self.atomic_numbers = [1, 8]

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos = data["pos"]
        N = pos.size(0)
        dev = pos.device
        ptr = data["ptr"]
        B = ptr.size(0) - 1
        return {
            "total_energy": torch.randn(B, dtype=torch.float64, device=dev),
            "forces": torch.randn(N, 3, dtype=torch.float64, device=dev),
        }


class _MockSchNetInner(nn.Module):
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos = inputs["_positions"]
        N = pos.size(0)
        dev = pos.device
        idx_m = inputs["_idx_m"]
        B = int(idx_m.max().item()) + 1 if idx_m.numel() > 0 else 1
        return {
            "energy": torch.randn(B, 1, dtype=torch.float64, device=dev),
            "forces": torch.randn(N, 3, dtype=torch.float64, device=dev),
        }


class _MockTorchANIInner(nn.Module):
    def forward(self, species: torch.Tensor, coordinates: torch.Tensor):
        B = species.size(0)
        energies = coordinates.pow(2).sum(dim=(1, 2)).unsqueeze(-1) * 0.001
        return species, energies


class TestWrapperExportRoundTrip:
    """
    Test that wrapping + TorchScript export + reload produces a model
    that still obeys the NAMD output contract.
    """

    def _save_mock(self, mock_module, tmp_path, name):
        scripted = torch.jit.script(mock_module)
        path = str(tmp_path / name)
        scripted.save(path)
        return path

    def test_mace_export_roundtrip(self, tmp_path):
        from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper
        from src.export import export_wrapped

        mock_path = self._save_mock(_MockMACEInner(), tmp_path, "mock_mace.pt")
        wrapper = MACE_TS_Wrapper(mock_path, device="cpu").eval()

        out_path = str(tmp_path / "exported_mace.pt")
        export_wrapped(wrapper, out_path, model_type="MACE")

        # Reload and test
        reloaded = torch.jit.load(out_path)
        coords = torch.randn(3, 3, dtype=torch.float64)
        Z = torch.tensor([8, 1, 1], dtype=torch.int64)
        pc = torch.zeros((0, 3), dtype=torch.float64)
        pq = torch.zeros(0, dtype=torch.float64)

        energy, forces, charges = reloaded(coords, Z, pc, pq)
        assert energy.dtype == torch.float64
        assert forces.shape == (3, 3)
        assert charges.shape == (3,)

    def test_nequip_export_roundtrip(self, tmp_path):
        from src.wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper
        from src.export import export_wrapped

        mock_path = self._save_mock(_MockNequIPInner(), tmp_path, "mock_nequip.pt")
        wrapper = NequIP_Allegro_Wrapper(mock_path, device="cpu").eval()

        out_path = str(tmp_path / "exported_nequip.pt")
        export_wrapped(wrapper, out_path, model_type="NequIP")

        reloaded = torch.jit.load(out_path)
        coords = torch.randn(3, 3, dtype=torch.float64)
        Z = torch.tensor([8, 1, 1], dtype=torch.int64)
        pc = torch.zeros((0, 3), dtype=torch.float64)
        pq = torch.zeros(0, dtype=torch.float64)

        energy, forces, charges = reloaded(coords, Z, pc, pq)
        assert energy.dtype == torch.float64
        assert forces.shape == (3, 3)
        assert charges.shape == (3,)

    def test_schnet_export_roundtrip(self, tmp_path):
        from src.wrappers.wrap_schnetpack import SchNetPack_Wrapper
        from src.export import export_wrapped

        mock_path = self._save_mock(_MockSchNetInner(), tmp_path, "mock_schnet.pt")
        wrapper = SchNetPack_Wrapper(
            model_path=mock_path, r_max=5.0, device="cpu",
        ).eval()

        out_path = str(tmp_path / "exported_schnet.pt")
        export_wrapped(wrapper, out_path, model_type="SchNetPack")

        reloaded = torch.jit.load(out_path)
        coords = torch.randn(3, 3, dtype=torch.float64)
        Z = torch.tensor([8, 1, 1], dtype=torch.int64)
        pc = torch.zeros((0, 3), dtype=torch.float64)
        pq = torch.zeros(0, dtype=torch.float64)

        energy, forces, charges = reloaded(coords, Z, pc, pq)
        assert energy.dtype == torch.float64
        assert forces.shape == (3, 3)
        assert charges.shape == (3,)

    def test_torchani_export_roundtrip(self, tmp_path):
        from src.wrappers.wrap_torchani import TorchANI_Wrapper
        from src.export import export_wrapped

        mock_path = self._save_mock(_MockTorchANIInner(), tmp_path, "mock_ani.pt")
        wrapper = TorchANI_Wrapper(
            model_path=mock_path, device="cpu", element_list=[1, 8],
        ).eval()

        out_path = str(tmp_path / "exported_ani.pt")
        export_wrapped(wrapper, out_path, model_type="TorchANI")

        reloaded = torch.jit.load(out_path)
        coords = torch.tensor([
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ], dtype=torch.float64)
        Z = torch.tensor([8, 1, 1], dtype=torch.int64)
        pc = torch.zeros((0, 3), dtype=torch.float64)
        pq = torch.zeros(0, dtype=torch.float64)

        energy, forces, charges = reloaded(coords, Z, pc, pq)
        assert energy.dtype == torch.float64
        assert forces.shape == (3, 3)
        assert charges.shape == (3,)

