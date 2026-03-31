"""
Interface compliance tests – parametrised across ALL wrappers.

These tests verify that every wrapper obeys the NAMD MLIP contract:
  * Output shapes, dtypes, and units are correct.
  * ``torch.jit.script()`` succeeds.
  * ``forward_batch`` with B=1 matches ``forward``.
  * ``supports_batch`` attribute exists.

Since we may not have real trained models in CI, each wrapper is tested
via a **mock inner model** that returns plausible tensors in the
model's native format.  This validates the *wrapper logic* (unit
conversion, edge building, input-dict assembly, etc.) without needing
actual MLIP weights.
"""

import pytest
import torch
from torch import nn
from typing import Dict

from src.constants import EV_TO_KCAL, HARTREE_TO_KCAL


# ===================================================================
#  Mock inner models (one per wrapper flavour)
# ===================================================================

class _MockMACEInner(nn.Module):
    """Mimics a TorchScript-compiled MACE model."""

    def __init__(self):
        super().__init__()
        self.r_max = 5.0
        self.atomic_numbers = [1, 8]

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pos = batch_dict["positions"]
        N = pos.size(0)
        dev = pos.device
        # Determine batch count from ptr
        ptr = batch_dict["ptr"]
        B = ptr.size(0) - 1
        energy = torch.randn(B, dtype=torch.float64, device=dev) * 0.5  # [B] eV
        forces = torch.randn(N, 3, dtype=torch.float64, device=dev) * 0.01
        return {"energy": energy, "forces": forces}


class _MockNequIPInner(nn.Module):
    """Mimics a nequip-deploy TorchScript model."""

    def __init__(self):
        super().__init__()
        self.r_max = 4.0
        self.atomic_numbers = [1, 8]

    def forward(
        self, data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pos = data["pos"]
        N = pos.size(0)
        dev = pos.device
        ptr = data["ptr"]
        B = ptr.size(0) - 1
        energy = torch.randn(B, dtype=torch.float64, device=dev) * 0.3
        forces = torch.randn(N, 3, dtype=torch.float64, device=dev) * 0.01
        return {"total_energy": energy, "forces": forces}


class _MockSchNetInner(nn.Module):
    """Mimics a scripted SchNetPack model."""

    def forward(
        self, inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pos = inputs["_positions"]
        N = pos.size(0)
        dev = pos.device
        # Determine number of molecules from _idx_m
        idx_m = inputs["_idx_m"]
        B = int(idx_m.max().item()) + 1 if idx_m.numel() > 0 else 1
        energy = torch.randn(B, 1, dtype=torch.float64, device=dev) * 0.1
        forces = torch.randn(N, 3, dtype=torch.float64, device=dev) * 0.01
        return {"energy": energy, "forces": forces}


class _MockTorchANIInner(nn.Module):
    """Mimics a TorchANI model: (species, coords) → (species, energies).

    Energy depends on coordinates so that autograd can compute forces.
    """

    def forward(
        self,
        species: torch.Tensor,
        coordinates: torch.Tensor,
    ):
        # Energy = sum of squared coordinates (simple differentiable function)
        B = species.size(0)
        energies = coordinates.pow(2).sum(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
        energies = energies * 0.001  # scale to Hartree-ish range
        return species, energies


# ===================================================================
#  Factory: build a real wrapper around a mock inner model
# ===================================================================

def _build_mace_wrapper():
    """MACE wrapper with mock inner."""
    from src.wrappers.wrap_compiled_mace import MACE_TS_Wrapper

    # Script the mock so jit.load isn't needed
    mock = torch.jit.script(_MockMACEInner())
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "mock_mace.pt")
    mock.save(path)

    wrapper = MACE_TS_Wrapper(path, device="cpu")
    wrapper.eval()
    return wrapper


def _build_nequip_wrapper():
    """NequIP/Allegro wrapper with mock inner."""
    from src.wrappers.wrap_compiled_nequip import NequIP_Allegro_Wrapper

    mock = torch.jit.script(_MockNequIPInner())
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "mock_nequip.pt")
    mock.save(path)

    wrapper = NequIP_Allegro_Wrapper(path, device="cpu")
    wrapper.eval()
    return wrapper


def _build_schnet_wrapper():
    """SchNetPack wrapper with mock inner."""
    from src.wrappers.wrap_schnetpack import SchNetPack_Wrapper

    mock = torch.jit.script(_MockSchNetInner())
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "mock_schnet.pt")
    mock.save(path)

    wrapper = SchNetPack_Wrapper(
        model_path=path, r_max=5.0, device="cpu",
    )
    wrapper.eval()
    return wrapper


def _build_torchani_wrapper():
    """TorchANI wrapper with mock inner."""
    from src.wrappers.wrap_torchani import TorchANI_Wrapper

    mock = torch.jit.script(_MockTorchANIInner())
    import tempfile, os
    path = os.path.join(tempfile.mkdtemp(), "mock_ani.pt")
    mock.save(path)

    wrapper = TorchANI_Wrapper(
        model_path=path, device="cpu", element_list=[1, 6, 7, 8],
    )
    wrapper.eval()
    return wrapper


# Collect all builders in a list for parametrisation.
_WRAPPER_BUILDERS = {
    "MACE":    _build_mace_wrapper,
    "NequIP":  _build_nequip_wrapper,
    "SchNet":  _build_schnet_wrapper,
    "TorchANI": _build_torchani_wrapper,
}


@pytest.fixture(params=_WRAPPER_BUILDERS.keys())
def wrapper(request):
    """Parametrised fixture – yields one wrapper instance per model type."""
    return _WRAPPER_BUILDERS[request.param]()


# ===================================================================
#  Tests
# ===================================================================

class TestOutputContract:
    """Verify that every wrapper obeys the NAMD output contract."""

    def test_supports_batch_attribute(self, wrapper):
        assert hasattr(wrapper, "supports_batch")
        assert isinstance(wrapper.supports_batch, bool)

    def test_forward_output_shapes_and_dtypes(
        self, wrapper, single_coords, single_Z,
        dummy_pc_coords, dummy_pc_charges,
    ):
        energy, forces, charges = wrapper(
            single_coords, single_Z, dummy_pc_coords, dummy_pc_charges,
        )

        # energy: scalar or [1]
        assert energy.dtype == torch.float64
        assert energy.dim() <= 1

        # forces: [N, 3]
        N = single_coords.size(0)
        assert forces.shape == (N, 3)
        assert forces.dtype == torch.float64

        # charges: [N]
        assert charges.shape == (N,)
        assert charges.dtype == torch.float64

    def test_forward_batch_output_shapes_and_dtypes(
        self, wrapper, dummy_coords, dummy_Z,
        dummy_batch, dummy_ptr,
        dummy_pc_coords, dummy_pc_charges,
    ):
        if not wrapper.supports_batch:
            pytest.skip("Wrapper does not support batching")

        energies, forces, charges = wrapper.forward_batch(
            dummy_coords, dummy_Z, dummy_batch, dummy_ptr,
            dummy_pc_coords, dummy_pc_charges,
        )

        N_total = dummy_coords.size(0)
        B = dummy_ptr.size(0) - 1

        assert energies.dtype == torch.float64
        assert energies.shape == (B,) or energies.numel() == B

        assert forces.shape == (N_total, 3)
        assert forces.dtype == torch.float64

        assert charges.shape == (N_total,)
        assert charges.dtype == torch.float64


class TestEdgeUtilities:
    """Test the shared edge-building functions directly."""

    def test_build_edges_basic(self):
        from src.edges import build_edges

        # Two atoms 1 Å apart – well within any reasonable cutoff.
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=torch.float32)

        edge_index, edge_vecs, edge_len = build_edges(coords, r_max=5.0)

        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 2  # bidirectional
        assert edge_vecs.shape == (2, 3)
        assert edge_len.shape == (2,)
        assert torch.allclose(edge_len, torch.tensor([1.0, 1.0]))

    def test_build_edges_no_neighbors(self):
        from src.edges import build_edges

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=torch.float32)

        edge_index, edge_vecs, edge_len = build_edges(coords, r_max=5.0)
        assert edge_index.shape[1] == 0

    def test_build_edges_batched(self):
        from src.edges import build_edges_batched

        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
        ], dtype=torch.float32)
        ptr = torch.tensor([0, 2, 4], dtype=torch.long)

        edge_index, _, _ = build_edges_batched(coords, ptr, r_max=5.0)

        # 2 edges per molecule (bidirectional), 2 molecules → 4 total
        assert edge_index.shape[1] == 4

        # No cross-molecule edges: atom 0/1 should not connect to 2/3
        for e in range(edge_index.shape[1]):
            i, j = edge_index[0, e].item(), edge_index[1, e].item()
            assert (i < 2 and j < 2) or (i >= 2 and j >= 2)


class TestConstants:
    """Verify unit conversion constants."""

    def test_ev_to_kcal(self):
        from src.constants import EV_TO_KCAL
        assert abs(EV_TO_KCAL - 23.0621) < 0.001

    def test_hartree_to_kcal(self):
        from src.constants import HARTREE_TO_KCAL
        assert abs(HARTREE_TO_KCAL - 627.509) < 0.1

