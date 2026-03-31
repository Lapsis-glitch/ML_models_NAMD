"""
NAMD-compatible wrapper for **SchNetPack** (≥ 2.0) models.

SchNetPack models are exported via ``torch.jit.script`` (or loaded from
a TorchScript archive produced during training).  The wrapper translates
NAMD's ``(coords, Z)`` interface into SchNetPack's input dict format and
converts the output to kcal/mol.

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges)
        -> (energy_kcal, forces_kcal_A, charges_e)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges)
        -> (energies, forces, charges)

SchNetPack specifics:
  * Input dict uses ``_positions``, ``_atomic_numbers``, ``_idx_i``,
    ``_idx_j``, ``_offsets``, ``_cell``, ``_n_atoms``, ``_idx_m`` keys.
  * The neighbor list is provided as COO index pairs (``_idx_i``,
    ``_idx_j``) — same topology as our shared ``build_edges`` but with
    different key names.
  * Default native units are eV (configurable at training time).
    The wrapper accepts a ``units`` parameter to handle non-default
    training unit systems.
"""

import argparse

import torch
from torch import nn
from typing import Dict

from ..constants import EV_TO_KCAL
from ..edges import build_edges, build_edges_batched
from ..export import export_wrapped


class SchNetPack_Wrapper(nn.Module):
    """
    Wrap a SchNetPack (≥ 2.0) TorchScript model for NAMD.

    Args:
        model_path:       Path to a scripted SchNetPack ``.pt`` file.
        r_max:            Cutoff radius in Å.  Must match the cutoff
                          used during training.
        device:           ``"cpu"`` or ``"cuda"``.
        energy_key:       Key in the model's output dict for energy.
        forces_key:       Key in the model's output dict for forces.
        energy_units_to_kcal:
            Multiplicative factor to convert the model's native energy
            unit to kcal/mol.  Default ``EV_TO_KCAL`` (23.0621).
    """

    def __init__(
        self,
        model_path: str,
        r_max: float,
        device: str = "cpu",
        energy_key: str = "energy",
        forces_key: str = "forces",
        energy_units_to_kcal: float = EV_TO_KCAL,
    ):
        super().__init__()

        self.inner = torch.jit.load(model_path, map_location=device)
        self.inner.eval()

        self.r_max: float = r_max
        self.energy_key: str = energy_key
        self.forces_key: str = forces_key

        self.conv_factor = torch.tensor(
            energy_units_to_kcal, dtype=torch.float64,
        )

        # Cached tensors
        self._cached_N: int = -1
        self._cached_idx_m: torch.Tensor = torch.empty(0)
        self._cached_n_atoms: torch.Tensor = torch.empty(0)
        self._cached_offsets_dummy: torch.Tensor = torch.empty(0)
        self._cached_cell: torch.Tensor = torch.empty(0)

        self.supports_batch: bool = True

    # -----------------------------------------------------------------
    #  Forward (single molecule)
    # -----------------------------------------------------------------

    def forward(
        self,
        coords: torch.Tensor,
        Z: torch.Tensor,
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
    ):
        dev = coords.device
        N = coords.size(0)

        coords32 = coords.to(torch.float32)
        Z = Z.to(torch.int64)

        # Cache invariant tensors
        if N != self._cached_N:
            self._cached_N = N
            self._cached_idx_m   = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_n_atoms = torch.tensor([N], dtype=torch.long, device=dev)
            self._cached_cell    = torch.zeros((3, 3), dtype=torch.float32, device=dev)

        # Build edges (FP32)
        edge_index, edge_vecs32, _ = build_edges(coords32, self.r_max)
        idx_i = edge_index[0]
        idx_j = edge_index[1]

        # SchNetPack offsets (zeros for non-periodic)
        offsets = torch.zeros(
            (edge_index.size(1), 3), dtype=torch.float32, device=dev,
        )

        inputs: Dict[str, torch.Tensor] = {
            "_positions":       coords32,
            "_atomic_numbers":  Z,
            "_idx_i":           idx_i,
            "_idx_j":           idx_j,
            "_offsets":         offsets,
            "_cell":            self._cached_cell,
            "_n_atoms":         self._cached_n_atoms,
            "_idx_m":           self._cached_idx_m,
        }

        out = self.inner(inputs)

        energy_raw = out[self.energy_key]
        forces_raw = out[self.forces_key]

        if energy_raw is None:
            raise RuntimeError("SchNetPack model returned energy=None")
        if forces_raw is None:
            raise RuntimeError("SchNetPack model returned forces=None")

        conv = self.conv_factor.to(dev)
        # SchNetPack often returns energy as [B, 1]; squeeze to scalar / [B].
        energy  = energy_raw.to(torch.float64).squeeze() * conv
        forces  = forces_raw.to(torch.float64) * conv
        charges = torch.zeros(N, dtype=torch.float64, device=dev)

        return energy, forces, charges

    # -----------------------------------------------------------------
    #  Batched forward
    # -----------------------------------------------------------------

    @torch.jit.export
    def forward_batch(
        self,
        coords: torch.Tensor,
        Z: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
    ):
        """
        Evaluate SchNetPack for a batch of molecules.

        Args:
            coords:     [N_total, 3]  float64  concatenated positions.
            Z:          [N_total]     int64    atomic numbers.
            batch:      [N_total]     int64    molecule index per atom.
            ptr:        [B+1]         int64    molecule boundaries.
            pc_coords:  [P, 3]        float64  (ignored).
            pc_charges: [P]           float64  (ignored).

        Returns:
            energies:  [B]           float64  kcal/mol.
            forces:    [N_total, 3]  float64  kcal/mol/Å.
            charges:   [N_total]     float64  e.
        """
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1

        coords32 = coords.to(torch.float32)
        Z = Z.to(torch.int64)

        cell = torch.zeros((3, 3), dtype=torch.float32, device=dev)

        # Per-molecule atom counts
        n_atoms = ptr[1:] - ptr[:-1]  # [B]

        # Block-diagonal edges
        edge_index, _, _ = build_edges_batched(coords32, ptr, self.r_max)
        idx_i = edge_index[0]
        idx_j = edge_index[1]
        offsets = torch.zeros(
            (edge_index.size(1), 3), dtype=torch.float32, device=dev,
        )

        inputs: Dict[str, torch.Tensor] = {
            "_positions":       coords32,
            "_atomic_numbers":  Z,
            "_idx_i":           idx_i,
            "_idx_j":           idx_j,
            "_offsets":         offsets,
            "_cell":            cell,
            "_n_atoms":         n_atoms,
            "_idx_m":           batch,
        }

        out = self.inner(inputs)

        energy_raw = out[self.energy_key]
        forces_raw = out[self.forces_key]

        if energy_raw is None:
            raise RuntimeError("SchNetPack model returned energy=None")
        if forces_raw is None:
            raise RuntimeError("SchNetPack model returned forces=None")

        conv     = self.conv_factor.to(dev)
        # SchNetPack often returns energy as [B, 1]; squeeze to [B].
        energies = energy_raw.to(torch.float64).squeeze(-1) * conv
        forces   = forces_raw.to(torch.float64) * conv
        charges  = torch.zeros(N_total, dtype=torch.float64, device=dev)

        return energies, forces, charges


# -------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a wrapped SchNetPack model for NAMD",
    )
    parser.add_argument("--model", required=True,
                        help="Path to scripted SchNetPack .pt file")
    parser.add_argument("--r-max", type=float, required=True,
                        help="Cutoff radius in Å (must match training)")
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--energy-key", default="energy",
                        help="Output dict key for energy")
    parser.add_argument("--forces-key", default="forces",
                        help="Output dict key for forces")

    args = parser.parse_args()

    wrapper = SchNetPack_Wrapper(
        model_path=args.model,
        r_max=args.r_max,
        device=args.device,
        energy_key=args.energy_key,
        forces_key=args.forces_key,
    ).eval()
    export_wrapped(wrapper, args.out, model_type="SchNetPack")

