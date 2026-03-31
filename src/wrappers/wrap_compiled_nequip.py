"""
NAMD-compatible wrapper for TorchScript-deployed **NequIP** and **Allegro** models.

Both frameworks share the same deployed-model I/O convention (the NequIP
``AtomicData`` dict), so a single wrapper class covers both.  Models are
deployed to TorchScript via::

    nequip-deploy build --train-dir <dir> deployed.pth

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges)
        -> (energy_kcal, forces_kcal_A, charges_e)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges)
        -> (energies, forces, charges)

Key differences from MACE:
  * Input dict uses NequIP keys (``pos``, ``atom_types``,
    ``edge_cell_shift``, …) instead of MACE keys.
  * ``atom_types`` are **0-indexed type IDs**, not raw atomic numbers.
    The wrapper builds the Z → type-index lookup at ``__init__`` time.
  * No ``node_attrs`` one-hot tensor is needed.
  * The deployed model is called with positional args, not keyword flags.
"""

import argparse

import torch
from torch import nn
from typing import Dict

from ..constants import EV_TO_KCAL
from ..edges import build_edges, build_edges_batched
from ..export import export_wrapped


# -------------------------------------------------------------------
#  Helpers to extract metadata from a NequIP-deployed model
# -------------------------------------------------------------------

def _get_r_max(model: torch.jit.ScriptModule) -> float:
    """Extract cutoff from a nequip-deployed model."""
    if hasattr(model, "r_max"):
        return float(model.r_max)
    raise RuntimeError(
        "Deployed NequIP/Allegro model has no 'r_max' attribute. "
        "Make sure you used `nequip-deploy build`."
    )


def _get_type_map(model: torch.jit.ScriptModule) -> Dict[int, int]:
    """
    Build a mapping ``{atomic_number: type_index}`` from the deployed
    model's metadata.

    NequIP stores ``type_names`` (e.g. ``['H', 'C', 'N', 'O']``) and
    optionally ``chemical_symbol_to_type`` on the deployed model.
    We also accept a plain ``atomic_numbers`` list (like MACE stores).
    """
    # Try the attribute that nequip-deploy writes (list of atomic numbers
    # corresponding to type indices 0, 1, 2, …).
    if hasattr(model, "atomic_numbers"):
        z_list = [int(z) for z in model.atomic_numbers]
        return {z: i for i, z in enumerate(z_list)}

    # Fallback: chemical_symbol_to_type stored as a string attr.
    # Not always present – raise informative error if missing.
    raise RuntimeError(
        "Cannot determine type map from deployed model. "
        "Expected an 'atomic_numbers' attribute on the model."
    )


class NequIP_Allegro_Wrapper(nn.Module):
    """
    Wrap a NequIP- or Allegro-deployed TorchScript model for NAMD.

    The same class works for both frameworks because ``nequip-deploy``
    produces models with an identical forward signature.

    Args:
        deployed_path:  Path to the ``.pth`` produced by ``nequip-deploy``.
        device:         ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, deployed_path: str, device: str = "cpu"):
        super().__init__()

        self.inner = torch.jit.load(deployed_path, map_location=device)
        self.inner.eval()

        self.r_max: float = _get_r_max(self.inner)

        # Build Z → 0-indexed type-index lookup tensor.
        # z_to_type_map[z] = type_index  (size = max_z + 1).
        type_map = _get_type_map(self.inner)
        max_z = max(type_map.keys())
        z_to_type = torch.full((max_z + 1,), -1, dtype=torch.long)
        for z, idx in type_map.items():
            z_to_type[z] = idx
        self.z_to_type = z_to_type

        # Conversion factor
        self.ev_to_kcal = torch.tensor(EV_TO_KCAL, dtype=torch.float64)

        # Cached tensors (populated on first forward call)
        self._cached_N: int = -1
        self._cached_batch: torch.Tensor = torch.empty(0)
        self._cached_ptr:   torch.Tensor = torch.empty(0)
        self._cached_cell:  torch.Tensor = torch.empty(0)

        self.supports_batch: bool = True

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _z_to_types(self, Z: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers [N] to 0-indexed type indices [N]."""
        z_map = self.z_to_type.to(Z.device)
        return z_map[Z]

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
        atom_types = self._z_to_types(Z)

        # Cache constant tensors
        if N != self._cached_N:
            self._cached_N = N
            self._cached_batch = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_ptr   = torch.tensor([0, N], dtype=torch.long, device=dev)
            self._cached_cell  = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        # Edge construction in FP32
        edge_index, edge_vecs32, _ = build_edges(coords32, self.r_max)
        edge_cell_shift = torch.zeros(
            (edge_index.size(1), 3), dtype=torch.float64, device=dev,
        )

        # NequIP AtomicData dict
        data: Dict[str, torch.Tensor] = {
            "pos":             coords,
            "edge_index":      edge_index,
            "atom_types":      atom_types,
            "edge_cell_shift": edge_cell_shift,
            "cell":            self._cached_cell,
            "batch":           self._cached_batch,
            "ptr":             self._cached_ptr,
        }

        out = self.inner(data)

        energy_raw = out["total_energy"]
        forces_raw = out["forces"]

        if energy_raw is None:
            raise RuntimeError("Deployed model returned total_energy=None")
        if forces_raw is None:
            raise RuntimeError("Deployed model returned forces=None")

        ev2kcal = self.ev_to_kcal.to(dev)
        energy  = energy_raw.to(torch.float64) * ev2kcal
        forces  = forces_raw.to(torch.float64) * ev2kcal
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
        Evaluate NequIP/Allegro for a batch of molecules.

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

        coords32 = coords.to(torch.float32)
        Z = Z.to(torch.int64)
        atom_types = self._z_to_types(Z)

        cell = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        # Block-diagonal edge construction
        edge_index, edge_vecs32, _ = build_edges_batched(
            coords32, ptr, self.r_max,
        )
        edge_cell_shift = torch.zeros(
            (edge_index.size(1), 3), dtype=torch.float64, device=dev,
        )

        data: Dict[str, torch.Tensor] = {
            "pos":             coords,
            "edge_index":      edge_index,
            "atom_types":      atom_types,
            "edge_cell_shift": edge_cell_shift,
            "cell":            cell,
            "batch":           batch,
            "ptr":             ptr,
        }

        out = self.inner(data)

        energy_raw = out["total_energy"]
        forces_raw = out["forces"]

        if energy_raw is None:
            raise RuntimeError("Deployed model returned total_energy=None")
        if forces_raw is None:
            raise RuntimeError("Deployed model returned forces=None")

        ev2kcal  = self.ev_to_kcal.to(dev)
        energies = energy_raw.to(torch.float64) * ev2kcal
        forces   = forces_raw.to(torch.float64) * ev2kcal
        charges  = torch.zeros(N_total, dtype=torch.float64, device=dev)

        return energies, forces, charges


# -------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a wrapped NequIP/Allegro model for NAMD",
    )
    parser.add_argument("--deployed", required=True,
                        help="Path to nequip-deploy .pth file")
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", default="NequIP/Allegro",
                        help="Model label for diagnostics")

    args = parser.parse_args()

    wrapper = NequIP_Allegro_Wrapper(args.deployed, device=args.device).eval()
    export_wrapped(wrapper, args.out, model_type=args.label)

