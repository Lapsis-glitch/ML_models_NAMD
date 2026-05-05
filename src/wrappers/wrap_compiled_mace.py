"""
NAMD-compatible wrapper for TorchScript-compiled **MACE** models.

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges)
        -> (energy_kcal, forces_kcal_A, charges_e)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges)
        -> (energies, forces, charges)

Performance optimisations:
  1. FP32 edge building for better throughput.
  2. Cached constant tensors (batch, ptr, cell, node_attrs).
  3. Fused eV → kcal/mol conversion on-device.
"""

import argparse

import torch
from torch import nn

from ..constants import EV_TO_KCAL
from ..edges import build_edges, build_edges_batched
from ..export import export_wrapped


class MACE_TS_Wrapper(nn.Module):
    """
    Wrap a TorchScript-compiled MACE model for NAMD.

    Inputs (single molecule):
        coords:     [N, 3] float64
        Z:          [N]    int64
        pc_coords:  [P, 3] float64   (ignored – kept for interface compat.)
        pc_charges: [P]    float64   (ignored)
    """

    def __init__(self, compiled_path: str, device: str = "cpu"):
        super().__init__()

        # Load compiled TorchScript model
        self.inner = torch.jit.load(compiled_path, map_location=device)
        self.inner.eval()

        # Cutoff
        if hasattr(self.inner, "r_max"):
            r = self.inner.r_max
            self.r_max = float(r) if not isinstance(r, float) else r
        else:
            raise RuntimeError("Compiled model has no r_max attribute")

        # Model's element ordering (TorchScript-friendly)
        self.atomic_numbers = torch.tensor(
            [int(z) for z in self.inner.atomic_numbers],
            dtype=torch.int64,
        )
        self.n_elements = int(self.atomic_numbers.numel())

        # --- Cached tensors (populated on first forward() call) ---
        self._cached_N: int = -1
        self._cached_Z:          torch.Tensor = torch.empty(0, dtype=torch.int64)
        self._cached_batch:      torch.Tensor = torch.empty(0)
        self._cached_ptr:        torch.Tensor = torch.empty(0)
        self._cached_num_nodes:  torch.Tensor = torch.empty(0)
        self._cached_cell:       torch.Tensor = torch.empty(0)
        self._cached_node_attrs: torch.Tensor = torch.empty(0)

        # Conversion factor on-device
        self.ev_to_kcal = torch.tensor(EV_TO_KCAL, dtype=torch.float64)

        # Flag checked by the C++ side to enable batched dispatch.
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

        # Cached constant tensors — keyed on both N and Z.  Different
        # molecules with the same atom count but different elements must
        # get fresh node_attrs, otherwise MACE sees stale one-hots.
        shape_changed = N != self._cached_N
        z_changed = shape_changed or not torch.equal(Z, self._cached_Z)

        if shape_changed:
            self._cached_N = N
            self._cached_batch     = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_ptr       = torch.tensor([0, N], dtype=torch.long, device=dev)
            self._cached_num_nodes = torch.tensor([N], dtype=torch.long, device=dev)
            self._cached_cell      = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        if z_changed:
            self._cached_Z = Z.clone()
            atomic_numbers_dev = self.atomic_numbers.to(dev)
            match = Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)
            self._cached_node_attrs = match.to(torch.float64)

        # FP32 edge construction, cast back to float64 for the model.
        edge_index, edge_vecs32, edge_len32 = build_edges(coords32, self.r_max)
        edge_vecs = edge_vecs32.to(torch.float64)
        edge_len  = edge_len32.to(torch.float64)
        shifts = torch.zeros_like(edge_vecs)

        batch_dict = {
            "positions":      coords,
            "atomic_numbers": Z,
            "node_attrs":     self._cached_node_attrs,
            "edge_index":     edge_index,
            "edge_vectors":   edge_vecs,
            "edge_lengths":   edge_len,
            "shifts":         shifts,
            "cell":           self._cached_cell,
            "batch":          self._cached_batch,
            "ptr":            self._cached_ptr,
            "num_nodes":      self._cached_num_nodes,
        }

        out = self.inner(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
        )

        energy_opt = out.get("energy")
        forces_opt = out.get("forces")

        ev2kcal = self.ev_to_kcal.to(dev)

        if energy_opt is not None:
            energy = energy_opt.to(torch.float64) * ev2kcal
        else:
            energy = torch.zeros(1, dtype=torch.float64, device=dev)

        if forces_opt is not None:
            forces = forces_opt.to(torch.float64) * ev2kcal
        else:
            forces = torch.zeros(N, 3, dtype=torch.float64, device=dev)

        charges_opt = out.get("charges")
        if charges_opt is not None:
            charges = charges_opt.to(torch.float64)
        else:
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
        Evaluate MACE for a batch of molecules in one GPU kernel launch.

        Args:
            coords:     [N_total, 3]  float64  concatenated positions.
            Z:          [N_total]     int64    concatenated atomic numbers.
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

        # One-hot node attributes for all atoms.
        atomic_numbers_dev = self.atomic_numbers.to(dev)
        match = Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)
        node_attrs = match.to(torch.float64)

        num_nodes = torch.tensor([N_total], dtype=torch.long, device=dev)
        cell = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        # Block-diagonal edge construction.
        edge_index, edge_vecs32, edge_len32 = build_edges_batched(
            coords32, ptr, self.r_max,
        )
        edge_vecs = edge_vecs32.to(torch.float64)
        edge_len  = edge_len32.to(torch.float64)
        shifts = torch.zeros_like(edge_vecs)

        batch_dict = {
            "positions":      coords,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "edge_vectors":   edge_vecs,
            "edge_lengths":   edge_len,
            "shifts":         shifts,
            "cell":           cell,
            "batch":          batch,
            "ptr":            ptr,
            "num_nodes":      num_nodes,
        }

        out = self.inner(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
        )

        energy_opt = out.get("energy")
        forces_opt = out.get("forces")

        ev2kcal  = self.ev_to_kcal.to(dev)

        if energy_opt is not None:
            energies = energy_opt.to(torch.float64) * ev2kcal
        else:
            energies = torch.zeros(ptr.size(0) - 1, dtype=torch.float64, device=dev)

        if forces_opt is not None:
            forces = forces_opt.to(torch.float64) * ev2kcal
        else:
            forces = torch.zeros(N_total, 3, dtype=torch.float64, device=dev)

        charges_opt = out.get("charges")
        if charges_opt is not None:
            charges = charges_opt.to(torch.float64)
        else:
            charges = torch.zeros(N_total, dtype=torch.float64, device=dev)

        return energies, forces, charges


# -------------------------------------------------------------------
#  CLI entry point (kept for backwards compat; prefer src/cli.py)
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a wrapped MACE TorchScript model for NAMD",
    )
    parser.add_argument("--compiled", required=True,
                        help="Path to compiled MACE .pt file")
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    wrapper = MACE_TS_Wrapper(args.compiled, device=args.device).eval()
    export_wrapped(wrapper, args.out, model_type="MACE")
