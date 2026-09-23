"""
NAMD-compatible wrapper for TorchScript-compiled **MACE** models.

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges, cell)
        -> (energy_kcal, forces_kcal_A, charges_e, virial_kcal)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        -> (energies, forces, charges, virials)

The virial is only meaningful for a periodic box, so it is zero whenever the
cell is.  NAMD forms its own virial by summing f_i (x) r_i over absolute
positions, which counts the wrong displacement for any pair whose interaction
crosses a box face.  The strain derivative MACE can supply instead has no such
problem, because it differentiates the energy with respect to a deformation of
the box itself and so sees the imaged separations rather than the raw ones.

Performance optimisations:
  1. FP32 edge building for better throughput.
  2. Cached constant tensors (batch, ptr, cell, node_attrs).
  3. Fused eV → kcal/mol conversion on-device.
"""

import argparse

import torch
from torch import nn

from ..constants import EV_TO_KCAL
from ..edges import (build_edges, build_edges_batched, build_edges_pbc,
                     build_edges_batched_pbc, cell_is_periodic)
from ..export import export_wrapped


class MACE_TS_Wrapper(nn.Module):
    """
    Wrap a TorchScript-compiled MACE model for NAMD.

    Inputs (single molecule):
        coords:     [N, 3] float64
        Z:          [N]    int64
        pc_coords:  [P, 3] float64   (ignored – kept for interface compat.)
        pc_charges: [P]    float64   (ignored)
        cell:       [1, 3, 3] or [3, 3] float64, lattice vectors as rows,
                    all zeros for a non-periodic system.
    """

    def __init__(self, compiled_path: str, device: str = "cpu"):
        super().__init__()

        # Load compiled TorchScript model
        self.inner = torch.jit.load(compiled_path, map_location=device)
        self.inner.eval()

        try:
            first_param = next(self.inner.parameters())
            self.model_uses_fp32: bool = first_param.dtype == torch.float32
        except StopIteration:
            self.model_uses_fp32 = False

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
        # Keyed on atom count N only; node_attrs is recomputed each step
        # (sync-free) rather than cached against Z.
        self._cached_N: int = -1
        self._cached_batch:      torch.Tensor = torch.empty(0)
        self._cached_ptr:        torch.Tensor = torch.empty(0)
        self._cached_num_nodes:  torch.Tensor = torch.empty(0)
        self._cached_cell:       torch.Tensor = torch.empty(0)

        # Conversion factor on-device
        self.ev_to_kcal = torch.tensor(EV_TO_KCAL, dtype=torch.float64)

        # Flag checked by the C++ side to enable batched dispatch.
        self.supports_batch: bool = True

        # Says this model takes a cell argument.  NAMD works this out from the
        # forward() signature anyway and only uses the flag to cross-check, so
        # if the two ever disagree the model fails to load rather than running
        # with a mishandled cell.
        self.supports_pbc: bool = True

    def _model_float_dtype(self):
        if self.model_uses_fp32:
            return torch.float32
        return torch.float64

    # -----------------------------------------------------------------
    #  Forward (single molecule)
    # -----------------------------------------------------------------

    def forward(
        self,
        coords: torch.Tensor,
        Z: torch.Tensor,
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
        cell: torch.Tensor,
    ):
        dev = coords.device
        N = coords.size(0)
        model_dtype = self._model_float_dtype()

        coords32 = coords.to(torch.float32)
        coords_model = coords.to(model_dtype)
        Z = Z.to(torch.int64)

        # NAMD sends the box as [1, 3, 3], all zeros when the system is not
        # periodic.  Anything else would be a caller bug rather than something
        # to guess about, so just take the first entry.
        cell3 = cell.reshape(-1, 3, 3)[0].to(model_dtype)
        periodic = cell_is_periodic(cell3)

        # Constant tensors keyed on atom count N only (allocation-time
        # caching, no host sync).  The old code also keyed on Z via a
        # per-step torch.equal(Z, ...) — a Python bool that forces a
        # GPU→CPU sync every MD step.  node_attrs is instead recomputed
        # below: it is cheap, sync-free, and always correct.
        if N != self._cached_N:
            self._cached_N = N
            self._cached_batch     = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_ptr       = torch.tensor([0, N], dtype=torch.long, device=dev)
            self._cached_num_nodes = torch.tensor([N], dtype=torch.long, device=dev)
            self._cached_cell      = torch.zeros((3, 3), dtype=model_dtype, device=dev)

        # One-hot node attributes — always recomputed (no sync, no caching).
        atomic_numbers_dev = self.atomic_numbers.to(dev)
        node_attrs = (Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)).to(model_dtype)

        # FP32 edge construction, then cast to the model's float dtype.
        # MACE reads its displacements as pos[edge_index[1]] - pos[edge_index[0]]
        # + shifts, with shifts in Angstrom, so that is what we hand it.
        if periodic:
            edge_index, edge_vecs32, edge_len32, unit_shifts32 = build_edges_pbc(
                coords32, cell3.to(torch.float32), self.r_max
            )
            unit_shifts = unit_shifts32.to(model_dtype)
            shifts = unit_shifts @ cell3
            cell_in = cell3
        else:
            edge_index, edge_vecs32, edge_len32 = build_edges(coords32, self.r_max)
            unit_shifts = torch.zeros(
                (edge_index.size(1), 3), dtype=model_dtype, device=dev
            )
            shifts = torch.zeros_like(unit_shifts)
            cell_in = self._cached_cell

        edge_vecs = edge_vecs32.to(model_dtype)
        edge_len  = edge_len32.to(model_dtype)

        batch_dict = {
            "positions":      coords_model,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "edge_vectors":   edge_vecs,
            "edge_lengths":   edge_len,
            "shifts":         shifts,
            # MACE recomputes shifts from unit_shifts whenever it is asked for
            # virials or stress, so these two have to be consistent with each
            # other and with the cell they came from.
            "unit_shifts":    unit_shifts,
            "cell":           cell_in,
            "batch":          self._cached_batch,
            "ptr":            self._cached_ptr,
            "num_nodes":      self._cached_num_nodes,
        }

        # The virial is asked for only under PBC.  MACE divides by det(cell) on
        # the way to a stress, and with a zero cell that division produces
        # infinities which its own clamp then turns into an all-zero answer, so
        # a non-periodic system would get a plausible-looking result that means
        # nothing.  Asking only when there is a real box avoids the whole thing
        # and keeps the cheaper code path for cluster calculations.
        out = self.inner(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=periodic,
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

        # Zero unless there is a box to strain.  NAMD ignores this output for a
        # cluster and falls back to its own sum, but the shape has to stay the
        # same either way because TorchScript allows only one return type.
        virial = torch.zeros((3, 3), dtype=torch.float64, device=dev)
        if periodic:
            virials_opt = out.get("virials")
            if virials_opt is not None:
                # Energy units, so the same eV to kcal/mol factor applies.  No
                # volume division: that would turn it into a stress.
                v = virials_opt.reshape(3, 3).to(torch.float64) * ev2kcal
                # The strain MACE differentiates against is symmetric by
                # construction, so this only removes round-off asymmetry.
                virial = 0.5 * (v + v.transpose(-1, -2))

        return energy, forces, charges, virial

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
        cells: torch.Tensor,
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
            cells:      [B, 3, 3]     one box per molecule, rows are lattice
                                      vectors; all zeros means non-periodic.

        Returns:
            energies:  [B]           float64  kcal/mol.
            forces:    [N_total, 3]  float64  kcal/mol/Å.
            charges:   [N_total]     float64  e.
            virials:   [B, 3, 3]     float64  kcal/mol, one per molecule,
                                     all zero when the batch is not periodic.
        """
        dev = coords.device
        N_total = coords.size(0)
        model_dtype = self._model_float_dtype()

        coords32 = coords.to(torch.float32)
        coords_model = coords.to(model_dtype)
        Z = Z.to(torch.int64)

        # One-hot node attributes for all atoms.
        atomic_numbers_dev = self.atomic_numbers.to(dev)
        match = Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)
        node_attrs = match.to(model_dtype)

        num_nodes = torch.tensor([N_total], dtype=torch.long, device=dev)

        B = ptr.size(0) - 1
        cells_model = cells.reshape(-1, 3, 3).to(model_dtype)
        # NAMD guarantees every walker in a batch agrees about periodicity, so
        # the first cell decides for the whole batch.
        periodic = cell_is_periodic(cells_model[0])

        # Block-diagonal edge construction.
        if periodic:
            edge_index, edge_vecs32, edge_len32, unit_shifts32 = build_edges_batched_pbc(
                coords32, ptr, cells_model.to(torch.float32), self.r_max,
            )
            unit_shifts = unit_shifts32.to(model_dtype)
            # Each edge is shifted by its OWN molecule's cell; the walkers'
            # boxes are not the same tensor.
            edge_cells = cells_model.index_select(0, batch.index_select(0, edge_index[0]))
            shifts = torch.einsum("ei,eij->ej", unit_shifts, edge_cells)
            cell = cells_model
        else:
            edge_index, edge_vecs32, edge_len32 = build_edges_batched(
                coords32, ptr, self.r_max,
            )
            unit_shifts = torch.zeros(
                (edge_index.size(1), 3), dtype=model_dtype, device=dev
            )
            shifts = torch.zeros_like(unit_shifts)
            cell = torch.zeros((B, 3, 3), dtype=model_dtype, device=dev)

        edge_vecs = edge_vecs32.to(model_dtype)
        edge_len  = edge_len32.to(model_dtype)

        batch_dict = {
            "positions":      coords_model,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "edge_vectors":   edge_vecs,
            "edge_lengths":   edge_len,
            "shifts":         shifts,
            "unit_shifts":    unit_shifts,
            "cell":           cell,
            "batch":          batch,
            "ptr":            ptr,
            "num_nodes":      num_nodes,
        }

        # Same reasoning as in forward(): a zero cell makes MACE's virial path
        # return zeros that look like an answer, so only ask when periodic.
        out = self.inner(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=periodic,
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

        # One virial per molecule, matching the one cell per molecule going in.
        virials = torch.zeros((B, 3, 3), dtype=torch.float64, device=dev)
        if periodic:
            virials_opt = out.get("virials")
            if virials_opt is not None:
                v = virials_opt.reshape(-1, 3, 3).to(torch.float64) * ev2kcal
                virials = 0.5 * (v + v.transpose(-1, -2))

        return energies, forces, charges, virials


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
