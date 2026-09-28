"""
NAMD-compatible wrapper for TorchScript-deployed **NequIP** and **Allegro** models.

Both frameworks share the same deployed-model I/O convention (the NequIP
``AtomicData`` dict), so a single wrapper class covers both.  Models are
deployed to TorchScript via::

    nequip-deploy build --train-dir <dir> deployed.pth

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges, cell)
        -> (energy_kcal, forces_kcal_A, charges_e, virial_kcal)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        -> (energies, forces, charges, virials)

The virial comes from the model itself.  NequIP differentiates the energy
against a strain of the box on every call and there is no flag to turn that
off, so the answer is already sitting in the output dict and reading it costs
nothing.  Finite difference against the energy puts the ``virial`` key at
-dE/d(strain) in eV, which is the sign NAMD wants, so all this wrapper does is
convert the unit and symmetrise.  See ``src/virial.py`` for the contract and
``tests/test_virial_nequip_ani.py`` for the measurement.

Key differences from MACE:
  * Input dict uses NequIP keys (``pos``, ``atom_types``,
    ``edge_cell_shift``, …) instead of MACE keys.
  * ``atom_types`` are **0-indexed type IDs**, not raw atomic numbers.
    The wrapper builds the Z → type-index lookup at ``__init__`` time.
  * No ``node_attrs`` one-hot tensor is needed.
  * The deployed model is called with positional args, not keyword flags.
  * ``edge_cell_shift`` holds the INTEGER cell offsets, not the Cartesian
    shift vectors.  NequIP multiplies them by the cell itself when it builds
    the edge vectors, so handing it ``unit_shifts @ cell`` (what MACE wants)
    would apply the cell twice.
"""

import argparse

import torch
from torch import nn
from typing import Dict, Optional

from ..constants import EV_TO_KCAL, SYMBOL_TO_Z
from ..edges import (build_edges, build_edges_batched, build_edges_pbc,
                     build_edges_batched_pbc, cell_is_periodic)
from ..export import export_wrapped
from ..virial import finalize, zero_virial, zero_virials


# -------------------------------------------------------------------
#  Helpers to extract metadata from a NequIP-deployed model
# -------------------------------------------------------------------

def _get_r_max(
    model: torch.jit.ScriptModule,
    override: Optional[float] = None,
) -> float:
    """Extract cutoff from a nequip-deployed model, or use *override*."""
    if override is not None:
        return float(override)
    if hasattr(model, "r_max"):
        return float(model.r_max)
    raise RuntimeError(
        "Deployed model has no 'r_max' attribute. Pass --r-max explicitly "
        "(NequIP-OAM-L: 6.0)."
    )


def _get_type_map(model: torch.jit.ScriptModule) -> Dict[int, int]:
    """
    Build a mapping ``{atomic_number: type_index}`` from the deployed
    model's metadata.

    Supports three flavors:
      1. ``atomic_numbers`` list  (old nequip-deploy / MACE-style)
      2. ``type_names`` list of element symbols  (NequIP 0.17 framework)
    """
    if hasattr(model, "atomic_numbers"):
        z_list = [int(z) for z in model.atomic_numbers]
        return {z: i for i, z in enumerate(z_list)}

    if hasattr(model, "type_names"):
        symbols = [str(s) for s in model.type_names]
        out: Dict[int, int] = {}
        for i, s in enumerate(symbols):
            if s not in SYMBOL_TO_Z:
                raise RuntimeError(f"Unknown element symbol in type_names: {s!r}")
            out[SYMBOL_TO_Z[s]] = i
        return out

    raise RuntimeError(
        "Cannot determine type map from deployed model. "
        "Expected an 'atomic_numbers' or 'type_names' attribute on the model."
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

    def __init__(
        self,
        deployed_path: str,
        device: str = "cpu",
        r_max: Optional[float] = None,
    ):
        super().__init__()

        self.inner = torch.jit.load(deployed_path, map_location=device)
        self.inner.eval()

        self.r_max: float = _get_r_max(self.inner, override=r_max)

        # Build Z → 0-indexed type-index lookup tensor.
        # z_to_type_map[z] = type_index  (size = max_z + 1).
        type_map = _get_type_map(self.inner)
        max_z = max(type_map.keys())
        z_to_type = torch.full((max_z + 1,), -1, dtype=torch.long)
        for z, idx in type_map.items():
            z_to_type[z] = idx
        self.z_to_type = z_to_type

        # NequIP 0.17 framework models lack 'r_max'/'atomic_numbers' attrs and
        # do not populate 'forces' in their output dict — we compute forces
        # via autograd and provide the extra input keys the new framework
        # expects.
        self._is_new_nequip: bool = not hasattr(self.inner, "r_max")

        # Conversion factor
        self.ev_to_kcal = torch.tensor(EV_TO_KCAL, dtype=torch.float64)

        # Cached tensors (populated on first forward call)
        self._cached_N: int = -1
        self._cached_batch: torch.Tensor = torch.empty(0)
        self._cached_ptr:   torch.Tensor = torch.empty(0)
        self._cached_cell:  torch.Tensor = torch.empty(0)

        self.supports_batch: bool = True

        # Says this model takes a cell argument.  NAMD reads the forward()
        # signature as well and refuses to load a model whose flag and
        # signature disagree, so the two must be kept in step.
        self.supports_pbc: bool = True

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _z_to_types(self, Z: torch.Tensor) -> torch.Tensor:
        """Convert atomic numbers [N] to 0-indexed type indices [N]."""
        z_map = self.z_to_type.to(Z.device)
        return z_map[Z]

    def _virial_from_model(
        self,
        out: Dict[str, torch.Tensor],
        cells: torch.Tensor,
        B: int,
        dev: torch.device,
    ) -> torch.Tensor:
        """
        Take the virial the model already worked out, as [B, 3, 3] kcal/mol.

        The strain derivative is part of NequIP's own forward pass, so nothing
        here re-runs the model.  Two spellings exist in the wild and they carry
        the same information: ``virial`` is what current deployments write, and
        older ones write ``stress``, which is the same quantity divided by minus
        the cell volume.  The second branch undoes that division.

        A deployment carrying neither key reports zeros, which NAMD reads as
        "no virial available" and answers by falling back to its own sum over
        absolute positions.  Only the legacy ``nequip-deploy`` branch can land
        there, and no such model was available to test it against.
        """
        ev2kcal = self.ev_to_kcal.to(dev)

        if "virial" in out:
            v = out["virial"].reshape(-1, 3, 3).to(torch.float64) * ev2kcal
            return finalize(v)

        if "stress" in out:
            # abs() because a left-handed cell has a negative determinant, and
            # the volume that relates a stress to a virial is a volume either
            # way round.
            vol = torch.linalg.det(cells.to(torch.float64)).abs()
            v = out["stress"].reshape(-1, 3, 3).to(torch.float64)
            return finalize(-v * vol.reshape(-1, 1, 1) * ev2kcal)

        return zero_virials(B, cells)

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
        Z = Z.to(torch.int64)
        atom_types = self._z_to_types(Z)

        # NAMD sends the box as [1, 3, 3], all zeros when the system is not
        # periodic.  Anything else would be a caller bug rather than something
        # to guess about, so just take the first entry.
        cell3 = cell.reshape(-1, 3, 3)[0]
        periodic = cell_is_periodic(cell3)

        if N != self._cached_N:
            self._cached_N = N
            self._cached_batch = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_ptr   = torch.tensor([0, N], dtype=torch.long, device=dev)
            self._cached_cell  = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        if self._is_new_nequip:
            # NequIP 0.17 framework: omit `edge_vectors` from the input dict
            # so the model takes its position-based force-computation branch
            # (which writes `forces` to the output dict via its own autograd).
            pos32 = coords.to(torch.float32)
            if periodic:
                cell32 = cell3.to(torch.float32)
                edge_index, _, _, unit_shifts = build_edges_pbc(
                    pos32, cell32, self.r_max,
                )
                # Straight through: NequIP does pos[j] - pos[i] and then adds
                # edge_cell_shift @ cell itself, so it wants the integer
                # offsets, not the Cartesian shift.
                edge_cell_shift = unit_shifts
                # This dict carries a `batch` key, and on that branch NequIP
                # index_selects the cell along dim 0, so the cell has to come
                # in with a leading batch dimension.
                cell_in = cell32.reshape(1, 3, 3)
            else:
                edge_index, _, _ = build_edges(pos32, self.r_max)
                E_np = edge_index.size(1)
                edge_cell_shift = torch.zeros(
                    (E_np, 3), dtype=torch.float32, device=dev,
                )
                cell_in = torch.zeros((1, 3, 3), dtype=torch.float32, device=dev)
            num_atoms = torch.tensor([N], dtype=torch.long, device=dev)

            ei = edge_index[0]
            ej = edge_index[1]
            keys = ei * N + ej
            rev_keys = ej * N + ei
            sorted_keys, sort_idx = torch.sort(keys)
            pos_in_sorted = torch.searchsorted(sorted_keys, rev_keys)
            edge_transpose_perm = sort_idx[pos_in_sorted]

            data: Dict[str, torch.Tensor] = {
                "pos":                 pos32,
                "edge_index":          edge_index,
                "atom_types":          atom_types,
                "edge_cell_shift":     edge_cell_shift,
                "edge_transpose_perm": edge_transpose_perm,
                "cell":                cell_in,
                "batch":               self._cached_batch,
                "num_atoms":           num_atoms,
            }
            out = self.inner(data)
            energy_raw = out["total_energy"]
            forces_raw = out["forces"]
        else:
            coords32 = coords.to(torch.float32)
            if periodic:
                edge_index, _, _, unit_shifts32 = build_edges_pbc(
                    coords32, cell3.to(torch.float32), self.r_max,
                )
                # Integer offsets again, in the dtype the rest of this branch
                # works in.  The cell gets a leading batch dimension for the
                # same reason as the new-framework branch above: this dict has
                # a `batch` key.
                edge_cell_shift = unit_shifts32.to(torch.float64)
                cell_in = cell3.to(torch.float64).reshape(1, 3, 3)
            else:
                edge_index, _, _ = build_edges(coords32, self.r_max)
                edge_cell_shift = torch.zeros(
                    (edge_index.size(1), 3), dtype=torch.float64, device=dev,
                )
                cell_in = self._cached_cell
            data: Dict[str, torch.Tensor] = {
                "pos":             coords,
                "edge_index":      edge_index,
                "atom_types":      atom_types,
                "edge_cell_shift": edge_cell_shift,
                "cell":            cell_in,
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

        # Zero unless there is a box to strain.  With no cell the model's own
        # stress step divides by a zero volume, so whatever ends up in the
        # output dict for a cluster is not a number to hand on.  NAMD ignores
        # this slot for a non-periodic system, but the shape has to stay the
        # same either way because TorchScript allows only one return type.
        virial = zero_virial(coords)
        if periodic:
            virial = self._virial_from_model(out, cell3.reshape(1, 3, 3), 1, dev)[0]

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
        Evaluate NequIP/Allegro for a batch of molecules.

        Args:
            coords:     [N_total, 3]  float64  concatenated positions.
            Z:          [N_total]     int64    atomic numbers.
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
        B = ptr.size(0) - 1
        Z = Z.to(torch.int64)
        atom_types = self._z_to_types(Z)

        # NAMD guarantees every walker in a batch agrees about periodicity, so
        # the first cell decides for the whole batch.
        cells3 = cells.reshape(-1, 3, 3)
        periodic = cell_is_periodic(cells3[0])

        if self._is_new_nequip:
            pos32 = coords.to(torch.float32)
            if periodic:
                cells32 = cells3.to(torch.float32)
                edge_index, _, _, unit_shifts = build_edges_batched_pbc(
                    pos32, ptr, cells32, self.r_max,
                )
                # Integer offsets straight through.  NequIP picks each edge's
                # own molecule's cell out of this stack via the batch index, so
                # walkers with different boxes stay separate.
                edge_cell_shift = unit_shifts
                cell = cells32
            else:
                edge_index, _, _ = build_edges_batched(
                    pos32, ptr, self.r_max,
                )
                E = edge_index.size(1)
                edge_cell_shift = torch.zeros((E, 3), dtype=torch.float32, device=dev)
                cell = torch.zeros((B, 3, 3), dtype=torch.float32, device=dev)
            n_atoms_per = ptr[1:] - ptr[:-1]

            ei = edge_index[0]
            ej = edge_index[1]
            keys = ei * N_total + ej
            rev_keys = ej * N_total + ei
            sorted_keys, sort_idx = torch.sort(keys)
            pos_in_sorted = torch.searchsorted(sorted_keys, rev_keys)
            edge_transpose_perm = sort_idx[pos_in_sorted]

            data: Dict[str, torch.Tensor] = {
                "pos":                 pos32,
                "edge_index":          edge_index,
                "atom_types":          atom_types,
                "edge_cell_shift":     edge_cell_shift,
                "edge_transpose_perm": edge_transpose_perm,
                "cell":                cell,
                "batch":               batch,
                "num_atoms":           n_atoms_per,
            }
            out = self.inner(data)
            energy_raw = out["total_energy"]
            forces_raw = out["forces"]
        else:
            coords32 = coords.to(torch.float32)
            if periodic:
                edge_index, _, _, unit_shifts32 = build_edges_batched_pbc(
                    coords32, ptr, cells3.to(torch.float32), self.r_max,
                )
                # Integer offsets, and one cell per walker so that each edge
                # gets shifted by its own molecule's box.
                edge_cell_shift = unit_shifts32.to(torch.float64)
                cell = cells3.to(torch.float64)
            else:
                cell = torch.zeros((3, 3), dtype=torch.float64, device=dev)
                edge_index, _, _ = build_edges_batched(
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

        # One virial per molecule, matching the one cell per molecule going in.
        # NequIP already keeps its strain derivative per graph, so the stack it
        # returns is in walker order and needs no splitting here.
        virials = zero_virials(B, coords)
        if periodic:
            virials = self._virial_from_model(out, cells3, B, dev)

        return energies, forces, charges, virials


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

