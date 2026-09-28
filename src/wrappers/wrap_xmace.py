"""
NAMD-compatible wrapper for TorchScript-compiled **X-MACE** excited-state
models (rhyan10/X-MACE, AutoencoderExcitedMACE).

X-MACE predicts ``n_energies`` electronic states simultaneously:
    energy: [B, n_states]
    forces: [N, n_states, 3]   (per-state autograd in compute_forces)

NAMD's interface only consumes a single energy / forces channel, so the
wrapper exposes one user-selected state via ``state_idx`` (``--state K``
on the CLI). State 0 is the ground state by default.

X-MACE weights are float32, in contrast to the foundation MACE-OFF
checkpoints — node_attrs and positions must be float32 going in, and we
cast the model output to float64 (kcal/mol) for NAMD.

Both entry points take a trailing cell argument.  X-MACE inherits MACE's
edge convention, so a periodic box is handed over as per-edge shift vectors
exactly the way the MACE wrapper does it.

WHERE THE PERIODIC VIRIAL COMES FROM
The compiled X-MACE forward advertises compute_virials and compute_stress, but
its get_outputs ends in ``return (forces, None, None, hessian)``: those two
outputs are hardwired to None, and the ``displacement`` it carries around is a
zero tensor that never touches the positions, the shifts or the cell.  So there
is no native virial to take, and asking for one would hand back nothing.  The
wrapper therefore derives the virial itself, with the shared strain helpers in
src/virial.py.

That needs the energy graph to still be alive after the model returns, and
X-MACE's own force pass would normally consume it: compute_forces differentiates
each electronic state in turn and lets the last one free the graph.  Its
``training`` argument is the switch for exactly that, and in this compiled model
it does nothing else, so a periodic step passes training=True to keep the graph
and a cluster step leaves it False and behaves as before.
"""

import argparse
from typing import List, Optional

import torch
from torch import nn

from ..constants import EV_TO_KCAL
from ..edges import (build_edges, build_edges_batched, build_edges_pbc,
                     build_edges_batched_pbc, cell_is_periodic)
from ..export import export_wrapped
from ..virial import apply_strain, finalize, make_strain, virial_from_strain


class XMACE_TS_Wrapper(nn.Module):
    """
    Wrap a TorchScript-compiled X-MACE (AutoencoderExcitedMACE) model
    for NAMD, exposing a single electronic state.
    """

    def __init__(
        self,
        compiled_path: str,
        state_idx: int = 0,
        device: str = "cpu",
    ):
        super().__init__()

        self.inner = torch.jit.load(compiled_path, map_location=device)
        self.inner.eval()

        if hasattr(self.inner, "r_max"):
            r = self.inner.r_max
            self.r_max = float(r) if not isinstance(r, float) else r
        else:
            raise RuntimeError("Compiled X-MACE model has no r_max attribute")

        self.atomic_numbers = torch.tensor(
            [int(z) for z in self.inner.atomic_numbers],
            dtype=torch.int64,
        )
        self.n_elements = int(self.atomic_numbers.numel())

        n_states = int(self.inner.n_energies) if hasattr(self.inner, "n_energies") else 1
        if state_idx < 0 or state_idx >= n_states:
            raise ValueError(
                f"state_idx={state_idx} out of range for n_energies={n_states}"
            )
        self.state_idx: int = int(state_idx)
        self.n_states: int = n_states

        # Cached constants (populated lazily).  Keyed on atom count N only;
        # node_attrs is recomputed each step (sync-free) rather than cached
        # against Z.
        self._cached_N: int = -1
        self._cached_batch:      torch.Tensor = torch.empty(0)
        self._cached_ptr:        torch.Tensor = torch.empty(0)
        self._cached_cell:       torch.Tensor = torch.empty(0)

        self.ev_to_kcal = torch.tensor(EV_TO_KCAL, dtype=torch.float64)

        self.supports_batch: bool = True

        # Says this model takes a cell argument.  NAMD works this out from the
        # forward() signature anyway and only uses the flag to cross-check, so
        # if the two ever disagree the model fails to load rather than running
        # with a mishandled cell.
        self.supports_pbc: bool = True

    # -----------------------------------------------------------------
    #  forward (single molecule)
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

        # X-MACE weights are float32; positions need grad for per-state forces.
        coords32 = coords.to(torch.float32).detach().requires_grad_(True)
        Z = Z.to(torch.int64)

        # NAMD sends the box as [1, 3, 3], all zeros when the system is not
        # periodic.  Anything else would be a caller bug rather than something
        # to guess about, so just take the first entry.
        cell3 = cell.reshape(-1, 3, 3)[0].to(torch.float32)
        periodic = cell_is_periodic(cell3)

        # Constant tensors keyed on atom count N only (no host sync).  The
        # old code also keyed on Z via a per-step torch.equal(Z, ...) — a
        # Python bool that forces a GPU→CPU sync every MD step.  node_attrs
        # is instead recomputed below: cheap, sync-free, always correct.
        if N != self._cached_N:
            self._cached_N = N
            self._cached_batch = torch.zeros(N, dtype=torch.long, device=dev)
            self._cached_ptr   = torch.tensor([0, N], dtype=torch.long, device=dev)
            self._cached_cell  = torch.zeros((3, 3), dtype=torch.float32, device=dev)

        # One-hot node attributes — always recomputed (no sync, no caching).
        atomic_numbers_dev = self.atomic_numbers.to(dev)
        node_attrs = (Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)).to(torch.float32)

        # X-MACE inherits MACE's edge convention: it forms each displacement
        # as (neighbour minus central atom) plus the shift, with the shift in
        # Angstrom, so that is what we hand it.
        #
        # Under a box the positions, the cell and the shifts all carry the same
        # zero strain D, so that differentiating the energy with respect to D
        # afterwards gives the virial.  Straining the positions on their own is
        # the classic way to get a wrong answer here, because the imaged
        # neighbours would stay put while their home atoms moved.
        if periodic:
            edge_index, _, _, unit_shifts = build_edges_pbc(
                coords32, cell3, self.r_max
            )
            D = make_strain(cell3)
            positions_in, cell_in, shifts = apply_strain(
                coords32, cell3, unit_shifts, D
            )
        else:
            edge_index, _, _ = build_edges(coords32, self.r_max)
            unit_shifts = torch.zeros(
                (edge_index.size(1), 3), dtype=torch.float32, device=dev
            )
            shifts = torch.zeros_like(unit_shifts)
            positions_in = coords32
            cell_in = self._cached_cell
            D = torch.zeros((3, 3), dtype=torch.float32, device=dev)

        data = {
            "positions":      positions_in,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "shifts":         shifts,
            # This model never reads unit_shifts; MACE does, when rebuilding
            # shifts for its own virial.  It is passed anyway so the input dict
            # stays the shape the MACE family expects, and it costs nothing.
            "unit_shifts":    unit_shifts,
            "cell":           cell_in,
            "batch":          self._cached_batch,
            "ptr":            self._cached_ptr,
        }

        # compute_virials is left off because this model has no virial to give;
        # see the note at the top of the file.  training=True is not a request
        # for training behaviour, it is the only way to stop X-MACE's own force
        # pass from freeing the graph the strain derivative still needs.
        out = self.inner(
            data,
            training=periodic,
            compute_force=True,
            compute_hessian=False,
            compute_virials=False,
            compute_stress=False,
        )

        ev2kcal = self.ev_to_kcal.to(dev)
        s = self.state_idx

        # Zero unless there is a box to strain.  NAMD ignores this output for a
        # cluster and falls back to its own sum, but the shape has to stay the
        # same either way because TorchScript allows only one return type.
        virial = torch.zeros((3, 3), dtype=torch.float64, device=dev)

        energy_opt = out.get("energy")
        if energy_opt is not None:
            # energy: [B, n_states] → scalar of selected state in kcal/mol
            e_state = energy_opt[:, s]
            energy = e_state.to(torch.float64) * ev2kcal
            if periodic:
                # Energy units, so the same eV to kcal/mol factor applies.  No
                # volume division: that would turn it into a stress.
                v = virial_from_strain(e_state, D, 1.0, retain_graph=False)
                virial = finalize(v.to(torch.float64) * ev2kcal)
        else:
            energy = torch.zeros(1, dtype=torch.float64, device=dev)

        forces_opt = out.get("forces")
        if forces_opt is not None:
            # forces: [N, n_states, 3] → [N, 3] for selected state
            forces = forces_opt[:, s, :].to(torch.float64) * ev2kcal
        else:
            forces = torch.zeros(N, 3, dtype=torch.float64, device=dev)

        if periodic:
            # Keeping the graph made X-MACE record its own backward pass as
            # well, and that record would stay reachable through the forces for
            # as long as NAMD holds them.  Cutting it here bounds the memory to
            # one step.
            forces = forces.detach()

        charges = torch.zeros(N, dtype=torch.float64, device=dev)
        return energy, forces, charges, virial

    # -----------------------------------------------------------------
    #  forward_batch
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
        Evaluate X-MACE for a batch of molecules.

        ``cells`` is [B, 3, 3], one box per molecule with the lattice vectors
        as rows; all zeros means the batch is not periodic.

        Returns (energies, forces, charges, virials), the last one [B, 3, 3]
        and all zeros when the batch is not periodic.
        """
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1

        coords32 = coords.to(torch.float32).detach().requires_grad_(True)
        Z = Z.to(torch.int64)

        atomic_numbers_dev = self.atomic_numbers.to(dev)
        match = Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)
        node_attrs = match.to(torch.float32)

        cells32 = cells.reshape(-1, 3, 3).to(torch.float32)
        # NAMD guarantees every walker in a batch agrees about periodicity, so
        # the first cell decides for the whole batch.
        periodic = cell_is_periodic(cells32[0])

        if periodic:
            edge_index, _, _, unit_shifts = build_edges_batched_pbc(
                coords32, ptr, cells32, self.r_max,
            )
            # One strain per molecule rather than one for the batch.  A shared
            # 3x3 would only ever yield the sum of the walkers' virials, which
            # looks like an answer and is not one, and NAMD needs them apart
            # because each replica runs its own barostat.  Molecule b's energy
            # touches only D[b], so a single backward separates them.
            D = torch.zeros_like(cells32).requires_grad_(True)
            sym = 0.5 * (D + D.transpose(-1, -2))
            positions_in = coords32 + torch.einsum(
                "ni,nij->nj", coords32, sym.index_select(0, batch)
            )
            cells_s = cells32 + torch.bmm(cells32, sym)
            # Each edge is shifted by its OWN molecule's cell; the walkers'
            # boxes are not the same tensor.
            edge_cells = cells_s.index_select(0, batch.index_select(0, edge_index[0]))
            shifts = torch.einsum("ei,eij->ej", unit_shifts, edge_cells)
            cell = cells_s
        else:
            edge_index, _, _ = build_edges_batched(coords32, ptr, self.r_max)
            unit_shifts = torch.zeros(
                (edge_index.size(1), 3), dtype=torch.float32, device=dev
            )
            shifts = torch.zeros_like(unit_shifts)
            positions_in = coords32
            cell = torch.zeros((3, 3), dtype=torch.float32, device=dev)
            D = torch.zeros((B, 3, 3), dtype=torch.float32, device=dev)

        data = {
            "positions":      positions_in,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "shifts":         shifts,
            "unit_shifts":    unit_shifts,
            "cell":           cell,
            "batch":          batch,
            "ptr":            ptr,
        }

        # Same reasoning as in forward(): training=True only asks X-MACE to
        # leave its graph standing so the strain derivative can still be taken.
        out = self.inner(
            data,
            training=periodic,
            compute_force=True,
            compute_hessian=False,
            compute_virials=False,
            compute_stress=False,
        )

        ev2kcal = self.ev_to_kcal.to(dev)
        s = self.state_idx

        # One virial per molecule, matching the one cell per molecule going in.
        virials = torch.zeros((B, 3, 3), dtype=torch.float64, device=dev)

        energy_opt = out.get("energy")
        if energy_opt is not None:
            e_state = energy_opt[:, s]
            energies = e_state.to(torch.float64) * ev2kcal
            if periodic:
                grads: List[Optional[torch.Tensor]] = torch.autograd.grad(
                    [e_state.sum()], [D], create_graph=False,
                    retain_graph=False, allow_unused=True,
                )
                g = grads[0]
                if g is not None:
                    virials = finalize((-g).to(torch.float64) * ev2kcal)
        else:
            energies = torch.zeros(B, dtype=torch.float64, device=dev)

        forces_opt = out.get("forces")
        if forces_opt is not None:
            forces = forces_opt[:, s, :].to(torch.float64) * ev2kcal
        else:
            forces = torch.zeros(N_total, 3, dtype=torch.float64, device=dev)

        if periodic:
            # See forward(): the retained graph would otherwise outlive the step.
            forces = forces.detach()

        charges = torch.zeros(N_total, dtype=torch.float64, device=dev)
        return energies, forces, charges, virials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a wrapped X-MACE TorchScript model for NAMD",
    )
    parser.add_argument("--compiled", required=True,
                        help="Path to compiled X-MACE .pt file")
    parser.add_argument("--state", type=int, default=0,
                        help="Electronic state index to expose (default: 0)")
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    wrapper = XMACE_TS_Wrapper(
        args.compiled, state_idx=args.state, device=args.device
    ).eval()
    export_wrapped(wrapper, args.out, model_type="X-MACE")
