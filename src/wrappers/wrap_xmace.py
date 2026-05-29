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
"""

import argparse

import torch
from torch import nn

from ..constants import EV_TO_KCAL
from ..edges import build_edges, build_edges_batched
from ..export import export_wrapped


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

    # -----------------------------------------------------------------
    #  forward (single molecule)
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

        # X-MACE weights are float32; positions need grad for per-state forces.
        coords32 = coords.to(torch.float32).detach().requires_grad_(True)
        Z = Z.to(torch.int64)

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

        edge_index, _, _ = build_edges(coords32, self.r_max)
        shifts = torch.zeros((edge_index.size(1), 3), dtype=torch.float32, device=dev)

        data = {
            "positions":      coords32,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "shifts":         shifts,
            "cell":           self._cached_cell,
            "batch":          self._cached_batch,
            "ptr":            self._cached_ptr,
        }

        out = self.inner(
            data,
            training=False,
            compute_force=True,
            compute_hessian=False,
            compute_virials=False,
            compute_stress=False,
        )

        ev2kcal = self.ev_to_kcal.to(dev)
        s = self.state_idx

        energy_opt = out.get("energy")
        if energy_opt is not None:
            # energy: [B, n_states] → scalar of selected state in kcal/mol
            energy = energy_opt[:, s].to(torch.float64) * ev2kcal
        else:
            energy = torch.zeros(1, dtype=torch.float64, device=dev)

        forces_opt = out.get("forces")
        if forces_opt is not None:
            # forces: [N, n_states, 3] → [N, 3] for selected state
            forces = forces_opt[:, s, :].to(torch.float64) * ev2kcal
        else:
            forces = torch.zeros(N, 3, dtype=torch.float64, device=dev)

        charges = torch.zeros(N, dtype=torch.float64, device=dev)
        return energy, forces, charges

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
    ):
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1

        coords32 = coords.to(torch.float32).detach().requires_grad_(True)
        Z = Z.to(torch.int64)

        atomic_numbers_dev = self.atomic_numbers.to(dev)
        match = Z.unsqueeze(1) == atomic_numbers_dev.unsqueeze(0)
        node_attrs = match.to(torch.float32)

        cell = torch.zeros((3, 3), dtype=torch.float32, device=dev)

        edge_index, _, _ = build_edges_batched(coords32, ptr, self.r_max)
        shifts = torch.zeros((edge_index.size(1), 3), dtype=torch.float32, device=dev)

        data = {
            "positions":      coords32,
            "atomic_numbers": Z,
            "node_attrs":     node_attrs,
            "edge_index":     edge_index,
            "shifts":         shifts,
            "cell":           cell,
            "batch":          batch,
            "ptr":            ptr,
        }

        out = self.inner(
            data,
            training=False,
            compute_force=True,
            compute_hessian=False,
            compute_virials=False,
            compute_stress=False,
        )

        ev2kcal = self.ev_to_kcal.to(dev)
        s = self.state_idx

        energy_opt = out.get("energy")
        if energy_opt is not None:
            energies = energy_opt[:, s].to(torch.float64) * ev2kcal
        else:
            energies = torch.zeros(B, dtype=torch.float64, device=dev)

        forces_opt = out.get("forces")
        if forces_opt is not None:
            forces = forces_opt[:, s, :].to(torch.float64) * ev2kcal
        else:
            forces = torch.zeros(N_total, 3, dtype=torch.float64, device=dev)

        charges = torch.zeros(N_total, dtype=torch.float64, device=dev)
        return energies, forces, charges


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
