"""
NAMD-compatible wrapper for **TorchANI** models.

TorchANI models (ANI-1x, ANI-1ccx, ANI-2x, or custom-trained) are
native ``nn.Module`` objects that accept ``(species, coordinates)`` and
return ``(species, energies)``.  Forces are obtained via
``torch.autograd.grad``.

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges)
        -> (energy_kcal, forces_kcal_A, charges_e)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges)
        -> (energies, forces, charges)

TorchANI specifics:
  * The model builds its own neighbor list internally (AEV computer),
    so the shared ``build_edges`` is **not** used here.
  * Native output units are **Hartree** → converted via
    ``HARTREE_TO_KCAL``.
  * Species are 0-indexed type IDs, not raw atomic numbers.
    The wrapper builds a Z → species-index lookup from the model.
  * Forces must be computed via ``torch.autograd.grad``.
  * Batched evaluation uses TorchANI's native ``[B, N, 3]`` padded
    format; variable-size molecules are padded with species index ``-1``.
"""

import argparse

import torch
from torch import nn
from typing import List, Optional

from ..constants import HARTREE_TO_KCAL
from ..export import export_wrapped


# -------------------------------------------------------------------
#  Species mapping utilities
# -------------------------------------------------------------------

# ANI-2x element order.  Other ANI variants may differ.
_ANI2X_ELEMENTS: List[int] = [1, 6, 7, 8, 16, 17]  # H C N O S Cl

# ANI-1x / ANI-1ccx element order.
_ANI1X_ELEMENTS: List[int] = [1, 6, 7, 8]  # H C N O


def _build_z_to_species(element_list: List[int]) -> torch.Tensor:
    """
    Return a lookup tensor ``t`` such that ``t[Z]`` gives the
    0-indexed species index for atomic number ``Z``.
    Unmapped entries are set to ``-1`` (TorchANI's padding value).
    """
    max_z = max(element_list)
    table = torch.full((max_z + 1,), -1, dtype=torch.long)
    for idx, z in enumerate(element_list):
        table[z] = idx
    return table


class TorchANI_Wrapper(nn.Module):
    """
    Wrap a TorchANI model for NAMD.

    The wrapper can load a **built-in** ANI model (e.g. ``"ANI-2x"``) or
    a custom-trained TorchScript archive.

    Args:
        model_path:     Path to a TorchScript-compiled TorchANI ``.pt``
                        file.  Pass ``""`` if using *builtin*.
        device:         ``"cpu"`` or ``"cuda"``.
        element_list:   Ordered list of atomic numbers corresponding to
                        TorchANI species indices 0, 1, 2, ….
                        Defaults to ANI-2x order ``[1,6,7,8,16,17]``.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        element_list: Optional[List[int]] = None,
    ):
        super().__init__()

        if element_list is None:
            element_list = _ANI2X_ELEMENTS

        self.inner = torch.jit.load(model_path, map_location=device)
        self.inner.eval()

        self.z_to_species = _build_z_to_species(element_list)

        # TorchANI outputs Hartree; we need kcal/mol.
        self.ha_to_kcal = torch.tensor(HARTREE_TO_KCAL, dtype=torch.float64)

        # r_max is informational only – the AEV computer owns the cutoff.
        # Try to read it from the model if available (not always present).
        self.r_max: float = 0.0

        self.supports_batch: bool = True

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _z_to_spec(self, Z: torch.Tensor) -> torch.Tensor:
        """Atomic numbers → TorchANI species indices."""
        table = self.z_to_species.to(Z.device)
        return table[Z]

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
        Z = Z.to(torch.int64)

        species = self._z_to_spec(Z).unsqueeze(0)              # [1, N]
        positions = coords.to(torch.float64).unsqueeze(0)       # [1, N, 3]
        positions = positions.requires_grad_(True)

        # TorchANI forward: (species, coordinates) → (species, energies)
        _, energy_ha = self.inner(species, positions)            # [1, 1]

        # Forces via autograd
        grad_list = torch.autograd.grad(
            [energy_ha.sum()],
            [positions],
            create_graph=False,
            retain_graph=False,
        )
        grad_opt = grad_list[0]
        assert grad_opt is not None, "autograd returned None gradient"
        grad = grad_opt                                           # [1, N, 3]

        forces_ha = -grad.squeeze(0)                             # [N, 3]
        energy_ha_scalar = energy_ha.squeeze()                   # scalar

        ha2kcal = self.ha_to_kcal.to(dev)
        energy  = energy_ha_scalar.to(torch.float64) * ha2kcal
        forces  = forces_ha.to(torch.float64) * ha2kcal
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
        Evaluate TorchANI for a batch of molecules.

        Molecules are padded to equal length and passed to TorchANI in
        its native ``[B, N_max, 3]`` format.  Padding atoms have
        species index ``-1``.

        Args:
            coords:     [N_total, 3]  float64
            Z:          [N_total]     int64
            batch:      [N_total]     int64
            ptr:        [B+1]         int64
            pc_coords:  [P, 3]        float64  (ignored)
            pc_charges: [P]           float64  (ignored)

        Returns:
            energies:  [B]           float64  kcal/mol.
            forces:    [N_total, 3]  float64  kcal/mol/Å.
            charges:   [N_total]     float64  e.
        """
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1
        Z = Z.to(torch.int64)
        spec_flat = self._z_to_spec(Z)  # [N_total]

        # Determine max molecule size for padding.
        n_atoms = ptr[1:] - ptr[:-1]           # [B]
        N_max = int(n_atoms.max().item())

        # Build padded species [B, N_max] and coords [B, N_max, 3].
        species_pad = torch.full(
            (B, N_max), -1, dtype=torch.long, device=dev,
        )
        coords_pad = torch.zeros(
            (B, N_max, 3), dtype=torch.float64, device=dev,
        )
        for b in range(B):
            s = int(ptr[b].item())
            e = int(ptr[b + 1].item())
            n = e - s
            species_pad[b, :n] = spec_flat[s:e]
            coords_pad[b, :n, :] = coords[s:e]

        coords_pad = coords_pad.requires_grad_(True)

        _, energies_ha = self.inner(species_pad, coords_pad)  # [B, 1]

        grad_list = torch.autograd.grad(
            [energies_ha.sum()],
            [coords_pad],
            create_graph=False,
            retain_graph=False,
        )
        grad_opt = grad_list[0]
        assert grad_opt is not None, "autograd returned None gradient"
        grad = grad_opt                                            # [B, N_max, 3]

        forces_pad = -grad  # [B, N_max, 3]

        # Un-pad back to concatenated layout.
        ha2kcal = self.ha_to_kcal.to(dev)
        energies = energies_ha.squeeze(-1).to(torch.float64) * ha2kcal  # [B]

        forces_list: List[torch.Tensor] = []
        for b in range(B):
            s = int(ptr[b].item())
            e = int(ptr[b + 1].item())
            n = e - s
            forces_list.append(forces_pad[b, :n, :])
        forces = torch.cat(forces_list, dim=0).to(torch.float64) * ha2kcal

        charges = torch.zeros(N_total, dtype=torch.float64, device=dev)

        return energies, forces, charges


# -------------------------------------------------------------------
#  CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a wrapped TorchANI model for NAMD",
    )
    parser.add_argument("--model", required=True,
                        help="Path to TorchScript-compiled TorchANI .pt file")
    parser.add_argument("--out", default="mlff_model.pt",
                        help="Output TorchScript file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--elements", default="1,6,7,8,16,17",
                        help="Comma-separated atomic numbers in species order")

    args = parser.parse_args()

    elem_list = [int(x) for x in args.elements.split(",")]

    wrapper = TorchANI_Wrapper(
        model_path=args.model,
        device=args.device,
        element_list=elem_list,
    ).eval()
    export_wrapped(wrapper, args.out, model_type="TorchANI")

