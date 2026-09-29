"""
NAMD-compatible wrapper for **SevenNet** (MDIL-SNU/SevenNet).

The inner model is SevenNet's LAMMPS serial deployment, the file
``sevenn.scripts.deploy.deploy`` writes (``src/compile_sevennet.py`` calls it)::

    inner({"x":          [N]    int64    0-based type index,
           "edge_index": [2, E] int64    row 0 = i, row 1 = j,
           "edge_vec":   [E, 3] float32  pos[j] - pos[i] (+ image shift), Å,
           "num_atoms":  [1]    int64})
        -> {"inferred_total_energy": []     eV,
            "atomic_energy":         [N, 1] eV, ...}

That is the same edge convention as ``src/edges.py``.  The deployment drops
SevenNet's force module (LAMMPS differentiates with respect to ``edge_vec``
itself), so this wrapper takes forces and the virial from autograd, exactly as
the template in ``README.md`` does.

The model is frozen, so it has no parameters to read metadata from.  The type
map, cutoff and dtype come from the ``_extra_files`` the deployment saves next
to the weights, which is also where ``pair_e3gnn.cpp`` reads them:

  * ``chemical_symbols_to_index`` - space-separated symbols; the position of a
    symbol is its type index.
  * ``cutoff`` - Å.
  * ``dtype`` - ``single`` (float32, every released model) or ``double``.
  * ``oeq`` - ``yes`` when deployed with OpenEquivariance kernels
    (``src.compile_sevennet --oeq``).  Their op ``libtorch_tp_jit::jit_conv_forward``
    must be registered before the model loads: the native library
    ``scripts/opt/nequip/oeq_native/liboeq_native.so`` (``--extra-libs`` /
    ``NAMD_MLFF_EXTRA_LIBS``), or ``import openequivariance`` in Python.  CUDA only.

Positions, strain and edge vectors stay float64 and are cast to the model's
dtype only at the call, as LAMMPS does, so large absolute coordinates don't
cost precision in the edge vectors.

Energies: SevenNet is strictly local and ``atomic_energy`` already carries
the per-atom shift and scale, so the wrapper sums ``atomic_energy`` itself, in
float64 (SevenNet's own ``inferred_total_energy`` is a float32 sum, which
rounds a 3000-atom energy by ~1e-2 kcal/mol).  The deployment sets
``is_batch_data=False``, so for a batch the per-molecule energies are
``atomic_energy`` summed over ``batch``.

Exposes the standard NAMD MLIP interface:
    forward(coords, Z, pc_coords, pc_charges, cell)
        -> (energy_kcal, forces_kcal_A, charges_e, virial_kcal)
    forward_batch(coords, Z, batch, ptr, pc_coords, pc_charges, cells)
        -> (energies, forces, charges, virials)
"""

import os
import zipfile
from typing import Dict, Tuple

import torch
from torch import nn

from ..constants import EV_TO_KCAL, SYMBOL_TO_Z
from ..edges import (build_edges, build_edges_batched, build_edges_pbc,
                     build_edges_batched_pbc, cell_is_periodic)
from ..virial import (make_strain, apply_strain, forces_and_virial,
                      finalize, zero_virial, zero_virials)


_DTYPES = {"single": torch.float32, "double": torch.float64}


def _read_extra_files(path: str) -> Dict[str, str]:
    """The ``_extra_files`` of a TorchScript archive, read from the zip without
    loading the model (which would need any custom ops it uses)."""
    meta: Dict[str, str] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) == 3 and parts[1] == "extra":
                meta[parts[2]] = zf.read(name).decode().strip()
    return meta


def uses_oeq(path: str) -> bool:
    """Whether a SevenNet deployment was built with OpenEquivariance kernels."""
    return _read_extra_files(path).get("oeq", "no") == "yes"


def read_sevennet_metadata(path: str) -> Tuple[Dict[int, int], float, torch.dtype]:
    """``({Z: type_index}, cutoff, dtype)`` from a SevenNet deployment."""
    meta = _read_extra_files(path)
    for key in ("chemical_symbols_to_index", "cutoff", "dtype"):
        meta.setdefault(key, "")
    if not meta["chemical_symbols_to_index"] or not meta["cutoff"]:
        raise RuntimeError(
            f"{path} has no SevenNet deployment metadata "
            "(chemical_symbols_to_index / cutoff). Build it with "
            "`python -m src.compile_sevennet` or `sevenn get_model`.")

    type_map: Dict[int, int] = {}
    for i, sym in enumerate(meta["chemical_symbols_to_index"].split()):
        if sym not in SYMBOL_TO_Z:
            raise RuntimeError(f"Unknown element symbol in type map: {sym!r}")
        type_map[SYMBOL_TO_Z[sym]] = i

    dtype_name = meta["dtype"] or "single"
    if dtype_name not in _DTYPES:
        raise RuntimeError(f"Unknown SevenNet dtype {dtype_name!r}")
    return type_map, float(meta["cutoff"]), _DTYPES[dtype_name]


OEQ_NATIVE_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                              "scripts", "opt", "nequip", "oeq_native", "liboeq_native.so")


def _oeq_op_registered() -> bool:
    try:
        torch.ops.libtorch_tp_jit.jit_conv_forward
    except (AttributeError, RuntimeError):
        return False
    return True


def load_oeq_library(path: str = OEQ_NATIVE_LIB) -> str:
    """Register the OpenEquivariance ops from the native (Python-free) library,
    the one NAMD loads via ``NAMD_MLFF_EXTRA_LIBS``.  Returns its absolute path
    ("" if the ops were already registered, e.g. by ``import openequivariance``;
    loading both in one process would register the namespace twice)."""
    if _oeq_op_registered():
        return ""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise RuntimeError(
            f"{path} not found. Build it once with "
            "`TORCH=<NAMD's libtorch> bash scripts/opt/nequip/oeq_native/build.sh`.")
    torch.ops.load_library(path)
    return path


class SevenNet_Wrapper(nn.Module):
    """
    Wrap a SevenNet serial deployment (``deployed_serial.pt``) for NAMD.

    Args:
        model_path: TorchScript file from ``src.compile_sevennet`` /
                    ``sevenn get_model``.
        device:     ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        super().__init__()
        type_map, r_max, model_dtype = read_sevennet_metadata(model_path)

        if uses_oeq(model_path) and not _oeq_op_registered():
            raise RuntimeError(
                f"{model_path} uses OpenEquivariance kernels. Register their ops "
                "first: load_oeq_library() (what src.cli does), --extra-libs "
                "scripts/opt/nequip/oeq_native/liboeq_native.so, or `import openequivariance`.")
        self.inner = torch.jit.load(model_path, map_location=device)
        self.inner.eval()

        self.r_max: float = r_max
        self.model_dtype: torch.dtype = model_dtype
        # Read by export_wrapped for its diagnostics.
        self.model_uses_fp32: bool = model_dtype == torch.float32

        # z_to_type[Z] = type index, -1 for elements the model doesn't know.
        z_to_type = torch.full((max(type_map.keys()) + 1,), -1, dtype=torch.long)
        for z, idx in type_map.items():
            z_to_type[z] = idx
        self.z_to_type = z_to_type.to(device)

        self.ev_to_kcal: float = EV_TO_KCAL

        self.supports_batch: bool = True
        self.supports_pbc: bool = True

    def _run_inner(self, types: torch.Tensor, edge_index: torch.Tensor,
                   edge_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        # The model writes its intermediates into the dict it is given, so it
        # gets a fresh one on every call.
        data: Dict[str, torch.Tensor] = {
            "x": types,
            "edge_index": edge_index,
            "edge_vec": edge_vec.to(self.model_dtype),
            "num_atoms": torch.full((1,), types.size(0), dtype=torch.int64,
                                    device=types.device),
        }
        return self.inner(data)

    @staticmethod
    def _edge_vectors(pos: torch.Tensor, edge_index: torch.Tensor,
                      shifts: torch.Tensor) -> torch.Tensor:
        # Recomputed from `pos` so forces can flow back through them.
        return (pos.index_select(0, edge_index[1])
                - pos.index_select(0, edge_index[0]) + shifts)

    # -----------------------------------------------------------------
    #  Single molecule
    # -----------------------------------------------------------------
    def forward(
        self,
        coords: torch.Tensor,       # [N, 3] float64, Å
        Z: torch.Tensor,            # [N]    int64
        pc_coords: torch.Tensor,    # [P, 3] float64 (ignored)
        pc_charges: torch.Tensor,   # [P]    float64 (ignored)
        cell: torch.Tensor,         # [1, 3, 3] float64, rows = lattice vectors; zeros = no box
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = coords.device
        N = coords.size(0)
        dt = torch.float64

        pos = coords.detach().to(dt).requires_grad_(True)
        types = self.z_to_type.index_select(0, Z.to(torch.int64))

        cell3 = cell.reshape(-1, 3, 3)[0].to(dt)
        periodic = cell_is_periodic(cell3)
        D = make_strain(cell3)

        if periodic:
            edge_index, _, _, unit_shifts = build_edges_pbc(
                pos.detach().to(torch.float32), cell3.to(torch.float32), self.r_max)
            pos_s, _, shifts = apply_strain(pos, cell3, unit_shifts.to(dt), D)
        else:
            edge_index, _, _ = build_edges(pos.detach().to(torch.float32), self.r_max)
            pos_s = pos
            shifts = torch.zeros((edge_index.size(1), 3), dtype=dt, device=dev)

        edge_vec = self._edge_vectors(pos_s, edge_index, shifts)
        atomic = self._run_inner(types, edge_index, edge_vec)["atomic_energy"]
        energy_ev = atomic.to(torch.float64).sum()

        forces, virial = forces_and_virial(energy_ev, pos, D, self.ev_to_kcal)

        energy = energy_ev * self.ev_to_kcal
        charges = torch.zeros(N, dtype=torch.float64, device=dev)
        virial = finalize(virial) if periodic else zero_virial(coords)
        return energy, forces, charges, virial

    # -----------------------------------------------------------------
    #  Batch of molecules (NAMD walkers)
    # -----------------------------------------------------------------
    @torch.jit.export
    def forward_batch(
        self,
        coords: torch.Tensor,       # [N_total, 3] float64
        Z: torch.Tensor,            # [N_total]    int64
        batch: torch.Tensor,        # [N_total]    int64, molecule index per atom
        ptr: torch.Tensor,          # [B + 1]      int64, molecule boundaries
        pc_coords: torch.Tensor,
        pc_charges: torch.Tensor,
        cells: torch.Tensor,        # [B, 3, 3]    one box per molecule
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = coords.device
        N_total = coords.size(0)
        B = ptr.size(0) - 1
        dt = torch.float64

        pos = coords.detach().to(dt).requires_grad_(True)
        types = self.z_to_type.index_select(0, Z.to(torch.int64))
        batch = batch.to(torch.int64)
        cells3 = cells.reshape(-1, 3, 3).to(dt)
        periodic = cell_is_periodic(cells3[0])

        D = torch.zeros((B, 3, 3), dtype=dt, device=dev).requires_grad_(True)

        if periodic:
            edge_index, _, _, unit_shifts = build_edges_batched_pbc(
                pos.detach().to(torch.float32), ptr, cells3.to(torch.float32), self.r_max)
            sym = 0.5 * (D + D.transpose(1, 2))
            pos_s = pos + torch.einsum("ni,nij->nj", pos, sym.index_select(0, batch))
            cells_s = cells3 + torch.bmm(cells3, sym)
            edge_mol = batch.index_select(0, edge_index[0])
            shifts = torch.einsum("ei,eij->ej", unit_shifts.to(dt),
                                  cells_s.index_select(0, edge_mol))
        else:
            edge_index, _, _ = build_edges_batched(
                pos.detach().to(torch.float32), ptr, self.r_max)
            pos_s = pos
            shifts = torch.zeros((edge_index.size(1), 3), dtype=dt, device=dev)

        edge_vec = self._edge_vectors(pos_s, edge_index, shifts)
        atomic = self._run_inner(types, edge_index, edge_vec)["atomic_energy"]
        energy_ev = torch.zeros(B, dtype=torch.float64, device=dev).index_add(
            0, batch, atomic.reshape(-1).to(torch.float64))              # [B]

        grads = torch.autograd.grad([energy_ev.sum()], [pos, D], allow_unused=True)
        g_pos = grads[0]
        g_D = grads[1]
        if g_pos is None:
            forces = torch.zeros((N_total, 3), dtype=torch.float64, device=dev)
        else:
            forces = -g_pos * self.ev_to_kcal

        virials = zero_virials(B, coords)
        if periodic and g_D is not None:
            V = -g_D * self.ev_to_kcal
            virials = 0.5 * (V + V.transpose(1, 2))

        energies = energy_ev * self.ev_to_kcal
        charges = torch.zeros(N_total, dtype=torch.float64, device=dev)
        return energies, forces, charges, virials
