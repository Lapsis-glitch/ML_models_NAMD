"""
Shared, TorchScript-compatible edge / neighbor-list construction.

All functions are free-standing (not methods) so any wrapper can import
and call them.  TorchScript inlines them at compile time.
"""

import torch
from typing import List, Tuple


# -----------------------------------------------------------------------
#  Single-molecule COO edge list  (O(N²), vectorised, FP32)
# -----------------------------------------------------------------------

def build_edges(
    coords: torch.Tensor,
    r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorised O(N²) neighbor list – no Python-level loops.

    Arithmetic runs in the dtype of *coords* (typically float32).
    Output tensors live on the same device as *coords*.

    Args:
        coords:  [N, 3]  atom positions.
        r_max:   cutoff radius.

    Returns:
        edge_index : [2, E]  int64    sender / receiver indices.
        edge_vecs  : [E, 3]  same dtype as coords – displacement vectors.
        edge_len   : [E]     same dtype as coords – edge lengths.
    """
    N = coords.size(0)
    dev = coords.device

    diff = coords.unsqueeze(0) - coords.unsqueeze(1)   # [N, N, 3]
    dist2 = (diff * diff).sum(dim=2)                    # [N, N]

    r_max2 = r_max * r_max
    mask = (dist2 < r_max2) & (dist2 > 0.0)

    edge_index = torch.nonzero(mask).t().to(torch.long)  # [2, E]

    if edge_index.size(1) == 0:
        edge_vecs = torch.zeros((0, 3), dtype=coords.dtype, device=dev)
        edge_len  = torch.zeros((0,),   dtype=coords.dtype, device=dev)
    else:
        ei = edge_index[0]
        ej = edge_index[1]
        edge_vecs = coords[ej] - coords[ei]
        edge_len  = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))

    return edge_index, edge_vecs, edge_len


# -----------------------------------------------------------------------
#  Batched (block-diagonal) COO edge list
# -----------------------------------------------------------------------

def build_edges_batched(
    coords: torch.Tensor,
    ptr: torch.Tensor,
    r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build edges for a batch of molecules.

    Each molecule's atoms occupy a contiguous block defined by *ptr*.
    Edges are only built within each molecule (block-diagonal neighbor
    list), so atoms from different walkers never interact.

    Args:
        coords:  [N_total, 3]  concatenated positions.
        ptr:     [B+1]         molecule boundaries (int64).
        r_max:   cutoff radius.

    Returns:
        edge_index : [2, E_total]  int64   global sender / receiver.
        edge_vecs  : [E_total, 3]  displacement vectors.
        edge_len   : [E_total]     edge lengths.
    """
    dev = coords.device
    r_max2 = r_max * r_max
    B = ptr.size(0) - 1

    all_ei:   List[torch.Tensor] = []
    all_ej:   List[torch.Tensor] = []
    all_vecs: List[torch.Tensor] = []
    all_lens: List[torch.Tensor] = []

    for b in range(B):
        start = int(ptr[b].item())
        end   = int(ptr[b + 1].item())
        mol_coords = coords[start:end]  # [n, 3]

        diff  = mol_coords.unsqueeze(0) - mol_coords.unsqueeze(1)  # [n,n,3]
        dist2 = (diff * diff).sum(dim=2)                            # [n,n]
        mask  = (dist2 < r_max2) & (dist2 > 0.0)

        local_idx = torch.nonzero(mask).t().to(torch.long)  # [2, E_mol]
        if local_idx.size(1) > 0:
            all_ei.append(local_idx[0] + start)
            all_ej.append(local_idx[1] + start)
            vecs = mol_coords[local_idx[1]] - mol_coords[local_idx[0]]
            all_vecs.append(vecs)
            all_lens.append(torch.sqrt((vecs * vecs).sum(dim=1)))

    if len(all_ei) > 0:
        edge_index = torch.stack([torch.cat(all_ei), torch.cat(all_ej)], dim=0)
        edge_vecs  = torch.cat(all_vecs, dim=0)
        edge_len   = torch.cat(all_lens, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
        edge_vecs  = torch.zeros((0, 3), dtype=coords.dtype, device=dev)
        edge_len   = torch.zeros((0,),   dtype=coords.dtype, device=dev)

    return edge_index, edge_vecs, edge_len

