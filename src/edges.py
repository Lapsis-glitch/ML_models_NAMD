"""
Shared, TorchScript-compatible edge / neighbor-list construction.

All functions are free-standing (not methods) so any wrapper can import
and call them.  TorchScript inlines them at compile time.
"""

import torch
from typing import Tuple


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

    Implemented as a single vectorised, block-diagonal masked build —
    no per-molecule Python loop and no ``.item()`` host syncs, so it
    does not stall the CUDA stream on every MD step (the throughput-
    critical path for multi-walker runs).  Edge ordering is identical to
    the old per-block concatenation: row-major ``nonzero`` over a
    block-diagonal mask groups edges by molecule exactly as the
    per-block loop did, and kept edges have identical ``dist2`` / vecs /
    lengths.  Tradeoff: allocates an ``[N_total, N_total]`` matrix, which
    is fine for QM-region sizes × a handful of walkers.

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

    # Per-atom molecule id, derived from ptr with NO host sync.  ptr.to(dev)
    # is a no-op when already co-located (NAMD hands coords/ptr in on the
    # model device); it just guards the device mismatch that the old
    # int(ptr[b].item()) path used to tolerate.
    ptr_dev = ptr.to(dev)
    counts = ptr_dev[1:] - ptr_dev[:-1]                                   # [B]
    batch = torch.repeat_interleave(torch.arange(B, device=dev), counts)  # [N_total]

    diff  = coords.unsqueeze(0) - coords.unsqueeze(1)                     # [N_total, N_total, 3]
    dist2 = (diff * diff).sum(dim=2)                                      # [N_total, N_total]
    same  = batch.unsqueeze(0) == batch.unsqueeze(1)                      # block-diagonal
    mask  = same & (dist2 < r_max2) & (dist2 > 0.0)

    edge_index = torch.nonzero(mask).t().to(torch.long)                  # [2, E]
    if edge_index.size(1) == 0:
        edge_vecs = torch.zeros((0, 3), dtype=coords.dtype, device=dev)
        edge_len  = torch.zeros((0,),   dtype=coords.dtype, device=dev)
    else:
        ei = edge_index[0]
        ej = edge_index[1]
        edge_vecs = coords[ej] - coords[ei]
        edge_len  = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))

    return edge_index, edge_vecs, edge_len

