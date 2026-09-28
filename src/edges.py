"""
Shared, TorchScript-compatible edge / neighbor-list construction.

All functions are free-standing (not methods) so any wrapper can import
and call them.  TorchScript inlines them at compile time.

Two families live here:

  build_edges / build_edges_batched
      Non-periodic.  An isolated cluster in vacuum.  These are unchanged and
      still what a model exported without cell support uses.

  build_edges_pbc / build_edges_batched_pbc
      Periodic, using the minimum-image convention.  These also return the
      per-edge shift vectors the models need, and they process the distance
      matrix in row blocks so memory stays bounded for a few thousand atoms.

Conventions used by the periodic builders (they match MACE, NequIP and
SchNetPack, all of which follow ASE here):

  * The cell is a 3x3 whose ROWS are the lattice vectors, so a fractional
    coordinate maps to Cartesian as ``x = f @ cell``.
  * ``edge_index[0]`` is the central atom (sender), ``edge_index[1]`` is the
    neighbour (receiver).
  * ``unit_shifts`` are integer cell offsets, and the displacement for an edge
    is ``pos[edge_index[1]] - pos[edge_index[0]] + unit_shifts @ cell``.
    NequIP wants ``unit_shifts`` and multiplies by the cell itself; MACE and
    SchNetPack want the Cartesian product.

Getting that shift SIGN backwards does not blow up: it moves the neighbour a
whole box away instead of pulling it alongside, so the pair just drops out of
the cutoff and the energy comes out quietly wrong.
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

    Two vectorised paths, no per-molecule Python loop:

    * Equal-size molecules (what NAMD replicas always send): the batch is
      viewed as ``[B, n, 3]`` and distances are taken per molecule as
      ``[B, n, n]``.  That is B times less work and memory than a
      block-diagonal ``[N_total, N_total]`` build (at 4 walkers x 6000
      atoms: ~3.5 GB of temporaries instead of ~14 GB), and row-major
      ``nonzero`` over (b, i, j) visits edges in exactly the order the
      block-diagonal build did, so edge_index / vecs / lengths are
      bit-identical to it.
    * Unequal sizes: the single block-diagonal masked build (the original
      implementation), which allocates ``[N_total, N_total]``.

    Choosing the path costs one host sync on ``ptr`` (tiny, and taken
    while the stream is idle at the start of a step); ``nonzero`` syncs
    anyway.

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
    N_total = coords.size(0)

    # ptr.to(dev) is a no-op when already co-located (NAMD hands coords/ptr
    # in on the model device); it just guards a device mismatch.
    ptr_dev = ptr.to(dev)
    counts = ptr_dev[1:] - ptr_dev[:-1]                                   # [B]

    # Equal-size fast path.  n * B == N_total alone does not prove equal
    # sizes, so check every molecule (one small host sync).
    if B > 0 and N_total % B == 0:
        n = N_total // B
        if bool((counts == n).all()):
            xb = coords.reshape(B, n, 3)
            diff_b = xb.unsqueeze(1) - xb.unsqueeze(2)                    # [B, n, n, 3]
            dist2_b = (diff_b * diff_b).sum(dim=3)                        # [B, n, n]
            mask_b = (dist2_b < r_max2) & (dist2_b > 0.0)
            nz = torch.nonzero(mask_b)                                    # [E, 3] (b, i, j)
            if nz.size(0) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=dev)
                edge_vecs = torch.zeros((0, 3), dtype=coords.dtype, device=dev)
                edge_len = torch.zeros((0,), dtype=coords.dtype, device=dev)
                return edge_index, edge_vecs, edge_len
            base = nz[:, 0] * n
            edge_index = torch.stack((base + nz[:, 1], base + nz[:, 2])).to(torch.long)
            ei = edge_index[0]
            ej = edge_index[1]
            edge_vecs = coords[ej] - coords[ei]
            edge_len = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))
            return edge_index, edge_vecs, edge_len

    # General path: one block-diagonal masked build over all atoms.
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


# -----------------------------------------------------------------------
#  Cell-list (binned) non-periodic edge lists -- opt-in, O(N)
# -----------------------------------------------------------------------

def build_edges_cell_batched(
    coords: torch.Tensor,
    ptr: torch.Tensor,
    r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Cell-list version of ``build_edges_batched``: same edge set, same
    (row-major) order, same vecs/lengths, but O(N) work and memory instead
    of O(N^2).  Opt-in: no existing wrapper calls it by default.

    Atoms are binned into cubes a hair larger than the cutoff (keyed by
    molecule, so walkers never see each other); every atom tests only the
    atoms of its 27 neighbouring cells.  Candidate distances use exactly the
    dense builders' arithmetic (``(x[j]-x[i])**2`` summed over xyz, compared
    ``< r_max**2`` and ``> 0``), and the result is sorted to the dense
    builders' row-major order, so the output matches ``build_edges`` /
    ``build_edges_batched`` edge for edge (checked in
    scripts/opt/schnet/nl/check_cell_nl.py).

    Worth it only for large systems (measured on the RTX 5080: above ~5000
    atoms single-walker, ~3000 atoms x 4 walkers); below that its fixed cost
    (~30 small kernels + one host sync for the max cell occupancy) exceeds the
    dense O(N^2) build.  Memory is O(N) either way it is used.

    Args / returns: as ``build_edges_batched``.
    """
    dev = coords.device
    N = coords.size(0)
    B = ptr.size(0) - 1
    if N == 0 or B <= 0:
        return (torch.zeros((2, 0), dtype=torch.long, device=dev),
                torch.zeros((0, 3), dtype=coords.dtype, device=dev),
                torch.zeros((0,), dtype=coords.dtype, device=dev))

    ptr_dev = ptr.to(dev)
    counts_m = ptr_dev[1:] - ptr_dev[:-1]
    mol = torch.repeat_interleave(torch.arange(B, device=dev), counts_m)    # [N]

    x = coords.detach()
    # Bin edge slightly above the cutoff: a pair inside the cutoff can then
    # never land two bins apart, whatever fp rounding does to (x - lo) / h.
    h = r_max * 1.001
    lo = x.min(dim=0).values
    # +1 so every neighbour bin index (ijk - 1) stays >= 0.  Cell ids use a
    # fixed stride K per axis instead of the true grid shape, so the grid shape
    # never has to come back to the host; cells are looked up by binary search
    # in the sorted id list, so no per-cell array of that size is allocated.
    ijk = torch.floor((x - lo) / h).to(torch.long) + 1                      # [N, 3]
    K = 1 << 16                  # bins per axis (65536 x 5 A >> any box); B * K**3 fits int64
    mol_key = mol * K
    cid = ((mol_key + ijk[:, 0]) * K + ijk[:, 1]) * K + ijk[:, 2]           # [N]
    cid_sorted, order = torch.sort(cid)                                     # atoms by cell

    k = torch.arange(27, device=dev)
    off = torch.stack((k // 9 - 1, (k // 3) % 3 - 1, k % 3 - 1), dim=1)     # [27, 3]
    nb = ijk.unsqueeze(1) + off.unsqueeze(0)                                # [N, 27, 3]
    nbc = ((mol_key.unsqueeze(1) + nb[:, :, 0]) * K + nb[:, :, 1]) * K + nb[:, :, 2]
    first = torch.searchsorted(cid_sorted, nbc)                             # [N, 27]
    last = torch.searchsorted(cid_sorted, nbc, right=True)
    cnt = last - first
    M = int(cnt.max())                                                      # the one host sync

    s = torch.arange(M, device=dev).view(1, 1, M)
    slot_ok = s < cnt.unsqueeze(2)                                          # [N, 27, M]
    slot = (first.unsqueeze(2) + s).clamp(max=N - 1)
    j = order[slot]                                                         # [N, 27, M]

    diff = x[j] - x.view(N, 1, 1, 3)                                        # x[j] - x[i]
    dist2 = (diff * diff).sum(dim=3)
    r_max2 = r_max * r_max
    mask = slot_ok & (dist2 < r_max2) & (dist2 > 0.0)

    nz = torch.nonzero(mask)                                                # [E, 3]
    if nz.size(0) == 0:
        return (torch.zeros((2, 0), dtype=torch.long, device=dev),
                torch.zeros((0, 3), dtype=coords.dtype, device=dev),
                torch.zeros((0,), dtype=coords.dtype, device=dev))
    ii = nz[:, 0]
    jj = j[nz[:, 0], nz[:, 1], nz[:, 2]]
    # Row-major (i, then j) order, like the dense builders.
    key, _ = torch.sort(ii * N + jj)
    ei = key // N
    ej = key - ei * N
    edge_index = torch.stack((ei, ej))
    edge_vecs = coords[ej] - coords[ei]
    edge_len = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))
    return edge_index, edge_vecs, edge_len


def build_edges_cell(
    coords: torch.Tensor,
    r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-molecule cell-list edges; same output as ``build_edges``."""
    ptr = torch.arange(2, dtype=torch.long, device=coords.device) * coords.size(0)
    return build_edges_cell_batched(coords, ptr, r_max)


# -----------------------------------------------------------------------
#  Single-molecule periodic edge list (minimum image, row-blocked)
# -----------------------------------------------------------------------

def build_edges_pbc(
    coords: torch.Tensor,
    cell: torch.Tensor,
    r_max: float,
    block: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Minimum-image neighbour list for a periodic box.

    Works in row blocks rather than building the whole [N, N, 3] difference
    tensor at once.  At a few thousand atoms the arithmetic is trivial for a
    GPU but the tensor is not: 3800 atoms need about 170 MB for the
    displacements alone, and again for the fractional coordinates and the
    rounded offsets, every step.  Blocking keeps that to one slice at a time
    and costs nothing, since this was never compute-bound.

    Edges come out in the same row-major order as the non-periodic builder,
    because the blocks are contiguous row ranges visited in order.

    Args:
        coords:  [N, 3] positions, same units as the cell.
        cell:    [3, 3] lattice vectors as ROWS.
        r_max:   cutoff radius.
        block:   how many central atoms to handle at once.

    Returns:
        edge_index  : [2, E]  int64, central atom then neighbour.
        edge_vecs   : [E, 3]  minimum-image displacements.
        edge_len    : [E]     their lengths.
        unit_shifts : [E, 3]  integer cell offsets, dtype of coords.
    """
    N = coords.size(0)
    dev = coords.device
    dtype = coords.dtype

    check_min_image(cell, r_max)

    # The cell is a fixed input rather than something we differentiate
    # through, so keep the inverse and the rounded offsets out of the graph.
    inv_cell = torch.linalg.inv(cell).detach()

    r_max2 = r_max * r_max

    idx_chunks: List[torch.Tensor] = []
    shift_chunks: List[torch.Tensor] = []

    for start in range(0, N, block):
        end = start + block
        if end > N:
            end = N

        ci = coords[start:end]                              # [C, 3]
        # diff[i, j] = coords[j] - ci[i], matching the non-periodic builder.
        diff = coords.unsqueeze(0) - ci.unsqueeze(1)        # [C, N, 3]

        # Pull each neighbour into whichever image sits nearest the centre.
        n = torch.round(diff @ inv_cell).detach()           # [C, N, 3]
        d = diff - n @ cell

        dist2 = (d * d).sum(dim=2)                          # [C, N]
        mask = (dist2 < r_max2) & (dist2 > 0.0)

        nz = torch.nonzero(mask)                            # [E_c, 2]
        if nz.size(0) > 0:
            local_i = nz[:, 0]
            j = nz[:, 1]
            gi = local_i + start
            idx_chunks.append(torch.stack((gi, j)))         # [2, E_c]
            # Subtracting n moved the neighbour towards us, so the shift that
            # reproduces that displacement from the raw positions is -n.
            shift_chunks.append(-n[local_i, j])             # [E_c, 3]

    if len(idx_chunks) == 0:
        edge_index  = torch.zeros((2, 0), dtype=torch.long, device=dev)
        edge_vecs   = torch.zeros((0, 3), dtype=dtype, device=dev)
        edge_len    = torch.zeros((0,),   dtype=dtype, device=dev)
        unit_shifts = torch.zeros((0, 3), dtype=dtype, device=dev)
        return edge_index, edge_vecs, edge_len, unit_shifts

    edge_index  = torch.cat(idx_chunks, dim=1).to(torch.long)
    unit_shifts = torch.cat(shift_chunks, dim=0).to(dtype)

    # Rebuild the displacements straight from the positions so they are on the
    # autograd graph; the blocked pass above ran partly detached.
    ei = edge_index[0]
    ej = edge_index[1]
    edge_vecs = coords[ej] - coords[ei] + unit_shifts @ cell
    edge_len  = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))

    return edge_index, edge_vecs, edge_len, unit_shifts


# -----------------------------------------------------------------------
#  Batched periodic edge list
# -----------------------------------------------------------------------

def build_edges_batched_pbc(
    coords: torch.Tensor,
    ptr: torch.Tensor,
    cells: torch.Tensor,
    r_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Periodic edges for a batch of molecules, one cell each.

    Each molecule carries its own cell because separate replicas run separate
    barostats and their boxes drift apart.

    Handled one molecule at a time rather than as a single block-diagonal
    build: every molecule has a different cell, so there is no shared
    minimum-image transform to vectorise over, and molecules never share
    edges anyway.

    Args:
        coords: [N_total, 3] concatenated positions.
        ptr:    [B+1] molecule boundaries.
        cells:  [B, 3, 3] lattice vectors as rows, per molecule.
        r_max:  cutoff radius.

    Returns:
        The same four tensors as build_edges_pbc, with edge_index in global
        (concatenated) numbering and edges grouped by molecule.
    """
    dev = coords.device
    dtype = coords.dtype
    B = ptr.size(0) - 1
    ptr_dev = ptr.to(dev)

    idx_chunks: List[torch.Tensor] = []
    vec_chunks: List[torch.Tensor] = []
    shift_chunks: List[torch.Tensor] = []

    for b in range(B):
        start = int(ptr_dev[b])
        end = int(ptr_dev[b + 1])
        if end <= start:
            continue

        sub = coords[start:end]
        ei, ev, _, us = build_edges_pbc(sub, cells[b], r_max)
        if ei.size(1) == 0:
            continue

        idx_chunks.append(ei + start)     # local -> global numbering
        vec_chunks.append(ev)
        shift_chunks.append(us)

    if len(idx_chunks) == 0:
        edge_index  = torch.zeros((2, 0), dtype=torch.long, device=dev)
        edge_vecs   = torch.zeros((0, 3), dtype=dtype, device=dev)
        edge_len    = torch.zeros((0,),   dtype=dtype, device=dev)
        unit_shifts = torch.zeros((0, 3), dtype=dtype, device=dev)
        return edge_index, edge_vecs, edge_len, unit_shifts

    edge_index  = torch.cat(idx_chunks, dim=1)
    edge_vecs   = torch.cat(vec_chunks, dim=0)
    unit_shifts = torch.cat(shift_chunks, dim=0)
    edge_len    = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))

    return edge_index, edge_vecs, edge_len, unit_shifts


# -----------------------------------------------------------------------
#  Periodic helpers
# -----------------------------------------------------------------------

def cell_is_periodic(cell: torch.Tensor) -> bool:
    """
    True when *cell* describes a real box.

    A cell of all zeros is how NAMD says "this system is not periodic", so
    that is what we test for.  Using the determinant rather than a sum of
    absolute values also rejects a degenerate (flat) cell, which would make
    the inverse blow up further down.

    The all-zero test runs first because it is what NAMD sends for every
    non-periodic step, and ``linalg_det`` on CUDA costs ~250 us of host time
    (cuSOLVER/MAGMA setup) — about 9% of a small-system SchNet call.  A zero
    cell has determinant 0 and was rejected before too, so the result is
    unchanged for every input.
    """
    if not bool((cell != 0).any()):
        return False
    return bool(torch.linalg.det(cell).abs() > 1e-8)


def min_perp_width(cell: torch.Tensor) -> torch.Tensor:
    """
    Smallest distance between opposite faces of the cell.

    For a cube this is just the side length, but for a sheared cell it can be
    much smaller than the shortest lattice vector, which is why the
    minimum-image check below uses this rather than ``cell.norm(dim=1)``.
    """
    vol = torch.linalg.det(cell).abs()
    a = cell[0]
    b = cell[1]
    c = cell[2]
    area_a = torch.linalg.cross(b, c).norm()
    area_b = torch.linalg.cross(c, a).norm()
    area_c = torch.linalg.cross(a, b).norm()
    widths = torch.stack((vol / area_a, vol / area_b, vol / area_c))
    return widths.min()


def check_min_image(cell: torch.Tensor, r_max: float) -> None:
    """
    Refuse to run if the box is too small for minimum imaging.

    Taking one image per pair is only right while the box is at least twice
    the cutoff across.  Below that an atom can see several images of the same
    neighbour, and we would keep only the closest and silently drop the rest.
    That shows up as an energy that is slightly too low and a drift that looks
    like bad thermostatting, so it is worth failing loudly instead.
    """
    width = min_perp_width(cell)
    if bool(width < 2.0 * r_max):
        raise ValueError(
            "Periodic cell is too small for the model cutoff: the box measures "
            + str(float(width))
            + " A between opposite faces but the cutoff needs at least "
            + str(2.0 * r_max)
            + " A. Use a larger box, or a model with a shorter cutoff."
        )


# -----------------------------------------------------------------------
#  Choosing a periodic builder
# -----------------------------------------------------------------------

def make_pbc_edge_builder(
    r_max: float,
    backend: str = "pure",
    block: int = 512,
    enforce_min_image: bool = True,
):
    """
    Return the periodic edge builder a wrapper should hold, as a Module whose
    forward has the same signature and return values as ``build_edges_pbc``.

    The default is the builder above, unchanged.  Passing ``backend="vesin"``
    swaps in the cell-list version from ``src/nl_vesin.py``, which is much
    faster once a system gets past a few hundred atoms.

    The import is done here rather than at the top of the file so that this
    module keeps working in environments without vesin installed.  See
    ``src/nl_vesin.py`` for what the vesin path does and does not change.
    """
    from .nl_vesin import make_pbc_edge_builder as _make
    return _make(r_max, backend, block, enforce_min_image)

