"""
Optional vesin-backed neighbour list, offered as a faster stand-in for
``build_edges_pbc``.

The pure builder in ``src/edges.py`` compares every atom against every other
atom.  That is fine for a few hundred atoms and it is easy to check by hand,
but the work grows with the square of the atom count, so a few thousand atoms
cost tens of milliseconds on every MD step.  vesin sorts the atoms into cells
first and only looks at nearby cells, so the work grows roughly linearly.

Nothing here changes the default.  The pure builder stays the one wrappers get
unless somebody asks for vesin by name at export time, and it stays the thing
we check the vesin output against.

What was checked against the installed vesin (0.6.1) rather than assumed:

  * The displacement convention is the same one ``src/edges.py`` produces,
    ``D = pos[P[:, 1]] - pos[P[:, 0]] + S @ cell``.  Confirmed by asking vesin
    for its own ``D`` alongside ``P`` and ``S`` and getting an exact match, so
    the pairs go in as ``edge_index = P.t()`` with no sign flip.
  * The import moved.  ``vesin.torch`` still works but warns that it is going
    away, and ``vesin_torch`` is the name to use now, so we try the new one
    first and fall back to the old one.
  * There are no CUDA kernels in the shipped library.  vesin accepts a CUDA
    tensor and hands the results back on the same device, but the search itself
    runs on the host, so a CUDA input means a copy down, a CPU search, and a
    copy back.  That copy waits for the GPU to catch up, which is worth knowing
    before putting this on a GPU-resident MD path.

The edge ORDER differs from the pure builder.  vesin is asked for an unsorted
list because sorting costs time and every consumer here scatters over the edges
rather than assuming a layout.  Edge count, lengths and vectors all agree; only
the order in which they arrive does not.

Deploying a model that holds one of these
-----------------------------------------
The saved model no longer stands on its own.  It refers to a C++ class that
lives in ``libvesin_torch.so``, so whatever loads the model has to load that
library first, before ``torch::jit::load``.  In C++ that is
``torch::jit::load_library`` or a plain ``dlopen``; in Python it is
``torch.ops.load_library(path)``, and importing vesin does it for you.  Without
it the load fails with "Unknown type name
'__torch__.torch.classes.vesin._NeighborList'".

The library is pinned to a libtorch minor version.  The wheel ships one build
per version in ``vesin_torch/torch-<major>.<minor>/lib/``, and picking the wrong
one fails at dlopen with an undefined symbol rather than at run time.  So the
model has to be exported from an environment whose torch minor version matches
the libtorch that NAMD links against, and the matching .so has to travel with
it.  That is the price of this backend, and it is why the default stays pure.
"""

import torch
from typing import List, Tuple

from .edges import build_edges_pbc, check_min_image


# vesin is optional.  Importing this module must never be the thing that breaks
# an environment that does not have it, so failure here is recorded and dealt
# with when somebody actually asks for the vesin backend.
_VESIN_IMPORT_ERROR = ""

try:
    from vesin_torch import NeighborList as _VesinNeighborList
    VESIN_AVAILABLE = True
except ImportError as _new_name_error:
    try:
        # Older vesin releases only had it under this name.
        from vesin.torch import NeighborList as _VesinNeighborList
        VESIN_AVAILABLE = True
    except ImportError:
        _VesinNeighborList = None
        VESIN_AVAILABLE = False
        _VESIN_IMPORT_ERROR = str(_new_name_error)


def vesin_version() -> str:
    """Version string of the installed vesin, or an empty string if absent."""
    if not VESIN_AVAILABLE:
        return ""
    try:
        import vesin_torch
        return str(vesin_torch.__version__)
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------
#  Shared conversion from a pair list to the edge contract
# -----------------------------------------------------------------------

def edges_from_pairs(
    coords: torch.Tensor,
    cell: torch.Tensor,
    pairs: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Turn a neighbour list into the four tensors every wrapper expects.

    Args:
        coords: [N, 3] positions.
        cell:   [3, 3] lattice vectors as rows.
        pairs:  [E, 2] central atom in column 0, neighbour in column 1.
        shifts: [E, 3] integer cell offsets, any integer dtype.

    Returns the same tuple as ``build_edges_pbc``.

    The displacements are rebuilt from *coords* rather than taken from the
    neighbour list, for the same reason the pure builder rebuilds them: the
    search runs off the autograd graph, and forces come from differentiating
    these vectors with respect to the positions.
    """
    dev = coords.device
    dtype = coords.dtype

    if pairs.size(0) == 0:
        edge_index  = torch.zeros((2, 0), dtype=torch.long, device=dev)
        edge_vecs   = torch.zeros((0, 3), dtype=dtype, device=dev)
        edge_len    = torch.zeros((0,),   dtype=dtype, device=dev)
        unit_shifts = torch.zeros((0, 3), dtype=dtype, device=dev)
        return edge_index, edge_vecs, edge_len, unit_shifts

    # The transpose is a view with a stride of two, and some downstream kernels
    # are happier with a plain contiguous block, so pay for the copy once.
    edge_index  = pairs.t().contiguous().to(torch.long)
    unit_shifts = shifts.to(dtype)

    ei = edge_index[0]
    ej = edge_index[1]
    edge_vecs = coords[ej] - coords[ei] + unit_shifts @ cell
    edge_len  = torch.sqrt((edge_vecs * edge_vecs).sum(dim=1))

    return edge_index, edge_vecs, edge_len, unit_shifts


# -----------------------------------------------------------------------
#  The two interchangeable builders
# -----------------------------------------------------------------------

class VesinPBCEdges(torch.nn.Module):
    """
    Periodic edge builder backed by vesin's cell list.

    Same call signature and same four return values as ``build_edges_pbc``, so
    a wrapper can hold one of these instead and change nothing else.

    The cutoff is fixed when the object is built, because vesin sizes its cells
    from it.  ``forward`` still takes ``r_max`` so the call site does not have
    to change, and it complains if the two disagree instead of quietly using
    the wrong one.
    """

    def __init__(
        self,
        r_max: float,
        enforce_min_image: bool = True,
    ):
        super().__init__()

        if not VESIN_AVAILABLE:
            raise ImportError(
                "The vesin edge builder was requested but vesin is not "
                "installed in this environment ("
                + _VESIN_IMPORT_ERROR
                + "). Install it with `pip install vesin vesin-torch`, or use "
                "backend='pure'."
            )

        self.r_max = float(r_max)

        # vesin is happy to report a pair through several periodic images, which
        # is more correct than the minimum-image convention the pure builder
        # uses.  Keeping the check on by default means both backends accept
        # exactly the same systems and return exactly the same edges, so one can
        # be used to check the other.  Turn it off only if you actually want
        # multi-image behaviour and understand that the two backends will then
        # disagree on a small box.
        self.enforce_min_image = bool(enforce_min_image)

        self.nl = _VesinNeighborList(
            cutoff=self.r_max,
            # Models need to see each pair from both ends.
            full_list=True,
            # Sorting is work we do not need, see the module docstring.
            sorted=False,
            # Do not leave this on "auto".  At a few thousand atoms "auto"
            # picks the brute-force search, which is the thing we came here to
            # get away from.
            algorithm="cell_list",
            # No Verlet skin.  A skin reuses a stale list until an atom has
            # moved far enough, which would quietly stop matching the pure
            # builder step for step.
            skin=0.0,
        )

    def forward(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor,
        r_max: float,
        block: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [N, 3] positions, same units as the cell.
            cell:   [3, 3] lattice vectors as rows.
            r_max:  cutoff, which has to match the one this was built with.
            block:  accepted and ignored, since there is no row blocking here.
                    It is in the signature only so this stays swappable with
                    the pure builder.
        """
        if r_max < self.r_max - 1e-6 or r_max > self.r_max + 1e-6:
            raise ValueError(
                "This vesin edge builder was created for a cutoff of "
                + str(self.r_max)
                + " A but was called with "
                + str(r_max)
                + " A. Rebuild it with the cutoff the model actually uses."
            )

        if self.enforce_min_image:
            check_min_image(cell, self.r_max)

        # vesin wants both arguments in the same dtype, and it works internally
        # in double regardless.  Detach because the search is a lookup, not
        # something to differentiate through; the vectors are rebuilt from the
        # live positions afterwards.
        points = coords.detach()
        box = cell.detach().to(coords.dtype)

        # "PS" gives the pair indices and the cell shifts.  copy=True because
        # the alternative hands back views into vesin's own buffers, which go
        # stale the next time this is called.
        out: List[torch.Tensor] = self.nl.compute(points, box, True, "PS", True)

        # These come back on the input device already, but say so explicitly so
        # the code does not depend on that staying true.
        pairs = out[0].to(coords.device)
        shifts = out[1].to(coords.device)

        return edges_from_pairs(coords, cell, pairs, shifts)


class PurePBCEdges(torch.nn.Module):
    """
    The existing minimum-image builder, wrapped so it can be swapped with the
    vesin one.  This is the default and the reference.
    """

    def __init__(self, r_max: float, block: int = 512):
        super().__init__()
        self.r_max = float(r_max)
        self.block = int(block)

    def forward(
        self,
        coords: torch.Tensor,
        cell: torch.Tensor,
        r_max: float,
        block: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return build_edges_pbc(coords, cell, r_max, block)


# -----------------------------------------------------------------------
#  Picking a backend
# -----------------------------------------------------------------------

def make_pbc_edge_builder(
    r_max: float,
    backend: str = "pure",
    block: int = 512,
    enforce_min_image: bool = True,
) -> torch.nn.Module:
    """
    Build the periodic edge builder a wrapper should hold.

    This is a factory rather than a global flag on purpose.  A wrapper is
    scripted and saved once, and whatever it holds at that moment is what ends
    up inside ``mlff_model.pt``, so the choice belongs at export time where it
    is visible, not in a module variable that some other import could flip
    underneath a half-built model.

    Args:
        r_max:   model cutoff.
        backend: "pure" for the O(N^2) minimum-image builder, which is the
                 default; "vesin" to insist on the cell list and fail loudly if
                 it is missing; "auto" to use vesin when it is importable and
                 fall back to pure when it is not.
        block:   row block size for the pure builder, ignored by vesin.
        enforce_min_image: keep the too-small-box check on the vesin path.  See
                 VesinPBCEdges for why it defaults to on.
    """
    if backend == "pure":
        return PurePBCEdges(r_max, block)

    if backend == "vesin":
        # Asking for vesin by name and silently getting something else is how
        # a benchmark ends up measuring the wrong thing, so this one raises.
        return VesinPBCEdges(r_max, enforce_min_image)

    if backend == "auto":
        if VESIN_AVAILABLE:
            return VesinPBCEdges(r_max, enforce_min_image)
        return PurePBCEdges(r_max, block)

    raise ValueError(
        "Unknown edge backend '" + str(backend) + "'. Use 'pure', 'vesin' or 'auto'."
    )


# -----------------------------------------------------------------------
#  Convenience for eager code and tests
# -----------------------------------------------------------------------

_BUILDER_CACHE = {}


def build_edges_pbc_vesin(
    coords: torch.Tensor,
    cell: torch.Tensor,
    r_max: float,
    block: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Drop-in replacement for ``build_edges_pbc`` with the same signature and the
    same four return values, for eager code and tests.

    Building a vesin neighbour list object allocates, so they are kept in a
    small cache keyed by cutoff.  That cache is a plain Python dict and TorchScript
    cannot see through it, which is why exported models hold a
    ``VesinPBCEdges`` module instead of calling this.
    """
    key = (float(r_max), True)
    builder = _BUILDER_CACHE.get(key)
    if builder is None:
        builder = VesinPBCEdges(r_max, True)
        _BUILDER_CACHE[key] = builder
    return builder(coords, cell, r_max, block)
