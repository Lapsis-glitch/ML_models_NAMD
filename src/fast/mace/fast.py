"""FastMACE: an exact re-evaluation of a (rebuilt) ScaleShiftMACE for NAMD inference.

Same weights, same maths, fewer FLOPs / fewer Python-op round trips.  Built on
top of a model from rebuild.py (e3nn or cuEquivariance flavour) and scripted as
the wrapper's `inner`, keeping MACE's dict-in / dict-out calling convention so
src/wrappers/wrap_compiled_mace.py needs no change to use it.

Exact rewrites (each checked numerically against the stock module when the
FastMACE is built, and end-to-end by the build script):

1. half_radial: MACE evaluates the radial MLP (conv_tp_weights, 8->64->64->64->W)
   and the Bessel/cutoff embedding on every DIRECTED edge.  For a non-periodic
   symmetric neighbour list the edge (i,j) and (j,i) have bit-identical lengths
   (|p_j-p_i| = |p_i-p_j| in IEEE arithmetic), so these are evaluated once per
   undirected pair and gathered back to the directed edges.  The conv TP itself
   (direction-dependent) still runs on all edges.  Only used when no virial /
   stress is requested (the wrapper asks for those exactly when periodic);
   periodic calls go through the stock model (self.full).  Sync-free: the pair
   list uses nonzero_static(size=E//2) + sort/searchsorted, and a device-side
   torch._assert_async checks every edge found its reverse.
2. species_skip: skip_tp is a FullyConnectedTensorProduct with the one-hot
   element attributes (10x0e), i.e. per node a linear map selected by species.
   The FCTP multiplies against all 10 elements (9 of them zeros).  Replaced by
   per-species block matrices (extracted from the module itself) applied to the
   rows of each present species only.  Needs the species counts on the host:
   one D2H sync per call (or, with cache_species=True, one per new atom count --
   valid only if the element list is fixed for a given N, as in a NAMD run).
3. plain_linear (cuEq models): cuEq Linear layers are block-diagonal per irrep;
   each is replaced by the equivalent plain torch matmul (weights extracted from
   the module).  Removes one Python custom-op round trip (~0.3-0.5 ms host each,
   fwd and bwd) per layer; the symmetric contraction and the (fused) conv TP stay
   on cuEquivariance kernels.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from e3nn import o3


# ---------------------------------------------------------------------------
#  irreps helpers
# ---------------------------------------------------------------------------
def _blocks(irreps) -> List[Tuple[int, int, int]]:
    """(mul, l, parity) per block, from an e3nn or cuEquivariance Irreps."""
    out = []
    for term in str(irreps).split("+"):
        mul, ir = term.strip().split("x")
        out.append((int(mul), int(ir[:-1]), 1 if ir[-1] == "e" else -1))
    return out


def _offsets(blocks):
    offs, o = [], 0
    for mul, l, _ in blocks:
        offs.append(o)
        o += mul * (2 * l + 1)
    return offs, o


def _pos(layout_irmul: bool, off: int, mul: int, d: int, u: int, m: int) -> int:
    return off + (m * mul + u if layout_irmul else u * d + m)


# ---------------------------------------------------------------------------
#  block (optionally species-selected) linear map, plain torch
# ---------------------------------------------------------------------------
class BlockLinear(nn.Module):
    """y = x @ T with T block structured per irrep; weights[s] per species.

    For each output block ob: y_ob[n, m, w] = sum_u X[n, m, u] M[s_n, ob][u, w]
    where X concatenates (along u) all input blocks with the same irrep.
    """

    def __init__(self, in_blocks, out_blocks, mats: List[torch.Tensor], srcs: List[List[int]],
                 irmul: bool, n_species: int):
        super().__init__()
        self.irmul: bool = irmul
        self.n_species: int = n_species
        io, self.d_in = _offsets(in_blocks)
        oo, self.d_out = _offsets(out_blocks)
        self.in_off: List[int] = io
        self.in_mul: List[int] = [b[0] for b in in_blocks]
        self.in_d: List[int] = [2 * b[1] + 1 for b in in_blocks]
        self.out_mul: List[int] = [b[0] for b in out_blocks]
        self.out_d: List[int] = [2 * b[1] + 1 for b in out_blocks]
        self.srcs: List[List[int]] = srcs
        # all per-output-block matrices [S, rows, cols] flattened into one buffer
        self.w_off: List[int] = []
        self.w_rows: List[int] = []
        self.w_cols: List[int] = []
        flat, o = [], 0
        for m in mats:
            self.w_off.append(o)
            self.w_rows.append(int(m.shape[1]))
            self.w_cols.append(int(m.shape[2]))
            flat.append(m.reshape(n_species, -1))
            o += int(m.shape[1] * m.shape[2])
        self.register_buffer("w", torch.cat(flat, dim=1).contiguous())

    def _block_in(self, x: torch.Tensor, ib: int) -> torch.Tensor:
        n = x.shape[0]
        mul = self.in_mul[ib]
        d = self.in_d[ib]
        xb = x.narrow(1, self.in_off[ib], mul * d)
        if self.irmul:
            return xb.reshape(n, d, mul)
        return xb.reshape(n, mul, d).transpose(1, 2)

    def run(self, x: torch.Tensor, s: int) -> torch.Tensor:
        """all rows of x belong to species s (s=0 for a species-free linear)."""
        n = x.shape[0]
        outs: List[torch.Tensor] = []
        for ob in range(len(self.srcs)):
            src = self.srcs[ob]
            if len(src) == 1:
                X = self._block_in(x, src[0])
            else:
                X = torch.cat([self._block_in(x, ib) for ib in src], dim=2)
            mat = self.w[s].narrow(0, self.w_off[ob], self.w_rows[ob] * self.w_cols[ob]).view(
                self.w_rows[ob], self.w_cols[ob])
            y = torch.matmul(X, mat)  # [n, d, mul_out]
            if self.irmul:
                outs.append(y.reshape(n, -1))
            else:
                outs.append(y.transpose(1, 2).reshape(n, -1))
        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x, 0)


class SpeciesLinear(nn.Module):
    """skip_tp replacement: rows grouped by species (perm/counts from FastMACE)."""

    def __init__(self, bl: BlockLinear):
        super().__init__()
        self.bl = bl

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor, perm: torch.Tensor,
                inv_perm: torch.Tensor, counts: List[int]) -> torch.Tensor:
        xs = x.index_select(0, perm)
        parts: List[torch.Tensor] = []
        start = 0
        for s, c in enumerate(counts):
            if c > 0:
                parts.append(self.bl.run(xs.narrow(0, start, c), s))
                start += c
        ys = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return ys.index_select(0, inv_perm)


class StockSkip(nn.Module):
    """stock FCTP skip with the SpeciesLinear call signature."""

    def __init__(self, tp):
        super().__init__()
        self.tp = tp

    def forward(self, x: torch.Tensor, node_attrs: torch.Tensor, perm: torch.Tensor,
                inv_perm: torch.Tensor, counts: List[int]) -> torch.Tensor:
        return self.tp(x, node_attrs)


# ---------------------------------------------------------------------------
#  probing: extract block matrices from a (black-box) linear module
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_block_linear(f, in_irreps, out_irreps, irmul: bool, n_species: int,
                         dtype, device, check_rows: int = 64) -> BlockLinear:
    """f(x, s) -> y, linear in x, for species s (ignored when n_species == 1)."""
    ib, ob = _blocks(in_irreps), _blocks(out_irreps)
    ioff, din = _offsets(ib)
    ooff, dout = _offsets(ob)
    mats, srcs = [], []
    for o, (mo, lo, po) in enumerate(ob):
        src = [i for i, (mi, li, pi) in enumerate(ib) if (li, pi) == (lo, po)]
        if not src:
            raise RuntimeError(f"output block {o} has no input of the same irrep")
        srcs.append(src)
        mats.append(torch.zeros(n_species, sum(ib[i][0] for i in src), mo, dtype=dtype))
    for s in range(n_species):
        rows = sum(b[0] for b in ib)
        X = torch.zeros(rows, din, dtype=dtype, device=device)
        r, rowinfo = 0, []
        for i, (mi, li, pi) in enumerate(ib):
            for u in range(mi):
                X[r, _pos(irmul, ioff[i], mi, 2 * li + 1, u, 0)] = 1.0
                rowinfo.append((i, u))
                r += 1
        Y = f(X, s)
        for o, (mo, lo, po) in enumerate(ob):
            base = 0
            for i in srcs[o]:
                mi = ib[i][0]
                first = sum(b[0] for b in ib[:i])
                cols = [_pos(irmul, ooff[o], mo, 2 * lo + 1, w, 0) for w in range(mo)]
                mats[o][s, base:base + mi] = Y[first:first + mi][:, cols].cpu()
                base += mi
    bl = BlockLinear(ib, ob, [m.to(device) for m in mats], srcs, irmul, n_species).to(device)
    # verify on random input, every species
    g = torch.Generator(device="cpu").manual_seed(0)
    worst = 0.0
    for s in range(n_species):
        x = torch.randn(check_rows, din, generator=g, dtype=dtype).to(device)
        ref = f(x, s)
        got = bl.run(x, s)
        err = (ref - got).abs().max().item() / max(ref.abs().max().item(), 1e-30)
        worst = max(worst, err)
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    if worst > tol:
        raise RuntimeError(f"block-linear extraction mismatch rel {worst:.2e}")
    return bl


class FusedConv(nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tp = tp

    def forward(self, node_feats: torch.Tensor, edge_attrs: torch.Tensor,
                tp_weights: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.tp(node_feats, edge_attrs, tp_weights, edge_index)


class GatherScatterConv(nn.Module):
    def __init__(self, tp):
        super().__init__()
        self.tp = tp

    def forward(self, node_feats: torch.Tensor, edge_attrs: torch.Tensor,
                tp_weights: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        mji = self.tp(node_feats.index_select(0, edge_index[0]), edge_attrs, tp_weights)
        # same scatter as mace.tools.scatter.scatter_sum (scatter_add_ along dim 0)
        idx = edge_index[1].unsqueeze(1).expand_as(mji)
        out = torch.zeros((node_feats.shape[0], mji.shape[1]), dtype=mji.dtype, device=mji.device)
        return out.scatter_add_(0, idx, mji)


class CueqProduct(nn.Module):
    """cuEq symmetric contraction (species as int32 index) + (plain) linear + sc."""

    def __init__(self, prod, linear: nn.Module):
        super().__init__()
        self.symc = prod.symmetric_contractions
        self.linear = linear
        self.use_sc: bool = bool(prod.use_sc)

    def forward(self, x: torch.Tensor, sc: Optional[torch.Tensor], node_attrs: torch.Tensor,
                species32: torch.Tensor) -> torch.Tensor:
        y = self.linear(self.symc(x.flatten(1), species32))
        if self.use_sc and sc is not None:
            return y + sc
        return y


class StockProduct(nn.Module):
    def __init__(self, prod):
        super().__init__()
        self.prod = prod

    def forward(self, x: torch.Tensor, sc: Optional[torch.Tensor], node_attrs: torch.Tensor,
                species32: torch.Tensor) -> torch.Tensor:
        return self.prod(node_feats=x, sc=sc, node_attrs=node_attrs)


# ---------------------------------------------------------------------------
#  one interaction + product layer
# ---------------------------------------------------------------------------
class FastLayer(nn.Module):
    def __init__(self, inter, prod, first: bool, irmul: bool, fused: bool,
                 skip: nn.Module, species_skip: bool,
                 linear_up: nn.Module, linear: nn.Module, prod_linear: Optional[nn.Module]):
        super().__init__()
        self.first: bool = first
        self.fused: bool = fused
        self.species_skip: bool = species_skip
        self.avg_num_neighbors: float = float(inter.avg_num_neighbors)
        self.linear_up = linear_up
        self.conv_tp_weights = inter.conv_tp_weights
        self.conv = FusedConv(inter.conv_tp) if fused else GatherScatterConv(inter.conv_tp)
        self.linear = linear
        self.skip = skip
        self.reshape = inter.reshape
        self.product = CueqProduct(prod, prod_linear) if prod_linear is not None else StockProduct(prod)

    def _skip(self, x: torch.Tensor, node_attrs: torch.Tensor, perm: torch.Tensor,
              inv_perm: torch.Tensor, counts: List[int]) -> torch.Tensor:
        return self.skip(x, node_attrs, perm, inv_perm, counts)

    def forward(self, node_feats: torch.Tensor, node_attrs: torch.Tensor, species32: torch.Tensor,
                edge_attrs: torch.Tensor, edge_feats: torch.Tensor, cutoff: torch.Tensor,
                edge_index: torch.Tensor, pair_id: Optional[torch.Tensor],
                perm: torch.Tensor, inv_perm: torch.Tensor, counts: List[int]) -> torch.Tensor:
        sc: Optional[torch.Tensor] = None
        if not self.first:
            sc = self._skip(node_feats, node_attrs, perm, inv_perm, counts)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats) * cutoff
        if pair_id is not None:
            tp_weights = tp_weights.index_select(0, pair_id)
        message = self.conv(node_feats, edge_attrs, tp_weights, edge_index)
        message = self.linear(message) / self.avg_num_neighbors
        if self.first:
            message = self._skip(message, node_attrs, perm, inv_perm, counts)
        x = self.reshape(message)
        return self.product(x, sc, node_attrs, species32)


class FastReadout(nn.Module):
    def __init__(self, ro, lin1: nn.Module, lin2: Optional[nn.Module]):
        super().__init__()
        self.nonlinear: bool = lin2 is not None
        self.lin1 = lin1
        if lin2 is not None:
            self.act = ro.non_linearity
            self.lin2 = lin2
        else:
            self.act = nn.Identity()
            self.lin2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinear:
            return self.lin2(self.act(self.lin1(x)))
        return self.lin1(x)


# ---------------------------------------------------------------------------
#  the model
# ---------------------------------------------------------------------------
class FastMACE(nn.Module):
    _c_counts: List[int]

    def __init__(self, m, half_radial: bool = True, species_skip: bool = True,
                 plain_linear: bool = True, cache_species: bool = False):
        super().__init__()
        from mace.modules.blocks import RealAgnosticInteractionBlock, LinearReadoutBlock
        dev = next(m.parameters()).device
        dt = next(m.parameters()).dtype
        cueq = type(m.interactions[0].linear_up).__module__.startswith("cuequivariance")
        irmul = cueq  # rebuild.py builds cuEq models with layout ir_mul
        plain_linear = plain_linear and cueq
        self.full = m
        self.half_radial: bool = half_radial
        self.species_skip: bool = species_skip
        self.cache_species: bool = cache_species
        self.r_max = m.r_max
        self.atomic_numbers = m.atomic_numbers
        self.n_species: int = int(m.atomic_numbers.numel())
        S = self.n_species
        eye = torch.eye(S, dtype=dt, device=dev)

        def lin(mod, irr_in, irr_out):
            if not plain_linear:
                return mod
            return extract_block_linear(lambda x, s: mod(x), irr_in, irr_out, irmul, 1, dt, dev)

        def skip(mod, irr_in, irr_out):
            if not species_skip:
                return StockSkip(mod)
            f = lambda x, s: mod(x, eye[s].expand(x.shape[0], S).contiguous())
            return SpeciesLinear(extract_block_linear(f, irr_in, irr_out, irmul, S, dt, dev))

        # node embedding: one-hot (10x0e) -> 128x0e; the plain version is a matmul
        ne = m.node_embedding.linear
        self.node_embedding = lin(ne, ne.irreps_in, ne.irreps_out)
        layers = []
        for inter, prod in zip(m.interactions, m.products):
            first = isinstance(inter, RealAgnosticInteractionBlock)
            fused = bool(getattr(inter, "conv_fusion", False)) or \
                type(inter.conv_tp).__name__ == "FusedConvTP"
            st = inter.skip_tp
            sk = skip(st, st.irreps_in1, st.irreps_out)
            lu = lin(inter.linear_up, inter.linear_up.irreps_in, inter.linear_up.irreps_out)
            li = lin(inter.linear, inter.linear.irreps_in, inter.linear.irreps_out)
            pl = None
            if cueq:
                pl = lin(prod.linear, prod.linear.irreps_in, prod.linear.irreps_out)
            layers.append(FastLayer(inter, prod, first, irmul, fused, sk, species_skip, lu, li, pl))
        self.layers = nn.ModuleList(layers)
        ros = []
        for ro, p in zip(m.readouts, m.products):
            nf = p.linear.irreps_out
            if isinstance(ro, LinearReadoutBlock):
                ros.append(FastReadout(ro, lin(ro.linear, nf, o3.Irreps("1x0e")), None))
            else:
                h = ro.hidden_irreps
                ros.append(FastReadout(ro, lin(ro.linear_1, nf, h), lin(ro.linear_2, h, o3.Irreps("1x0e"))))
        self.readouts = nn.ModuleList(ros)
        self.spherical_harmonics = m.spherical_harmonics
        self.radial_embedding = m.radial_embedding
        self.atomic_energies_fn = m.atomic_energies_fn
        self.scale_shift = m.scale_shift
        # species-group cache (only with cache_species=True)
        self._c_n: int = -1
        self._c_perm = torch.zeros(0, dtype=torch.long)
        self._c_inv = torch.zeros(0, dtype=torch.long)
        self._c_counts: List[int] = []

    # -- species grouping -------------------------------------------------
    def _groups(self, species: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        n = species.shape[0]
        if self.cache_species and n == self._c_n and self._c_perm.device == species.device:
            return self._c_perm, self._c_inv, self._c_counts
        perm = torch.sort(species, stable=True)[1]
        inv = torch.empty_like(perm)
        inv.scatter_(0, perm, torch.arange(n, device=species.device))
        counts: List[int] = torch.bincount(species, minlength=self.n_species).cpu().tolist()
        if self.cache_species:
            self._c_n = n
            self._c_perm = perm
            self._c_inv = inv
            self._c_counts = counts
        return perm, inv, counts

    def forward(self, data: Dict[str, torch.Tensor], training: bool = False,
                compute_force: bool = True, compute_virials: bool = False,
                compute_stress: bool = False, compute_displacement: bool = False,
                compute_hessian: bool = False) -> Dict[str, Optional[torch.Tensor]]:
        if compute_virials or compute_stress or compute_displacement or compute_hessian or training:
            return self.full(data, training=training, compute_force=compute_force,
                             compute_virials=compute_virials, compute_stress=compute_stress,
                             compute_displacement=compute_displacement,
                             compute_hessian=compute_hessian)
        positions = data["positions"]
        positions.requires_grad_(True)
        node_attrs = data["node_attrs"]
        edge_index = data["edge_index"]
        batch = data["batch"]
        num_graphs = data["ptr"].numel() - 1
        n = positions.shape[0]
        dev = positions.device

        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = positions.index_select(0, receiver) - positions.index_select(0, sender) + data["shifts"]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)

        node_e0 = self.atomic_energies_fn(node_attrs)[:, 0]
        # energies are accumulated in float64 (no-op for fp64 models; for the fp32 EXTRA
        # build it removes the O(1e1) kcal/mol fp32 summation error of E0 + interaction
        # energies on ~1e6 kcal/mol totals -- forces are unaffected)
        e0 = torch.zeros(num_graphs, dtype=torch.float64, device=dev).index_add_(
            0, batch, node_e0.to(torch.float64))

        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(vectors)

        pair_id: Optional[torch.Tensor] = None
        E = sender.shape[0]
        if self.half_radial and E > 0:
            hidx = torch.nonzero_static(sender < receiver, size=E // 2, fill_value=0).squeeze(1)
            lo = torch.minimum(sender, receiver)
            hi = torch.maximum(sender, receiver)
            keys = lo * n + hi
            skeys, order = torch.sort(keys.index_select(0, hidx))
            pos = torch.searchsorted(skeys, keys).clamp_(max=max(E // 2 - 1, 0))
            torch._assert_async(torch.all(skeys.index_select(0, pos) == keys))
            pair_id = order.index_select(0, pos)
            ef_len = lengths.index_select(0, hidx)
            ef_ei = edge_index.index_select(1, hidx)
        else:
            ef_len = lengths
            ef_ei = edge_index
        edge_feats, cutoff_opt = self.radial_embedding(ef_len, node_attrs, ef_ei, self.atomic_numbers)
        if cutoff_opt is None:
            cutoff = torch.ones_like(ef_len)
        else:
            cutoff = cutoff_opt

        species = torch.argmax(node_attrs, dim=1)
        species32 = species.to(torch.int32)
        if self.species_skip:
            perm, inv_perm, counts = self._groups(species)
        else:
            perm, inv_perm, counts = species, species, [0]

        node_es = torch.zeros(n, dtype=positions.dtype, device=dev)
        for layer, ro in zip(self.layers, self.readouts):
            node_feats = layer(node_feats, node_attrs, species32, edge_attrs, edge_feats, cutoff,
                               edge_index, pair_id, perm, inv_perm, counts)
            node_es = node_es + ro(node_feats)[:, 0]
        node_inter_es = self.scale_shift(node_es, torch.zeros_like(batch))
        inter_e = torch.zeros(num_graphs, dtype=torch.float64, device=dev).index_add_(
            0, batch, node_inter_es.to(torch.float64))
        total = e0 + inter_e
        forces: Optional[torch.Tensor] = None
        if compute_force:
            go: List[Optional[torch.Tensor]] = [torch.ones_like(inter_e)]
            g = torch.autograd.grad([inter_e], [positions], go,
                                    retain_graph=False, create_graph=False, allow_unused=True)[0]
            if g is None:
                forces = torch.zeros_like(positions)
            else:
                forces = -g
        out: Dict[str, Optional[torch.Tensor]] = {
            "energy": total, "forces": forces, "node_energy": None, "virials": None,
            "stress": None, "displacement": None, "hessian": None,
        }
        return out


