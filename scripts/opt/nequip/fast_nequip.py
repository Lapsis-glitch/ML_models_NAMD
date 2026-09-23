"""FastNequIP: exact (same weights, same math) rewrites of NequIP-OAM-L for inference.

Applied to the EAGER model after `enable_OpenEquivariance`, before TorchScript compilation
(see build_fast.py).  Nothing is retrained or approximated; outputs differ from the stock model only
by fp32 summation order (same noise floor as recompiling the stock model).

1. half_edges  -- every directed edge (i->j) has its reverse (j->i) in a proper neighbour list.  The
   edge MLP (Bessel(|r|) -> 8->128->1312 per layer), the Bessel/cutoff embedding and the spherical
   harmonics are computed once per UNDIRECTED pair; the OEQ TP-convolution is then called twice with the
   SAME per-pair weights: once for (dst=i, src=j, Y(r)) and once for (dst=j, src=i, Y(-r)), where
   Y_l(-r) = (-1)^l Y_l(r) bit-exactly (odd/even homogeneous polynomials of the normalised vector).
   No [E, 1312] gather is materialised, so the edge-MLP GEMMs (fwd + bwd) AND the per-edge weight
   memory are halved.  ZBL, which sums its (i,j)-symmetric pair energy onto the centre atom of every
   directed edge, scatters each half-edge energy onto both atoms instead (identical per-atom energies).
   The canonical half is (i < j) or (i == j and the integer cell shift is lexicographically positive),
   so periodic self-images work too.  Selected sync-free with nonzero_static(size = E // 2).
2. species_sc -- the self-connection FullyConnectedTensorProduct(x, node_attrs) with node_attrs =
   48-d type embedding is linear in x for a fixed atom type.  Its per-type block matrices are extracted
   once (probing the module in float64) and applied as one dense matmul per atom type present
   (kron(W_l, I_{2l+1}) blocks in e3nn's mul_ir layout).  One tiny D2H copy per call (type counts).
3. The `[:num_local_nodes]` slices of InteractionBlock (no-ops outside LAMMPS ghost exchange) are dropped.
"""
from typing import Dict, List

import torch

NODE_FEATURES = "node_features"
NODE_ATTRS = "node_attrs"
EDGE_INDEX = "edge_index"
EDGE_ATTRS = "edge_attrs"
EDGE_EMBEDDING = "edge_embedding"
ATOM_TYPE = "atom_types"
CELL_SHIFT = "edge_cell_shift"
EDGE_LENGTH = "edge_lengths"
NORM_LENGTH = "normed_edge_lengths"


class HalfEdgeSelect(torch.nn.Module):
    """First module of the network: keep one direction of every edge pair, and precompute the
    atom-type sort used by SpeciesSelfConnection."""

    half_edges: torch.jit.Final[bool]
    species_sc: torch.jit.Final[bool]

    def __init__(self, num_types: int, half_edges: bool, species_sc: bool):
        super().__init__()
        self.num_types = num_types
        self.half_edges = half_edges
        self.species_sc = species_sc

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.half_edges:
            ei = data["edge_index"]
            E = ei.size(1)
            i = ei[0]
            j = ei[1]
            if "edge_cell_shift" in data:
                s = data["edge_cell_shift"]
                pos_shift = (s[:, 0] > 0) | ((s[:, 0] == 0) & ((s[:, 1] > 0) | ((s[:, 1] == 0) & (s[:, 2] > 0))))
                keep = (i < j) | ((i == j) & pos_shift)
            else:
                keep = i < j
            # a symmetric edge list has exactly E/2 canonical edges; checked on the device, no sync
            torch._assert_async(keep.sum() * 2 == E)
            idx = torch.nonzero_static(keep, size=E // 2).view(-1)
            data["edge_index"] = ei.index_select(1, idx)
            if "edge_cell_shift" in data:
                data["edge_cell_shift"] = data["edge_cell_shift"].index_select(0, idx)
            if "edge_transpose_perm" in data:
                data.pop("edge_transpose_perm")
        if self.species_sc:
            t = data["atom_types"].view(-1)
            counts = torch.bincount(t, minlength=self.num_types).cpu()
            present = torch.nonzero(counts).view(-1)
            data["_sp_order"] = torch.argsort(t, stable=True)
            data["_sp_present"] = present
            data["_sp_counts"] = counts.index_select(0, present)
        return data


class SpeciesSelfConnection(torch.nn.Module):
    """Exact per-atom-type rewrite of FCTP(x, type_embedding): for atom type s the map is the dense
    block matrix D_s[in_k, out_k] = kron(W_k[s], I_{d_k}) (e3nn mul_ir layout), one mm per type present."""

    nl: torch.jit.Final[int]
    din: torch.jit.Final[int]
    dout: torch.jit.Final[int]
    i0: torch.jit.Final[int]
    i1: torch.jit.Final[int]
    i2: torch.jit.Final[int]
    i3: torch.jit.Final[int]
    o0: torch.jit.Final[int]
    o1: torch.jit.Final[int]
    o2: torch.jit.Final[int]
    o3: torch.jit.Final[int]

    def __init__(self, blocks, din: int, dout: int):
        """blocks: list of (in_off, out_off, d, [num_types, mul_in, mul_out] tensor), at most 4."""
        super().__init__()
        assert 1 <= len(blocks) <= 4
        self.nl = len(blocks)
        self.din = din
        self.dout = dout
        pad = blocks + [(0, 0, 1, torch.zeros(1, 1, 1, dtype=blocks[0][3].dtype))] * (4 - len(blocks))
        self.i0, self.o0 = pad[0][0], pad[0][1]
        self.i1, self.o1 = pad[1][0], pad[1][1]
        self.i2, self.o2 = pad[2][0], pad[2][1]
        self.i3, self.o3 = pad[3][0], pad[3][1]
        self.w0 = torch.nn.Parameter(pad[0][3].clone(), requires_grad=False)
        self.w1 = torch.nn.Parameter(pad[1][3].clone(), requires_grad=False)
        self.w2 = torch.nn.Parameter(pad[2][3].clone(), requires_grad=False)
        self.w3 = torch.nn.Parameter(pad[3][3].clone(), requires_grad=False)
        self.e0 = torch.nn.Parameter(torch.eye(pad[0][2], dtype=pad[0][3].dtype), requires_grad=False)
        self.e1 = torch.nn.Parameter(torch.eye(pad[1][2], dtype=pad[0][3].dtype), requires_grad=False)
        self.e2 = torch.nn.Parameter(torch.eye(pad[2][2], dtype=pad[0][3].dtype), requires_grad=False)
        self.e3 = torch.nn.Parameter(torch.eye(pad[3][2], dtype=pad[0][3].dtype), requires_grad=False)

    def dense(self, s: int) -> torch.Tensor:
        D = torch.zeros((self.din, self.dout), dtype=self.w0.dtype, device=self.w0.device)
        k = torch.kron(self.w0[s], self.e0)
        D[self.i0:self.i0 + k.size(0), self.o0:self.o0 + k.size(1)] = k
        if self.nl > 1:
            k = torch.kron(self.w1[s], self.e1)
            D[self.i1:self.i1 + k.size(0), self.o1:self.o1 + k.size(1)] = k
        if self.nl > 2:
            k = torch.kron(self.w2[s], self.e2)
            D[self.i2:self.i2 + k.size(0), self.o2:self.o2 + k.size(1)] = k
        if self.nl > 3:
            k = torch.kron(self.w3[s], self.e3)
            D[self.i3:self.i3 + k.size(0), self.o3:self.o3 + k.size(1)] = k
        return D

    def forward(self, x: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        present: List[int] = data["_sp_present"].tolist()
        counts: List[int] = data["_sp_counts"].tolist()
        if len(present) == 1:
            return torch.mm(x, self.dense(present[0]))
        order = data["_sp_order"]
        xs = x.index_select(0, order)
        outs: List[torch.Tensor] = []
        start = 0
        for k in range(len(present)):
            c = counts[k]
            outs.append(torch.mm(xs.narrow(0, start, c), self.dense(present[k])))
            start += c
        out_sorted = torch.cat(outs, 0)
        return torch.zeros_like(out_sorted).index_copy(0, order, out_sorted)


class FastInteractionBlock(torch.nn.Module):
    """InteractionBlock.forward with half-edge OEQ convolution and per-type self-connection."""

    half_edges: torch.jit.Final[bool]
    use_species_sc: torch.jit.Final[bool]
    has_sc: torch.jit.Final[bool]
    use_norm_module: torch.jit.Final[bool]
    has_alpha: torch.jit.Final[bool]
    alpha: torch.jit.Final[float]

    def __init__(self, old, half_edges: bool, sc_module, sh_signs: torch.Tensor):
        super().__init__()
        self.linear_1 = old.linear_1
        # nequip 0.17: AvgNumNeighborsNorm module; nequip 0.14 (the packaged OAM-L): scalar scatter_norm_factor
        self.use_norm_module = hasattr(old, "avg_num_neighbors_norm")
        self.avg_num_neighbors_norm = old.avg_num_neighbors_norm if self.use_norm_module else torch.nn.Identity()
        a = getattr(old, "scatter_norm_factor", None)
        self.has_alpha = a is not None
        self.alpha = float(a) if a is not None else 1.0
        self.edge_mlp = old.edge_mlp
        self.linear_2 = old.linear_2
        self.tp_conv = old.tp_scatter.tp_conv  # OEQ TensorProductConv
        self.model_dtype = old.tp_scatter.model_dtype
        self.half_edges = half_edges
        self.use_species_sc = sc_module is not None and not isinstance(sc_module, torch.nn.Identity) \
            and isinstance(sc_module, SpeciesSelfConnection)
        self.has_sc = old.sc is not None
        self.sc = sc_module if self.use_species_sc else (old.sc if old.sc is not None else torch.nn.Identity())
        self.sh_signs = torch.nn.Parameter(sh_signs.clone(), requires_grad=False)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data["node_features"]
        sc = x
        if self.has_sc:
            if self.use_species_sc:
                sc = self.sc(x, data)
            else:
                sc = self.sc(x, data["node_attrs"])
        x = self.linear_1(x)
        if self.use_norm_module:
            data["node_features"] = x
            data = self.avg_num_neighbors_norm(data)
            x = data["node_features"]
        elif self.has_alpha:
            x = self.alpha * x
        x = x.to(self.model_dtype)
        y = data["edge_attrs"].to(self.model_dtype)
        w = self.edge_mlp(data["edge_embedding"]).to(self.model_dtype)
        ei = data["edge_index"]
        m = self.tp_conv(x, y, w, ei[0], ei[1])
        if self.half_edges:
            m = m + self.tp_conv(x, y * self.sh_signs, w, ei[1], ei[0])
        x = self.linear_2(m)
        if self.has_sc:
            x = x + sc
        data["node_features"] = x
        return data


class HalfEdgeZBL(torch.nn.Module):
    """nequip ZBL on a half edge list: the pair energy goes to BOTH atoms (the stock module puts it on
    the centre atom of each of the two directed edges -> identical per-atom energies)."""

    use_cutoff_module: torch.jit.Final[bool]

    def __init__(self, old):
        super().__init__()
        self._zbl = old._zbl
        self.atomic_numbers = old.atomic_numbers
        self._qqr2exesquare = float(old._qqr2exesquare)
        # nequip 0.17 multiplies by self.cutoff(normed length); 0.14 (packaged OAM-L) by data["edge_cutoff"]
        self.use_cutoff_module = hasattr(old, "cutoff")
        self.cutoff = old.cutoff if self.use_cutoff_module else torch.nn.Identity()
        self.per_atom_energy_field = getattr(old, "per_atom_energy_field", "atomic_energy")
        self.model_dtype = getattr(old, "model_dtype", torch.get_default_dtype())

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ei = data["edge_index"]
        num_nodes = data[self.per_atom_energy_field].size(0)
        e = self._zbl(Z=self.atomic_numbers, r=data["edge_lengths"].view(-1), atom_types=data["atom_types"],
                      edge_index=ei, qqr2exesquare=self._qqr2exesquare).unsqueeze(-1)
        if self.use_cutoff_module:
            e = e * self.cutoff(data["normed_edge_lengths"]).to(self.model_dtype)
        else:
            e = e * data["edge_cutoff"]
        atomic = torch.zeros((num_nodes, 1), dtype=e.dtype, device=e.device)
        atomic = atomic.index_add(0, ei[0], e).index_add(0, ei[1], e)
        data[self.per_atom_energy_field] = atomic + data[self.per_atom_energy_field]
        return data


# ----------------------------------------------------------------------------------------------
#  surgery
# ----------------------------------------------------------------------------------------------
def _irreps_layout(irreps):
    """[(mul, 2l+1, offset)] of an e3nn Irreps in mul_ir layout."""
    out, off = [], 0
    for mul, ir in irreps:
        out.append((mul, ir.dim, off))
        off += mul * ir.dim
    return out, off


@torch.no_grad()
def extract_species_blocks(sc_mod, type_embeddings: torch.Tensor):
    """Probe FCTP(x, e_s) in float64 for every type s. Every output irrep block must come from the input
    block of the same irrep as kron(W, I_d); returns [(in_off, out_off, d, W[num_types, mul_in, mul_out])],
    din, dout and the worst relative reconstruction error of the dense map."""
    import copy
    m = copy.deepcopy(sc_mod).double()
    lin, din = _irreps_layout(m.irreps_in1)
    lout, dout = _irreps_layout(m.irreps_out)
    irs_in = [ir for _, ir in m.irreps_in1]
    pairs = []  # (in block idx, out block idx)
    for ko, (_, ir) in enumerate(m.irreps_out):
        pairs.append((irs_in.index(ir), ko))
    dev = type_embeddings.device
    eye = torch.eye(din, dtype=torch.float64, device=dev)
    Ws = [[] for _ in pairs]
    worst = 0.0
    for s in range(type_embeddings.size(0)):
        e = type_embeddings[s].double().expand(din, -1)
        D = m(eye, e)  # row k = map applied to basis vector k -> dense [din, dout]
        rebuilt = torch.zeros_like(D)
        for p, (ki, ko) in enumerate(pairs):
            mi, d, oi = lin[ki]
            mo, d2, oo = lout[ko]
            assert d == d2
            W = D[oi:oi + mi * d:d, oo:oo + mo * d:d].contiguous()  # m=0 component: [mi, mo]
            Ws[p].append(W.float())
            rebuilt[oi:oi + mi * d, oo:oo + mo * d] = torch.kron(W, torch.eye(d, dtype=W.dtype, device=dev))
        worst = max(worst, ((D - rebuilt).abs().max() / D.abs().max().clamp_min(1e-30)).item())
    assert worst < 1e-12, f"self-connection is not kron-block structured (rel err {worst})"
    blocks = [(lin[ki][2], lout[ko][2], lin[ki][1], torch.stack(Ws[p])) for p, (ki, ko) in enumerate(pairs)]
    return blocks, din, dout, worst


def make_fast(model, half_edges: bool = True, species_sc: bool = True):
    """In-place surgery on an eager NequIP GraphModel (OEQ modifier already applied)."""
    from collections import OrderedDict
    # the package loader returns torch.package classes (<torch_package_0>.nequip...): match by class name
    tname = lambda m: type(m).__name__

    net = model.model.func  # GraphModel -> ForceStressOutput -> SequentialGraphNetwork
    type_embed = net.type_embed
    num_types = type_embed.embed_module.num_embeddings
    dev = next(model.parameters()).device
    with torch.no_grad():
        type_emb = type_embed({"atom_types": torch.arange(num_types, device=dev)})["node_attrs"]
    report = {}
    new = OrderedDict()
    new["fast_half_select"] = HalfEdgeSelect(num_types, half_edges, species_sc)
    for name, mod in net._modules.items():
        if name.endswith("_convnet"):
            blk = mod.conv
            assert tname(blk) == "InteractionBlock" and tname(blk.tp_scatter) == "OpenEquivarianceTensorProductScatter", (name, tname(blk))
            sc_mod = None
            if species_sc and blk.sc is not None:
                blocks, din, dout, err = extract_species_blocks(blk.sc, type_emb)
                sc_mod = SpeciesSelfConnection(blocks, din, dout).to(dev)
                report[name] = err
            signs = []
            for mul, ir in blk.tp_scatter.irreps_edge_attr:
                signs += [(-1.0) ** ir.l] * (mul * ir.dim)
            mod.conv = FastInteractionBlock(blk, half_edges, sc_mod,
                                            torch.tensor(signs, dtype=blk.tp_scatter.model_dtype, device=dev))
            new[name] = mod
        elif tname(mod) == "ZBL" and half_edges:
            new[name] = HalfEdgeZBL(mod)
        else:
            new[name] = mod
    net._modules = new
    return report
