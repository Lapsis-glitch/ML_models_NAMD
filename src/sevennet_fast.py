"""
FastSevenNet: exact rewrites of a SevenNet model for inference, applied while
SevenNet's own LAMMPS serial deployment is built (``src.compile_sevennet --fast``).

The deployment uses SevenNet's OpenEquivariance convolutions (``use_oeq``), so
it needs the op library ``scripts/opt/nequip/oeq_native/liboeq_native.so`` at
run time (``NAMD_MLFF_EXTRA_LIBS``).  With OEQ the model is host-bound below
~1000 atoms (hundreds of tiny kernel launches per call); the rewrites cut the
op count.  Weights and maths are unchanged:

  linear  every e3nn ``Linear`` becomes one dense matmul with the equivalent
          block matrix, probed from the module in fp64 (kron(W_l, I_{2l+1})
          blocks, zeros elsewhere).  A multi-fidelity linear takes the modality
          one-hot as extra inputs; the modality is fixed at deploy time, so its
          rows become a constant bias (addmm).
  gate    e3nn ``Gate`` becomes split, act(scalars), gated * act(gates) expanded
          by one index_select; normalize2mom and path constants in one scale.
  radial  the radial MLPs of all convolutions run once, fused: one matmul for
          the shared first layer, one bmm per hidden layer, one matmul per
          convolution for the last; act constants and each convolution's
          1/denominator folded into the weights (the conv is linear in them).
          The convolutions read their weights from the data dict and call the
          OEQ op with int64 indices directly.
  fuse    each layer's self-connection and first self-interaction linear read
          the same features: one matmul for both.  A gate's output constants
          are folded into the rows of the matrix that consumes its output.

Every rewrite is checked in fp64 against what it replaces before the model is
scripted.  A model whose structure a rewrite doesn't know raises
``RewriteNotApplicable``; ``compile_sevennet`` then falls back to the OEQ
kernels without rewrites.
"""

import contextlib
import copy
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from torch import nn

REWRITES = ("linear", "gate", "radial", "fuse")
F64 = torch.float64


class RewriteNotApplicable(RuntimeError):
    """The model has a structure a rewrite doesn't handle."""


# ----------------------------------------------------------------------------- linear
class DenseLinear(nn.Module):
    """y = x @ W (+ b) for an e3nn Linear (W probed, see ``_dense_of``)."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.has_bias = bias is not None
        self.register_buffer("bias", bias if bias is not None else torch.zeros(0, dtype=weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_bias:
            return torch.addmm(self.bias, x, self.weight)
        return torch.mm(x, self.weight)


def _dense_of(lin: nn.Module) -> torch.Tensor:
    """fp64 W with lin(x) == x @ W, probed on the identity."""
    lin64 = copy.deepcopy(lin).to(F64).cpu()
    d_in = lin.irreps_in.dim
    with torch.no_grad():
        W = lin64(torch.eye(d_in, dtype=F64))
        if lin64(torch.zeros(1, d_in, dtype=F64)).abs().max().item() != 0.0:
            raise RewriteNotApplicable("linear: e3nn Linear with biases")
        x = torch.randn(7, d_in, dtype=F64)
        err = (lin64(x) - x @ W).abs().max().item()
    assert err < 1e-12, f"linear rewrite mismatch {err}"
    return W


def _modal_index(model: nn.Module) -> Optional[int]:
    prep = model._modules.get("modal_input_prepare")
    return None if prep is None else int(prep.modal_idx)


def rewrite_linears(model: nn.Module) -> int:
    from e3nn.o3 import Linear
    modal = _modal_index(model)
    n = 0
    for mod in model.modules():
        lin = getattr(mod, "linear", None)
        if not isinstance(lin, Linear):
            continue
        W = _dense_of(lin)
        n_modal = getattr(mod, "num_modalities", 0)
        if n_modal > 1:
            # input = [x | one-hot(modality)] (IrrepsLinear._patch_modal_to_data)
            if modal is None:
                raise RewriteNotApplicable("linear: multi-modal model deployed without --modal")
            d_x = W.shape[0] - n_modal
            mod.linear = DenseLinear(W[:d_x].contiguous(), W[d_x + modal].clone())
            mod.num_modalities = 0  # the modality is now in the bias; don't append it
        else:
            mod.linear = DenseLinear(W)
        n += 1
    return n


# ------------------------------------------------------------------------------- gate
def _act_kind(f) -> int:
    import torch.nn.functional as F
    if f is F.silu or isinstance(f, nn.SiLU):
        return 0
    if f is torch.tanh or isinstance(f, nn.Tanh):
        return 1
    raise RewriteNotApplicable(f"unsupported activation {f!r}")


def _act(x: torch.Tensor, kind: int) -> torch.Tensor:
    if kind == 0:
        return torch.nn.functional.silu(x)
    return torch.tanh(x)


def _single_act(activation):
    """(kind, cst) of an e3nn Activation with one activation for all its irreps."""
    kinds, csts = set(), set()
    for a in activation.acts:
        if a is None:
            raise RewriteNotApplicable("gate: identity activation")
        kinds.add(_act_kind(a.f))
        csts.add(1.0 if a._is_id else float(a.cst))
    if len(kinds) != 1 or len(csts) != 1:
        raise RewriteNotApplicable("gate: mixed activations in one block")
    return kinds.pop(), csts.pop()


class FastGate(nn.Module):
    """e3nn Gate as  [act_s(s) * out_scale_s,  q * act_g(g)[:, gate_of_feature] * out_scale_q]
    for x = [s | g | q].  ``fold_scales`` (the ``fuse`` rewrite) moves the output
    scales into the consumer's matrix; the gate then only multiplies by the gate."""

    def __init__(self, gate: nn.Module):
        super().__init__()
        from e3nn.o3 import Irreps
        cut = gate.sc.cut
        slices = Irreps(cut.irreps_in).slices()
        parts = []
        for ins in cut.instructions:
            idx = list(ins)
            if idx and idx != list(range(idx[0], idx[-1] + 1)):
                raise RewriteNotApplicable("gate: non-contiguous gate block")
            parts.append((slices[idx[0]].start, slices[idx[-1]].stop) if idx else None)
        (s0, s1), gpart, qpart = parts
        self.has_gates = gpart is not None
        dim = Irreps(cut.irreps_in).dim
        if s0 != 0 or (self.has_gates and not (s1 == gpart[0] and gpart[1] == qpart[0] and qpart[1] == dim)) \
                or (not self.has_gates and s1 != dim):
            raise RewriteNotApplicable("gate: features not laid out as [scalars | gates | gated]")
        self.split_sizes = [s1] + ([gpart[1] - gpart[0], qpart[1] - qpart[0]] if self.has_gates else [])
        self.act_s, cst_s = _single_act(gate.act_scalars)
        out_scale = [cst_s] * s1
        idx = []
        self.act_g = 0
        if self.has_gates:
            self.act_g, cst_g = _single_act(gate.act_gates)
            gated_irreps = Irreps(gate.irreps_gated)
            # the ElementwiseTensorProduct's per-path constant, probed on ones
            with torch.no_grad():
                c = copy.deepcopy(gate.mul).to(F64).cpu()(
                    torch.ones(1, gated_irreps.dim, dtype=F64),
                    torch.ones(1, gate.irreps_gates.dim, dtype=F64))[0]
            col = 0
            for mul, ir in gated_irreps:
                for u in range(mul):
                    idx += [col + u] * ir.dim
                col += mul
            out_scale += (c * cst_g).tolist()
        self.register_buffer("gate_index", torch.tensor(idx, dtype=torch.long))
        self.register_buffer("out_scale", torch.tensor(out_scale, dtype=F64))
        self.scaled = True

    def fold_scales(self) -> torch.Tensor:
        """Stop applying the output scales; return them for the consumer's rows."""
        scale = self.out_scale.clone()
        self.scaled = False
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = torch.split(x, self.split_sizes, dim=1)
        s = _act(parts[0], self.act_s)
        if self.has_gates:
            s = torch.cat([s, parts[2] * _act(parts[1], self.act_g).index_select(1, self.gate_index)], dim=1)
        if self.scaled:
            s = s * self.out_scale
        return s


def rewrite_gates(model: nn.Module) -> int:
    from e3nn.nn import Gate
    n = 0
    for mod in model.modules():
        g = getattr(mod, "gate", None)
        if isinstance(g, Gate):
            fast = FastGate(g)
            x = torch.randn(11, g.irreps_in.dim, dtype=F64)
            with torch.no_grad():
                err = (copy.deepcopy(g).to(F64).cpu()(x) - fast(x)).abs().max().item()
            assert err < 1e-12, f"gate rewrite mismatch {err}"
            mod.gate = fast
            n += 1
    return n


# ----------------------------------------------------------------------------- radial
def _fcn_layers(fcn: nn.Module):
    """[(W fp64 [h_in, h_out] with the layer's normalisation, act kind or None,
    act output constant)] of an e3nn FullyConnectedNet."""
    out = []
    for i in range(len(fcn.hs) - 1):
        L = getattr(fcn, f"layer{i}")
        W = L.weight.detach().to(F64).cpu()
        if L.act is not None:
            W = W / (L.h_in * L.var_in) ** 0.5
            cst = (1.0 if L.act._is_id else float(L.act.cst)) * L.var_out ** 0.5
            out.append((W, _act_kind(L.act.f), cst))
        else:
            out.append((W / (L.h_in * L.var_in / L.var_out) ** 0.5, None, 1.0))
    return out


class FastRadial(nn.Module):
    """All convolutions' radial MLPs at once (see the module docstring)."""

    def __init__(self, convs, key_in: str, keys_out):
        super().__init__()
        layers = [_fcn_layers(c.weight_nn) for c in convs]
        depth = len(layers[0])
        hs = [[W.shape[1] for W, _, _ in ls] for ls in layers]
        if depth < 2 or any(len(ls) != depth for ls in layers) or any(h[:-1] != hs[0][:-1] for h in hs):
            raise RewriteNotApplicable("radial: convolutions have different MLP shapes")
        kinds = {k for ls in layers for _, k, _ in ls[:-1]}
        if len(kinds) != 1 or None in kinds or any(ls[-1][1] is not None for ls in layers):
            raise RewriteNotApplicable("radial: unsupported MLP activations")
        self.kind = kinds.pop()
        self.C = len(convs)
        self.h = hs[0][0]
        self.n_hidden = depth - 2
        self.key_in = key_in
        self.keys_out = list(keys_out)
        mats = []
        for ls in layers:  # fold each act's output constant into the next layer's rows
            Ws, carry = [], 1.0
            for W, _, cst in ls:
                Ws.append(W * carry)
                carry = cst
            mats.append(Ws)
        self.register_buffer("w_first", torch.cat([m[0] for m in mats], dim=1))
        self.register_buffer("w_hidden", torch.stack(
            [torch.stack([m[k] for m in mats]) for k in range(1, depth - 1)])
            if depth > 2 else torch.zeros(0, dtype=F64))
        dens = [float(c.denominator.detach().double()) for c in convs]
        self.last = nn.ModuleList([DenseLinear(m[-1] / dens[c]) for c, m in enumerate(mats)])

    def mlp(self, emb: torch.Tensor) -> List[torch.Tensor]:
        E = emb.size(0)
        h = _act(torch.mm(emb, self.w_first), self.kind)             # [E, C*h]
        h = h.view(E, self.C, self.h).transpose(0, 1)               # [C, E, h]
        for k in range(self.n_hidden):
            h = _act(torch.bmm(h, self.w_hidden[k]), self.kind)
        hc = torch.unbind(h, 0)
        out: List[torch.Tensor] = []
        for c, lin in enumerate(self.last):
            out.append(lin(hc[c]))
        return out

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ws = self.mlp(data[self.key_in])
        for c in range(self.C):
            data[self.keys_out[c]] = ws[c]
        return data


class FastConv(nn.Module):
    """IrrepsScatterGatterFusedConvolution with precomputed (1/denominator-scaled)
    weights, calling the OEQ op with int64 indices."""

    def __init__(self, conv: nn.Module, key_weight: str):
        super().__init__()
        self.tp_conv = conv.convolution.tp_conv
        self.key_x = conv.key_x
        self.key_filter = conv.key_filter
        self.key_edge_idx = conv.key_edge_idx
        self.key_weight = key_weight
        self.out_dim: int = conv._out_dim

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = data[self.key_x]
        edge_filter = data[self.key_filter]
        weight = data[self.key_weight]
        edge_idx = data[self.key_edge_idx]
        if edge_idx.size(1) == 0:
            out = x.new_zeros(x.shape[0], self.out_dim)
            data[self.key_x] = out + (edge_filter.sum() + weight.sum()) * 0
        else:
            # OEQ rows = dst = edge_idx[0], cols = src = edge_idx[1]
            data[self.key_x] = self.tp_conv(x, edge_filter, weight, edge_idx[0], edge_idx[1])
        return data


def rewrite_radial(model: nn.Module) -> int:
    from sevenn.nn.convolution import IrrepsScatterGatterFusedConvolution
    names = [k for k, m in model._modules.items() if isinstance(m, IrrepsScatterGatterFusedConvolution)]
    if not names:
        raise RewriteNotApplicable("radial: needs the OEQ (scatter/gather fused) convolutions")
    convs = [model._modules[k] for k in names]
    if any(c.is_parallel for c in convs) or len({c.key_weight_input for c in convs}) != 1 \
            or not all(hasattr(c.convolution, "tp_conv") for c in convs):
        raise RewriteNotApplicable("radial: unsupported convolution setup")
    if "edge_embedding" not in model._modules:
        raise RewriteNotApplicable("radial: no edge_embedding module")
    keys = [f"_fast_conv_weight_{i}" for i in range(len(convs))]
    radial = FastRadial(convs, convs[0].key_weight_input, keys)
    emb = torch.rand(13, convs[0].weight_nn.hs[0], dtype=F64) * 2
    with torch.no_grad():
        got = radial.mlp(emb)
        for c, conv in enumerate(convs):
            want = copy.deepcopy(conv.weight_nn).to(F64).cpu()(emb) / float(conv.denominator)
            err = (got[c] - want).abs().max().item() / max(1.0, want.abs().max().item())
            assert err < 1e-12, f"radial rewrite mismatch {err}"
    new = OrderedDict()
    for k, m in model._modules.items():
        new[k] = FastConv(m, keys[names.index(k)]) if k in names else m
        if k == "edge_embedding":
            new["_fast_radial"] = radial
    model._modules.clear()
    model._modules.update(new)
    return len(convs)


# ------------------------------------------------------------------------------- fuse
class FusedIntro(nn.Module):
    """SelfConnectionLinearIntro + the following IrrepsLinear on the same input:
    [temp | x] = x @ [W_intro | W_si1] (+ [0 | b_si1])."""

    def __init__(self, intro: nn.Module, si1: nn.Module, key_temp: str):
        super().__init__()
        a, b = intro.linear, si1.linear
        bias = None
        if a.has_bias or b.has_bias:
            za = a.bias if a.has_bias else torch.zeros(a.weight.shape[1], dtype=a.weight.dtype)
            zb = b.bias if b.has_bias else torch.zeros(b.weight.shape[1], dtype=b.weight.dtype)
            bias = torch.cat([za, zb])
        self.lin = DenseLinear(torch.cat([a.weight, b.weight], dim=1), bias)
        self.sizes = [a.weight.shape[1], b.weight.shape[1]]
        self.key_x = intro.key_x
        self.key_temp = key_temp

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        y = torch.split(self.lin(data[self.key_x]), self.sizes, dim=1)
        data[self.key_temp] = y[0]
        data[self.key_x] = y[1]
        return data


def rewrite_fuse(model: nn.Module) -> str:
    import sevenn._keys as KEY
    from sevenn.nn.equivariant_gate import EquivariantGate
    from sevenn.nn.linear import IrrepsLinear
    from sevenn.nn.self_connection import SelfConnectionLinearIntro
    items = list(model._modules.items())
    new, fused, folded, i = OrderedDict(), 0, 0, 0
    while i < len(items):
        k, m = items[i]
        nxt = items[i + 1][1] if i + 1 < len(items) else None
        if (isinstance(m, SelfConnectionLinearIntro) and isinstance(nxt, IrrepsLinear)
                and isinstance(m.linear, DenseLinear) and isinstance(nxt.linear, DenseLinear)
                and nxt.key_input == m.key_x == nxt.key_output and nxt.num_modalities <= 1):
            f = FusedIntro(m, nxt, KEY.SELF_CONNECTION_TEMP)
            x = torch.randn(7, m.linear.weight.shape[0], dtype=F64)
            with torch.no_grad():
                t, y = torch.split(f.lin(x), f.sizes, dim=1)
                err = max((t - m.linear(x)).abs().max().item(), (y - nxt.linear(x)).abs().max().item())
            assert err < 1e-12, f"fuse (intro + si1) mismatch {err}"
            new[k] = f
            fused += 1
            i += 2
            continue
        new[k] = m
        i += 1
    model._modules.clear()
    model._modules.update(new)
    # fold each gate's output scales into the rows of the matrix that consumes it
    items = list(model._modules.items())
    for j, (k, m) in enumerate(items):
        if isinstance(m, EquivariantGate) and isinstance(m.gate, FastGate) and j + 1 < len(items):
            c = items[j + 1][1]
            if isinstance(c, FusedIntro) and c.key_x == m.key_x:
                lin = c.lin
            elif isinstance(c, IrrepsLinear) and isinstance(c.linear, DenseLinear) \
                    and c.key_input == m.key_x and c.num_modalities <= 1:
                lin = c.linear
            else:
                continue
            x = torch.randn(7, sum(m.gate.split_sizes), dtype=F64)
            with torch.no_grad():
                ref = lin(m.gate(x))
                lin.weight = lin.weight * m.gate.fold_scales()[:, None]
                err = (lin(m.gate(x)) - ref).abs().max().item() / max(1.0, ref.abs().max().item())
            assert err < 1e-12, f"fuse (gate scale fold) mismatch {err}"
            folded += 1
    return f"{fused} self-connection/self-interaction pairs fused, {folded} gate scales folded"


# ---------------------------------------------------------------------------- driver
def apply_rewrites(model: nn.Module, rewrites=REWRITES, log=print) -> None:
    """Rewrite a SevenNet model in place (deploy state: force_output removed,
    modality fixed).  Raises RewriteNotApplicable if the structure doesn't fit."""
    rewrites = list(rewrites)
    unknown = set(rewrites) - set(REWRITES)
    if unknown:
        raise ValueError(f"unknown rewrite(s) {sorted(unknown)}; choose from {REWRITES}")
    if "fuse" in rewrites and not {"linear", "gate"} <= set(rewrites):
        raise ValueError("fuse needs the linear and gate rewrites")
    dtype = next(model.parameters()).dtype
    for r in REWRITES:  # fixed order
        if r not in rewrites:
            continue
        if r == "linear":
            log(f"[fast] linear: {rewrite_linears(model)} e3nn Linear -> dense matmul")
        elif r == "gate":
            log(f"[fast] gate: {rewrite_gates(model)} e3nn Gate -> FastGate")
        elif r == "radial":
            log(f"[fast] radial: {rewrite_radial(model)} convolutions share one fused radial MLP")
        elif r == "fuse":
            log(f"[fast] fuse: {rewrite_fuse(model)}")
    # the new modules were built in fp64; cast their float buffers to the model dtype
    for mod in model.modules():
        if isinstance(mod, (DenseLinear, FastGate, FastRadial)):
            for name, b in mod.named_buffers(recurse=False):
                if b.is_floating_point():
                    setattr(mod, name, b.to(dtype))


@contextlib.contextmanager
def rewriting_deploy(rewrites=REWRITES, log=print):
    """Within this context, ``sevenn.scripts.deploy.deploy`` applies the rewrites
    to the model just before it scripts it (after SevenNet has removed the force
    module and fixed the modality), so the output is an ordinary deployment."""
    import e3nn.util.jit as ej
    orig_script = ej.script
    done = []

    def script(mod, *args, **kwargs):
        if not done:
            done.append(True)
            apply_rewrites(mod, rewrites, log)
        return orig_script(mod, *args, **kwargs)

    ej.script = script
    try:
        yield
    finally:
        ej.script = orig_script
