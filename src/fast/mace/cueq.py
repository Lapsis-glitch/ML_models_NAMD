"""TorchScript fixes for mace 0.3.15's cuEquivariance path.

1. EquivariantProductBasisBlock decides "use cuEq?" at run time via
   hasattr(self, "cueq_config") -- the config is a Python dataclass that
   TorchScript drops, so the SCRIPTED model silently takes the e3nn branch and
   feeds the one-hot float node_attrs to the cuEq kernel as indices
   ("RAFT failure ... unexpected index buffer dtype").  ScriptableCueqProduct
   hard-wires the cuEq branch (layout ir_mul -> no transpose) and computes the
   species index with argmax instead of torch.nonzero (same result for a
   one-hot row, but no device->host sync).
2. reshape_irreps has the same hasattr(self, "cueq_config") pattern, so the
   scripted model reshapes messages as mul_ir although the cuEq kernels
   produce ir_mul (garbage energies, ~60 eV off).  ReshapeIrMul hard-wires
   the ir_mul branch.
3. with_cueq_conv_fusion() monkey-patches forward on the instance
   (types.MethodType), which TorchScript cannot compile.  FusedConvTP is the
   same call as a real nn.Module.
"""
import torch
from torch import nn
from typing import List, Optional


class ScriptableCueqProduct(nn.Module):
    def __init__(self, prod):
        super().__init__()
        assert prod.cueq_config is not None and prod.cueq_config.layout_str == "ir_mul"
        assert not getattr(prod, "use_agnostic_product", False)
        self.symmetric_contractions = prod.symmetric_contractions
        self.linear = prod.linear
        self.use_sc: bool = bool(prod.use_sc)

    def forward(self, node_feats: torch.Tensor, sc: Optional[torch.Tensor],
                node_attrs: torch.Tensor) -> torch.Tensor:
        index_attrs = torch.argmax(node_attrs, dim=1).to(torch.int32)
        node_feats = self.symmetric_contractions(node_feats.flatten(1), index_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)


class ReshapeIrMul(nn.Module):
    def __init__(self, r):
        super().__init__()
        assert r.cueq_config is not None and r.cueq_config.layout_str == "ir_mul"
        self.dims: List[int] = [int(x) for x in r.dims]
        self.muls: List[int] = [int(x) for x in r.muls]

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out: List[torch.Tensor] = []
        batch = tensor.shape[0]
        for mul, d in zip(self.muls, self.dims):
            out.append(tensor[:, ix: ix + mul * d].reshape(batch, d, mul))
            ix += mul * d
        return torch.cat(out, dim=-2)


class FusedConvTP(nn.Module):
    def __init__(self, sp):
        super().__init__()
        # undo the instance-level monkey patch so the module scripts normally
        for k in ("forward", "original_forward"):
            if k in sp.__dict__:
                del sp.__dict__[k]
        self.weight_numel: int = int(sp.weight_numel)
        del sp.weight_numel
        self.sp = sp

    def forward(self, node_feats: torch.Tensor, edge_attrs: torch.Tensor,
                tp_weights: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        return self.sp([tp_weights, node_feats, edge_attrs],
                       {1: sender}, {0: node_feats}, {0: receiver})[0]


def make_scriptable(m):
    for i in range(len(m.products)):
        m.products[i] = ScriptableCueqProduct(m.products[i])
    from mace.modules.irreps_tools import reshape_irreps
    for inter in m.interactions:
        if isinstance(inter.reshape, reshape_irreps):
            inter.reshape = ReshapeIrMul(inter.reshape)
        if getattr(inter, "conv_fusion", False):
            inter.conv_tp = FusedConvTP(inter.conv_tp)
    # nothing else may still depend on a dropped Python config at run time
    left = [n for n, mod in m.named_modules() if isinstance(mod, reshape_irreps)]
    assert not left, left
    return m
