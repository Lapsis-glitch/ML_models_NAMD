"""Rebuild MACE-OFF under the allegro env (mace 0.3.15, e3nn 0.6) from the
version-neutral dump written by extract_state.py (run in MACE_312).

rebuild(state, dtype)            -> e3nn ScaleShiftMACE, same weights
rebuild(state, dtype, cueq=True) -> same model built with cuEquivariance
                                    kernels (weights remapped by mace's own
                                    convert_e3nn_cueq.transfer_weights)
"""
import torch
from e3nn import o3
from mace.modules import models as mm, blocks as mb


def _config(d):
    c = dict(d["config"])
    c["gate"] = torch.nn.functional.silu if c["gate"] == "silu" else c["gate"]
    c["interaction_cls"] = getattr(mb, c["interaction_cls"])
    c["interaction_cls_first"] = getattr(mb, c["interaction_cls_first"])
    c["hidden_irreps"] = o3.Irreps(c["hidden_irreps"])
    c["MLP_irreps"] = o3.Irreps(c["MLP_irreps"])
    c.setdefault("use_reduced_cg", False)  # MACE-OFF23 predates reduced-CG U matrices
    return c


def _weight_key_for_U(ukey, correlation):
    """contractions.X.U_matrix_nu -> the weight tensor that multiplies it."""
    base, nu = ukey.rsplit(".U_matrix_", 1)
    nu = int(nu)
    if nu == correlation:
        return f"{base}.weights_max"
    return f"{base}.weights.{correlation - 1 - nu}"


def rebase_U(sd, fresh, correlation, tol=1e-10):
    """MACE-OFF23 was trained with generalized-CG U matrices from an older
    e3nn/mace.  For one path (L=1 output, nu=3) the stored basis differs from
    the one mace 0.3.15 generates (same span, different linear combination) --
    the cuEq converter assumes the fresh basis.  Re-express the model in the
    fresh basis: U_old = U_new @ A  ->  w_new = A @ w_old.  The function the
    model computes is unchanged (checked by the caller); returns a new dict and
    the list of rebased keys."""
    sd = dict(sd)
    done = []
    for k, u_new in fresh.items():
        if ".U_matrix_" not in k or k not in sd:
            continue
        u_old = sd[k].to(torch.float64)
        u_new = u_new.to(torch.float64)
        if u_old.shape != u_new.shape:
            raise RuntimeError(f"U shape mismatch {k}")
        if (u_old - u_new).abs().max() < tol:
            continue
        K = u_new.shape[-1]
        A = torch.linalg.lstsq(u_new.reshape(-1, K), u_old.reshape(-1, K)).solution
        res = (u_new.reshape(-1, K) @ A - u_old.reshape(-1, K)).abs().max().item()
        if res > 1e-9:
            raise RuntimeError(f"{k}: old U basis not in span of new one (res {res:.2e})")
        wk = _weight_key_for_U(k, correlation)
        w = sd[wk]
        sd[wk] = torch.einsum("jk,ekc->ejc", A, w.to(torch.float64)).to(w.dtype)
        sd[k] = u_new.to(sd[k].dtype)
        done.append((k, wk, res))
    return sd, done


def _freeze(m, dtype, device):
    m = m.to(dtype=dtype, device=device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def rebuild(state_path, dtype=torch.float64, cueq=False, device="cpu",
            conv_fusion=None, rebase=False):
    d = torch.load(state_path, weights_only=False, map_location="cpu")
    c = _config(d)
    cls = getattr(mm, d["class"])
    prev = torch.get_default_dtype()
    # Always rebuild + rebase the e3nn model in float64: an fp32-constructed model
    # has fp32-rounded U matrices (~1e-8 off) and rebase_U's span check would fail.
    # The float64 model is cast to `dtype` at the end (cuEq models are constructed
    # directly in `dtype` so their math_dtype matches, then get the weights copied).
    torch.set_default_dtype(torch.float64)
    try:
        m = cls(**c)
        sd = d["state_dict"]
        if cueq or rebase:
            sd, _ = rebase_U(sd, m.state_dict(), c["correlation"])
        missing, unexpected = m.load_state_dict(sd, strict=False)
        # *_zeroed flags are new in mace 0.3.15: computed at construction from
        # the (fixed) CG U-matrices, so they carry no trained state.
        missing = [k for k in missing if not k.endswith("_zeroed")]
        if missing or unexpected:
            raise RuntimeError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
        m = _freeze(m, torch.float64, "cpu")
        if not cueq:
            return _freeze(m, dtype, device)

        from mace.modules.wrapper_ops import CuEquivarianceConfig
        from mace.cli.convert_e3nn_cueq import transfer_weights
        cc = dict(c)
        cc["cueq_config"] = CuEquivarianceConfig(
            enabled=True, layout="ir_mul", group="O3_e3nn", optimize_all=True,
            conv_fusion=(device == "cuda") if conv_fusion is None else conv_fusion,
        )
        torch.set_default_dtype(dtype)
        t = cls(**cc)
        t = t.to(dtype=dtype)
        transfer_weights(m, t,
                         num_product_irreps=len(c["hidden_irreps"].slices()) - 1,
                         correlation=c["correlation"],
                         num_layers=c["num_interactions"],
                         use_reduced_cg=c["use_reduced_cg"])
        return _freeze(t, dtype, device)
    finally:
        torch.set_default_dtype(prev)
