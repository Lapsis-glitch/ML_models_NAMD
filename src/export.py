"""
Shared export logic for all MLIP wrappers.

Wraps a Python ``nn.Module`` with ``torch.jit.script``, saves the
resulting TorchScript archive, and prints diagnostics.

NOTE: We intentionally do **not** call
``torch.jit.optimize_for_inference()`` because it strips
``@torch.jit.export`` methods (e.g. ``forward_batch``) and custom
attributes (``supports_batch``).  The performance benefit is
negligible for GPU-inference workloads.
"""

import torch
from torch import nn
from pathlib import Path


def export_wrapped(
    wrapper: nn.Module,
    out_path: str,
    *,
    model_type: str = "unknown",
) -> None:
    """
    TorchScript-compile *wrapper* and save to *out_path*.

    Args:
        wrapper:    An ``nn.Module`` wrapper (already ``.eval()``).
        out_path:   Destination ``.pt`` file.
        model_type: Human-readable label printed in diagnostics.
    """
    # Ensure parent directory exists.
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    scripted = torch.jit.script(wrapper)
    scripted.save(out_path)

    r_max = getattr(wrapper, "r_max", None)
    supports_batch = getattr(wrapper, "supports_batch", False)

    print(f"[{model_type}] Exported wrapped TorchScript model to: {out_path}")
    if r_max is not None:
        print(f"  cutoff (r_max): {r_max} Å")
    print(f"  supports_batch: {supports_batch}")
    print(f"  Unit conversion is fused into model output (kcal/mol).")
    print(f"  Edge computation runs in float32; model runs in float64.")

