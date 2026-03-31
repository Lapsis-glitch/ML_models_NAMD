"""
Thin TorchScript wrapper that attaches ``r_max`` and ``atomic_numbers``
metadata to a NequIP / Allegro model saved via manual
``torch.jit.script``.

When ``nequip-compile --mode torchscript`` is unavailable (PyTorch ≥ 2.10),
the training scripts fall back to scripting the eager model directly.
That model lacks the ``r_max`` / ``atomic_numbers`` attributes that the
NAMD wrappers expect.  This module bridges the gap.

Usage (from a training script)::

    from src.training._nequip_metadata_wrapper import save_with_metadata
    save_with_metadata(scripted_inner, path, r_max=5.0, z_list=[1, 8])
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import nn


class NequIPMetadataWrapper(nn.Module):
    """
    Thin scriptable wrapper that exposes ``r_max`` (float) and
    ``atomic_numbers`` (``List[int]``) as attributes, then delegates
    ``forward`` to the inner model.
    """

    def __init__(
        self,
        inner: torch.jit.ScriptModule,
        r_max: float,
        atomic_numbers: List[int],
    ):
        super().__init__()
        self.inner = inner
        self.r_max: float = r_max
        self.atomic_numbers: List[int] = atomic_numbers

    def forward(
        self, data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.inner(data)


def save_with_metadata(
    scripted_inner: torch.jit.ScriptModule,
    path: str | Path,
    r_max: float,
    z_list: List[int],
) -> None:
    """
    Wrap *scripted_inner* with metadata and save to *path*.

    The resulting ``.pth`` file is loadable by
    :class:`src.wrappers.wrap_compiled_nequip.NequIP_Allegro_Wrapper`.
    """
    wrapper = NequIPMetadataWrapper(scripted_inner, r_max, z_list)
    wrapper.eval()
    scripted = torch.jit.script(wrapper)
    scripted.save(str(path))

