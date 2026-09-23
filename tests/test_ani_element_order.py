"""The default TorchANI element list must match ANI-2x's own species order.

A wrong order silently evaluates atoms with another element's network
(the old default [1,6,7,8,16,17] scored Cl with the F network).
"""
import pytest

torchani = pytest.importorskip("torchani")

from src.wrappers.wrap_torchani import _ANI2X_ELEMENTS


def test_ani2x_default_matches_torchani():
    model = torchani.models.ANI2x(model_index=0)
    assert _ANI2X_ELEMENTS == model.atomic_numbers.tolist()
