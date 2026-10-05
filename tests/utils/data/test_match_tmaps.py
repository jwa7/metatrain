import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.data.match_tmaps import match_layout


def test_jit_script_match_layout():
    pytest.skip("Not yet torchscriptable.")
    torch.jit.script(match_layout)


def _scalar_tmap(value: float) -> TensorMap:
    """A scalar (non-spherical) per-structure target, as ``energy`` is stored."""
    return TensorMap(
        keys=Labels("_", torch.tensor([[0]])),
        blocks=[
            TensorBlock(
                values=torch.full((2, 1), value),
                samples=Labels("system", torch.tensor([[0], [1]])),
                components=[],
                properties=Labels("energy", torch.tensor([[0]])),
            )
        ],
    )


def test_match_layout_scalar_target_without_cg_coeffs():
    """Scalar targets must not be routed through the spherical coupling branch.

    ``match_layout`` used to test for the *absence* of an ``o3_lambda`` key
    dimension when deciding whether a TensorMap was spherical, so a scalar
    target like ``energy`` was treated as spherical and the call failed with
    "Cannot couple or uncouple spherical tensor blocks without Clebsch-Gordan
    coefficients" -- which made every scalar-target run abort while the scaler
    removed the composition baseline.
    """
    tmap1 = _scalar_tmap(1.0)
    tmap2 = _scalar_tmap(2.0)

    matched = match_layout(tmap1, tmap2, cg_coeffs=None)

    assert matched.keys == tmap1.keys
    torch.testing.assert_close(matched.block(0).values, tmap1.block(0).values)
