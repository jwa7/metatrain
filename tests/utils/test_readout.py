"""Tests for the general-purpose readout utilities (:mod:`metatrain.utils.readout`).

Cover the optional ``bias`` argument (default off, for equivariant models), the
gated ("one-hot") vs ungated linear map, the gathered/sorted grouped-matmul
paths, the MoE readout, and TorchScript compatibility.
"""

import pytest
import torch

from metatrain.utils.readout import LinearReadout, MoEReadout


def test_bias_defaults_off_and_is_absent():
    readout = LinearReadout(4, 3, n_groups=1, gated=False)
    assert readout.bias is None
    assert "bias" not in readout.state_dict()

    x = torch.randn(5, 4)
    out = readout(x, torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(out, x @ readout.weight.t())


def test_bias_on_matches_affine_linear():
    readout = LinearReadout(4, 3, n_groups=1, gated=False, bias=True)
    assert readout.bias is not None
    assert set(readout.state_dict().keys()) == {"weight", "bias"}

    x = torch.randn(5, 4)
    out = readout(x, torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(out, x @ readout.weight.t() + readout.bias)


@pytest.mark.parametrize("bias", [False, True])
def test_gated_one_hot_matches_per_row_reference(bias):
    n_groups = 3
    readout = LinearReadout(4, 2, n_groups=n_groups, gated=True, bias=bias)
    if bias:
        assert tuple(readout.bias.shape) == (n_groups, 2)
    else:
        assert readout.bias is None

    x = torch.randn(7, 4)
    idx = torch.randint(0, n_groups, (7,))
    out = readout(x, idx)

    weight = readout.weight[idx]  # (7, out, in)
    expected = torch.bmm(x.unsqueeze(1), weight.transpose(-2, -1)).squeeze(1)
    if bias:
        expected = expected + readout.bias[idx]
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("bias", [False, True])
def test_gated_sorted_matches_gathered(bias):
    # The sorted grouped-matmul path and the gathered path must be identical,
    # with and without bias (the sorted path handles the optional bias too).
    n_groups = 3
    readout = LinearReadout(
        4,
        2,
        n_groups=n_groups,
        gated=True,
        bias=bias,
        sorted_min_rows=1,
        sorted_min_rows_per_group=1,
    )
    x = torch.randn(20, 4)
    idx = torch.randint(0, n_groups, (20,))

    out_sorted = readout(x, idx)  # thresholds = 1 -> sorted path
    readout.sorted_min_rows = 10**9  # force the gathered path
    out_gathered = readout(x, idx)
    torch.testing.assert_close(out_sorted, out_gathered)


@pytest.mark.parametrize("bias", [False, True])
def test_gated_chunking_is_result_invariant(bias):
    n_groups = 4
    readout = LinearReadout(
        4, 3, n_groups=n_groups, gated=True, bias=bias, chunk_size=1
    )
    x = torch.randn(10, 4)
    idx = torch.randint(0, n_groups, (10,))

    out_chunked = readout(x, idx)
    readout.chunk_size = 1000  # single chunk
    out_single = readout(x, idx)
    torch.testing.assert_close(out_chunked, out_single)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("gated", [False, True])
def test_linear_readout_torchscript(bias, gated):
    n_groups = 3
    readout = LinearReadout(4, 2, n_groups=n_groups, gated=gated, bias=bias)
    scripted = torch.jit.script(readout)
    x = torch.randn(6, 4)
    idx = torch.randint(0, n_groups, (6,))
    torch.testing.assert_close(readout(x, idx), scripted(x, idx))


@pytest.mark.parametrize("bias", [False, True])
def test_moe_readout_bias_and_torchscript(bias):
    readout = MoEReadout(
        4,
        3,
        n_species=5,
        num_experts=4,
        num_routed_experts=3,
        num_topk_experts=2,
        bias=bias,
    )
    for expert in list(readout.routed_experts) + list(readout.shared_experts):
        assert (expert.bias is not None) == bias

    scripted = torch.jit.script(readout)
    x = torch.randn(6, 4)
    idx = torch.randint(0, 5, (6,))
    torch.testing.assert_close(readout(x, idx), scripted(x, idx))
