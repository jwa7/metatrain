"""Tests for the per-block heads (``head_type`` / ``d_head``) and the atom-type
gated readouts (``readout_type``) of the PET model.

These exercise forward correctness (last-layer feature widths), TorchScript
compatibility, and hyper validation across the new configurations.
"""

import copy

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.pet import PET
from metatrain.pet.modules.readouts import LinearReadout
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


ATOMIC_TYPES = [1, 6, 7, 8]


def _hypers(**overrides):
    h = copy.deepcopy(get_default_hypers("pet")["model"])
    h.update(
        dict(
            d_pet=8,
            d_node=8,
            d_head=8,
            d_feedforward=8,
            num_heads=1,
            num_attention_layers=1,
            num_gnn_layers=1,
        )
    )
    h.update(overrides)
    return h


def _energy_info():
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=ATOMIC_TYPES,
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )


def _multispherical_info():
    # A three-block (o3_lambda = 2, 1, 0) per-atom target.
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=ATOMIC_TYPES,
        targets={
            "spherical_tensor": get_generic_target_info(
                "spherical_tensor",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": 2, "o3_sigma": 1},
                                {"o3_lambda": 1, "o3_sigma": 1},
                                {"o3_lambda": 0, "o3_sigma": 1},
                            ]
                        }
                    },
                    "num_subtargets": 3,
                    "sample_kind": "atom",
                },
            )
        },
    )


def _make_system(model):
    system = System(
        types=torch.tensor(ATOMIC_TYPES),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
        ),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


def _run(model, target):
    system = _make_system(model)
    model = model.to(system.positions.dtype).eval()
    ll_name = f"mtt::aux::{target}_last_layer_features"
    outputs = {
        target: ModelOutput(sample_kind="atom"),
        ll_name: ModelOutput(sample_kind="atom"),
    }
    with torch.no_grad():
        result = model([_make_system(model)], outputs)
    return result, ll_name


# num_readout_layers = 1 (feedforward featurizer default), so with a symmetric
# d_head the per-target last-layer feature width is d_head_node + d_head_edge.


@pytest.mark.parametrize(
    "overrides, target, expected_ll_width",
    [
        # energy: single block -> per_block == per_target width
        (dict(), "energy", 8 + 8),
        (dict(head_type="per_block"), "energy", 8 + 8),
        (dict(head_type="per_block", d_head={"node": 4, "edge": 6}), "energy", 4 + 6),
        (
            dict(readout_type={"atom_type_gating": "one-hot", "hypers": {}}),
            "energy",
            8 + 8,
        ),
        (
            dict(
                readout_type={
                    "atom_type_gating": "moe",
                    "hypers": {
                        "num_experts": 4,
                        "num_routed_experts": 3,
                        "num_topk_experts": 2,
                    },
                }
            ),
            "energy",
            8 + 8,
        ),
        # multispherical: three blocks. per_target shares one head across blocks;
        # per_block concatenates one (node+edge) feature per block.
        (dict(), "spherical_tensor", 8 + 8),
        (dict(head_type="per_block"), "spherical_tensor", 3 * (8 + 8)),
        (
            dict(head_type="per_block", d_head={"node": 4, "edge": 6}),
            "spherical_tensor",
            3 * (4 + 6),
        ),
        (
            dict(
                head_type="per_block",
                d_head={"node": 4, "edge": 6},
                readout_type={"atom_type_gating": "one-hot", "hypers": {}},
            ),
            "spherical_tensor",
            3 * (4 + 6),
        ),
    ],
)
def test_forward_and_ll_features(overrides, target, expected_ll_width):
    dsinfo = _energy_info() if target == "energy" else _multispherical_info()
    model = PET(_hypers(**overrides), dsinfo)
    result, ll_name = _run(model, target)
    assert ll_name in result
    assert result[ll_name].block(0).values.shape[-1] == expected_ll_width
    # last_layer_feature_size scalar matches the per_target (per-block) width.
    assert model.last_layer_feature_size == overrides_ll_scalar(overrides)


def overrides_ll_scalar(overrides):
    d_head = overrides.get("d_head", 8)
    if isinstance(d_head, dict):
        return d_head["node"] + d_head["edge"]
    return d_head + d_head


@pytest.mark.parametrize(
    "overrides",
    [
        dict(head_type="per_block", d_head={"node": 4, "edge": 6}),
        dict(readout_type={"atom_type_gating": "one-hot", "hypers": {}}),
        dict(
            readout_type={
                "atom_type_gating": "moe",
                "hypers": {
                    "num_experts": 4,
                    "num_routed_experts": 3,
                    "num_topk_experts": 2,
                },
            }
        ),
    ],
)
def test_torchscript(overrides):
    model = PET(_hypers(**overrides), _multispherical_info())
    system = _make_system(model)
    model = model.to(system.positions.dtype).eval()
    scripted = torch.jit.script(model)
    outputs = {"spherical_tensor": ModelOutput(sample_kind="atom")}
    with torch.no_grad():
        scripted([_make_system(scripted)], outputs)


def test_head_type_per_target_dict():
    # head_type as a dict keyed by target name: the listed target uses per_block,
    # unlisted targets fall back to the default per_target.
    model = PET(
        _hypers(head_type={"spherical_tensor": "per_block"}),
        _multispherical_info(),
    )
    result, ll_name = _run(model, "spherical_tensor")
    # per_block over 3 blocks with symmetric d_head=8.
    assert result[ll_name].block(0).values.shape[-1] == 3 * (8 + 8)


def test_head_type_dict_fallback_to_default():
    # A target not listed in the head_type dict falls back to per_target.
    model = PET(
        _hypers(head_type={"some_other_target": "per_block"}),
        _multispherical_info(),
    )
    result, ll_name = _run(model, "spherical_tensor")
    assert result[ll_name].block(0).values.shape[-1] == (8 + 8)


def test_readout_type_per_target_dict():
    # readout_type keyed by target name (no top-level atom_type_gating key) is
    # interpreted per target.
    model = PET(
        _hypers(
            readout_type={
                "spherical_tensor": {"atom_type_gating": "one-hot", "hypers": {}}
            }
        ),
        _multispherical_info(),
    )
    # Node readout is one-hot -> n_species weight groups.
    layer0 = model.backend.node_last_layers["spherical_tensor"][0]
    first_key = next(iter(layer0.keys()))
    readout = layer0[first_key]
    assert readout.gated and readout.weight.shape[0] == len(ATOMIC_TYPES)
    # A forward pass works.
    result, _ = _run(model, "spherical_tensor")
    assert "spherical_tensor" in result


def _set_chunk_size(model, chunk_size):
    for module in model.modules():
        if isinstance(module, LinearReadout):
            module.chunk_size = chunk_size


def test_one_hot_chunk_size_propagates():
    model = PET(
        _hypers(
            readout_type={
                "atom_type_gating": "one-hot",
                "hypers": {"chunk_size": 4},
            }
        ),
        _energy_info(),
    )
    readouts = [m for m in model.modules() if isinstance(m, LinearReadout)]
    gated = [m for m in readouts if m.gated]
    assert gated and all(m.chunk_size == 4 for m in gated)


def test_one_hot_chunk_size_numerically_equivalent():
    # Chunking the per-row gather must not change the result: chunk_size=1 (many
    # chunks) must match a single-chunk gather.
    model = PET(
        _hypers(readout_type={"atom_type_gating": "one-hot", "hypers": {}}),
        _multispherical_info(),
    )
    system = _make_system(model)
    model = model.to(system.positions.dtype).eval()
    outputs = {"spherical_tensor": ModelOutput(sample_kind="atom")}

    _set_chunk_size(model, 1000)  # single chunk (> n_rows)
    with torch.no_grad():
        full = model([_make_system(model)], outputs)["spherical_tensor"].block(0).values
    _set_chunk_size(model, 1)  # one row per chunk
    with torch.no_grad():
        chunked = (
            model([_make_system(model)], outputs)["spherical_tensor"].block(0).values
        )
    torch.testing.assert_close(full, chunked, atol=0.0, rtol=0.0)


def test_one_hot_small_chunk_torchscript():
    model = PET(
        _hypers(
            readout_type={
                "atom_type_gating": "one-hot",
                "hypers": {"chunk_size": 2},
            }
        ),
        _multispherical_info(),
    )
    system = _make_system(model)
    model = model.to(system.positions.dtype).eval()
    scripted = torch.jit.script(model)
    outputs = {"spherical_tensor": ModelOutput(sample_kind="atom")}
    with torch.no_grad():
        scripted([_make_system(scripted)], outputs)


def test_num_head_layers_must_be_positive():
    with pytest.raises(ValueError, match="num_head_layers must be >= 1"):
        PET(_hypers(num_head_layers=0), _energy_info())


def test_moe_rejects_pair_targets():
    moe_spec = {
        "atom_type_gating": "moe",
        "hypers": {
            "num_experts": 4,
            "num_routed_experts": 3,
            "num_topk_experts": 2,
        },
    }
    model = PET(_hypers(readout_type=moe_spec), _energy_info())
    with pytest.raises(ValueError, match="only supported for per-atom"):
        model.backend._make_readout(
            8, 4, moe_spec, is_pair_target=True, is_edge_readout=True
        )


def test_one_hot_pair_conditioning_shapes():
    spec = {"atom_type_gating": "one-hot", "hypers": {}}
    model = PET(_hypers(readout_type=spec), _energy_info())
    # Edge readout of a pair target: n_species**2 weight groups.
    pair = model.backend._make_readout(
        8, 4, spec, is_pair_target=True, is_edge_readout=True
    )
    assert pair.weight.shape[0] == len(ATOMIC_TYPES) ** 2
    # Node/central-atom conditioning: n_species groups.
    node = model.backend._make_readout(
        8, 4, spec, is_pair_target=True, is_edge_readout=False
    )
    assert node.weight.shape[0] == len(ATOMIC_TYPES)


def test_num_head_layers_changes_head_depth():
    # num_head_layers=3 -> 3 Linear layers in each head MLP. For head_type=
    # "per_target" node_heads[target][layer] is the shared head (Sequential).
    model = PET(_hypers(num_head_layers=3), _energy_info())
    head = model.backend.node_heads["energy"][0]
    n_linear = sum(1 for m in head if isinstance(m, torch.nn.Linear))
    assert n_linear == 3
