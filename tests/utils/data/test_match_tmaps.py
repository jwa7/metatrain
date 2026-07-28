import pytest
import torch

from metatrain.utils.data.match_tmaps import match_layout


def test_jit_script_match_layout():
    pytest.skip("Not yet torchscriptable.")
    torch.jit.script(match_layout)
