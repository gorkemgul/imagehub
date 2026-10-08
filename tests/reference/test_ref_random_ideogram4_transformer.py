"""Ideogram 4 transformer structural parity with random weights."""

import pytest
import torch
from torch import nn

from imagehub.models.fp8 import load_fp8_state_dict, swap_linears_to_fp8
from imagehub.models.ideogram4_transformer import Ideogram4Transformer

pytestmark = pytest.mark.reference
ideogram4 = pytest.importorskip("ideogram4")
constants = pytest.importorskip("ideogram4.constants")
quantized_loading = pytest.importorskip("ideogram4.quantized_loading")
Ideogram4Config = ideogram4.Ideogram4Config
ReferenceModel = ideogram4.Ideogram4Transformer
LLM_TOKEN_INDICATOR = constants.LLM_TOKEN_INDICATOR
OUTPUT_IMAGE_INDICATOR = constants.OUTPUT_IMAGE_INDICATOR
load_reference_fp8_state_dict = quantized_loading.load_fp8_state_dict
quantize_weight_to_fp8 = quantized_loading.quantize_weight_to_fp8
swap_reference_linears_to_fp8 = quantized_loading.swap_linears_to_fp8

TINY = {
    "emb_dim": 24,
    "num_layers": 2,
    "num_heads": 2,
    "intermediate_size": 32,
    "adanln_dim": 8,
    "in_channels": 4,
    "llm_features_dim": 10,
    "rope_theta": 10_000,
    "mrope_section": [2, 1, 1],
    "norm_eps": 1e-5,
}
MEDIUM = TINY | {
    "emb_dim": 128,
    "num_layers": 1,
    "num_heads": 4,
    "intermediate_size": 192,
    "adanln_dim": 32,
    "llm_features_dim": 96,
    "mrope_section": [4, 3, 3],
}


def _reference_config(config):
    return Ideogram4Config(**(config | {"mrope_section": tuple(config["mrope_section"])}))


def _inputs(config):
    batch_size, text_length, height, width = 2, 3, 2, 3
    image_length = height * width
    length = text_length + image_length
    llm_features = torch.randn(batch_size, length, config["llm_features_dim"])
    x = torch.randn(batch_size, length, config["in_channels"])
    t = torch.tensor([0.2, 0.8])
    position_ids = torch.zeros(batch_size, length, 3, dtype=torch.long)
    position_ids[:, :text_length, 0] = torch.arange(text_length)
    position_ids[:, text_length:, 0] = 64
    position_ids[:, text_length:, 1] = torch.arange(height).repeat_interleave(width)
    position_ids[:, text_length:, 2] = torch.arange(width).repeat(height)
    segment_ids = torch.zeros(batch_size, length, dtype=torch.long)
    indicator = torch.full((batch_size, length), OUTPUT_IMAGE_INDICATOR)
    indicator[:, :text_length] = LLM_TOKEN_INDICATOR
    return {
        "llm_features": llm_features,
        "x": x,
        "t": t,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
    }


def _fp8_state(model):
    state = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_name = f"{name}.weight"
            state[weight_name], state[f"{weight_name}_scale"] = quantize_weight_to_fp8(
                state[weight_name]
            )
    return state


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
@pytest.mark.parametrize("fp8", [False, True], ids=["float32", "fp8"])
def test_ideogram4_transformer_random_weights_exact(config, fp8):
    torch.manual_seed(0)
    reference = ReferenceModel(_reference_config(config)).eval()
    model = Ideogram4Transformer(config).eval()
    state = _fp8_state(reference) if fp8 else reference.state_dict()
    if fp8:
        swap_reference_linears_to_fp8(reference, state, torch.float32)
        swap_linears_to_fp8(model, state, torch.float32)
        load_reference_fp8_state_dict(
            reference, state, torch.device("cpu"), torch.float32, assign=True
        )
        load_fp8_state_dict(model, state, torch.device("cpu"), torch.float32, assign=True)
    else:
        model.load_checkpoint(state)

    inputs = _inputs(config)
    with torch.inference_mode():
        expected = reference(**inputs)
        actual = model(**inputs)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
