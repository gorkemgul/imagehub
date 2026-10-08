"""Krea 2 transformer structural parity with random weights."""

import pytest
import torch

from imagehub.models.krea2_transformer import (
    Krea2Transformer2DModel,
    prepare_krea2_position_ids,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY = {
    "in_channels": 8,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "intermediate_size": 48,
    "timestep_embed_dim": 16,
    "text_hidden_dim": 24,
    "num_text_layers": 3,
    "text_num_attention_heads": 4,
    "text_num_key_value_heads": 2,
    "text_intermediate_size": 32,
    "num_layerwise_text_blocks": 1,
    "num_refiner_text_blocks": 1,
    "axes_dims_rope": [2, 2, 4],
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}
MEDIUM = TINY | {
    "num_layers": 3,
    "attention_head_dim": 32,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "intermediate_size": 384,
    "timestep_embed_dim": 64,
    "text_hidden_dim": 96,
    "num_text_layers": 6,
    "text_num_attention_heads": 8,
    "text_num_key_value_heads": 4,
    "text_intermediate_size": 160,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "axes_dims_rope": [8, 12, 12],
}


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
def test_krea2_transformer_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.Krea2Transformer2DModel(**config).eval()
    model = Krea2Transformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size, text_length, height, width = 2, 7, 3, 5
    hidden_states = torch.randn(batch_size, height * width, config["in_channels"])
    encoder_hidden_states = torch.randn(
        batch_size,
        text_length,
        config["num_text_layers"],
        config["text_hidden_dim"],
    )
    encoder_attention_mask = torch.tensor([[True] * text_length, [True] * 5 + [False] * 2])
    timestep = torch.tensor([0.00125, 0.8765])
    expected_position_ids = diffusers.Krea2Pipeline.prepare_position_ids(
        text_length, height, width, torch.device("cpu")
    )
    position_ids = prepare_krea2_position_ids(text_length, height, width, torch.device("cpu"))
    torch.testing.assert_close(position_ids, expected_position_ids, atol=0, rtol=0)

    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            position_ids=expected_position_ids,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]
        actual = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            attention_kwargs=None,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
