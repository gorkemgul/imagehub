"""Kandinsky 5 transformer structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.kandinsky5_transformer import Kandinsky5Transformer3DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

PUBLISHED = json.loads(
    (Path(__file__).parents[1] / "fixtures/kandinsky5_configs/transformer/config.json").read_text()
)
TINY = {
    "in_visual_dim": 4,
    "out_visual_dim": 4,
    "time_dim": 64,
    "patch_size": [1, 2, 2],
    "model_dim": 128,
    "ff_dim": 256,
    "num_text_blocks": 2,
    "num_visual_blocks": 2,
    "axes_dims": [16, 24, 24],
    "visual_cond": False,
    "in_text_dim": 32,
    "in_text_dim2": 32,
    "attention_type": "regular",
}
PUBLISHED_SMALL = {
    **{key: value for key, value in PUBLISHED.items() if not key.startswith("_")},
    "model_dim": 512,
    "ff_dim": 1024,
    "num_text_blocks": 2,
    "num_visual_blocks": 2,
    "axes_dims": [16, 24, 24],
    "patch_size": [1, 2, 2],
}


def _random_pair(config):
    reference = diffusers.Kandinsky5Transformer3DModel(**config).eval()
    generator = torch.Generator().manual_seed(0)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.05, generator=generator)
    ours = Kandinsky5Transformer3DModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    return reference, ours


def _inputs(config, shape, tokens, *, batch=1):
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(
        batch, shape[0], shape[1], shape[2], config["in_visual_dim"], generator=generator
    )
    encoder_hidden_states = torch.randn(batch, tokens, config["in_text_dim"], generator=generator)
    pooled = torch.randn(batch, config["in_text_dim2"], generator=generator)
    timestep = torch.tensor([500.0] * batch)
    visual_rope_pos = [
        torch.arange(shape[0]),
        torch.arange(shape[1] // 2),
        torch.arange(shape[2] // 2),
    ]
    text_rope_pos = torch.arange(tokens)
    return hidden_states, encoder_hidden_states, pooled, timestep, visual_rope_pos, text_rope_pos


@pytest.mark.parametrize(
    ("shape", "tokens"),
    [((1, 4, 4), 4), ((1, 8, 8), 7), ((2, 4, 6), 5)],
    ids=["small-square", "larger-square", "non-square"],
)
def test_tiny_transformer_matches_reference(shape, tokens):
    reference, ours = _random_pair(TINY)
    (
        hidden_states,
        encoder_hidden_states,
        pooled,
        timestep,
        visual_rope_pos,
        text_rope_pos,
    ) = _inputs(TINY, shape, tokens)
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states,
            timestep,
            pooled,
            visual_rope_pos,
            text_rope_pos,
        ).sample
        actual = ours(
            hidden_states,
            encoder_hidden_states,
            timestep,
            pooled,
            visual_rope_pos,
            text_rope_pos,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_published_depth_transformer_matches_reference_at_small_sizes():
    reference, ours = _random_pair(PUBLISHED_SMALL)
    (
        hidden_states,
        encoder_hidden_states,
        pooled,
        timestep,
        visual_rope_pos,
        text_rope_pos,
    ) = _inputs(PUBLISHED_SMALL, (1, 4, 4), 11, batch=2)
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states,
            timestep,
            pooled,
            visual_rope_pos,
            text_rope_pos,
        ).sample
        actual = ours(
            hidden_states,
            encoder_hidden_states,
            timestep,
            pooled,
            visual_rope_pos,
            text_rope_pos,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
