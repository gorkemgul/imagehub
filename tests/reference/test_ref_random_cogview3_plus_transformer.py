"""CogView3-Plus transformer random-weight parity."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.cogview3_plus_transformer import CogView3PlusTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

PUBLISHED = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/cogview3_plus_configs/transformer_config.json"
    ).read_text()
)
TINY = {
    "patch_size": 2,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "text_embed_dim": 12,
    "time_embed_dim": 16,
    "condition_dim": 4,
    "pos_embed_max_size": 8,
    "sample_size": 8,
}
MEDIUM = TINY | {
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "text_embed_dim": 96,
    "time_embed_dim": 128,
    "condition_dim": 32,
}
FULL = {key: value for key, value in PUBLISHED.items() if key != "pooled_projection_dim"} | {
    "num_layers": 2
}


@pytest.mark.parametrize(
    ("config", "height", "width", "text_length"),
    [(TINY, 6, 10, 16), (MEDIUM, 4, 6, 32), (FULL, 8, 12, 24)],
    ids=["tiny", "medium-width", "published-config-two-layers"],
)
def test_cogview3_plus_transformer_random_weights_exact(config, height, width, text_length):
    torch.manual_seed(0)
    reference = diffusers.CogView3PlusTransformer2DModel(**config).eval()
    model = CogView3PlusTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size = 2
    hidden_states = torch.randn(batch_size, config["in_channels"], height, width)
    encoder_hidden_states = torch.randn(batch_size, text_length, config["text_embed_dim"])
    timestep = torch.tensor([1, 753])
    original_size = torch.tensor([[768.0, 1280.0], [1024.0, 1536.0]])
    target_size = torch.tensor([[768.0, 1280.0], [896.0, 1344.0]])
    crop_coords = torch.tensor([[0.0, 0.0], [64.0, 96.0]])

    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
        )[0]
        actual = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            original_size,
            target_size,
            crop_coords,
        )

    assert torch.equal(actual, expected)
    assert torch.equal(model.patch_embed.pos_embed, reference.patch_embed.pos_embed)
