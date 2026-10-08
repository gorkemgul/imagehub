"""CogView4 transformer layout and random-weight parity."""

import pytest
import torch

from imagehub.models.cogview4_transformer import CogView4Transformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

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
    "rope_axes_dim": [8, 8],
}
MEDIUM = TINY | {
    "num_layers": 2,
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "text_embed_dim": 96,
    "time_embed_dim": 128,
    "condition_dim": 32,
    "rope_axes_dim": [16, 16],
}


@pytest.mark.parametrize(
    ("config", "height", "width", "text_length", "use_mask"),
    [(TINY, 6, 10, 16, False), (MEDIUM, 4, 6, 32, True)],
    ids=["tiny", "medium-width-masked"],
)
def test_cogview4_transformer_random_weights_exact(config, height, width, text_length, use_mask):
    torch.manual_seed(0)
    reference = diffusers.CogView4Transformer2DModel(**config).eval()
    model = CogView4Transformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size = 2
    hidden_states = torch.randn(batch_size, config["in_channels"], height, width)
    encoder_hidden_states = torch.randn(batch_size, text_length, config["text_embed_dim"])
    timestep = torch.tensor([1.0, 753.25])
    original_size = torch.tensor([[768.0, 1280.0], [1024.0, 1536.0]])
    target_size = torch.tensor([[768.0, 1280.0], [896.0, 1344.0]])
    crop_coords = torch.tensor([[0.0, 0.0], [64.0, 96.0]])
    attention_mask = None
    if use_mask:
        attention_mask = torch.ones(batch_size, text_length)
        attention_mask[1, -8:] = 0

    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        actual = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            original_size,
            target_size,
            crop_coords,
            attention_mask=attention_mask,
        )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
