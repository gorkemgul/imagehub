"""GLM-Image transformer random-weight parity with diffusers."""

import pytest
import torch

from imagehub.models.glm_image_transformer import GlmImageTransformer2DModel

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
    "prior_vq_quantizer_codebook_size": 32,
}
MEDIUM = TINY | {
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "text_embed_dim": 96,
    "time_embed_dim": 128,
    "condition_dim": 32,
    "prior_vq_quantizer_codebook_size": 64,
}
# The published configuration with fewer layers, at small spatial sizes.
PUBLISHED = {
    "patch_size": 2,
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 2,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "text_embed_dim": 1472,
    "time_embed_dim": 512,
    "condition_dim": 256,
    "prior_vq_quantizer_codebook_size": 16384,
}


def _run(config, height, width, text_length, batch_size, drop):
    torch.manual_seed(0)
    reference = diffusers.GlmImageTransformer2DModel(**config).eval()
    model = GlmImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    patches = (height // 2) * (width // 2)
    hidden_states = torch.randn(batch_size, config["in_channels"], height, width)
    encoder_hidden_states = torch.randn(batch_size, text_length, config["text_embed_dim"])
    prior_token_id = torch.randint(0, config["prior_vq_quantizer_codebook_size"], (batch_size, patches))
    prior_token_drop = torch.full_like(prior_token_id, drop, dtype=torch.bool)
    timestep = torch.tensor([1.0, 753.25, 999.0][:batch_size])
    target_size = torch.tensor([[768.0, 1280.0], [1024.0, 1536.0], [512.0, 512.0]][:batch_size])
    crop_coords = torch.zeros(batch_size, 2)
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            prior_token_id=prior_token_id,
            prior_token_drop=prior_token_drop.clone(),
            timestep=timestep,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
        )[0]
        actual = model(
            hidden_states,
            encoder_hidden_states,
            prior_token_id,
            prior_token_drop,
            timestep,
            target_size,
            crop_coords,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("drop", [False, True], ids=["conditional", "dropped-prior"])
@pytest.mark.parametrize(
    ("config", "height", "width", "text_length", "batch_size"),
    [(TINY, 6, 10, 7, 2), (MEDIUM, 8, 12, 33, 3), (PUBLISHED, 8, 12, 5, 2)],
    ids=["tiny", "medium", "published-config"],
)
def test_glm_image_transformer_random_weights_exact(
    config, height, width, text_length, batch_size, drop
):
    _run(config, height, width, text_length, batch_size, drop)
