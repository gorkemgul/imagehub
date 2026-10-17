"""Original Qwen-Image transformer structural parity with random weights."""

import pytest
import torch

from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY = {
    "patch_size": 2,
    "in_channels": 8,
    "out_channels": 2,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "guidance_embeds": False,
    "axes_dims_rope": [2, 2, 4],
}
PUBLISHED_WIDTH = {
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 1,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 3584,
    "guidance_embeds": False,
    "axes_dims_rope": [16, 56, 56],
}


@pytest.mark.parametrize("config", [TINY, PUBLISHED_WIDTH], ids=["tiny", "published-width"])
def test_qwen_image_transformer_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.QwenImageTransformer2DModel(**config).eval()
    model = QwenImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size, text_length, height, width = 2, 5, 2, 3
    hidden_states = torch.randn(batch_size, height * width, config["in_channels"])
    encoder_hidden_states = torch.randn(batch_size, text_length, config["joint_attention_dim"])
    encoder_hidden_states_mask = torch.tensor(
        [[True] * text_length, [True] * (text_length - 2) + [False] * 2]
    )
    timestep = torch.tensor([0.9375, 0.125])
    img_shapes = [[(1, height, width)]] * batch_size
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]
        actual = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
