"""Qwen-Image 2.1 transformer structural parity with random weights."""

import pytest
import torch

from imagehub.models.qwen_image21_transformer import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip(
        "needs the diffusers commit pinned for Qwen-Image 2.1 (.venv-ref-qwen-image-2.1)",
        allow_module_level=True,
    )

TINY = {
    "patch_size": 1,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "context_in_dim": 12,
    "mlp_ratio": 3,
    "axes_dims_rope": [2, 2, 4],
    "eps": 1e-6,
    "causal_condition": True,
}
MEDIUM = TINY | {
    "num_layers": 1,
    "attention_head_dim": 64,
    "num_attention_heads": 8,
    "context_in_dim": 96,
    "axes_dims_rope": [8, 28, 28],
}


@pytest.mark.parametrize("config", [TINY, MEDIUM], ids=["tiny", "medium-width"])
def test_qwen_image21_transformer_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.QwenImage21Transformer2DModel(**config).eval()
    model = QwenImage21Transformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch_size, text_length, height, width = 2, 7, 3, 4
    target_length = height * width
    latent = torch.randn(1, target_length, config["in_channels"])
    hidden_states = latent.expand(batch_size, -1, -1).clone()
    encoder_hidden_states = torch.randn(batch_size, text_length, config["context_in_dim"])
    encoder_hidden_states_mask = torch.tensor([[True] * text_length, [True] * 5 + [False] * 2])
    img_mask = torch.cat(
        [
            torch.zeros(batch_size, text_length, dtype=torch.bool),
            torch.ones(batch_size, target_length // 4, dtype=torch.bool),
        ],
        dim=1,
    )
    img_shapes = [[(1, height, width)]] * batch_size
    timesteps = [
        torch.tensor([0.95, 0.95]),
        torch.tensor([0.7, 0.7]),
        torch.tensor([0.35, 0.35]),
        torch.tensor([0.05, 0.05]),
    ]

    reference_cache = diffusers.models.transformers.transformer_qwenimage21.QwenImage21KVCache(
        config["num_layers"]
    )
    model_cache = QwenImage21KVCache(config["num_layers"])
    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            mode = "extract" if index == 0 else "cached"
            expected_uncached = reference(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                return_dict=False,
            )[0]
            actual_uncached = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
            )
            expected_cached = reference(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                kv_cache=reference_cache,
                kv_cache_mode=mode,
                return_dict=False,
            )[0]
            actual_cached = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                kv_cache=model_cache,
                kv_cache_mode=mode,
            )

            torch.testing.assert_close(actual_uncached, expected_uncached, atol=0, rtol=0)
            torch.testing.assert_close(actual_cached, expected_cached, atol=0, rtol=0)
            hidden_states = hidden_states - actual_cached[:, -target_length:] * 0.1
