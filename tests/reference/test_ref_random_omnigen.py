"""OmniGen transformer exact parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.omnigen_inputs import IMAGE_PLACEHOLDER, build_inputs
from imagehub.models.omnigen_transformer import OmniGenTransformer2DModel
from imagehub.tokenization_omnigen import OmniGenTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = Path(__file__).parents[1] / "fixtures"
FACTORS = {
    "short_factor": [1.05, 1.1, 1.2, 1.4],
    "long_factor": [1.5, 2.5, 4.0, 8.0],
    "type": "su",
}
TINY = {
    "in_channels": 4,
    "patch_size": 2,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 48,
    "num_layers": 2,
    "pad_token_id": 32000,
    "vocab_size": 32064,
    "max_position_embeddings": 4096,
    "original_max_position_embeddings": 64,
    "rope_scaling": FACTORS,
    "pos_embed_max_size": 24,
    "time_step_dim": 16,
}


def _compare(config, prompt, with_image, latent_hw, image_hw, record, name, seed=0):
    torch.manual_seed(seed)
    reference = diffusers.OmniGenTransformer2DModel(**config).eval()
    model = OmniGenTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    tokenizer = OmniGenTokenizer(FIXTURES / "omnigen_tokenizer")
    channels = config["in_channels"]
    images = [torch.randn(3, *image_hw)] if with_image else None
    height, width = latent_hw[0] * 8, latent_hw[1] * 8
    processed = build_inputs(
        tokenizer, prompt, images, height=height, width=width, use_img_cfg=with_image
    )
    copies = 3 if with_image else 2
    latents = torch.randn(1, channels, *latent_hw)
    batch = torch.cat([latents] * copies)
    img_latents = [
        torch.randn(1, channels, image_hw[0] // 8, image_hw[1] // 8)
        for _ in processed.input_pixel_values
    ]
    timestep = torch.full((copies,), 0.37)
    args = (
        processed.input_ids,
        img_latents,
        processed.input_image_sizes,
        processed.attention_mask,
        processed.position_ids,
    )
    with torch.inference_mode():
        expected = reference(batch, timestep, *args, return_dict=False)[0]
        actual = model(batch, timestep, *args)
    record(name, max_abs_diff=float((actual - expected).abs().max()))
    assert torch.equal(actual, expected)


def test_omnigen_tiny_text_only_exact(record):
    _compare(TINY, "a red cube on a table", False, (8, 6), (32, 32), record, "omnigen_tiny_text")


def test_omnigen_tiny_with_image_exact(record):
    prompt = f"a portrait like {IMAGE_PLACEHOLDER}, in oil"
    _compare(TINY, prompt, True, (8, 6), (32, 48), record, "omnigen_tiny_image")


def test_omnigen_tiny_long_rope_exact(record):
    # The sequence is longer than original_max_position_embeddings, so the long factors apply.
    prompt = "a very detailed " * 12 + "scene"
    _compare(TINY, prompt, False, (12, 12), (32, 32), record, "omnigen_tiny_long_rope")


@pytest.mark.parametrize("with_image", [False, True])
def test_omnigen_published_config_two_layers_exact(record, with_image):
    config = json.loads((FIXTURES / "omnigen_configs/transformer_config.json").read_text())
    config = {k: v for k, v in config.items() if not k.startswith("_")} | {"num_layers": 2}
    prompt = f"turn {IMAGE_PLACEHOLDER} into a sketch" if with_image else "a cat"
    _compare(
        config, prompt, with_image, (8, 8), (32, 32), record, f"omnigen_full_two_layers_{with_image}"
    )
