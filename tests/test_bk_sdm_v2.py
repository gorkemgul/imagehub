import json
from pathlib import Path

import pytest
import torch
from helpers import TINY_UNET_CONFIG, write_tiny_sd

from imagehub import pipeline
from imagehub.backends.stable_diffusion import StableDiffusionBackend
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel

FIXTURES = Path(__file__).parent / "fixtures"
VARIANTS = ("base", "small", "tiny")
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
TINY_CONFIGS = {
    "base": {
        "attention_head_dim": [2, 4, 8, 8],
        "layers_per_block": 1,
        "use_linear_projection": True,
    },
    "small": {
        "attention_head_dim": [2, 4, 8, 8],
        "layers_per_block": 1,
        "mid_block_type": None,
        "use_linear_projection": True,
    },
    "tiny": {
        "attention_head_dim": [2, 4, 8],
        "block_out_channels": [32, 64, 64],
        "down_block_types": ["CrossAttnDownBlock2D"] * 3,
        "up_block_types": ["CrossAttnUpBlock2D"] * 3,
        "layers_per_block": 1,
        "mid_block_type": None,
        "use_linear_projection": True,
    },
}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("component", COMPONENTS)
def test_published_component_layouts_match_on_meta(variant, component):
    config = json.loads(
        (FIXTURES / f"bk_sdm_v2_configs/{variant}/{component}.json").read_text(encoding="utf-8")
    )
    published = json.loads(
        (FIXTURES / f"bk_sdm_v2_{variant}_tensor_shapes.json").read_text(encoding="utf-8")
    )[component]
    cls = COMPONENTS[component]
    with torch.device("meta"):
        model = cls(config)
    prepared = model.prepare_state_dict(
        {name: torch.empty(entry["shape"], device="meta") for name, entry in published.items()}
    )
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    actual = {name: tuple(tensor.shape) for name, tensor in prepared.items()}
    assert actual == expected


@pytest.mark.parametrize("variant", VARIANTS)
def test_tiny_bk_sdm_v2_variants_generate(variant, tmp_path):
    config = TINY_UNET_CONFIG | TINY_CONFIGS[variant] | {"sample_size": 8}
    root = write_tiny_sd(tmp_path / variant, unet_config=config)
    result = pipeline("t2i", backend=f"bk-sdm-v2-{variant}", model=root)(
        "a cat", seed=3, steps=2, width=64, height=64
    )
    assert len(result.images) == 1
    assert result.images[0].tensor.shape == (64, 64, 3)
    assert result.metadata["scheduler"] == "pndm"


@pytest.mark.parametrize("variant", VARIANTS)
def test_invalid_option_fails_before_loading(variant, tmp_path):
    root = write_tiny_sd(tmp_path / variant, unet_config=TINY_CONFIGS[variant])
    pipe = pipeline("t2i", backend=f"bk-sdm-v2-{variant}", model=root)
    with pytest.raises(TypeError, match="stepz"):
        pipe("a cat", stepz=2)
    assert not pipe.backend.is_loaded


@pytest.mark.parametrize("variant", VARIANTS)
def test_published_defaults(variant):
    backend = StableDiffusionBackend("unused", model_id=f"bk-sdm-v2-{variant}")
    assert backend.option_defaults["steps"] == 50
    assert backend.option_defaults["guidance"] == 7.5
