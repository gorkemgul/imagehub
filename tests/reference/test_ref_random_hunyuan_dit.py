"""HunyuanDiT transformer structural parity with random weights."""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import snapshot_dir

from imagehub.checkpoints import read_json
from imagehub.models.hunyuan_dit_transformer import (
    HunyuanDiT2DModel,
    get_2d_rotary_pos_embed,
    get_resize_crop_region_for_grid,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY = {
    "sample_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 4,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "hidden_size": 32,
    "mlp_ratio": 2.0,
    "learn_sigma": True,
    "cross_attention_dim": 24,
    "cross_attention_dim_t5": 32,
    "pooled_projection_dim": 20,
    "text_len": 5,
    "text_len_t5": 7,
    "activation_fn": "gelu-approximate",
    "norm_type": "layer_norm",
}
MEDIUM = TINY | {
    "sample_size": 16,
    "num_layers": 6,
    "attention_head_dim": 32,
    "num_attention_heads": 4,
    "hidden_size": 128,
    "mlp_ratio": 2.5,
    "cross_attention_dim": 64,
    "cross_attention_dim_t5": 80,
    "pooled_projection_dim": 48,
    "text_len": 7,
    "text_len_t5": 9,
}


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    with (directory / "diffusion_pytorch_model.safetensors").open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


@pytest.mark.parametrize(
    "model_id", ["hunyuan-dit-v1.1-distilled", "hunyuan-dit-v1.2", "hunyuan-dit-v1.2-distilled"]
)
def test_hunyuan_dit_published_layout_and_header(record, model_id):
    root = snapshot_dir(model_id) / "transformer"
    config = read_json(root / "config.json")
    clean_config = {name: value for name, value in config.items() if not name.startswith("_")}
    with torch.device("meta"):
        reference = diffusers.HunyuanDiT2DModel(**clean_config).state_dict()
        model = HunyuanDiT2DModel(config)
        actual = model.state_dict()
    published = _checkpoint_shapes(root)
    expected = {name: tuple(value.shape) for name, value in reference.items()}
    native = {name: tuple(value.shape) for name, value in actual.items()}
    result = {
        "missing": sorted(expected.keys() - published.keys()),
        "extra": sorted(published.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & published.keys() if expected[name] != published[name]
        ),
        "tensors": len(published),
    }
    name = "header_hunyuan_dit_transformer"
    record(name if model_id == "hunyuan-dit-v1.1-distilled" else f"{name}_{model_id}", **result)
    assert native == expected
    assert not result["missing"]
    assert not result["extra"]
    assert not result["mismatch"]
    model.load_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in published.items()}
    )


@pytest.mark.parametrize(
    "config",
    [
        TINY,
        MEDIUM,
        TINY | {"use_style_cond_and_image_meta_size": False},
        MEDIUM | {"use_style_cond_and_image_meta_size": False},
    ],
    ids=["tiny", "medium", "tiny-no-style", "medium-no-style"],
)
def test_hunyuan_dit_transformer_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.HunyuanDiT2DModel(**config).eval()
    model = HunyuanDiT2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch = 2
    latent_height, latent_width = 6, 10
    hidden_states = torch.randn(batch, config["in_channels"], latent_height, latent_width)
    timestep = torch.tensor([500.0, 250.0])
    encoder_hidden_states = torch.randn(batch, config["text_len"], config["cross_attention_dim"])
    encoder_hidden_states_t5 = torch.randn(
        batch, config["text_len_t5"], config["cross_attention_dim_t5"]
    )
    text_embedding_mask = torch.tensor(
        [[1] * config["text_len"], [1, 1, 1] + [0] * (config["text_len"] - 3)]
    )
    text_embedding_mask_t5 = torch.tensor(
        [[1] * (config["text_len_t5"] - 2) + [0, 0], [1, 1] + [0] * (config["text_len_t5"] - 2)]
    )
    image_meta_size = torch.tensor([[768, 1280, 768, 1280, 0, 0]] * batch, dtype=torch.float32)
    style = torch.zeros(batch, dtype=torch.long)

    grid_size = (latent_height // config["patch_size"], latent_width // config["patch_size"])
    crops = get_resize_crop_region_for_grid(grid_size, 32)
    actual_rotary = get_2d_rotary_pos_embed(config["attention_head_dim"], crops, grid_size)
    expected_crops = (
        diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid(
            grid_size, 32
        )
    )
    expected_rotary = diffusers.models.embeddings.get_2d_rotary_pos_embed(
        config["attention_head_dim"], expected_crops, grid_size, output_type="pt"
    )
    assert crops == expected_crops
    torch.testing.assert_close(actual_rotary[0], expected_rotary[0], atol=0, rtol=0)
    torch.testing.assert_close(actual_rotary[1], expected_rotary[1], atol=0, rtol=0)

    kwargs = {
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "text_embedding_mask": text_embedding_mask,
        "encoder_hidden_states_t5": encoder_hidden_states_t5,
        "text_embedding_mask_t5": text_embedding_mask_t5,
        "image_meta_size": image_meta_size,
        "style": style,
        "image_rotary_emb": actual_rotary,
    }
    with torch.inference_mode():
        expected = reference(hidden_states, **kwargs).sample
        actual = model(hidden_states, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
