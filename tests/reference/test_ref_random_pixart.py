"""PixArt transformer layout and random-weight parity."""

import inspect
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import snapshot_dir

from imagehub.checkpoints import read_json
from imagehub.models.pixart_transformer import PixArtTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

TINY = {
    "num_attention_heads": 4,
    "attention_head_dim": 9,
    "in_channels": 4,
    "out_channels": 8,
    "num_layers": 2,
    "dropout": 0.0,
    "norm_num_groups": 8,
    "cross_attention_dim": 36,
    "attention_bias": True,
    "sample_size": 8,
    "patch_size": 2,
    "activation_fn": "gelu-approximate",
    "num_embeds_ada_norm": 1000,
    "upcast_attention": False,
    "norm_type": "ada_norm_single",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 24,
    "attention_type": "default",
    "use_additional_conditions": False,
}
ALPHA = TINY | {"use_additional_conditions": True}
MEDIUM = TINY | {
    "num_attention_heads": 8,
    "attention_head_dim": 16,
    "num_layers": 4,
    "cross_attention_dim": 128,
    "caption_channels": 96,
    "sample_size": 16,
    "interpolation_scale": 2,
}


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    with (directory / "diffusion_pytorch_model.safetensors").open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


@pytest.mark.parametrize("model_id", ["pixart-sigma-xl-1024", "pixart-alpha-xl-1024"])
def test_published_layout_and_header(model_id, record):
    root = snapshot_dir(model_id) / "transformer"
    config = read_json(root / "config.json")
    accepted = inspect.signature(diffusers.PixArtTransformer2DModel.__init__).parameters
    clean_config = {
        name: value
        for name, value in config.items()
        if not name.startswith("_") and name in accepted
    }
    with torch.device("meta"):
        reference = diffusers.PixArtTransformer2DModel(**clean_config).state_dict()
        model = PixArtTransformer2DModel(config)
        actual = model.state_dict()
    published = _checkpoint_shapes(root)
    expected = {name: tuple(value.shape) for name, value in reference.items()}
    if "caption_projection.y_embedding" in published:
        expected["caption_projection.y_embedding"] = published["caption_projection.y_embedding"]
    native = {name: tuple(value.shape) for name, value in actual.items()}
    result = {
        "missing": sorted(expected.keys() - published.keys()),
        "extra": sorted(published.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & published.keys() if expected[name] != published[name]
        ),
        "tensors": len(published),
    }
    record(f"header_{model_id}_transformer", **result)
    assert native == expected
    assert not result["missing"]
    assert not result["extra"]
    assert not result["mismatch"]
    model.load_checkpoint(
        {name: torch.empty(shape, device="meta") for name, shape in published.items()}
    )


@pytest.mark.parametrize("config", [TINY, ALPHA, MEDIUM], ids=["sigma", "alpha", "medium"])
def test_random_weights_exact(config):
    torch.manual_seed(0)
    reference = diffusers.PixArtTransformer2DModel(**config).eval()
    model = PixArtTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    batch = 2
    hidden_states = torch.randn(batch, config["in_channels"], 6, 10)
    encoder_hidden_states = torch.randn(batch, 7, config["caption_channels"])
    timestep = torch.tensor([500.0, 250.0])
    encoder_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    added_cond_kwargs = None
    if config["use_additional_conditions"]:
        added_cond_kwargs = {
            "resolution": torch.tensor([[768.0, 1280.0]] * batch),
            "aspect_ratio": torch.tensor([[0.6]] * batch),
        }

    with torch.inference_mode():
        expected = reference(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
        actual = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
            added_cond_kwargs,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
