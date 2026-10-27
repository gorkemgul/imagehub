import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.models.stable_cascade_unet import StableCascadeUNet
from tests.test_stable_cascade_unet import DECODER_TINY, PRIOR_TINY

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

FIXTURES = fixture_path("stable_cascade_configs")


def _build(config):
    torch.manual_seed(0)
    ref = diffusers.StableCascadeUNet(**{k: v for k, v in config.items() if not k.startswith("_")})
    ref.eval()
    with torch.no_grad():
        for name, value in ref.named_parameters():
            if name.endswith(("gamma", "beta")):
                value.normal_(std=0.5)
    ours = StableCascadeUNet(config).eval()
    ours.load_checkpoint(ref.state_dict())
    return ref, ours


def _compare(record, name, ref, ours, **kwargs):
    with torch.inference_mode():
        expected = ref(**kwargs, return_dict=False)[0]
        actual = ours(**kwargs)
    assert torch.isfinite(expected).all()
    difference = (actual - expected).abs().max().item()
    record(name, max_abs_difference=difference, tolerance=0.0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("size", [(12, 12), (13, 11)], ids=["12x12", "13x11"])
def test_prior_like_tiny(record, size):
    ref, ours = _build(PRIOR_TINY)
    torch.manual_seed(1)
    _compare(
        record,
        f"stable_cascade_unet_prior_tiny_{size[0]}x{size[1]}",
        ref,
        ours,
        sample=torch.randn(2, 4, *size),
        timestep_ratio=torch.tensor([0.3, 0.8]),
        clip_text_pooled=torch.randn(2, 1, 16),
        clip_text=torch.randn(2, 77, 20),
        clip_img=torch.randn(2, 1, 12),
    )


def test_decoder_like_tiny(record):
    ref, ours = _build(DECODER_TINY)
    torch.manual_seed(1)
    _compare(
        record,
        "stable_cascade_unet_decoder_tiny",
        ref,
        ours,
        sample=torch.randn(2, 4, 36, 38),
        timestep_ratio=torch.tensor([0.3, 0.8]),
        clip_text_pooled=torch.randn(2, 1, 16),
        effnet=torch.randn(2, 6, 5, 5),
        pixels=torch.randn(2, 3, 8, 8),
    )


def test_decoder_like_tiny_default_pixels(record):
    ref, ours = _build(DECODER_TINY)
    torch.manual_seed(2)
    _compare(
        record,
        "stable_cascade_unet_decoder_tiny_default_pixels",
        ref,
        ours,
        sample=torch.randn(2, 4, 36, 38),
        timestep_ratio=torch.tensor([0.5, 0.1]),
        clip_text_pooled=torch.randn(2, 1, 16),
        effnet=torch.randn(2, 6, 5, 5),
    )


def test_prior_lite_full_size(record):
    config = json.loads((FIXTURES / "prior_lite_config.json").read_text(encoding="utf-8"))
    ref, ours = _build(config)
    torch.manual_seed(1)
    _compare(
        record,
        "stable_cascade_unet_prior_lite",
        ref,
        ours,
        sample=torch.randn(1, 16, 12, 12),
        timestep_ratio=torch.tensor([0.6]),
        clip_text_pooled=torch.randn(1, 1, 1280),
        clip_text=torch.randn(1, 77, 1280),
        clip_img=torch.randn(1, 1, 768),
    )


def test_decoder_lite_full_size(record):
    config = json.loads((FIXTURES / "decoder_lite_config.json").read_text(encoding="utf-8"))
    ref, ours = _build(config)
    torch.manual_seed(1)
    _compare(
        record,
        "stable_cascade_unet_decoder_lite",
        ref,
        ours,
        sample=torch.randn(1, 4, 32, 32),
        timestep_ratio=torch.tensor([0.4]),
        clip_text_pooled=torch.randn(1, 1, 1280),
        effnet=torch.randn(1, 16, 4, 4),
    )
