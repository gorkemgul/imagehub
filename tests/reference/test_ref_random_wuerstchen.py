"""Random-weight parity for the Würstchen v2 prior and decoder networks."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.wuerstchen import WuerstchenDiffNeXt, WuerstchenPrior
from tests.test_wuerstchen_models import DECODER_TINY, PRIOR_TINY

pytestmark = pytest.mark.reference
wuerstchen = pytest.importorskip("diffusers.pipelines.deprecated.wuerstchen")

FIXTURES = Path(__file__).parents[1] / "fixtures/wuerstchen_configs"


def _published(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _build(reference_cls, our_cls, config):
    torch.manual_seed(0)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    reference = reference_cls(**config).eval()
    with torch.no_grad():
        for tensor in reference.parameters():
            if not tensor.any():  # zero-initialised output layers and modulation
                tensor.normal_(std=0.05)
    ours = our_cls(config).eval()
    ours.load_checkpoint(reference.state_dict())
    return reference, ours


def _compare(record, name, reference, ours, **kwargs):
    with torch.inference_mode():
        expected = reference(**kwargs)
        actual = ours(**kwargs)
    assert torch.isfinite(expected).all()
    record(name, max_abs_difference=(actual - expected).abs().max().item(), tolerance=0.0)
    assert torch.equal(actual, expected)


def _prior_inputs(config, size, batch=2):
    return {
        "x": torch.randn(batch, config.get("c_in", 16), *size),
        "r": torch.tensor([0.3, 0.8, 0.5][:batch]),
        "c": torch.randn(batch, 77, config.get("c_cond", 1024)),
    }


@pytest.mark.parametrize("size", [(5, 6), (12, 12)], ids=["5x6", "12x12"])
def test_prior_tiny(record, size):
    reference, ours = _build(wuerstchen.WuerstchenPrior, WuerstchenPrior, PRIOR_TINY)
    torch.manual_seed(1)
    _compare(
        record,
        f"wuerstchen_prior_tiny_{size[0]}x{size[1]}",
        reference,
        ours,
        **_prior_inputs(PRIOR_TINY, size),
    )


def test_prior_published_small_size(record):
    config = _published("prior-repo/prior_config.json")
    reference, ours = _build(wuerstchen.WuerstchenPrior, WuerstchenPrior, config)
    torch.manual_seed(1)
    _compare(
        record,
        "wuerstchen_prior_published_4x5",
        reference,
        ours,
        **_prior_inputs(config, (4, 5), batch=1),
    )


def _decoder_inputs(config, size, effnet_size, batch=2):
    return {
        "x": torch.randn(batch, config["c_in"], *size),
        "r": torch.tensor([0.3, 0.8, 0.5][:batch]),
        "effnet": torch.randn(batch, config["effnet_embd"], *effnet_size),
        "clip": torch.randn(batch, 77, config["clip_embd"]),
    }


@pytest.mark.parametrize(
    ("size", "effnet_size"), [((32, 48), (3, 4)), ((64, 64), (5, 5))], ids=["32x48", "64x64"]
)
def test_decoder_tiny(record, size, effnet_size):
    reference, ours = _build(wuerstchen.WuerstchenDiffNeXt, WuerstchenDiffNeXt, DECODER_TINY)
    torch.manual_seed(1)
    _compare(
        record,
        f"wuerstchen_decoder_tiny_{size[0]}x{size[1]}",
        reference,
        ours,
        **_decoder_inputs(DECODER_TINY, size, effnet_size),
    )


def test_decoder_tiny_x_cat_and_split_outputs(record):
    reference, ours = _build(wuerstchen.WuerstchenDiffNeXt, WuerstchenDiffNeXt, DECODER_TINY)
    torch.manual_seed(2)
    inputs = _decoder_inputs(DECODER_TINY, (32, 32), (3, 3)) | {
        "x": torch.randn(2, 2, 32, 32),
        "x_cat": torch.randn(2, 2, 32, 32),
    }
    _compare(record, "wuerstchen_decoder_tiny_x_cat", reference, ours, **inputs)
    with torch.inference_mode():
        expected = reference(**inputs, return_noise=False)
        actual = ours(**inputs, return_noise=False)
    assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])


def test_decoder_published_small_size(record):
    config = _published("decoder-repo/decoder_config.json")
    reference, ours = _build(wuerstchen.WuerstchenDiffNeXt, WuerstchenDiffNeXt, config)
    torch.manual_seed(1)
    _compare(
        record,
        "wuerstchen_decoder_published_32x48",
        reference,
        ours,
        **_decoder_inputs(config, (32, 48), (3, 4), batch=1),
    )
