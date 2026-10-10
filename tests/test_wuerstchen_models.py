import json
from pathlib import Path

import pytest
import torch

from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.wuerstchen import WuerstchenDiffNeXt, WuerstchenPrior

FIXTURES = Path(__file__).parent / "fixtures/wuerstchen_configs"

PRIOR_TINY = {"c_in": 4, "c": 32, "c_cond": 24, "c_r": 16, "depth": 2, "nhead": 4, "dropout": 0.1}
DECODER_TINY = {
    "c_in": 4,
    "c_out": 4,
    "c_r": 16,
    "patch_size": 2,
    "c_cond": 24,
    "c_hidden": [8, 16, 32, 32],
    "nhead": [-1, 2, 4, 4],
    "blocks": [1, 1, 2, 1],
    "level_config": ["CT", "CTA", "CTA", "CTA"],
    "inject_effnet": [False, True, True, True],
    "effnet_embd": 6,
    "clip_embd": 20,
    "kernel_size": 3,
    "dropout": 0.1,
}


def _published(config_file: str, tensors_file: str):
    config = json.loads((FIXTURES / config_file).read_text(encoding="utf-8"))
    tensors = json.loads((FIXTURES / tensors_file).read_text(encoding="utf-8"))
    return config, {key: tuple(shape) for key, shape in tensors.items()}


@pytest.mark.parametrize(
    ("cls", "config_file", "tensors_file", "count"),
    [
        (WuerstchenPrior, "prior-repo/prior_config.json", "prior_tensors.json", 648),
        (WuerstchenDiffNeXt, "decoder-repo/decoder_config.json", "decoder_tensors.json", 990),
        (PaellaVQModel, "decoder-repo/vqgan_config.json", "vqgan_tensors.json", 122),
    ],
    ids=["prior", "decoder", "vqgan"],
)
def test_published_layout(cls, config_file, tensors_file, count):
    config, expected = _published(config_file, tensors_file)
    with torch.device("meta"):
        model = cls(config)
    actual = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    assert len(expected) == count
    assert sorted(set(expected) - set(actual)) == []
    assert sorted(set(actual) - set(expected)) == []
    assert {k: (actual[k], expected[k]) for k in actual if actual[k] != expected[k]} == {}


def test_meta_construction_then_checkpoint_load_leaves_no_meta_tensors():
    source = WuerstchenPrior(PRIOR_TINY)
    with torch.device("meta"):
        model = WuerstchenPrior(PRIOR_TINY)
    model.load_checkpoint(source.state_dict())
    assert all(value.device.type == "cpu" for value in (*model.parameters(), *model.buffers()))
    assert torch.equal(model.projection.weight, source.projection.weight)


def test_prior_forward_shape():
    model = WuerstchenPrior(PRIOR_TINY).eval()
    out = model(torch.randn(2, 4, 5, 6), torch.tensor([0.3, 0.8]), torch.randn(2, 7, 24))
    assert out.shape == (2, 4, 5, 6)


def test_decoder_forward_shape_and_noise_split():
    model = WuerstchenDiffNeXt(DECODER_TINY).eval()
    x, r = torch.randn(2, 4, 32, 48), torch.tensor([0.3, 0.8])
    effnet, clip = torch.randn(2, 6, 3, 4), torch.randn(2, 7, 20)
    noise = model(x, r, effnet, clip)
    assert noise.shape == x.shape
    a, b = model(x, r, effnet, clip, return_noise=False)
    assert torch.equal(noise, (x - a) / b)


@pytest.mark.parametrize(
    "override",
    [
        {"unknown_option": 1},
        {"level_config": ["CT", "CTX", "CTA", "CTA"]},
        {"blocks": [1, 1]},
        {"nhead": [-1, -1, 4, 4]},
    ],
)
def test_unsupported_decoder_options_are_rejected(override):
    with pytest.raises(ValueError):
        WuerstchenDiffNeXt(DECODER_TINY | override)


def test_unsupported_prior_option_is_rejected():
    with pytest.raises(ValueError):
        WuerstchenPrior(PRIOR_TINY | {"unknown_option": 1})
