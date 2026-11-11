import json

import pytest
import torch
from fixture_files import fixture_path
from helpers import write_tiny_hunyuanimage

from imagehub import pipeline
from imagehub.backends.hunyuanimage import (
    AdaptiveProjectedMixGuidance,
    HunyuanImageBackend,
    extract_glyph_text,
    sigmas_for,
)
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.models.autoencoder_kl_hunyuanimage import AutoencoderKLHunyuanImage
from imagehub.models.hunyuanimage_transformer import HunyuanImageTransformer2DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.types import T2IRequest

CONFIGS = fixture_path("hunyuanimage_configs")


def _json(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers")
    return write_tiny_hunyuanimage(tmp_path_factory.mktemp("tiny-hunyuanimage") / "snapshot")


def test_published_tensor_headers_match_the_native_models():
    with torch.device("meta"):
        for model, name in (
            (HunyuanImageTransformer2DModel(_json("transformer_config.json")), "transformer"),
            (AutoencoderKLHunyuanImage(_json("vae_config.json")), "vae"),
            (T5EncoderModel(_json("text_encoder_2_config.json")), "text_encoder_2"),
        ):
            published = _json(f"{name}_tensors.json")
            assert {n: list(v.shape) for n, v in model.state_dict().items()} == published
        text = Qwen25VLTextModel(_json("text_encoder_config.json"))
        published = _json("text_encoder_tensors.json")
        prepared = text.prepare_state_dict(
            {n: torch.empty(s, device="meta") for n, s in published.items()}
        )
        assert set(prepared) == set(text.state_dict())


def test_catalog_entry_and_defaults(tmp_path):
    spec = model_info("hunyuanimage-2-1")
    assert spec.family == "hunyuanimage"
    assert spec.catalog_id == "hunyuanimage-2-1"
    assert spec.repo_id == "hunyuanvideo-community/HunyuanImage-2.1-Diffusers"
    assert spec.revision == "7e7b7a177de58591aeaffca0929f4765003d7ced"
    assert spec.license == "Tencent Hunyuan Community License"
    assert spec.config == {"defaults": {"steps": 50, "width": 2048, "height": 2048}}
    assert len(spec.files) == 28
    options = HunyuanImageBackend(tmp_path, model_id="hunyuanimage-2-1").validate_options(
        T2IRequest("a cat"), {}
    )
    assert (options["steps"], options["width"], options["guidance"]) == (50, 2048, None)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 100}, ValueError, "multiple of 32"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": -1}, ValueError, "guidance"),
        ({"negative_prompt": 3}, ValueError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 64, 64, 64)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 64, 3, 3)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = HunyuanImageBackend(tmp_path)
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_glyph_text_extraction():
    assert extract_glyph_text("a cat") is None
    assert extract_glyph_text("sign 'a' and \"b\"") == 'Text "a". Text "b". '
    assert extract_glyph_text("‘x’ “y”") == 'Text "x". Text "y". '


def test_sigmas_follow_linspace():
    assert sigmas_for(4) == [1.0, 0.75, 0.5, 0.25]


def test_guidance_scale_one_disables_guidance():
    config = _json("guider_guider_config.json")
    guider = AdaptiveProjectedMixGuidance(config, 1.0)
    assert not guider.cfg_enabled
    cond = torch.randn(1, 4, 2, 2)
    assert guider.combine(0, cond, None) is cond


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a HunyuanImage snapshot"):
        HunyuanImageBackend(tmp_path, model_id="hunyuanimage-2-1").load()


def test_generates_images_with_metadata(tiny):
    result = pipeline("t2i", backend="hunyuanimage", model=tiny)(
        "a cat", seed=3, steps=2, width=32, height=32
    )
    assert len(result.images) == 1
    image = result.images[0]
    assert (image.width, image.height) == (32, 32)
    assert image.tensor.dtype == torch.uint8
    metadata = result.metadata
    assert metadata["seed"] == 3 and metadata["steps"] == 2
    assert metadata["guidance"] == 3.5 and metadata["guider"] == "default"
    assert metadata["scheduler"] == "flow_match_euler"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"
    quoted = pipeline("t2i", backend="hunyuanimage", model=tiny)(
        "a sign 'cat'", seed=3, steps=2, width=32, height=32
    )
    assert quoted.metadata["guider"] == "ocr"


def test_seed_is_reproducible_and_requests_are_isolated(tiny):
    run = pipeline("t2i", backend="hunyuanimage", model=tiny)
    first = run("a cat", seed=1, steps=2, width=32, height=32).images[0].tensor
    other = run("a cat", seed=2, steps=2, width=32, height=32).images[0].tensor
    again = run("a cat", seed=1, steps=2, width=32, height=32).images[0].tensor
    assert torch.equal(first, again)
    assert not torch.equal(first, other)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny, grad_before):
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipeline("t2i", backend="hunyuanimage", model=tiny)(
            "a cat", seed=0, steps=1, width=32, height=32
        )
        assert torch.is_grad_enabled() is grad_before
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)
