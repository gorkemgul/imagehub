import re

import pytest

from imagehub.catalog import MODEL_CATALOG, ModelSpec, available_models, model_info
from imagehub.types import Task


def test_catalog_contents():
    assert available_models() == {
        "t2i": (
            "sd-v1-1",
            "sd-v1-2",
            "sd-v1-3",
            "sd-v1-4",
            "sd-v1-5",
            "sdxl-base-1.0",
            "ssd-1b",
        )
    }
    for spec in (value for value in MODEL_CATALOG.values() if value.family == "stable-diffusion"):
        assert spec.task is Task.T2I
        assert spec.family == "stable-diffusion"
        assert spec.catalog_id == "stable-diffusion-1"
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.source == f"https://huggingface.co/{spec.repo_id}"
        assert spec.license == "CreativeML-OpenRAIL-M"
        assert len(set(spec.files)) == len(spec.files) == 12
        assert not any(name.endswith((".py", ".ckpt")) or "fp16" in name for name in spec.files)
        assert not any(
            name.startswith(("safety_checker/", "feature_extractor/")) for name in spec.files
        )

    sdxl = model_info("sdxl-base-1.0")
    assert sdxl.task is Task.T2I
    assert sdxl.family == "stable-diffusion-xl"
    assert sdxl.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert sdxl.revision == "462165984030d82259a11f4367a4eed129e94a7b"
    assert sdxl.provenance == "author"
    assert sdxl.catalog_id == "sdxl"
    assert sdxl.format == "diffusers"
    assert sdxl.license == "CreativeML-OpenRAIL++-M"
    assert len(set(sdxl.files)) == len(sdxl.files) == 18

    ssd1b = model_info("ssd-1b")
    assert ssd1b.task is Task.T2I
    assert ssd1b.family == "stable-diffusion-xl"
    assert ssd1b.repo_id == "segmind/SSD-1B"
    assert ssd1b.revision == "60987f37e94cd59c36b1cba832b9f97b57395a10"
    assert ssd1b.provenance == "author"
    assert ssd1b.catalog_id == "ssd-1b"
    assert ssd1b.format == "diffusers"
    assert ssd1b.license == "Apache-2.0"
    assert len(set(ssd1b.files)) == len(ssd1b.files) == 18


def test_provenance_and_formats():
    assert model_info("sd-v1-5").provenance == "community-mirror"
    assert model_info("sd-v1-5").revision == "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    for version in (1, 2, 3, 4):
        assert model_info(f"sd-v1-{version}").provenance == "author"
    assert "unet/diffusion_pytorch_model.bin" in model_info("sd-v1-1").files
    assert "unet/diffusion_pytorch_model.safetensors" in model_info("sd-v1-4").files


def test_unknown_model_lists_available():
    with pytest.raises(ValueError, match="sd-v1-4, sd-v1-5"):
        model_info("sd-v15")


def test_spec_validation():
    base = model_info("sd-v1-4")
    with pytest.raises(ValueError, match="revision"):
        ModelSpec(**{**base.__dict__, "revision": "main"})
    with pytest.raises(ValueError, match="provenance"):
        ModelSpec(**{**base.__dict__, "provenance": "unknown"})
