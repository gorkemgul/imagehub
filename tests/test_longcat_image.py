from pathlib import Path

import pytest
import torch

from imagehub.backends.longcat_image import (
    LongCatImageBackend,
    _apply_cfg,
    _pack_latents,
    _prepare_latent_image_ids,
    _unpack_latents,
)
from imagehub.catalog import model_info
from imagehub.types import T2IRequest


def test_catalog_entry_and_defaults():
    spec = model_info("longcat-image")
    assert spec.family == "longcat-image"
    assert spec.repo_id == "meituan-longcat/LongCat-Image"
    assert spec.revision == "d2ea50b79a930074c37b9b97ce45e3b2ea8cf4d8"
    assert spec.provenance == "author"
    assert spec.catalog_id == "longcat-image"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {
            "steps": 50,
            "guidance": 4.5,
            "enable_cfg_renorm": True,
            "cfg_renorm_min": 0.0,
            "enable_prompt_rewrite": True,
        }
    }
    assert len(set(spec.files)) == len(spec.files)
    assert "text_encoder/model-00005-of-00005.safetensors" in spec.files
    assert not any(
        name == "config.json" or name == "README.md" or name.startswith("assets/")
        for name in spec.files
    )


def test_backend_uses_catalog_defaults():
    backend = LongCatImageBackend(Path("missing"), model_id="longcat-image")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert (
        options["steps"],
        options["guidance"],
        options["enable_cfg_renorm"],
        options["cfg_renorm_min"],
        options["enable_prompt_rewrite"],
    ) == (50, 4.5, True, 0.0, True)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 1000}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"cfg_renorm_min": float("inf")}, ValueError, "cfg_renorm_min"),
        ({"cfg_renorm_min": -0.1}, ValueError, "cfg_renorm_min"),
        ({"cfg_renorm_min": 1.1}, ValueError, "cfg_renorm_min"),
        ({"enable_cfg_renorm": 1}, TypeError, "enable_cfg_renorm"),
        ({"enable_prompt_rewrite": 1}, TypeError, "enable_prompt_rewrite"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        (
            {"seed": 1, "latents": torch.zeros(1, 16, 128, 128)},
            ValueError,
            "either",
        ),
        ({"latents": torch.zeros(1, 16, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = LongCatImageBackend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_pack_and_unpack_latents():
    latents = torch.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6)
    packed = _pack_latents(latents, 1, 2, 4, 6)
    assert packed.shape == (1, 6, 8)
    assert packed[0, 0].tolist() == [0, 1, 6, 7, 24, 25, 30, 31]
    assert torch.equal(_unpack_latents(packed, 32, 48, 8), latents)


def test_latent_image_ids_start_after_text_range():
    ids = _prepare_latent_image_ids(2, 2, torch.device("cpu"), torch.float32)
    assert ids.tolist() == [
        [1.0, 512.0, 512.0],
        [1.0, 512.0, 513.0],
        [1.0, 513.0, 512.0],
        [1.0, 513.0, 513.0],
    ]


def test_cfg_and_cfg_renormalization():
    conditional = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]])
    unconditional = torch.tensor([[[1.0, 0.0], [4.0, 2.0]]])
    guided = unconditional + 4.5 * (conditional - unconditional)
    assert torch.equal(
        _apply_cfg(conditional, unconditional, 4.5, False, 0.0),
        guided,
    )
    expected_scale = (
        conditional.norm(dim=-1, keepdim=True) / (guided.norm(dim=-1, keepdim=True) + 1e-8)
    ).clamp(min=0.25, max=1.0)
    torch.testing.assert_close(
        _apply_cfg(conditional, unconditional, 4.5, True, 0.25),
        guided * expected_scale,
        rtol=0,
        atol=0,
    )
