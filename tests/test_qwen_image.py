import json

import pytest
import torch
from helpers import write_tiny_qwen_image, write_tiny_qwen_image21, write_tiny_qwen_image_edit

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    QwenImageBackend,
    QwenImageComponents,
    _vision_pixels,
    encode_prompt,
)
from imagehub.catalog import model_info
from imagehub.checkpoints import SNAPSHOT_FILE, load_weights, save_safetensors
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def tiny_qwen_image(tmp_path_factory):
    pytest.importorskip("transformers")  # builds the tiny text encoder
    return write_tiny_qwen_image21(tmp_path_factory.mktemp("tiny-qwen-image"))


@pytest.fixture(scope="module")
def tiny_original_qwen_image(tmp_path_factory):
    pytest.importorskip("transformers")
    return write_tiny_qwen_image(tmp_path_factory.mktemp("tiny-original-qwen-image"))


@pytest.fixture(scope="module")
def tiny_qwen_image_edit(tmp_path_factory):
    pytest.importorskip("transformers")
    return write_tiny_qwen_image_edit(tmp_path_factory.mktemp("tiny-qwen-image-edit"))


def test_edit_release_generates_from_an_image(tiny_qwen_image_edit):
    source = Image(torch.arange(32 * 48 * 3, dtype=torch.int64).reshape(32, 48, 3).to(torch.uint8))
    pipe = pipeline("t2i", backend="qwen-image-edit", model=tiny_qwen_image_edit)
    first = pipe("turn it blue", image=source, seed=7, steps=1, width=32, height=32)
    second = pipe("turn it blue", image=source, seed=7, steps=1, width=32, height=32)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert first.metadata["image_input"] is True


def test_edit_vision_pixels_match_fused_uint8_preprocessing():
    source = Image(torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8))
    config = {
        "patch_size": 14,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "min_pixels": 3136,
        "max_pixels": 12_845_056,
        "rescale_factor": 1 / 255,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }
    actual, grid = _vision_pixels(source, config)

    pixels = source.tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
    pixels = torch.nn.functional.interpolate(
        pixels, size=(56, 84), mode="bicubic", align_corners=False, antialias=True
    )[0]
    mean = torch.tensor(config["image_mean"])[:, None, None] * 255
    std = torch.tensor(config["image_std"])[:, None, None] * 255
    pixels = (pixels.float() - mean) / std
    expected = pixels.reshape(3, 2, 2, 14, 3, 2, 14).permute(1, 4, 2, 5, 0, 3, 6)
    expected = expected.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1)

    assert torch.equal(grid, torch.tensor([[1, 4, 6]]))
    assert torch.equal(actual, expected.reshape(24, 1176))


def test_edit_release_requires_an_image_before_loading(tmp_path):
    backend = QwenImageBackend(tmp_path, model_id="qwen-image-edit")
    with pytest.raises(ValueError, match="image is required"):
        pipeline("t2i", backend=backend)("turn it blue", steps=1, width=32, height=32)
    assert not backend.is_loaded


def test_original_release_generates_repeatably(tiny_original_qwen_image):
    pipe = pipeline("t2i", backend="qwen-image", model=tiny_original_qwen_image)
    first = pipe("a cat", seed=9, steps=2, true_cfg=1.0, width=32, height=48)
    second = pipe("a cat", seed=9, steps=2, true_cfg=1.0, width=32, height=48)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert (first.images[0].width, first.images[0].height) == (32, 48)
    assert first.metadata["seed"] == 9


def test_original_defaults_and_snapshot_dispatch(tiny_original_qwen_image):
    backend = QwenImageBackend(tiny_original_qwen_image)
    selected = backend.validate_options(T2IRequest("a cat"), {})
    assert selected["steps"] == 50
    assert selected["true_cfg"] == 4.0
    assert backend.original_release


def test_generates_images_with_metadata(tiny_qwen_image):
    result = pipeline("t2i", backend="qwen-image", model=tiny_qwen_image)(
        "a cat", seed=3, steps=2, width=32, height=32, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.images[0].channels == 4
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["true_cfg"] == 1.0
    assert result.metadata["scheduler"] == "flow_match_euler"


def test_seed_and_latents_are_reproducible(tiny_qwen_image):
    pipe = pipeline("t2i", backend="qwen-image-2.1", model=tiny_qwen_image)
    a = pipe("a cat", seed=11, steps=2, width=32, height=32)
    b = pipe("a cat", seed=11, steps=2, width=32, height=32)
    latents = torch.randn(1, 1, 64, 2, 2, generator=torch.Generator("cpu").manual_seed(11))
    latents = latents.view(1, 64, 4).transpose(1, 2)
    c = pipe("a cat", latents=latents, steps=2, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 32"),
        ({"steps": 0}, ValueError, "steps"),
        ({"true_cfg": float("nan")}, ValueError, "true_cfg"),
        ({"true_cfg": -0.1}, ValueError, "true_cfg"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 4, 64)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 16, 64)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
    ],
)
def test_invalid_options_fail_before_loading(tmp_path, options, error, match):
    backend = QwenImageBackend(tmp_path)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_catalog_entry_and_snapshot_hint(tmp_path):
    original = model_info("qwen-image")
    assert original.repo_id == "Qwen/Qwen-Image"
    assert original.revision == "75e0b4be04f60ec59a75f475837eced720f823b6"
    assert original.license == "Apache-2.0"
    assert len(original.files) == 29
    spec = model_info("qwen-image-2.1")
    assert spec.family == "qwen-image" and spec.catalog_id == "qwen-image-2"
    assert spec.repo_id == "Qwen/Qwen-Image-2.1"
    assert spec.revision == "d26bb61231c349cf6b7896fa83353113880e1ba3"
    assert spec.license == "Qwen Research License"
    assert spec.reference["diffusers"] == {"git_commit": "7997e4eabb1f880fa5697e2b24de09a892fffad5"}
    assert len(spec.files) == 25
    edit = model_info("qwen-image-edit")
    assert edit.family == "qwen-image" and edit.catalog_id == "qwen-image-edit"
    assert edit.repo_id == "Qwen/Qwen-Image-Edit"
    assert edit.revision == "ac7f9318f633fc4b5778c59367c8128225f1e3de"
    assert edit.image_input == "required" and edit.license == "Apache-2.0"
    assert edit.reference["diffusers"]["pipeline"] == "QwenImageEditPipeline"
    plus = model_info("qwen-image-edit-2509")
    assert plus.repo_id == "Qwen/Qwen-Image-Edit-2509"
    assert plus.revision == "d3968ef930e841f4c73640fb8afa3b306a78167e"
    assert plus.reference["diffusers"]["pipeline"] == "QwenImageEditPlusPipeline"
    pipe = pipeline("t2i", backend="qwen-image-2.1", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained qwen-image-2\.1 "):
        pipe("a cat")


def test_snapshot_metadata(tiny_qwen_image):
    (tiny_qwen_image / SNAPSHOT_FILE).write_text(
        '{"model_id": "qwen-image-2.1", "revision": "d26bb61231c349cf6b7896fa83353113880e1ba3"}'
    )
    result = pipeline("t2i", backend="qwen-image-2.1", model=tiny_qwen_image)(
        "a cat", seed=1, steps=1, width=32, height=32
    )
    assert result.metadata["model_id"] == "qwen-image-2.1"


def test_prompt_template_drops_system_tokens(tiny_qwen_image):
    components = QwenImageComponents.load(
        tiny_qwen_image, device=torch.device("cpu"), dtype=torch.float32
    )
    embeds, mask, image_mask = encode_prompt(components, "a cat", torch.device("cpu"), 2)
    assert embeds.shape[0] == 2
    assert mask is None
    assert not image_mask.any()
    full = components.tokenizer.tokenize(
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    )
    system = components.tokenizer.tokenize(
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
    )
    assert embeds.shape[1] == len(full) - len(system)


def test_text_loader_rejects_non_vision_unexpected_tensor(tiny_qwen_image, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text((tiny_qwen_image / "text_encoder/config.json").read_text())
    state = load_weights(tiny_qwen_image / "text_encoder")
    state["rogue.weight"] = torch.zeros(1)
    save_safetensors(state, broken / "model.safetensors")
    with pytest.raises(CheckpointError, match="rogue.weight"):
        QwenImageComponents.load_text_encoder(
            broken, device=torch.device("cpu"), dtype=torch.float32
        )


def test_bfloat16_checkpoint_converts_to_requested_dtype(tiny_qwen_image, tmp_path):
    converted = tmp_path / "bf16"
    converted.mkdir()
    config = json.loads((tiny_qwen_image / "text_encoder/config.json").read_text())
    (converted / "config.json").write_text(json.dumps(config))
    state = {
        name: value.to(torch.bfloat16)
        for name, value in load_weights(tiny_qwen_image / "text_encoder").items()
    }
    save_safetensors(state, converted / "model.safetensors")
    model = QwenImageComponents.load_text_encoder(
        converted, device=torch.device("cpu"), dtype=torch.float32
    )
    assert next(model.parameters()).dtype == torch.float32
