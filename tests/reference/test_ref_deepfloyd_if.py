"""Published DeepFloyd IF snapshot checks (opt in with IMAGEHUB_REFERENCE_TESTS=1)."""

import gc
import json

import pytest
import torch
from helpers import FIXTURES, max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.models.t5 import T5EncoderModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")

CPU = torch.device("cpu")
MODEL_ID = "deepfloyd-if-i-xl"
PROMPT = "a quiet landscape"
STEPS = 4


def _check(record, name, actual, expected) -> None:
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
def test_published_unet_header(stage: str, record) -> None:
    root = snapshot_dir(MODEL_ID)
    model_root = root if stage == "stage_i" else root / "stage_ii"
    path = FIXTURES / "deepfloyd_if_configs" / stage
    with torch.device("meta"):
        model = UNet2DConditionModel(json.loads((path / "unet_config.json").read_text()))
    expected = json.loads((path / "unet.safetensors.header.json").read_text())
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert (model_root / "unet").is_dir()
    record(
        f"header_deepfloyd_if_{stage}",
        missing=len(expected.keys() - actual.keys()),
        extra=len(actual.keys() - expected.keys()),
        mismatch=sum(expected[k] != actual[k] for k in expected.keys() & actual.keys()),
    )
    assert actual == expected


def test_published_tokenizer_and_shared_t5_output(record) -> None:
    """Stage II deliberately reuses stage I's encoder, as the published configs do."""
    root = snapshot_dir(MODEL_ID)
    tokenizer = T5Tokenizer(root / "tokenizer")
    ids, mask = tokenizer([PROMPT], max_length=77)
    reference_tokenizer = transformers.T5Tokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_tokens = reference_tokenizer(
        [PROMPT], padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    tokens_exact = torch.equal(ids, expected_tokens.input_ids) and torch.equal(
        mask, expected_tokens.attention_mask.bool()
    )
    record("deepfloyd_if_tokenizer", exact=tokens_exact, prompts=1, tolerance="exact")
    assert tokens_exact

    encoder = T5EncoderModel.from_pretrained(root / "text_encoder", device=CPU)
    reference_encoder = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        actual = encoder(ids, mask)[0]
        expected = reference_encoder(ids, attention_mask=mask).last_hidden_state
    _check(record, "deepfloyd_if_t5", actual, expected)


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
def test_published_unet_one_step(stage: str, record) -> None:
    root = snapshot_dir(MODEL_ID)
    model_root = root if stage == "stage_i" else root / "stage_ii"
    model = UNet2DConditionModel.from_pretrained(model_root / "unet", device=CPU)
    reference_model = diffusers.UNet2DConditionModel.from_pretrained(
        model_root / "unet", local_files_only=True, torch_dtype=torch.float32
    )
    config = json.loads((model_root / "unet" / "config.json").read_text())
    encoder = T5EncoderModel.from_pretrained(root / "text_encoder", device=CPU)
    tokenizer = T5Tokenizer(root / "tokenizer")
    ids, mask = tokenizer([PROMPT], max_length=77)
    context = encoder(ids, mask)[0]
    size = 64 if stage == "stage_i" else 256
    sample = torch.randn(
        1, config["in_channels"], size, size, generator=torch.Generator().manual_seed(0)
    )
    labels = torch.tensor([250]) if stage == "stage_ii" else None
    with torch.inference_mode():
        actual = model(sample, torch.tensor(999), context, class_labels=labels)
        expected = reference_model(sample, 999, context, class_labels=labels).sample
    assert actual.shape == (1, config["out_channels"], size, size)
    _check(record, f"deepfloyd_if_{stage}_unet", actual, expected)


def _reference_pixels(root, *, seed: int):
    first = diffusers.IFPipeline.from_pretrained(
        root,
        local_files_only=True,
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
        watermarker=None,
    )
    second = diffusers.IFSuperResolutionPipeline.from_pretrained(
        root / "stage_ii",
        local_files_only=True,
        torch_dtype=torch.float32,
        tokenizer=None,
        text_encoder=None,
        safety_checker=None,
        feature_extractor=None,
        watermarker=None,
    )
    first.set_progress_bar_config(disable=True)
    second.set_progress_bar_config(disable=True)
    prompt_embeds, negative_prompt_embeds = first.encode_prompt(
        PROMPT, True, device=CPU, clean_caption=False
    )
    generator = torch.Generator("cpu").manual_seed(seed)
    stage_i = first(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=STEPS,
        guidance_scale=7.0,
        generator=generator,
        output_type="pt",
    ).images
    result = second(
        image=stage_i,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=STEPS,
        guidance_scale=4.0,
        noise_level=250,
        generator=generator,
        output_type="pt",
    ).images
    pixels = ((result / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
    del first, second, prompt_embeds, negative_prompt_embeds, stage_i, result
    return pixels


def test_end_to_end(record) -> None:
    root = snapshot_dir(MODEL_ID)
    expected = _reference_pixels(root, seed=0)
    gc.collect()

    actual = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, seed=0, steps=STEPS, superresolution_steps=STEPS
    )
    pixels = torch.stack([image.tensor for image in actual.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record("e2e_deepfloyd-if-i-xl", passed=passed, max_pixel_diff=pixel_diff)
    assert passed


def test_seed_end_to_end(record) -> None:
    root = snapshot_dir(MODEL_ID)
    expected = _reference_pixels(root, seed=42)
    gc.collect()

    actual = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, seed=42, steps=STEPS, superresolution_steps=STEPS
    )
    pixels = torch.stack([image.tensor for image in actual.images])
    passed = torch.equal(pixels, expected)
    record("seed_e2e_deepfloyd-if-i-xl", passed=passed, seed=42)
    assert passed
