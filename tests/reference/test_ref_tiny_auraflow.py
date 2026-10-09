"""Tiny-snapshot AuraFlow end-to-end parity against diffusers' AuraFlowPipeline."""

import pytest
import torch
from helpers import write_tiny_auraflow

from imagehub import pipeline
from imagehub.backends.auraflow import AuraFlowComponents, decode_latents, denoise
from imagehub.tokenization_auraflow import AuraFlowTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

VAE_SCALE = 8
PROMPT = "A photo of a cat, sitting on the mat in the sun."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_auraflow(tmp_path_factory.mktemp("auraflow") / "tiny")
    reference = diffusers.AuraFlowPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = AuraFlowComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, reference, components


def _reference_call(reference, seed, output_type, **kwargs):
    return reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type=output_type,
        **kwargs,
    ).images


def _check(tiny, record, name, *, seed=7, **options):
    root, reference, components = tiny
    width = options.get("width", 64)
    height = options.get("height", 64)
    kwargs = {
        "negative_prompt": options.get("negative_prompt"),
        "height": height,
        "width": width,
        "max_sequence_length": options.get("max_sequence_length", 24),
    }
    if "steps" in options:
        kwargs["num_inference_steps"] = options["steps"]
    if "guidance" in options:
        kwargs["guidance_scale"] = options["guidance"]
    expected_latents = _reference_call(reference, seed, "latent", **kwargs)
    expected_images = _reference_call(reference, seed, "pil", **kwargs)

    noise = torch.randn(
        1,
        4,
        height // VAE_SCALE,
        width // VAE_SCALE,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=kwargs["negative_prompt"],
        steps=options.get("steps", 50),
        guidance=options.get("guidance", 3.5),
        max_sequence_length=kwargs["max_sequence_length"],
        latents=noise,
    )
    assert torch.equal(actual_latents, expected_latents)
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))
    actual_pixels = decode_latents(components.vae, actual_latents)
    assert torch.equal(actual_pixels, expected_pixels)
    assert expected_pixels.float().std() > 1.0

    result = pipeline("t2i", backend="auraflow", model=root)(
        PROMPT,
        seed=seed,
        **(
            {"width": width, "height": height, "max_sequence_length": kwargs["max_sequence_length"]}
            | options
        ),
    )
    pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(pixels, expected_pixels)
    record(
        name,
        latent_max_abs=float((actual_latents - expected_latents).abs().max()),
        pixels_equal=True,
        shape=list(pixels.shape),
    )


def test_defaults(tiny, record):
    _check(tiny, record, "auraflow_tiny_default")


def test_guidance_one_disables_cfg(tiny, record):
    _check(tiny, record, "auraflow_tiny_guidance_1", steps=4, guidance=1.0)


def test_non_square(tiny, record):
    _check(tiny, record, "auraflow_tiny_non_square", steps=4, width=96, height=48)
    _check(tiny, record, "auraflow_tiny_portrait", steps=4, width=48, height=80, seed=3)


def test_negative_prompt_and_guidance(tiny, record):
    _check(
        tiny,
        record,
        "auraflow_tiny_negative_prompt",
        steps=3,
        guidance=5.0,
        negative_prompt="blurry, low quality",
    )


def test_truncated_prompt(tiny, record):
    _check(tiny, record, "auraflow_tiny_truncated", steps=3, max_sequence_length=6)


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, record, seed):
    _check(tiny, record, f"auraflow_tiny_seed_{seed}", seed=seed, steps=3)


def test_one_step(tiny, record):
    _check(tiny, record, "auraflow_tiny_one_step", steps=1)


def test_tiny_tokenizer_matches_reference(tiny):
    root, reference, _ = tiny
    ours = AuraFlowTokenizer(root / "tokenizer")
    for text in (PROMPT, "", "the cat in the hat", "  spaced   out "):
        assert ours.tokenize(text) == reference.tokenizer.encode(text)
