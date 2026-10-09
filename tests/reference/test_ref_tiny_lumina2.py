"""Tiny-snapshot Lumina-Image 2.0 end-to-end parity against diffusers' Lumina2Pipeline."""

import pytest
import torch
from helpers import write_tiny_lumina2

from imagehub import pipeline
from imagehub.backends.lumina2 import (
    DEFAULT_SYSTEM_PROMPT,
    Lumina2Components,
    decode_latents,
    denoise,
)
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

LATENT_CHANNELS = 4
PROMPT = "A photo of a cat, sitting on the mat in the sun."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_lumina2(tmp_path_factory.mktemp("lumina2") / "tiny")
    reference = diffusers.Lumina2Pipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = Lumina2Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, reference, components


def _reference_call(reference, seed, output_type, **kwargs):
    return reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type=output_type,
        **kwargs,
    ).images


def _check(tiny, record, name, *, seed=7, steps=4, **options):
    root, reference, components = tiny
    width = options.get("width", 128)
    height = options.get("height", 128)
    kwargs = {
        "negative_prompt": options.get("negative_prompt", ""),
        "num_inference_steps": steps,
        "guidance_scale": options.get("guidance", 4.0),
        "height": height,
        "width": width,
        "cfg_trunc_ratio": options.get("cfg_trunc_ratio", 1.0),
        "cfg_normalization": options.get("cfg_normalization", True),
        "max_sequence_length": options.get("max_sequence_length", 256),
    }
    system_prompt = options.get("system_prompt")
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    expected_latents = _reference_call(reference, seed, "latent", **kwargs)
    expected_images = _reference_call(reference, seed, "pil", **kwargs)

    noise = torch.randn(
        1,
        LATENT_CHANNELS,
        height // 8,
        width // 8,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=kwargs["negative_prompt"],
        system_prompt=DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt,
        steps=steps,
        guidance=kwargs["guidance_scale"],
        cfg_trunc_ratio=kwargs["cfg_trunc_ratio"],
        cfg_normalization=kwargs["cfg_normalization"],
        max_sequence_length=kwargs["max_sequence_length"],
        latents=noise,
    )
    assert torch.equal(actual_latents, expected_latents)
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))
    assert torch.equal(decode_latents(components.vae, actual_latents), expected_pixels)
    assert expected_pixels.float().std() > 1.0

    result = pipeline("t2i", backend="lumina2", model=root)(
        PROMPT, seed=seed, steps=steps, **({"width": width, "height": height} | options)
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
    _check(tiny, record, "lumina2_tiny_default")


def test_cfg_truncation(tiny, record):
    _check(tiny, record, "lumina2_tiny_cfg_trunc", steps=5, cfg_trunc_ratio=0.5)
    _check(tiny, record, "lumina2_tiny_cfg_trunc_all", steps=3, cfg_trunc_ratio=0.0)


def test_cfg_normalization_off(tiny, record):
    _check(tiny, record, "lumina2_tiny_no_normalization", cfg_normalization=False)


def test_guidance_one_disables_cfg(tiny, record):
    _check(tiny, record, "lumina2_tiny_guidance_1", guidance=1.0)


def test_non_square(tiny, record):
    _check(tiny, record, "lumina2_tiny_non_square", width=192, height=96)
    _check(tiny, record, "lumina2_tiny_portrait", width=80, height=160, seed=3)


def test_custom_system_prompt_negative_prompt_and_sequence_length(tiny, record):
    _check(
        tiny,
        record,
        "lumina2_tiny_system_prompt",
        system_prompt="Describe the image.",
        negative_prompt="Blurry, low quality",
        max_sequence_length=24,
    )


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, record, seed):
    _check(tiny, record, f"lumina2_tiny_seed_{seed}", seed=seed, steps=3)


def test_one_step(tiny, record):
    _check(tiny, record, "lumina2_tiny_one_step", steps=1)


def test_tiny_tokenizer_matches_reference(tiny):
    root, reference, _ = tiny
    ours = GemmaTokenizer(root / "tokenizer")
    for text in (DEFAULT_SYSTEM_PROMPT + " <Prompt Start> " + PROMPT, ""):
        assert ours.tokenize(text) == reference.tokenizer.encode(text)
