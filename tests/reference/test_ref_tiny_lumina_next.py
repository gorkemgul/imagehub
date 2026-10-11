"""Tiny-snapshot Lumina-Next end-to-end parity against diffusers' LuminaPipeline."""

import pytest
import torch
from helpers import write_tiny_lumina_next

from imagehub import pipeline
from imagehub.backends.lumina_next import LuminaNextComponents, decode_latents, denoise
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

LATENT_CHANNELS = 4
PROMPT = "A photo of a Cat, sitting on the mat in the sun."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_lumina_next(tmp_path_factory.mktemp("lumina-next") / "tiny")
    reference = diffusers.LuminaPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = LuminaNextComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, reference, components


def _reference_call(reference, seed, output_type, **kwargs):
    return reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type=output_type,
        clean_caption=False,
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
        "scaling_watershed": options.get("scaling_watershed", 1.0),
        "proportional_attn": options.get("proportional_attn", True),
    }
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
        steps=steps,
        guidance=kwargs["guidance_scale"],
        scaling_watershed=kwargs["scaling_watershed"],
        proportional_attn=kwargs["proportional_attn"],
        latents=noise,
    )
    assert torch.equal(actual_latents, expected_latents)
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))
    assert torch.equal(decode_latents(components.vae, actual_latents), expected_pixels)
    assert expected_pixels.float().std() > 1.0

    result = pipeline("t2i", backend="lumina-next", model=root)(
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
    _check(tiny, record, "lumina_next_tiny_default")


def test_guidance_one_disables_cfg(tiny, record):
    _check(tiny, record, "lumina_next_tiny_guidance_1", guidance=1.0)


def test_negative_prompt(tiny, record):
    _check(tiny, record, "lumina_next_tiny_negative", negative_prompt="Blurry, low quality")
    _check(
        tiny,
        record,
        "lumina_next_tiny_long_negative",
        negative_prompt="blurry " * 30,
        steps=2,
    )


def test_non_square_scales_the_rotary_positions(tiny, record):
    _check(tiny, record, "lumina_next_tiny_non_square", width=192, height=96)
    _check(tiny, record, "lumina_next_tiny_portrait", width=80, height=160, seed=3)


@pytest.mark.parametrize("watershed", [0.0, 0.35, 0.7, 1.0, 2.0])
def test_scaling_watershed(tiny, record, watershed):
    _check(
        tiny,
        record,
        f"lumina_next_tiny_watershed_{watershed}",
        width=192,
        height=96,
        steps=5,
        scaling_watershed=watershed,
    )


def test_proportional_attention_off(tiny, record):
    _check(
        tiny,
        record,
        "lumina_next_tiny_no_proportional",
        width=192,
        height=96,
        proportional_attn=False,
    )


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, record, seed):
    _check(tiny, record, f"lumina_next_tiny_seed_{seed}", seed=seed, steps=3)


def test_one_step(tiny, record):
    _check(tiny, record, "lumina_next_tiny_one_step", steps=1)


def test_scheduler_timesteps_match_reference(tiny):
    from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler

    _, reference, components = tiny
    for steps in (1, 2, 7, 30):
        reference.scheduler.set_timesteps(steps)
        ours = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
        ours.set_timesteps(steps)
        assert torch.equal(ours.timesteps, reference.scheduler.timesteps)
        assert torch.equal(ours.sigmas, reference.scheduler.sigmas)


def test_tiny_tokenizer_matches_reference(tiny):
    root, reference, _ = tiny
    ours = GemmaTokenizer(root / "tokenizer")
    assert ours.add_eos_token
    for text in (PROMPT.lower(), "", "a b c d e f g h i j k l"):
        assert ours.tokenize(text) == reference.tokenizer.encode(text)
        assert ours.tokenize(text, 6) == reference.tokenizer.encode(
            text, truncation=True, max_length=6
        )
