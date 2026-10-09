"""Tiny-snapshot aMUSEd end-to-end parity against diffusers' AmusedPipeline."""

import pytest
import torch
from helpers import write_tiny_amused

from imagehub import pipeline
from imagehub.backends.amused import AmusedComponents, decode_tokens, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

PROMPT = "A photo of a cat, sitting on the mat in the sun."


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "resampling"])
def tiny(request, tmp_path_factory):
    root = write_tiny_amused(tmp_path_factory.mktemp("amused") / "tiny", downsample=request.param)
    reference = diffusers.AmusedPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = AmusedComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
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
    size = options.get("size", 32)
    kwargs = {
        "num_inference_steps": steps,
        "guidance_scale": options.get("guidance", 10.0),
        "height": size,
        "width": size,
        "micro_conditioning_aesthetic_score": options.get("aesthetic_score", 6),
        "micro_conditioning_crop_coord": options.get("crop_coord", (0, 0)),
        "temperature": options.get("temperature", (2, 0)),
    }
    if "negative_prompt" in options:
        kwargs["negative_prompt"] = options["negative_prompt"]
    expected_tokens = _reference_call(reference, seed, "latent", **kwargs)
    expected_images = _reference_call(reference, seed, "pil", **kwargs)

    actual_tokens = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=options.get("negative_prompt", ""),
        steps=steps,
        guidance=kwargs["guidance_scale"],
        height=size,
        width=size,
        aesthetic_score=kwargs["micro_conditioning_aesthetic_score"],
        crop_coord=kwargs["micro_conditioning_crop_coord"],
        temperature=kwargs["temperature"],
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    assert torch.equal(actual_tokens, expected_tokens)
    assert len(set(actual_tokens.flatten().tolist())) > 1
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))
    assert torch.equal(decode_tokens(components.vqvae, actual_tokens), expected_pixels)
    assert expected_pixels.float().std() > 1.0

    result = pipeline("t2i", backend="amused", model=root)(
        PROMPT,
        seed=seed,
        steps=steps,
        width=size,
        height=size,
        **{key: value for key, value in options.items() if key != "size"},
    )
    pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(pixels, expected_pixels)
    record(name, tokens_equal=True, pixels_equal=True, shape=list(pixels.shape))


def test_defaults(tiny, record):
    _check(tiny, record, "amused_tiny_default")


def test_pipeline_defaults_match_the_reference_signature(tiny, record):
    root, reference, _ = tiny
    expected = _reference_call(reference, 2, "pil")
    result = pipeline("t2i", backend="amused", model=root)(PROMPT, seed=2)
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected]))
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
    assert result.metadata["steps"] == 12 and result.metadata["guidance"] == 10.0
    record("amused_tiny_signature_defaults", pixels_equal=True)


def test_guidance_one_disables_cfg(tiny, record):
    _check(tiny, record, "amused_tiny_guidance_1", guidance=1.0)


def test_negative_prompt(tiny, record):
    _check(tiny, record, "amused_tiny_negative_prompt", negative_prompt="blurry, low quality")


def test_micro_conditioning_and_temperature(tiny, record):
    _check(
        tiny,
        record,
        "amused_tiny_micro_conditioning",
        aesthetic_score=3,
        crop_coord=(2, 5),
        temperature=3,
        guidance=5.0,
    )


def test_larger_grid(tiny, record):
    _check(tiny, record, "amused_tiny_larger_grid", size=64, steps=5)


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, record, seed):
    _check(tiny, record, f"amused_tiny_seed_{seed}", seed=seed, steps=3)


def test_one_step(tiny, record):
    _check(tiny, record, "amused_tiny_one_step", steps=1)
