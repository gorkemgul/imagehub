"""Tiny-snapshot Sana end-to-end parity against diffusers' SanaPipeline."""

import pytest
import torch
from helpers import write_tiny_sana

from imagehub import pipeline
from imagehub.backends.sana import (
    ASPECT_RATIO_BINS,
    COMPLEX_HUMAN_INSTRUCTION,
    SanaComponents,
    classify_height_width_bin,
    decode_latents,
    denoise,
)
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pixart = pytest.importorskip("diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha")
np = pytest.importorskip("numpy")

VAE_SCALE = 32
PROMPT = "A photo of a cat, sitting on the mat in the sun."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_sana(tmp_path_factory.mktemp("sana") / "tiny")
    reference = diffusers.SanaPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = SanaComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
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
    width = options.get("width", 1024)
    height = options.get("height", 1024)
    binning = options.get("use_resolution_binning", True)
    kwargs = {
        "negative_prompt": options.get("negative_prompt", ""),
        "num_inference_steps": steps,
        "guidance_scale": options.get("guidance", 4.5),
        "height": height,
        "width": width,
        "use_resolution_binning": binning,
        "max_sequence_length": options.get("max_sequence_length", 300),
    }
    instruction = options.get("complex_human_instruction")
    if instruction is not None:
        kwargs["complex_human_instruction"] = list(instruction)
    expected_latents = _reference_call(reference, seed, "latent", **kwargs)
    expected_images = _reference_call(reference, seed, "pil", **kwargs)

    run_height, run_width = (
        classify_height_width_bin(height, width, 16) if binning else (height, width)
    )
    noise = torch.randn(
        1,
        8,
        run_height // VAE_SCALE,
        run_width // VAE_SCALE,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=kwargs["negative_prompt"],
        steps=steps,
        guidance=kwargs["guidance_scale"],
        max_sequence_length=kwargs["max_sequence_length"],
        instruction=COMPLEX_HUMAN_INSTRUCTION if instruction is None else tuple(instruction),
        latents=noise,
    )
    assert torch.equal(actual_latents, expected_latents)
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))
    actual_pixels = decode_latents(
        components.vae, actual_latents, output_width=width, output_height=height, resize=binning
    )
    assert torch.equal(actual_pixels, expected_pixels)
    assert expected_pixels.float().std() > 1.0

    result = pipeline("t2i", backend="sana", model=root)(PROMPT, seed=seed, steps=steps, **options)
    pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(pixels, expected_pixels)
    record(
        name,
        latent_max_abs=float((actual_latents - expected_latents).abs().max()),
        pixels_equal=True,
        shape=list(pixels.shape),
    )


def test_default_flow_binning_and_guidance(tiny, record):
    _check(tiny, record, "sana_tiny_default", width=512, height=512)


def test_default_request_size_is_resized_back(tiny, record):
    _check(tiny, record, "sana_tiny_1024_request")


def test_resolution_binning_off(tiny, record):
    _check(
        tiny, record, "sana_tiny_no_binning", width=320, height=256, use_resolution_binning=False
    )


def test_guidance_one_disables_cfg(tiny, record):
    _check(tiny, record, "sana_tiny_guidance_1", width=512, height=512, guidance=1.0)


def test_non_square_request_is_binned_and_cropped(tiny, record):
    _check(tiny, record, "sana_tiny_non_square", width=600, height=400)
    _check(tiny, record, "sana_tiny_portrait", width=300, height=500, seed=3)


def test_empty_complex_human_instruction(tiny, record):
    _check(
        tiny,
        record,
        "sana_tiny_no_instruction",
        width=512,
        height=512,
        complex_human_instruction=[],
    )


def test_custom_instruction_negative_prompt_and_sequence_length(tiny, record):
    _check(
        tiny,
        record,
        "sana_tiny_custom_instruction",
        width=480,
        height=544,
        complex_human_instruction=["Describe the image.", "Prompt: "],
        negative_prompt="Blurry, low quality",
        max_sequence_length=24,
    )


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, record, seed):
    _check(tiny, record, f"sana_tiny_seed_{seed}", width=512, height=512, seed=seed, steps=3)


def test_one_step(tiny, record):
    _check(tiny, record, "sana_tiny_one_step", width=512, height=512, steps=1)


def test_sample_size_32_uses_the_1024_bins(tmp_path, record):
    root = write_tiny_sana(tmp_path / "tiny32", sample_size=32)
    reference = diffusers.SanaPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "num_inference_steps": 2,
        "guidance_scale": 4.5,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 32,
    }
    expected = _reference_call(reference, 5, "pil", **kwargs)[0]
    result = pipeline("t2i", backend="sana", model=root)(
        PROMPT, seed=5, steps=2, max_sequence_length=32
    )
    assert torch.equal(result.images[0].tensor, torch.from_numpy(np.array(expected)))
    record("sana_tiny_sample_size_32", shape=list(result.images[0].tensor.shape))


def test_bin_tables_and_classification_match_reference():
    processor = diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha
    assert ASPECT_RATIO_BINS[32] == processor.ASPECT_RATIO_1024_BIN
    assert ASPECT_RATIO_BINS[16] == processor.ASPECT_RATIO_512_BIN
    image_processor = diffusers.image_processor.PixArtImageProcessor(vae_scale_factor=32)
    for sample_size, table in ASPECT_RATIO_BINS.items():
        for height in range(64, 2400, 97):
            for width in range(64, 2400, 131):
                assert classify_height_width_bin(
                    height, width, sample_size
                ) == image_processor.classify_height_width_bin(height, width, ratios=table)


def test_tiny_tokenizer_matches_reference(tiny):
    root, reference, _ = tiny
    ours = GemmaTokenizer(root / "tokenizer")
    for text in (PROMPT, "", "\n".join(COMPLEX_HUMAN_INSTRUCTION) + PROMPT.lower()):
        assert ours.tokenize(text) == reference.tokenizer.encode(text)
