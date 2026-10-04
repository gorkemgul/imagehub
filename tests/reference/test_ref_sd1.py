"""Parity of the native SD 1.x components and pipeline against diffusers on CPU fp32.

Requires IMAGEHUB_REFERENCE_TESTS=1 and IMAGEHUB_SD_V1_<n> pointing at snapshots made by
``python -m imagehub.pretrained sd-v1-<n> <dir>``.
"""

import pytest
import torch
from helpers import checkpoint_dir, max_diff

from imagehub import pipeline
from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

# Maximum absolute difference allowed against diffusers 0.40.0 / transformers 5.18.0, CPU fp32.
# Rule (spec §6.2): 0 if the measured difference is 0, otherwise the next power of ten above it.
# Random-weight runs at full SD 1.x size measured exactly 0 for every entry on 2026-10-04.
TOLERANCES = {
    "text_encoder": 0.0,
    "unet": 0.0,
    "vae_decode": 0.0,
    "vae_encode": 0.0,
    "e2e_latents": 0.0,
    "e2e_pixels": 0,
}
PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")


def load_pair(version):
    root = checkpoint_dir(version)
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
    ref = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    ref.set_progress_bar_config(disable=True)
    return root, ours, ref


@pytest.fixture(scope="module")
def sd14():
    return load_pair(4)


def check(record, name, actual, expected, tolerance_key):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=TOLERANCES[tolerance_key])
    torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


def test_text_encoder(sd14, record):
    _, ours, ref = sd14
    prompts = [PROMPT, "", "Bir kedi, yağlı boya", "emoji 🐱🚀 " * 30]
    ids = torch.tensor([ours.tokenizer(p) for p in prompts])
    with torch.inference_mode():
        check(
            record, "text_encoder", ours.text_encoder(ids), ref.text_encoder(ids)[0], "text_encoder"
        )


def test_unet_single_step(sd14, record):
    _, ours, ref = sd14
    g = torch.Generator().manual_seed(0)
    sample = torch.randn(2, 4, 64, 64, generator=g)
    ids = torch.tensor([ours.tokenizer(""), ours.tokenizer(PROMPT)])
    with torch.inference_mode():
        context = ref.text_encoder(ids)[0]
        worst = None
        pairs = []
        for t in (981, 501, 1):
            timestep = torch.tensor(t)
            actual = ours.unet(sample, timestep, context)
            expected = ref.unet(sample, timestep, context).sample
            measured = max_diff(actual, expected)
            if worst is None or measured["max_abs"] > worst["max_abs"]:
                worst = measured
            pairs.append((actual, expected))
    record("unet", **worst, tolerance=TOLERANCES["unet"], timesteps=[981, 501, 1])
    for actual, expected in pairs:
        torch.testing.assert_close(actual, expected, atol=TOLERANCES["unet"], rtol=0)


def test_vae(sd14, record):
    _, ours, ref = sd14
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, 64, 64, generator=g)
    image = torch.rand(1, 3, 512, 512, generator=g) * 2 - 1
    with torch.inference_mode():
        decode = (ours.vae.decode(latents), ref.vae.decode(latents).sample)
        encode = (ours.vae.encode(image), ref.vae.encode(image).latent_dist.mean)
    for name, (actual, expected) in (("vae_decode", decode), ("vae_encode", encode)):
        record(name, **max_diff(actual, expected), tolerance=TOLERANCES[name])
    for name, (actual, expected) in (("vae_decode", decode), ("vae_encode", encode)):
        torch.testing.assert_close(actual, expected, atol=TOLERANCES[name], rtol=0)


def test_seed_noise(record):
    from diffusers.utils.torch_utils import randn_tensor

    shape = (2, 4, 64, 64)
    expected = randn_tensor(
        shape, generator=torch.Generator("cpu").manual_seed(42), dtype=torch.float32
    )
    assert torch.equal(initial_latents(42, shape), expected)
    record("seed_noise", exact=True)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5])
def test_end_to_end(version, record):
    _, ours, ref = load_pair(version)
    latents = initial_latents(1234, (1, 4, 64, 64))
    with torch.inference_mode():
        ref_latents = ref(
            PROMPT,
            num_inference_steps=50,
            guidance_scale=7.5,
            latents=latents.clone(),
            output_type="latent",
        ).images
        decoded = ref.vae.decode(ref_latents / ref.vae.config.scaling_factor).sample
        ref_np = ref.image_processor.postprocess(decoded, output_type="np")
    ref_pixels = torch.from_numpy((ref_np * 255).round().astype("uint8"))
    our_latents = denoise(
        ours, prompt=PROMPT, negative_prompt=None, steps=50, guidance=7.5, latents=latents.clone()
    )
    our_pixels = decode_latents(ours, our_latents)
    latent_diff = max_diff(our_latents, ref_latents)
    pixel_diff = int((our_pixels.int() - ref_pixels.int()).abs().max())
    passed = (
        latent_diff["max_abs"] <= TOLERANCES["e2e_latents"]
        and pixel_diff <= TOLERANCES["e2e_pixels"]
    )
    record(
        f"e2e_sd-v1-{version}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=50,
        guidance=7.5,
        size=[512, 512],
        noise_seed=1234,
        tolerance={"latents": TOLERANCES["e2e_latents"], "pixels": TOLERANCES["e2e_pixels"]},
    )
    assert passed, (latent_diff, pixel_diff)


def test_seed_end_to_end_v15(record):
    root, _, ref = load_pair(5)
    with torch.inference_mode():
        expected = ref(
            PROMPT,
            num_inference_steps=50,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend="sd-v1-5", model=root)(PROMPT, seed=42).images[0].tensor
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        "seed_e2e_sd-v1-5",
        pixels_max_abs=pixel_diff,
        seed=42,
        passed=pixel_diff <= TOLERANCES["e2e_pixels"],
    )
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
