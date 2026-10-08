"""Parity of SD 2.1 components and pipeline against diffusers on CPU fp32."""

import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.ddim import DDIMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

TOLERANCES = {
    "text_encoder": 0.0,
    "unet": 0.0,
    "vae_decode": 0.0,
    "vae_encode": 0.0,
    "e2e_latents": 0.0,
    "e2e_pixels": 0,
}
PROMPT = "a photograph of an astronaut riding a horse"
PROMPTS = [PROMPT, "", "Bir kedi, yağlı boya", "emoji 🐱🚀 " * 30, "!"]
CPU = torch.device("cpu")
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
WEIGHT_FILES = ("model.safetensors", "diffusion_pytorch_model.safetensors")


@pytest.fixture(scope="module")
def sd21():
    root = snapshot_dir("sd-v2-1")
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
    ref = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    ref.set_progress_bar_config(disable=True)
    return root, ours, ref


def _check(record, name, actual, expected, tolerance_key):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=TOLERANCES[tolerance_key])
    torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


def _checkpoint_shapes(directory):
    path = next(path for name in WEIGHT_FILES if (path := directory / name).is_file())
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


def test_header(record):
    root = snapshot_dir("sd-v2-1")
    result = {}
    for component, model_cls in COMPONENTS.items():
        with torch.device("meta"):
            model = model_cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        actual = {name: tuple(tensor.shape) for name, tensor in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(actual),
        }
    record("header_sd-v2-1", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(sd21, record):
    _, ours, ref = sd21
    mismatches = []
    for prompt in PROMPTS:
        expected = ref.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)[
            "input_ids"
        ]
        if ours.tokenizer(prompt) != expected:
            mismatches.append(prompt)
    record("sd21_tokenizer", prompts=len(PROMPTS), mismatches=mismatches, tolerance="exact")
    assert not mismatches


def test_text_encoder(sd21, record):
    _, ours, ref = sd21
    ids = torch.tensor([ours.tokenizer(prompt) for prompt in PROMPTS])
    with torch.inference_mode():
        actual = ours.text_encoder(ids)
        expected = ref.text_encoder(ids)[0]
    _check(record, "sd21_text_encoder", actual, expected, "text_encoder")


def test_unet_one_step_and_one_more(sd21, record):
    _, ours, ref = sd21
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(2, 4, 96, 96, generator=generator)
    context = torch.randn(2, 77, 1024, generator=generator)
    scheduler = type(ref.scheduler).from_config(ref.scheduler.config)
    scheduler.set_timesteps(50)
    timesteps = [scheduler.timesteps[0], scheduler.timesteps[25], scheduler.timesteps[-1]]
    pairs = []
    with torch.inference_mode():
        for timestep in timesteps:
            actual = ours.unet(sample, timestep, context)
            expected = ref.unet(sample, timestep, context).sample
            pairs.append((actual, expected, max_diff(actual, expected)))
    worst = max((measured for _, _, measured in pairs), key=lambda value: value["max_abs"])
    record(
        "sd21_unet",
        **worst,
        tolerance=TOLERANCES["unet"],
        timesteps=[float(timestep) for timestep in timesteps],
    )
    for actual, expected, _ in pairs:
        torch.testing.assert_close(actual, expected, atol=TOLERANCES["unet"], rtol=0)


def test_vae(sd21, record):
    _, ours, ref = sd21
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, 96, 96, generator=generator)
    image = torch.rand(1, 3, 768, 768, generator=generator) * 2 - 1
    with torch.inference_mode():
        pairs = {
            "vae_decode": (ours.vae.decode(latents), ref.vae.decode(latents).sample),
            "vae_encode": (ours.vae.encode(image), ref.vae.encode(image).latent_dist.mean),
        }
    for name, (actual, expected) in pairs.items():
        _check(record, f"sd21_{name}", actual, expected, name)


def test_scheduler(sd21, record):
    _, ours, ref = sd21
    measurements = {}
    schedules = []
    for steps in (25, 50):
        actual = DDIMScheduler(ours.scheduler_config)
        expected = type(ref.scheduler).from_config(ref.scheduler.config)
        actual.set_timesteps(steps)
        expected.set_timesteps(steps)
        measurements[str(steps)] = {"timesteps": max_diff(actual.timesteps, expected.timesteps)}
        schedules.append((actual, expected))
    record("sd21_scheduler", steps=[25, 50], measurements=measurements, tolerance=0.0)
    for actual, expected in schedules:
        torch.testing.assert_close(actual.timesteps, expected.timesteps, atol=0, rtol=0)
    torch.testing.assert_close(
        schedules[0][0].alphas_cumprod, schedules[0][1].alphas_cumprod, atol=0, rtol=0
    )


def _run_end_to_end(sd21, record, *, steps, guidance, negative_prompt, name):
    _, ours, ref = sd21
    latents = initial_latents(0, (1, 4, 96, 96))
    with torch.inference_mode():
        ref_latents = ref(
            PROMPT,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=768,
            width=768,
            latents=latents.clone(),
            output_type="latent",
        ).images
        decoded = ref.vae.decode(ref_latents / ref.vae.config.scaling_factor).sample
        ref_np = ref.image_processor.postprocess(decoded, output_type="np")
    expected_pixels = torch.from_numpy((ref_np * 255).round().astype("uint8"))
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        latents=latents.clone(),
    )
    actual_pixels = decode_latents(ours, actual_latents)
    latent_diff = max_diff(actual_latents, ref_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0.0 and pixel_diff == 0
    record(
        name,
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=guidance,
        size=[768, 768],
        noise_seed=0,
        tolerance={"latents": TOLERANCES["e2e_latents"], "pixels": TOLERANCES["e2e_pixels"]},
    )
    assert passed, (latent_diff, pixel_diff)


def test_end_to_end_defaults(sd21, record):
    _run_end_to_end(
        sd21,
        record,
        steps=50,
        guidance=7.5,
        negative_prompt=None,
        name="e2e_sd-v2-1",
    )


def test_seed_end_to_end(sd21, record):
    root, _, ref = sd21
    with torch.inference_mode():
        expected = ref(
            PROMPT,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend="sd-v2-1", model=root)(PROMPT, seed=42).images[0].tensor
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record("seed_e2e_sd-v2-1", pixels_max_abs=pixel_diff, seed=42, passed=pixel_diff == 0)
    assert pixel_diff == 0
