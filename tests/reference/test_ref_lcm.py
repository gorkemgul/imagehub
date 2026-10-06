"""Parity for LCM Dreamshaper v7 on CPU float32."""

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
from imagehub.schedulers.lcm import LCMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")


def load_pair():
    root = snapshot_dir("lcm-dreamshaper-v7")
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
    ref = diffusers.LatentConsistencyModelPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    ref.set_progress_bar_config(disable=True)
    return root, ours, ref


@pytest.fixture(scope="module")
def lcm_pair():
    return load_pair()


def exact(record, name, actual, expected, **details):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0, **details)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lcm_published_tensor_headers_match_models(record):
    root = snapshot_dir("lcm-dreamshaper-v7")
    classes = {
        "text_encoder": CLIPTextModel,
        "unet": UNet2DConditionModel,
        "vae": AutoencoderKL,
    }
    result = {}
    for component, cls in classes.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        path = next((root / component).glob("*.safetensors"))
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes = {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(actual),
        }
    record("header_lcm-dreamshaper-v7", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_lcm_text_encoder(lcm_pair, record):
    _, ours, ref = lcm_pair
    ids = torch.tensor([ours.tokenizer(PROMPT), ours.tokenizer("")])
    with torch.inference_mode():
        exact(record, "lcm_text_encoder", ours.text_encoder(ids), ref.text_encoder(ids)[0])


def test_lcm_unet_step(lcm_pair, record):
    _, ours, ref = lcm_pair
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(2, 4, 96, 96, generator=generator)
    ids = torch.tensor([ours.tokenizer(PROMPT)] * 2)
    context = ref.text_encoder(ids)[0]
    guidance = torch.tensor([7.0, 7.0])
    timestep_cond = ref.get_guidance_scale_embedding(
        guidance, embedding_dim=ref.unet.config.time_cond_proj_dim
    )
    with torch.inference_mode():
        actual = ours.unet(sample, torch.tensor(999), context, timestep_cond=timestep_cond)
        expected = ref.unet(
            sample,
            torch.tensor(999),
            context,
            timestep_cond=timestep_cond,
        ).sample
    exact(record, "lcm_unet", actual, expected, timestep=999)


def test_lcm_vae(lcm_pair, record):
    _, ours, ref = lcm_pair
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, 96, 96, generator=generator)
    image = torch.rand(1, 3, 768, 768, generator=generator) * 2 - 1
    with torch.inference_mode():
        actual_decode = ours.vae.decode(latents)
        expected_decode = ref.vae.decode(latents).sample
        actual_encode = ours.vae.encode(image)
        expected_encode = ref.vae.encode(image).latent_dist.mean
    exact(record, "lcm_vae_decode", actual_decode, expected_decode)
    exact(record, "lcm_vae_encode", actual_encode, expected_encode)


def test_lcm_scheduler(lcm_pair, record):
    _, ours, ref = lcm_pair
    actual_scheduler = LCMScheduler(ours.scheduler_config)
    expected_scheduler = diffusers.LCMScheduler.from_config(ref.scheduler.config)
    actual_scheduler.set_timesteps(4)
    expected_scheduler.set_timesteps(4)
    assert torch.equal(actual_scheduler.timesteps, expected_scheduler.timesteps)
    sample = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    expected = sample.clone()
    actual_generator = torch.Generator().manual_seed(42)
    expected_generator = torch.Generator().manual_seed(42)
    worst = 0.0
    for timestep in actual_scheduler.timesteps:
        output = torch.randn_like(sample)
        expected, expected_denoised = expected_scheduler.step(
            output, timestep, expected, generator=expected_generator, return_dict=False
        )
        sample, actual_denoised = actual_scheduler.step(
            output, timestep, sample, generator=actual_generator
        )
        worst = max(worst, float((sample - expected).abs().max()))
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_denoised, expected_denoised, rtol=0, atol=0)
    record("lcm_scheduler", max_abs=worst, tolerance=0.0, steps=4)


def test_lcm_end_to_end(lcm_pair, record):
    root, ours, ref = lcm_pair
    latents = initial_latents(1234, (1, 4, 96, 96))
    ref_generator = torch.Generator().manual_seed(9)
    with torch.inference_mode():
        expected_latents = ref(
            PROMPT,
            num_inference_steps=4,
            guidance_scale=8.0,
            latents=latents.clone(),
            generator=ref_generator,
            output_type="latent",
        ).images
        decoded = ref.vae.decode(expected_latents / ref.vae.config.scaling_factor).sample
        expected_np = ref.image_processor.postprocess(decoded, output_type="np")
    expected_pixels = torch.from_numpy((expected_np * 255).round().astype("uint8"))
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=None,
        steps=4,
        guidance=8.0,
        latents=latents.clone(),
        generator=torch.Generator().manual_seed(9),
    )
    actual_pixels = decode_latents(ours, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    record(
        "e2e_lcm-dreamshaper-v7",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=latent_diff["max_abs"] == 0 and pixel_diff == 0,
        prompt=PROMPT,
        steps=4,
        guidance=8.0,
        size=[768, 768],
        tolerance={"latents": 0.0, "pixels": 0},
    )
    assert latent_diff["max_abs"] == 0 and pixel_diff == 0
    assert root == snapshot_dir("lcm-dreamshaper-v7")


def test_lcm_seed_42(lcm_pair, record):
    root, _, ref = lcm_pair
    with torch.inference_mode():
        expected = ref(
            PROMPT,
            num_inference_steps=4,
            guidance_scale=8.0,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = (
        pipeline("t2i", backend="lcm-dreamshaper-v7", model=root)(PROMPT, seed=42).images[0].tensor
    )
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        "seed_e2e_lcm-dreamshaper-v7", pixels_max_abs=pixel_diff, seed=42, passed=pixel_diff == 0
    )
    assert pixel_diff == 0
