import json
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline
from imagehub.backends.stable_diffusion import decode_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "case",
    [
        dict(prompt="a cat", negative_prompt=None, steps=4, guidance=5.0, num=1, hw=(32, 32)),
        dict(prompt="a dog", negative_prompt="blurry", steps=3, guidance=5.0, num=2, hw=(32, 48)),
        dict(prompt="no cfg", negative_prompt=None, steps=3, guidance=1.0, num=1, hw=(32, 32)),
        dict(
            prompt="crop",
            negative_prompt=None,
            steps=3,
            guidance=5.0,
            num=1,
            hw=(32, 40),
            original_size=(64, 80),
            crops_coords_top_left=(3, 4),
            target_size=(48, 56),
        ),
    ],
    ids=["zero-negative", "negative-batch-nonsquare", "no-cfg", "micro-conditioning"],
)
def test_tiny_sdxl_generation_matches_diffusers(tmp_path, case):
    root = write_tiny_sdxl(tmp_path / "tiny")
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
    )
    ref.set_progress_bar_config(disable=True)
    h, w = case["hw"]
    common = {
        key: case[key]
        for key in ("original_size", "crops_coords_top_left", "target_size")
        if key in case
    }
    expected = ref(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        num_images_per_prompt=case["num"],
        height=h,
        width=w,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
        **common,
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="stable-diffusion-xl", model=root)(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        num_images=case["num"],
        height=h,
        width=w,
        seed=42,
        **common,
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("model_id", "fixture", "guidance"),
    [
        ("sdxl-turbo", "sdxl_turbo_configs", 0.0),
        ("playground-v2", "playground_v2_configs", 3.0),
    ],
)
def test_tiny_ancestral_generation_matches_diffusers_exactly(tmp_path, model_id, fixture, guidance):
    config = json.loads(
        (Path(__file__).parents[1] / "fixtures" / fixture / "scheduler_config.json").read_text()
    )
    root = write_tiny_sdxl(tmp_path / "tiny-ancestral", scheduler_config=config, sample_size=4)
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
    )
    ref.set_progress_bar_config(disable=True)
    expected = ref(
        "a cat",
        num_inference_steps=2,
        guidance_scale=guidance,
        height=32,
        width=32,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend=model_id, model=root)(
        "a cat", seed=42, steps=2, guidance=guidance, height=32, width=32
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)


def test_tiny_playground_generation_matches_diffusers_exactly(tmp_path):
    config = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/playground_v25_configs/scheduler_config.json"
        ).read_text()
    )
    vae_config = {
        "latents_mean": [-1.6574, 1.886, -1.383, 2.5155],
        "latents_std": [8.4927, 5.9022, 6.5498, 5.2299],
        "scaling_factor": 0.5,
    }
    root = write_tiny_sdxl(
        tmp_path / "tiny-playground",
        scheduler_config=config,
        vae_config=vae_config,
        sample_size=4,
    )
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
    )
    ref.set_progress_bar_config(disable=True)
    expected = ref(
        "a cat",
        num_inference_steps=2,
        guidance_scale=3.0,
        height=32,
        width=32,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="playground-v2.5", model=root)(
        "a cat", seed=42, steps=2, height=32, width=32
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("timesteps", [[800], [999, 749, 499, 249]])
def test_tiny_lcm_generation_with_explicit_timesteps_matches_diffusers(tmp_path, timesteps):
    root = write_tiny_sdxl(tmp_path / f"tiny-lcm-{len(timesteps)}", sample_size=4)
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
    )
    ref.scheduler = diffusers.LCMScheduler.from_config(ref.scheduler.config)
    ref.set_progress_bar_config(disable=True)
    latents = torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(7))
    expected_latents = ref(
        "a cat",
        num_inference_steps=len(timesteps),
        timesteps=timesteps,
        guidance_scale=0.0,
        height=32,
        width=32,
        latents=latents.clone(),
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="latent",
    ).images
    decoded = ref.vae.decode(expected_latents / ref.vae.config.scaling_factor).sample.detach()
    expected = ref.image_processor.postprocess(decoded, output_type="np")
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    components = StableDiffusionXLComponents.load(
        root,
        device=torch.device("cpu"),
        dtype=torch.float32,
        scheduler_overrides={"_class_name": "LCMScheduler"},
    )
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=len(timesteps),
        timesteps=timesteps,
        guidance=0.0,
        latents=latents.clone(),
        original_size=(32, 32),
        crops_coords_top_left=(0, 0),
        target_size=(32, 32),
        generator=torch.Generator("cpu").manual_seed(42),
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    actual = decode_latents(components, actual_latents)
    assert torch.equal(actual, expected)
