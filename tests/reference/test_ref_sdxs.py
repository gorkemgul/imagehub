"""Parity of SDXS-512-0.9 against diffusers on CPU float32."""

import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion import StableDiffusionComponents, encode_prompt
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.euler import EulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

MODEL_ID = "sdxs-512-0.9"
PROMPT = "portrait photo of a girl, highly detailed face, golden hour"
PROMPTS = [PROMPT, "", "Bir kedi, yağlı boya", "emoji 🐱🚀 " * 30, "!"]
CPU = torch.device("cpu")
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderTiny,
}


@pytest.fixture(scope="module")
def sdxs():
    root = snapshot_dir(MODEL_ID)
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
    reference = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        local_files_only=True,
        torch_dtype=torch.float32,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    reference.set_progress_bar_config(disable=True)
    return root, ours, reference


def _checkpoint_shapes(path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


def test_header(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, model_class in COMPONENTS.items():
        with torch.device("meta"):
            model = model_class(read_json(root / component / "config.json"))
        filename = (
            "model.safetensors"
            if component == "text_encoder"
            else "diffusion_pytorch_model.safetensors"
        )
        published = _checkpoint_shapes(root / component / filename)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in published.items()}
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
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer_and_text_encoder(sdxs, record):
    _, ours, reference = sdxs
    expected_ids = [
        reference.tokenizer(prompt, padding="max_length", max_length=77, truncation=True)[
            "input_ids"
        ]
        for prompt in PROMPTS
    ]
    actual_ids = [ours.tokenizer(prompt) for prompt in PROMPTS]
    ids = torch.tensor(actual_ids)
    with torch.inference_mode():
        expected = reference.text_encoder(ids)[0]
        actual = ours.text_encoder(ids)
    difference = max_diff(actual, expected)
    record(
        f"sdxs_tokenizer_text_encoder_{MODEL_ID}",
        tokenizer_mismatches=sum(a != b for a, b in zip(actual_ids, expected_ids, strict=True)),
        text_encoder=difference,
        tolerance=0.0,
    )
    assert actual_ids == expected_ids
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_unet_vae_and_scheduler(sdxs, record):
    _, ours, reference = sdxs
    generator = torch.Generator("cpu").manual_seed(0)
    sample = torch.randn(1, 4, 64, 64, generator=generator)
    context = encode_prompt(ours, PROMPT, CPU)
    latents = torch.randn(1, 4, 64, 64, generator=generator)
    native_scheduler = EulerDiscreteScheduler(ours.scheduler_config)
    reference_scheduler = type(reference.scheduler).from_config(reference.scheduler.config)
    native_scheduler.set_timesteps(1)
    reference_scheduler.set_timesteps(1)
    timestep = native_scheduler.timesteps[0]
    with torch.inference_mode():
        actual_unet = ours.unet(sample, timestep, context)
        expected_unet = reference.unet(sample, timestep, context).sample
        actual_vae = ours.vae.decode(latents)
        expected_vae = reference.vae.decode(latents).sample
    measurements = {
        "unet": max_diff(actual_unet, expected_unet),
        "vae_decode": max_diff(actual_vae, expected_vae),
        "timesteps": max_diff(native_scheduler.timesteps, reference_scheduler.timesteps),
        "sigmas": max_diff(native_scheduler.sigmas, reference_scheduler.sigmas),
    }
    record(f"sdxs_components_{MODEL_ID}", **measurements, tolerance=0.0)
    torch.testing.assert_close(actual_unet, expected_unet, atol=0, rtol=0)
    torch.testing.assert_close(actual_vae, expected_vae, atol=0, rtol=0)
    torch.testing.assert_close(
        native_scheduler.timesteps, reference_scheduler.timesteps, atol=0, rtol=0
    )
    torch.testing.assert_close(native_scheduler.sigmas, reference_scheduler.sigmas, atol=0, rtol=0)


def _end_to_end(sdxs, record, *, seed, name):
    root, _, reference = sdxs
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=seed).images[0].tensor
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    passed = torch.equal(actual, expected)
    record(
        name,
        passed=passed,
        pixels_max_abs=pixel_diff,
        seed=seed,
        steps=1,
        guidance=0.0,
        size=[512, 512],
        tolerance=0,
    )
    assert passed


def test_end_to_end(sdxs, record):
    _end_to_end(sdxs, record, seed=0, name=f"e2e_{MODEL_ID}")


def test_seed_end_to_end(sdxs, record):
    _end_to_end(sdxs, record, seed=42, name=f"seed_e2e_{MODEL_ID}")
