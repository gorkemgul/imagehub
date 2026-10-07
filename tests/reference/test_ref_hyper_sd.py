"""Hyper-SD parity against the published diffusers loading recipe on CPU fp32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion import decode_latents, initial_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents, denoise
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
safetensors = pytest.importorskip("safetensors.torch")
pytest.importorskip("transformers")

MODEL_ID = "hyper-sdxl-1step"
PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")
TOLERANCES = {"unet": 0.0, "e2e_latents": 0.0, "e2e_pixels": 0}


def _checkpoint_shapes(path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


@pytest.fixture(scope="module")
def hyper_sd():
    root = snapshot_dir(MODEL_ID)
    # The snapshot is a standard diffusers layout, so the reference loads it with
    # from_pretrained. The model card instead copies the U-Net into a fresh model with
    # load_state_dict; that changes tensor alignment and CPU summation order (about 1e-4).
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root,
        local_files_only=True,
        add_watermarker=False,
        torch_dtype=torch.float32,
    )
    ref.scheduler = diffusers.LCMScheduler.from_config(ref.scheduler.config)
    ref.to(CPU)
    ref.set_progress_bar_config(disable=True)
    ours = StableDiffusionXLComponents.load(
        root,
        device=CPU,
        dtype=torch.float32,
        scheduler_overrides={"_class_name": "LCMScheduler"},
    )
    yield root, ours, ref
    del ours, ref
    gc.collect()


def test_header(record):
    root = snapshot_dir(MODEL_ID)
    with torch.device("meta"):
        model = __import__(
            "imagehub.models.unet_2d", fromlist=["UNet2DConditionModel"]
        ).UNet2DConditionModel(read_json(root / "unet" / "config.json"))
    shapes = _checkpoint_shapes(root / "unet" / "diffusion_pytorch_model.safetensors")
    prepared = model.prepare_state_dict(
        {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
    )
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    actual = {name: tuple(tensor.shape) for name, tensor in prepared.items()}
    result = {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual),
    }
    record(f"header_{MODEL_ID}", **result)
    assert not result["missing"] and not result["extra"] and not result["mismatch"]


def test_unet_one_step_1024(hyper_sd, record):
    _, ours, ref = hyper_sd
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 4, 128, 128, generator=generator)
    context = torch.randn(1, 77, 2048, generator=generator)
    text_embeds = torch.randn(1, 1280, generator=generator)
    time_ids = torch.randn(1, 6, generator=generator)
    timestep = torch.tensor(800)
    with torch.inference_mode():
        actual = ours.unet(sample, timestep, context, text_embeds=text_embeds, time_ids=time_ids)
        expected = ref.unet(
            sample,
            timestep,
            context,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample
    measured = max_diff(actual, expected)
    record(
        "hyper_sd_unet",
        **measured,
        timestep=800,
        size=[1024, 1024],
        tolerance=TOLERANCES["unet"],
    )
    torch.testing.assert_close(actual, expected, atol=TOLERANCES["unet"], rtol=0)


def test_end_to_end_defaults(hyper_sd, record):
    _, ours, ref = hyper_sd
    defaults = MODEL_CATALOG[MODEL_ID].config["defaults"]
    steps = int(defaults["steps"])
    guidance = float(defaults["guidance"])
    timesteps = list(defaults["timesteps"])
    latents = initial_latents(1234, (1, 4, 128, 128))
    with torch.inference_mode():
        expected_latents = ref(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            timesteps=timesteps,
            height=1024,
            width=1024,
            latents=latents.clone(),
            output_type="latent",
        ).images
        expected_decoded = ref.vae.decode(expected_latents / ref.vae.config.scaling_factor).sample
        expected_np = ref.image_processor.postprocess(expected_decoded, output_type="np")
    expected_pixels = torch.from_numpy((expected_np * 255).round().astype("uint8"))
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=None,
        steps=steps,
        guidance=guidance,
        timesteps=timesteps,
        latents=latents.clone(),
        original_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        target_size=(1024, 1024),
    )
    actual_pixels = decode_latents(ours, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = (
        latent_diff["max_abs"] <= TOLERANCES["e2e_latents"]
        and pixel_diff <= TOLERANCES["e2e_pixels"]
    )
    record(
        f"e2e_{MODEL_ID}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=guidance,
        timesteps=timesteps,
        size=[1024, 1024],
        noise_seed=1234,
        tolerance={
            "latents": TOLERANCES["e2e_latents"],
            "pixels": TOLERANCES["e2e_pixels"],
        },
    )
    assert passed, (latent_diff, pixel_diff)


def test_seed_end_to_end(hyper_sd, record):
    root, _, ref = hyper_sd
    defaults = MODEL_CATALOG[MODEL_ID].config["defaults"]
    steps = int(defaults["steps"])
    guidance = float(defaults["guidance"])
    timesteps = list(defaults["timesteps"])
    with torch.inference_mode():
        expected = ref(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            timesteps=timesteps,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=42).images[0].tensor
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{MODEL_ID}",
        pixels_max_abs=pixel_diff,
        seed=42,
        steps=steps,
        guidance=guidance,
        timesteps=timesteps,
        size=[1024, 1024],
        passed=pixel_diff <= TOLERANCES["e2e_pixels"],
    )
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
