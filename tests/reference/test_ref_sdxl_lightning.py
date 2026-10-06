"""SDXL-Lightning parity against the published diffusers loading recipe on CPU fp32."""

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

MODEL_IDS = (
    "sdxl-lightning-2step",
    "sdxl-lightning-4step",
    "sdxl-lightning-8step",
)
PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")
TOLERANCES = {"unet": 0.0, "e2e_latents": 0.0, "e2e_pixels": 0}


def _checkpoint_shapes(path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


@pytest.fixture(scope="module", params=MODEL_IDS)
def lightning(request):
    model_id = request.param
    root = snapshot_dir(model_id)
    unet = diffusers.UNet2DConditionModel.from_config(root, subfolder="unet").to(
        device=CPU, dtype=torch.float32
    )
    state = safetensors.load_file(
        root / "unet" / "diffusion_pytorch_model.safetensors", device="cpu"
    )
    unet.load_state_dict(state)
    del state
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root,
        unet=unet,
        local_files_only=True,
        add_watermarker=False,
        torch_dtype=torch.float32,
    )
    ref.scheduler = diffusers.EulerDiscreteScheduler.from_config(
        ref.scheduler.config, timestep_spacing="trailing"
    )
    ref.to(CPU)
    ref.set_progress_bar_config(disable=True)
    ours = StableDiffusionXLComponents.load(
        root,
        device=CPU,
        dtype=torch.float32,
        scheduler_overrides={"timestep_spacing": "trailing"},
    )
    yield model_id, root, ours, ref
    del ours, ref, unet
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_header(model_id, record):
    root = snapshot_dir(model_id)
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
    record(f"header_{model_id}", **result)
    assert not result["missing"] and not result["extra"] and not result["mismatch"]


def test_unet_one_step_1024(lightning, record):
    model_id, _, ours, ref = lightning
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 4, 128, 128, generator=generator)
    context = torch.randn(1, 77, 2048, generator=generator)
    text_embeds = torch.randn(1, 1280, generator=generator)
    time_ids = torch.randn(1, 6, generator=generator)
    timestep = torch.tensor(999)
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
        f"sdxl_lightning_unet_{model_id}",
        **measured,
        timestep=999,
        size=[1024, 1024],
        tolerance=TOLERANCES["unet"],
    )
    torch.testing.assert_close(actual, expected, atol=TOLERANCES["unet"], rtol=0)


def test_end_to_end_defaults(lightning, record):
    model_id, _, ours, ref = lightning
    defaults = MODEL_CATALOG[model_id].config["defaults"]
    steps = int(defaults["steps"])
    guidance = float(defaults["guidance"])
    latents = initial_latents(1234, (1, 4, 128, 128))
    with torch.inference_mode():
        expected_latents = ref(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
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
        f"e2e_{model_id}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=guidance,
        size=[1024, 1024],
        noise_seed=1234,
        tolerance={
            "latents": TOLERANCES["e2e_latents"],
            "pixels": TOLERANCES["e2e_pixels"],
        },
    )
    assert passed, (latent_diff, pixel_diff)


def test_seed_end_to_end(lightning, record):
    model_id, root, _, ref = lightning
    defaults = MODEL_CATALOG[model_id].config["defaults"]
    steps = int(defaults["steps"])
    guidance = float(defaults["guidance"])
    with torch.inference_mode():
        expected = ref(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend=model_id, model=root)(PROMPT, seed=42).images[0].tensor
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        pixels_max_abs=pixel_diff,
        seed=42,
        steps=steps,
        guidance=guidance,
        size=[1024, 1024],
        passed=pixel_diff <= TOLERANCES["e2e_pixels"],
    )
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
