"""SDXL refiner parity on CPU float32 against diffusers' image-to-image pipeline.

Requires IMAGEHUB_REFERENCE_TESTS=1 and IMAGEHUB_SNAPSHOT_SDXL_REFINER_1_0 pointing at a
snapshot made by ``python -m imagehub.pretrained sdxl-refiner-1.0 <dir>``. The tests skip
cleanly without it.
"""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.sdxl_refiner import preprocess_refiner_image, refine
from imagehub.backends.stable_diffusion import decode_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.euler import EulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
PILImage = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_ID = "sdxl-refiner-1.0"
PROMPT = "a photograph of an astronaut riding a horse"
IMAGE_SIZE = (1024, 1024)
FIXTURE = Path(__file__).parents[1] / "fixtures" / "sdxl_refiner_tensor_shapes.json"
COMPONENTS = {
    "text_encoder_2": CLIPTextModelWithProjection,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "Bir kedi, yağlı boya; İSTANBUL'da ĞÜŞİÖÇ ığüşöç",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "x!<|endoftext|>y <|startoftext|> mid",
    "word " * 100,
    "",
    "!",
    "Hi CAT!",
]


def make_image(width, height, variant=0):
    """A deterministic smooth gradient with a few filled shapes."""
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    channels = (x, y, (x + y) / 2) if variant == 0 else (y, (x + y) / 2, 255 - x)
    image = PILImage.fromarray(np.stack(channels, axis=-1).round().astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (width // 8, height // 8, width // 8 + width // 4, height // 8 + height // 3),
        fill=(220, 40, 40),
    )
    draw.rectangle(
        (width // 2, height // 2, width // 2 + width // 3, height // 2 + height // 4),
        fill=(30, 60, 200),
    )
    draw.polygon(
        [(width // 4, height - 10), (width // 2, height // 2 + 20), (3 * width // 4, height - 10)],
        fill=(240, 220, 40),
    )
    return image


def _check(record, name, actual, expected, **extra):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0, **extra)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {n: tuple(e["shape"]) for n, e in header.items() if n != "__metadata__"}


@pytest.fixture(scope="module")
def refiner():
    root = snapshot_dir(MODEL_ID)
    ours = StableDiffusionXLComponents.load(
        root, device=CPU, dtype=torch.float32, model_id=MODEL_ID, refiner=True
    )
    ref = diffusers.StableDiffusionXLImg2ImgPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False, torch_dtype=torch.float32
    )
    ref.to(CPU)
    ref.set_progress_bar_config(disable=True)
    yield root, ours, ref
    del ours, ref
    gc.collect()


def test_published_tensor_headers_match_models(record):
    root = snapshot_dir(MODEL_ID)
    published = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = {}
    for component, cls in COMPONENTS.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        directory = root / component
        weights = next(directory.glob("*.safetensors"))
        shapes = _checkpoint_shapes(weights)
        assert shapes == {n: tuple(e["shape"]) for n, e in published[component].items()}
        prepared = model.prepare_state_dict(
            {n: torch.empty(s, device="meta") for n, s in shapes.items()}
        )
        expected = {n: tuple(t.shape) for n, t in model.state_dict().items()}
        got = {n: tuple(t.shape) for n, t in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - got.keys()),
            "extra": sorted(got.keys() - expected.keys()),
            "mismatch": sorted(
                f"{n}: checkpoint {got[n]} != model {expected[n]}"
                for n in expected.keys() & got.keys()
                if got[n] != expected[n]
            ),
            "tensors": len(got),
        }
    record(f"header_{MODEL_ID}", **result)
    for component, entry in result.items():
        assert not entry["missing"] and not entry["extra"] and not entry["mismatch"], component


def test_tokenizer(refiner, record):
    _, ours, ref = refiner
    assert ours.tokenizer is None and ref.tokenizer is None
    mismatches = [
        text
        for text in PROMPTS
        if ours.tokenizer_2(text)
        != ref.tokenizer_2(text, padding="max_length", max_length=77, truncation=True)["input_ids"]
    ]
    record(
        f"sdxl_tokenizers_{MODEL_ID}",
        prompts=len(PROMPTS),
        tokenizers=1,
        mismatches=mismatches,
        tolerance="exact",
    )
    assert not mismatches, mismatches[:5]


def test_text_encoder_2(refiner, record):
    _, ours, ref = refiner
    ids = torch.tensor([ours.tokenizer_2(prompt) for prompt in PROMPTS])
    with torch.inference_mode():
        pooled, _, hidden = ours.text_encoder_2(ids, output_hidden_states=True)
        expected = ref.text_encoder_2(ids, output_hidden_states=True)
    hidden_diff = max_diff(hidden[-2], expected.hidden_states[-2])
    pooled_diff = max_diff(pooled, expected.text_embeds)
    worst = max((hidden_diff, pooled_diff), key=lambda value: value["max_abs"])
    record(
        f"sdxl_text_encoder_2_{MODEL_ID}",
        **worst,
        hidden_states=hidden_diff,
        text_embeds=pooled_diff,
        tolerance=0.0,
    )
    torch.testing.assert_close(hidden[-2], expected.hidden_states[-2], atol=0, rtol=0)
    torch.testing.assert_close(pooled, expected.text_embeds, atol=0, rtol=0)


def test_unet_single_step(refiner, record):
    _, ours, ref = refiner
    generator = torch.Generator().manual_seed(0)
    size = int(ours.unet.config["sample_size"])
    sample = torch.randn(2, 4, size, size, generator=generator)
    context = torch.randn(2, 77, 1280, generator=generator)
    text_embeds = torch.randn(2, 1280, generator=generator)
    time_ids = torch.randn(2, 5, generator=generator)
    pairs = []
    with torch.inference_mode():
        for value in (999, 500, 1):
            timestep = torch.tensor(value)
            actual = ours.unet(
                sample, timestep, context, text_embeds=text_embeds, time_ids=time_ids
            )
            expected = ref.unet(
                sample,
                timestep,
                context,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample
            pairs.append((actual, expected, max_diff(actual, expected)))
    worst = max((measured for _, _, measured in pairs), key=lambda value: value["max_abs"])
    record(f"sdxl_unet_{MODEL_ID}", **worst, tolerance=0.0, timesteps=[999, 500, 1])
    for actual, expected, _ in pairs:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_vae(refiner, record):
    _, ours, ref = refiner
    generator = torch.Generator().manual_seed(1)
    size = int(ours.unet.config["sample_size"])
    latents = torch.randn(1, 4, size, size, generator=generator)
    image = torch.rand(1, 3, size * 8, size * 8, generator=generator) * 2 - 1
    with torch.inference_mode():
        decode = ours.vae.decode(latents), ref.vae.decode(latents).sample
        encode = ours.vae.encode(image), ref.vae.encode(image).latent_dist.mean
        ref_generator = torch.Generator("cpu").manual_seed(3)
        our_generator = torch.Generator("cpu").manual_seed(3)
        sample = (
            ours.vae.encode_sample(image, our_generator),
            ref.vae.encode(image).latent_dist.sample(ref_generator),
        )
    for name, (actual, expected) in (
        (f"sdxl_vae_decode_{MODEL_ID}", decode),
        (f"sdxl_vae_encode_{MODEL_ID}", encode),
        (f"sdxl_refiner_vae_sample_{MODEL_ID}", sample),
    ):
        record(name, **max_diff(actual, expected), tolerance=0.0)
    for actual, expected in (decode, encode, sample):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(ref_generator.get_state(), our_generator.get_state())


@pytest.mark.parametrize(("width", "height"), [(1024, 1024), (1000, 700), (640, 963)])
def test_preprocess(refiner, record, width, height):
    _, _, ref = refiner
    image = make_image(width, height, 0 if width >= height else 1)
    expected = ref.image_processor.preprocess(image)
    actual = preprocess_refiner_image(image)
    record(
        f"sdxl_refiner_preprocess_{width}x{height}_{MODEL_ID}",
        **max_diff(actual, expected),
        tolerance=0.0,
        shape=list(actual.shape),
    )
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_scheduler(refiner, record):
    _, ours, ref = refiner
    measurements = {}
    for steps, strength in ((50, 0.3), (30, 0.5)):
        actual = EulerDiscreteScheduler(ours.scheduler_config)
        expected = type(ref.scheduler).from_config(ref.scheduler.config)
        actual.set_timesteps(steps)
        expected.set_timesteps(steps)
        ref.scheduler = expected
        timesteps, _ = ref.get_timesteps(steps, strength, CPU)
        start = steps - len(timesteps)
        actual.set_begin_index(start)
        torch.manual_seed(0)
        original, noise = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
        batch = timesteps[:1].repeat(2)
        noised = actual.add_noise(original, noise, batch)
        reference_noised = expected.add_noise(original, noise, batch)
        measurements[f"{steps}_{strength}"] = {
            "timesteps": max_diff(actual.timesteps, expected.timesteps),
            "sigmas": max_diff(actual.sigmas, expected.sigmas),
            "add_noise": max_diff(noised, reference_noised),
            "start": start,
        }
        torch.testing.assert_close(actual.timesteps, expected.timesteps, atol=0, rtol=0)
        torch.testing.assert_close(actual.sigmas, expected.sigmas, atol=0, rtol=0)
        torch.testing.assert_close(noised, reference_noised, atol=0, rtol=0)
    record(f"sdxl_scheduler_{MODEL_ID}", measurements=measurements, tolerance=0.0)


@pytest.fixture(scope="module")
def reference_run(refiner):
    """The reference pipeline at its defaults (50 steps, strength 0.3, guidance 5.0), seed 0."""
    _, _, ref = refiner
    image = make_image(*IMAGE_SIZE)
    with torch.inference_mode():
        latents = ref(
            PROMPT,
            image=image,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="latent",
        ).images
        decoded = ref.vae.decode(latents / ref.vae.config.scaling_factor).sample
        pixels = ref.image_processor.postprocess(decoded, output_type="np")
    return image, latents, torch.from_numpy((pixels * 255).round().astype("uint8"))


def test_end_to_end(refiner, reference_run, record):
    _, ours, _ = refiner
    image, ref_latents, ref_pixels = reference_run
    our_latents = refine(
        ours,
        prompt=PROMPT,
        negative_prompt=None,
        steps=50,
        strength=0.3,
        guidance=5.0,
        pixels=preprocess_refiner_image(image),
        num_images=1,
        original_size=None,
        crops_coords_top_left=(0, 0),
        target_size=None,
        negative_original_size=None,
        negative_crops_coords_top_left=(0, 0),
        negative_target_size=None,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    our_pixels = decode_latents(ours, our_latents)
    latent_diff = max_diff(our_latents, ref_latents)
    pixel_diff = int((our_pixels.int() - ref_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0.0 and pixel_diff == 0
    record(
        f"e2e_{MODEL_ID}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=50,
        strength=0.3,
        guidance=5.0,
        input_size=list(IMAGE_SIZE),
        input="synthetic gradient with three filled shapes",
        noise_seed=0,
        tolerance={"latents": 0.0, "pixels": 0},
        runtime_note="50 steps at strength 0.3 run 15 denoising steps.",
    )
    assert passed, (latent_diff, pixel_diff)


def test_seed_end_to_end(refiner, reference_run, record):
    root, _, _ = refiner
    image, _, expected = reference_run
    result = pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, image=image, seed=0)
    pixel_diff = int((result.images[0].tensor.int() - expected[0].int()).abs().max())
    record(
        f"seed_e2e_{MODEL_ID}",
        pixels_max_abs=pixel_diff,
        seed=0,
        steps=50,
        strength=0.3,
        guidance=5.0,
        size=list(IMAGE_SIZE),
        passed=pixel_diff == 0,
        runtime_note="50 steps at strength 0.3 run 15 denoising steps.",
    )
    assert pixel_diff == 0
