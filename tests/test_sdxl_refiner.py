import json
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline
from imagehub.backends.sdxl_refiner import refiner_input_size, refiner_start
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLBackend
from imagehub.catalog import MODEL_CATALOG, model_info
from imagehub.exceptions import CheckpointError
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.euler import EulerDiscreteScheduler
from imagehub.types import T2IRequest

PILImage = pytest.importorskip("PIL.Image")
FIXTURES = Path(__file__).parent / "fixtures"
REFINER = "sdxl-refiner-1.0"


def make_image(width, height, mode="RGB", seed=0):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (height * width * channels,), generator=g, dtype=torch.uint8)
    return PILImage.frombytes(mode, (width, height), bytes(data.tolist()))


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_tiny_sdxl(tmp_path_factory.mktemp("refiner") / "snap", refiner=True)


def run(snapshot, **options):
    pipe = pipeline("t2i", backend=REFINER, model=snapshot)
    return pipe("a cat", **({"steps": 6, "image": make_image(32, 24), "seed": 1} | options))


def test_catalog_entry():
    spec = model_info(REFINER)
    assert spec.family == "stable-diffusion-xl" and spec.catalog_id == "sdxl"
    assert spec.repo_id == "stabilityai/stable-diffusion-xl-refiner-1.0"
    assert spec.revision == "5d4cfe854c9a9a87939ff3653551c2b3c99a4356"
    assert spec.provenance == "author" and not spec.gated and spec.format == "diffusers"
    assert spec.license == "CreativeML-OpenRAIL++-M"
    assert spec.image_input == "required" and spec.config == {"pipeline": "refiner"}
    assert len(set(spec.files)) == len(spec.files) == 12
    assert not any(name.startswith(("text_encoder/", "tokenizer/")) for name in spec.files)
    assert not any("fp16" in name or name.startswith("vae_1_0/") for name in spec.files)
    assert [m for m, s in MODEL_CATALOG.items() if s.image_input == "required"] == [REFINER]


def test_published_configs_match_the_published_tensor_layout():
    root = FIXTURES / "sdxl_refiner_configs"
    published = json.loads((FIXTURES / "sdxl_refiner_tensor_shapes.json").read_text())
    for component, cls in (
        ("text_encoder_2", CLIPTextModelWithProjection),
        ("unet", UNet2DConditionModel),
        ("vae", AutoencoderKL),
    ):
        config = json.loads((root / component / "config.json").read_text())
        with torch.device("meta"):
            model = cls(config)
        prepared = model.prepare_state_dict(
            {n: torch.empty(e["shape"], device="meta") for n, e in published[component].items()}
        )
        assert {n: list(t.shape) for n, t in prepared.items()} == {
            n: list(t.shape) for n, t in model.state_dict().items()
        }


def test_scheduler_and_model_index_of_the_release():
    root = FIXTURES / "sdxl_refiner_configs"
    index = json.loads((root / "model_index.json").read_text())
    assert index["requires_aesthetics_score"] is True
    assert index["force_zeros_for_empty_prompt"] is False
    assert index["text_encoder"] == [None, None] and index["tokenizer"] == [None, None]
    assert json.loads((root / "scheduler" / "scheduler_config.json").read_text()) == json.loads(
        (FIXTURES / "sdxl_configs" / "scheduler.json").read_text()
    )


def test_defaults_come_from_the_pipeline_signature():
    backend = StableDiffusionXLBackend(Path("missing"), model_id=REFINER)
    options = backend.validate_options(T2IRequest("x"), {"image": make_image(16, 16)})
    assert (options["steps"], options["guidance"], options["strength"]) == (50, 5.0, 0.3)
    assert (options["aesthetic_score"], options["negative_aesthetic_score"]) == (6.0, 2.5)
    assert options["image"].mode == "RGB"


def test_image_is_required_and_validated_before_loading():
    backend = StableDiffusionXLBackend(Path("missing"), model_id=REFINER)
    good = make_image(16, 16)
    for options, error in (
        ({}, ValueError),
        ({"image": "cat.png"}, TypeError),
        ({"image": good, "strength": 1.5}, ValueError),
        ({"image": good, "strength": -0.1}, ValueError),
        ({"image": good, "strength": float("nan")}, ValueError),
        ({"image": good, "strength": 0.01}, ValueError),
        ({"image": good, "steps": 0}, ValueError),
        ({"image": good, "guidance": -1}, ValueError),
        ({"image": good, "aesthetic_score": "high"}, ValueError),
        ({"image": make_image(4, 16)}, ValueError),
        ({"image": good, "seed": -1}, ValueError),
        ({"image": good, "target_size": (1, 2, 3)}, ValueError),
        ({"image": good, "width": 32}, TypeError),
        ({"image": good, "height": 32}, TypeError),
        ({"image": good, "latents": torch.zeros(1, 4, 2, 2)}, TypeError),
        ({"image": good, "timesteps": [999]}, TypeError),
        ({"image": good, "unknown": 1}, TypeError),
    ):
        with pytest.raises(error):
            backend.validate_options(T2IRequest("x"), options)
    assert not backend.is_loaded


def test_other_sdxl_models_still_reject_image_and_strength():
    for model_id in ("sdxl-base-1.0", "sdxl-turbo", None):
        backend = StableDiffusionXLBackend(Path("missing"), model_id=model_id)
        for option in ("image", "strength", "aesthetic_score"):
            with pytest.raises(TypeError, match=option):
                backend.validate_options(T2IRequest("x"), {option: 1})
        assert not backend.is_loaded
    with pytest.raises(TypeError):
        pipeline("t2i", backend="stable-diffusion-xl", model=Path("missing"))(
            "x", image=make_image(8, 8)
        )


def test_start_index_and_size_follow_the_reference_arithmetic():
    assert [refiner_start(50, s) for s in (0.3, 0.5, 1.0, 0.99)] == [35, 25, 0, 1]
    assert refiner_start(10, 0.01) == 10
    assert refiner_input_size(make_image(61, 45)) == (40, 56)
    assert refiner_input_size(make_image(64, 64)) == (64, 64)


def test_snapshot_missing_the_refiner_components(tmp_path):
    backend = StableDiffusionXLBackend(tmp_path, model_id=REFINER)
    with pytest.raises(CheckpointError, match="not a Stable Diffusion XL snapshot"):
        backend.load()


def test_generates_at_the_input_size_with_metadata(snapshot):
    result = run(snapshot, image=make_image(61, 45))
    assert result.images[0].tensor.shape == (40, 56, 3)
    meta = result.metadata
    assert meta["image_input"] is True and meta["seed"] == 1
    assert (meta["input_height"], meta["input_width"]) == (40, 56)
    assert (meta["strength"], meta["steps"], meta["guidance"]) == (0.3, 6, 5.0)
    assert meta["scheduler"] == "euler" and meta["device"] == "cpu"


def test_seed_repeats_and_changes(snapshot):
    first = run(snapshot)
    assert torch.equal(first.images[0].tensor, run(snapshot).images[0].tensor)
    assert not torch.equal(first.images[0].tensor, run(snapshot, seed=2).images[0].tensor)
    assert not torch.equal(
        first.images[0].tensor, run(snapshot, image=make_image(32, 24, seed=5)).images[0].tensor
    )


def test_image_types_and_modes(snapshot):
    source = make_image(32, 24)
    expected = run(snapshot, image=source).images[0].tensor
    wrapped = Image(
        torch.frombuffer(bytearray(source.tobytes()), dtype=torch.uint8).view(24, 32, 3)
    )
    assert torch.equal(run(snapshot, image=wrapped).images[0].tensor, expected)
    rgba = make_image(32, 24, "RGBA")
    assert torch.equal(
        run(snapshot, image=rgba).images[0].tensor,
        run(snapshot, image=rgba.convert("RGB")).images[0].tensor,
    )


def test_strength_guidance_and_batch(snapshot):
    low = run(snapshot, strength=0.3).images[0].tensor
    assert not torch.equal(low, run(snapshot, strength=0.9).images[0].tensor)
    batch = run(snapshot, num_images=2)
    assert len(batch.images) == 2
    assert not torch.equal(batch.images[0].tensor, batch.images[1].tensor)


def test_euler_begin_index_and_add_noise():
    config = json.loads((FIXTURES / "sdxl_configs" / "scheduler.json").read_text())
    scheduler = EulerDiscreteScheduler(config)
    scheduler.set_timesteps(10)
    scheduler.set_begin_index(7)
    original, noise = torch.randn(2, 4, 3, 3), torch.randn(2, 4, 3, 3)
    noised = scheduler.add_noise(original, noise, scheduler.timesteps[7:8].repeat(2))
    assert torch.equal(noised, original + noise * scheduler.sigmas[7])
    sample = torch.randn(2, 4, 3, 3)
    scaled = scheduler.scale_model_input(sample, scheduler.timesteps[7])
    assert scheduler.step_index == 7
    assert torch.equal(scaled, sample / (scheduler.sigmas[7] ** 2 + 1) ** 0.5)
    scheduler.set_timesteps(10)
    assert scheduler.begin_index is None
    # Without a begin index the timestep is looked up, as before.
    again = scheduler.add_noise(original, noise, scheduler.timesteps[3:4].repeat(2))
    assert torch.equal(again, original + noise * scheduler.sigmas[3])


def test_vae_sample_uses_mean_and_clamped_logvar():
    vae = AutoencoderKL({"block_out_channels": [32, 32, 64, 64], "norm_num_groups": 8}).eval()
    pixels = torch.randn(1, 3, 16, 16)
    with torch.inference_mode():
        mean, logvar = vae.encode_moments(pixels)
        assert torch.equal(mean, vae.encode(pixels))
        sample = vae.encode_sample(pixels, torch.Generator("cpu").manual_seed(0))
    noise = torch.randn(mean.shape, generator=torch.Generator("cpu").manual_seed(0))
    assert torch.equal(sample, mean + torch.exp(0.5 * logvar) * noise)
    assert float(logvar.min()) >= -30.0 and float(logvar.max()) <= 20.0
