"""DreamShaper 8 parity against diffusers 0.40.0 on CPU float32.

Real-checkpoint tests require ``IMAGEHUB_SNAPSHOT_DREAMSHAPER_8``. They never download files.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion import StableDiffusionComponents, _scheduler
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parents[1] / "fixtures" / "dreamshaper_8_configs"
PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE_PROMPT = "blurry, low quality"
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}


def _exact(record, name, actual, expected, **details):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0, **details)
    assert torch.equal(actual, expected)


def _checkpoint_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


def test_published_tensor_headers_match_models(record):
    root = snapshot_dir("dreamshaper-8")
    result = {}
    for component, cls in COMPONENTS.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        filename = (
            "model.safetensors"
            if component == "text_encoder"
            else "diffusion_pytorch_model.safetensors"
        )
        shapes = _checkpoint_shapes(root / component / filename)
        published = json.loads((FIXTURES / f"{component}_tensors.json").read_text())
        assert shapes == {name: tuple(shape) for name, shape in published.items()}
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
    record("header_dreamshaper-8", **result)
    for entry in result.values():
        assert not entry["missing"]
        assert not entry["extra"]
        assert not entry["mismatch"]


@pytest.fixture(scope="module")
def real_pipelines():
    root = snapshot_dir("dreamshaper-8")
    ours = StableDiffusionComponents.load(
        root, device=CPU, dtype=torch.float32, model_id="dreamshaper-8"
    )
    reference = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    return root, ours, reference


def test_real_components(real_pipelines, record):
    _, ours, reference = real_pipelines
    prompts = [PROMPT, "Bir kedi İstanbul'da", "emoji 🐱 and   spaces", ""]
    for prompt in prompts:
        expected_ids = reference.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True
        )["input_ids"]
        assert ours.tokenizer(prompt) == expected_ids
    record("sd_tokenizer_dreamshaper-8", prompts=len(prompts), mismatches=[], tolerance="exact")

    ids = torch.tensor([ours.tokenizer(prompt) for prompt in prompts])
    generator = torch.Generator().manual_seed(0)
    with torch.inference_mode():
        _exact(
            record,
            "sd_text_encoder_dreamshaper-8",
            ours.text_encoder(ids),
            reference.text_encoder(ids)[0],
        )
        sample = torch.randn(1, 4, 8, 8, generator=generator)
        context = torch.randn(1, 77, 768, generator=generator)
        _exact(
            record,
            "sd_unet_dreamshaper-8",
            ours.unet(sample, torch.tensor(500), context),
            reference.unet(sample, torch.tensor(500), context).sample,
            timesteps=[500],
        )
        latents = torch.randn(1, 4, 8, 8, generator=generator)
        _exact(
            record,
            "sd_vae_decode_dreamshaper-8",
            ours.vae.decode(latents),
            reference.vae.decode(latents).sample,
        )

    actual_scheduler, _ = _scheduler(ours.scheduler_config)
    expected_scheduler = reference.scheduler
    actual_scheduler.set_timesteps(4)
    expected_scheduler.set_timesteps(4)
    actual = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(1))
    expected = actual.clone()
    for timestep in expected_scheduler.timesteps:
        output = torch.randn(actual.shape, generator=torch.Generator().manual_seed(int(timestep)))
        actual = actual_scheduler.step(output, timestep, actual)
        expected = expected_scheduler.step(output, timestep, expected).prev_sample
    _exact(record, "sd_scheduler_dreamshaper-8", actual, expected, steps=4)


def _e2e(root, reference, record, *, seed, record_name):
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = (
        pipeline("t2i", backend="dreamshaper-8", model=root)(PROMPT, seed=seed).images[0].tensor
    )
    difference = int((actual.int() - expected.int()).abs().max())
    passed = difference == 0
    record(
        record_name,
        pixels_max_abs=difference,
        passed=passed,
        seed=seed,
        steps=50,
        guidance=7.5,
        size=[512, 512],
        tolerance=0,
    )
    assert passed


def test_end_to_end_seed_0(real_pipelines, record):
    root, _, reference = real_pipelines
    _e2e(root, reference, record, seed=0, record_name="e2e_dreamshaper-8")


def test_end_to_end_seed_42(real_pipelines, record):
    root, _, reference = real_pipelines
    _e2e(root, reference, record, seed=42, record_name="seed_e2e_dreamshaper-8")
