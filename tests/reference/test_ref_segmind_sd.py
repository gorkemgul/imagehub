"""Segmind Small-SD and Tiny-SD parity against diffusers 0.40.0 on CPU float32.

Real-checkpoint tests use ``IMAGEHUB_SNAPSHOT_SEGMIND_SMALL_SD`` and
``IMAGEHUB_SNAPSHOT_SEGMIND_TINY_SD``. Random-weight tests need no snapshots.
"""

import json
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir, write_tiny_sd

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
FIXTURES = Path(__file__).parents[1] / "fixtures"
PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE_PROMPT = "blurry, low quality"
MODEL_IDS = ("segmind-small-sd", "segmind-tiny-sd")
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
SMALL_SCHEDULER = {
    "_class_name": "DEISMultistepScheduler",
    "algorithm_type": "deis",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "solver_order": 2,
    "solver_type": "logrho",
    "steps_offset": 1,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_karras_sigmas": False,
}
TINY_SCHEDULER = SMALL_SCHEDULER | {
    "_class_name": "DPMSolverMultistepScheduler",
    "algorithm_type": "dpmsolver++",
    "solver_type": "midpoint",
    "variance_type": None,
}
TINY_NO_MID_UNET = {
    "block_out_channels": [32, 64, 64],
    "down_block_types": ["CrossAttnDownBlock2D"] * 3,
    "up_block_types": ["CrossAttnUpBlock2D"] * 3,
    "layers_per_block": 1,
    "mid_block_type": None,
}


def _exact(record, name, actual, expected, **details):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0, **details)
    assert torch.equal(actual, expected)


def _pickle_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    state = torch.load(path, map_location="meta", weights_only=True)
    return {name: tuple(tensor.shape) for name, tensor in state.items()}


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_published_tensor_headers_match_models(model_id, record):
    root = snapshot_dir(model_id)
    fixture_name = model_id.replace("-", "_") + "_tensor_shapes.json"
    published = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    result = {}
    for component, cls in COMPONENTS.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        filename = (
            "pytorch_model.bin" if component == "text_encoder" else "diffusion_pytorch_model.bin"
        )
        shapes = _pickle_shapes(root / component / filename)
        expected_published = {
            name: tuple(entry["shape"]) for name, entry in published[component].items()
        }
        assert shapes == expected_published
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
    record(f"header_{model_id}", **result)
    for entry in result.values():
        assert not entry["missing"]
        assert not entry["extra"]
        assert not entry["mismatch"]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_unet_random_weights_match_diffusers_exactly(model_id):
    folder = model_id.replace("segmind-", "segmind_").replace("-", "_") + "_configs"
    config = json.loads((FIXTURES / folder / "unet.json").read_text(encoding="utf-8"))
    torch.manual_seed(0)
    reference = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    sample = torch.randn(1, 4, 8, 8)
    context = torch.randn(1, 7, 768)
    with torch.inference_mode():
        expected = reference(sample, torch.tensor(500), context).sample
        actual = ours(sample, torch.tensor(500), context)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("model_id", "scheduler_config", "unet_config"),
    [
        ("segmind-small-sd", SMALL_SCHEDULER, {"layers_per_block": 1}),
        ("segmind-tiny-sd", TINY_SCHEDULER, TINY_NO_MID_UNET),
    ],
)
@pytest.mark.parametrize(
    ("guidance", "negative_prompt", "size"),
    [(5.0, NEGATIVE_PROMPT, (32, 48)), (1.0, None, (40, 32))],
)
def test_tiny_pipeline_matches_diffusers_exactly(
    tmp_path, model_id, scheduler_config, unet_config, guidance, negative_prompt, size
):
    root = write_tiny_sd(
        tmp_path / model_id,
        unet_config=unet_config,
        scheduler_config=scheduler_config,
    )
    reference = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    reference.set_progress_bar_config(disable=True)
    height, width = size
    expected = reference(
        PROMPT,
        negative_prompt=negative_prompt,
        num_inference_steps=3,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend=model_id, model=root)(
        PROMPT,
        negative_prompt=negative_prompt,
        steps=3,
        guidance=guidance,
        height=height,
        width=width,
        seed=42,
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)


@pytest.fixture(scope="module", params=MODEL_IDS)
def real_pipelines(request):
    model_id = request.param
    root = snapshot_dir(model_id)
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
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
    return model_id, root, ours, reference


def test_real_components(real_pipelines, record):
    model_id, _, ours, reference = real_pipelines
    prompts = [PROMPT, "Bir kedi İstanbul'da", "emoji 🐱 and   spaces", ""]
    for prompt in prompts:
        expected_ids = reference.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True
        )["input_ids"]
        assert ours.tokenizer(prompt) == expected_ids
    record(f"sd_tokenizer_{model_id}", prompts=len(prompts), mismatches=[], tolerance="exact")
    ids = torch.tensor([ours.tokenizer(prompt) for prompt in prompts])
    with torch.inference_mode():
        _exact(
            record,
            f"sd_text_encoder_{model_id}",
            ours.text_encoder(ids),
            reference.text_encoder(ids)[0],
        )
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(1, 4, 8, 8, generator=generator)
        context = torch.randn(1, 77, 768, generator=generator)
        _exact(
            record,
            f"sd_unet_{model_id}",
            ours.unet(sample, torch.tensor(500), context),
            reference.unet(sample, torch.tensor(500), context).sample,
            timesteps=[500],
        )
        latents = torch.randn(1, 4, 8, 8, generator=generator)
        _exact(
            record,
            f"sd_vae_decode_{model_id}",
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
    _exact(record, f"sd_scheduler_{model_id}", actual, expected, steps=4)


def _e2e(model_id, root, reference, record, *, seed, record_name):
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=4,
            guidance_scale=5.0,
            height=64,
            width=64,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seed=seed,
            steps=4,
            guidance=5.0,
            height=64,
            width=64,
        )
        .images[0]
        .tensor
    )
    difference = int((actual.int() - expected.int()).abs().max())
    passed = difference == 0
    record(
        record_name,
        pixels_max_abs=difference,
        passed=passed,
        seed=seed,
        steps=4,
        guidance=5.0,
        size=[64, 64],
        tolerance=0,
    )
    assert passed


def test_end_to_end_seed_0(real_pipelines, record):
    model_id, root, _, reference = real_pipelines
    _e2e(model_id, root, reference, record, seed=0, record_name=f"e2e_{model_id}")


def test_end_to_end_seed_42(real_pipelines, record):
    model_id, root, _, reference = real_pipelines
    _e2e(model_id, root, reference, record, seed=42, record_name=f"seed_e2e_{model_id}")
