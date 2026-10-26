"""BK-SDM parity against diffusers 0.40.0 StableDiffusionPipeline on CPU fp32.

Real-checkpoint tests use one snapshot variable per release:
``IMAGEHUB_SNAPSHOT_BK_SDM_BASE``, ``IMAGEHUB_SNAPSHOT_BK_SDM_SMALL`` and
``IMAGEHUB_SNAPSHOT_BK_SDM_TINY``. Random-weight and tiny-pipeline tests only need the opt-in
reference dependencies.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir, write_tiny_sd

from imagehub import pipeline
from imagehub.backends.stable_diffusion import StableDiffusionComponents
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.pndm import PNDMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parents[1] / "fixtures"
VARIANTS = ("base", "small", "tiny")
MODEL_IDS = tuple(f"bk-sdm-{variant}" for variant in VARIANTS)
PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE_PROMPT = "blurry, low quality"
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
TINY_CONFIGS = {
    "base": {"layers_per_block": 1},
    "small": {"layers_per_block": 1, "mid_block_type": None},
    "tiny": {
        "block_out_channels": [32, 64, 64],
        "down_block_types": ["CrossAttnDownBlock2D"] * 3,
        "up_block_types": ["CrossAttnUpBlock2D"] * 3,
        "layers_per_block": 1,
        "mid_block_type": None,
    },
}


def _exact(record, name, actual, expected, **details):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0, **details)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


def _load_pair(model_id):
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
    return root, ours, reference


@pytest.mark.parametrize("variant", VARIANTS)
def test_published_tensor_headers_match_models(variant, record):
    model_id = f"bk-sdm-{variant}"
    root = snapshot_dir(model_id)
    published = json.loads(
        (FIXTURES / f"bk_sdm_{variant}_tensor_shapes.json").read_text(encoding="utf-8")
    )
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


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("published", [False, True], ids=["tiny", "published"])
def test_unet_random_weights_match_diffusers_exactly(variant, published):
    if published:
        config = json.loads(
            (FIXTURES / f"bk_sdm_configs/{variant}/unet.json").read_text(encoding="utf-8")
        )
        size = (8, 8)
    else:
        config = {
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": [32, 64, 64, 64],
            "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
            "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
            "layers_per_block": 1,
            "attention_head_dim": 2,
            "cross_attention_dim": 32,
            "norm_num_groups": 8,
            "sample_size": 8,
        } | TINY_CONFIGS[variant]
        size = (13, 9)
    torch.manual_seed(0)
    reference = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    sample = torch.randn(1, 4, *size)
    context = torch.randn(1, 7, config["cross_attention_dim"])
    with torch.inference_mode():
        actual = ours(sample, torch.tensor(500), context)
        expected = reference(sample, torch.tensor(500), context).sample
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    ("guidance", "negative_prompt", "size"),
    [(7.5, NEGATIVE_PROMPT, (64, 80)), (1.0, None, (72, 64))],
    ids=["guided-nonsquare", "no-guidance-nonsquare"],
)
def test_tiny_pipeline_matches_diffusers_exactly(
    tmp_path, variant, guidance, negative_prompt, size
):
    root = write_tiny_sd(
        tmp_path / f"tiny-{variant}", unet_config=TINY_CONFIGS[variant] | {"sample_size": 8}
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
        num_inference_steps=2,
        guidance_scale=guidance,
        height=height,
        width=width,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend=f"bk-sdm-{variant}", model=root)(
        PROMPT,
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        height=height,
        width=width,
        seed=42,
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_tokenizer_and_text_encoder(model_id, record):
    _, ours, reference = _load_pair(model_id)
    prompts = [PROMPT, "Bir kedi İstanbul'da", "emoji 🐱 and   spaces", ""]
    for prompt in prompts:
        expected_ids = reference.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True
        )["input_ids"]
        assert ours.tokenizer(prompt) == expected_ids
    record(f"bk_sdm_tokenizer_{model_id}", prompts=len(prompts), mismatches=[], tolerance="exact")
    ids = torch.tensor([ours.tokenizer(prompt) for prompt in prompts])
    with torch.inference_mode():
        actual = ours.text_encoder(ids)
        expected = reference.text_encoder(ids)[0]
    _exact(record, f"bk_sdm_text_encoder_{model_id}", actual, expected)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_unet_vae_and_scheduler(model_id, record):
    _, ours, reference = _load_pair(model_id)
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 4, 16, 16, generator=generator)
    context = torch.randn(1, 77, 768, generator=generator)
    with torch.inference_mode():
        actual = ours.unet(sample, torch.tensor(500), context)
        expected = reference.unet(sample, torch.tensor(500), context).sample
    _exact(record, f"bk_sdm_unet_{model_id}", actual, expected, timesteps=[500])
    latents = torch.randn(1, 4, 8, 8, generator=generator)
    with torch.inference_mode():
        actual_decode = ours.vae.decode(latents)
        expected_decode = reference.vae.decode(latents).sample
    _exact(record, f"bk_sdm_vae_decode_{model_id}", actual_decode, expected_decode)
    image = torch.rand(1, 3, 64, 64, generator=generator) * 2 - 1
    with torch.inference_mode():
        actual_encode = ours.vae.encode(image)
        expected_encode = reference.vae.encode(image).latent_dist.mean
    _exact(record, f"bk_sdm_vae_encode_{model_id}", actual_encode, expected_encode)
    ours_scheduler = PNDMScheduler(ours.scheduler_config)
    reference_scheduler = diffusers.PNDMScheduler.from_config(reference.scheduler.config)
    ours_scheduler.set_timesteps(4)
    reference_scheduler.set_timesteps(4)
    _exact(
        record,
        f"bk_sdm_scheduler_{model_id}",
        ours_scheduler.timesteps,
        reference_scheduler.timesteps,
        steps=[4],
    )


def _e2e(model_id, record, *, seed, record_name):
    root = snapshot_dir(model_id)
    reference = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=4,
            guidance_scale=7.5,
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
            guidance=7.5,
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
        guidance=7.5,
        size=[64, 64],
        tolerance=0,
        runtime_note="Four 64x64 steps keep the real-checkpoint comparison short.",
    )
    assert passed


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end_seed_0(model_id, record):
    _e2e(model_id, record, seed=0, record_name=f"e2e_{model_id}")


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end_seed_42(model_id, record):
    _e2e(model_id, record, seed=42, record_name=f"seed_e2e_{model_id}")
