"""Segmind-Vega parity against diffusers 0.40.0 StableDiffusionXLPipeline on CPU fp32.

Real-checkpoint tests require ``IMAGEHUB_SNAPSHOT_SEGMIND_VEGA``. The random-weight and tiny
pipeline tests need only the opt-in reference dependencies.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir, write_tiny_sdxl

from imagehub import pipeline
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.euler import EulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parents[1] / "fixtures"
PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE_PROMPT = "blurry, low quality"
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "text_encoder_2": CLIPTextModelWithProjection,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
TINY_VEGA_UNET = {
    "transformer_layers_per_block": [1, 1, 2],
    "mid_block_type": "UNetMidBlock2D",
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


def _header_result(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    published = json.loads((FIXTURES / "segmind_vega_tensor_shapes.json").read_text())
    for component, cls in COMPONENTS.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        filename = (
            "model.safetensors"
            if component.startswith("text_encoder")
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
    return result


def test_published_tensor_headers_match_models(record):
    result = _header_result(snapshot_dir("segmind-vega"))
    record("header_segmind-vega", **result)
    for entry in result.values():
        assert not entry["missing"]
        assert not entry["extra"]
        assert not entry["mismatch"]


@pytest.mark.parametrize("published", [False, True], ids=["tiny", "published"])
def test_unet_random_weights_match_diffusers_exactly(published):
    if published:
        config = json.loads((FIXTURES / "segmind_vega_configs/unet.json").read_text())
        sizes = [(8, 8)]
    else:
        config = {
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": [32, 64, 64],
            "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
            "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
            "transformer_layers_per_block": [1, 1, 2],
            "attention_head_dim": [2, 4, 4],
            "cross_attention_dim": 64,
            "use_linear_projection": True,
            "addition_embed_type": "text_time",
            "addition_time_embed_dim": 8,
            "projection_class_embeddings_input_dim": 80,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "mid_block_type": "UNetMidBlock2D",
        }
        sizes = [(16, 16), (13, 9)]
    torch.manual_seed(0)
    reference = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    context = torch.randn(1, 7, config["cross_attention_dim"])
    text_dim = (
        config["projection_class_embeddings_input_dim"] - 6 * config["addition_time_embed_dim"]
    )
    text_embeds = torch.randn(1, text_dim)
    time_ids = torch.randn(1, 6)
    for height, width in sizes:
        sample = torch.randn(1, 4, height, width)
        with torch.inference_mode():
            actual = ours(
                sample,
                torch.tensor(500),
                context,
                text_embeds=text_embeds,
                time_ids=time_ids,
            )
            expected = reference(
                sample,
                torch.tensor(500),
                context,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("guidance", "negative_prompt", "size"),
    [(5.0, NEGATIVE_PROMPT, (32, 48)), (1.0, None, (40, 32))],
    ids=["guided-nonsquare", "no-guidance-nonsquare"],
)
def test_tiny_pipeline_matches_diffusers_exactly(tmp_path, guidance, negative_prompt, size):
    root = write_tiny_sdxl(tmp_path / "tiny-vega", sample_size=4, unet_config=TINY_VEGA_UNET)
    reference = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
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
    result = pipeline("t2i", backend="segmind-vega", model=root)(
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


@pytest.fixture(scope="module")
def real_pipelines():
    root = snapshot_dir("segmind-vega")
    ours = StableDiffusionXLComponents.load(root, device=CPU, dtype=torch.float32)
    reference = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    return root, ours, reference


def test_tokenizers_and_text_encoders(real_pipelines, record):
    _, ours, reference = real_pipelines
    prompts = [PROMPT, "Bir kedi İstanbul'da", "emoji 🐱 and   spaces", ""]
    for index, (ours_tokenizer, ref_tokenizer) in enumerate(
        ((ours.tokenizer, reference.tokenizer), (ours.tokenizer_2, reference.tokenizer_2)), 1
    ):
        for prompt in prompts:
            expected_ids = ref_tokenizer(
                prompt, padding="max_length", max_length=77, truncation=True
            )["input_ids"]
            assert ours_tokenizer(prompt) == expected_ids, (index, prompt)
    record(
        "sdxl_tokenizers_segmind-vega",
        prompts=len(prompts),
        tokenizers=2,
        mismatches=[],
        tolerance="exact",
    )
    ids_1 = torch.tensor([ours.tokenizer(prompt) for prompt in prompts])
    ids_2 = torch.tensor([ours.tokenizer_2(prompt) for prompt in prompts])
    with torch.inference_mode():
        _, ours_hidden_1 = ours.text_encoder(ids_1, output_hidden_states=True)
        expected_1 = reference.text_encoder(ids_1, output_hidden_states=True)
        ours_pooled_2, _, ours_hidden_2 = ours.text_encoder_2(ids_2, output_hidden_states=True)
        expected_2 = reference.text_encoder_2(ids_2, output_hidden_states=True)
    _exact(
        record,
        "sdxl_text_encoder_segmind-vega",
        ours_hidden_1[-2],
        expected_1.hidden_states[-2],
    )
    _exact(
        record,
        "sdxl_text_encoder_2_segmind-vega",
        ours_hidden_2[-2],
        expected_2.hidden_states[-2],
    )
    torch.testing.assert_close(ours_pooled_2, expected_2.text_embeds, atol=0, rtol=0)


def test_unet_vae_and_scheduler(real_pipelines, record):
    _, ours, reference = real_pipelines
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 4, 16, 16, generator=generator)
    context = torch.randn(1, 77, 2048, generator=generator)
    text_embeds = torch.randn(1, 1280, generator=generator)
    time_ids = torch.randn(1, 6, generator=generator)
    with torch.inference_mode():
        actual = ours.unet(
            sample,
            torch.tensor(500),
            context,
            text_embeds=text_embeds,
            time_ids=time_ids,
        )
        expected = reference.unet(
            sample,
            torch.tensor(500),
            context,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample
    _exact(record, "sdxl_unet_segmind-vega", actual, expected, timesteps=[500])
    latents = torch.randn(1, 4, 8, 8, generator=generator)
    with torch.inference_mode():
        actual_decode = ours.vae.decode(latents)
        expected_decode = reference.vae.decode(latents).sample
    _exact(record, "sdxl_vae_decode_segmind-vega", actual_decode, expected_decode)
    image = torch.rand(1, 3, 64, 64, generator=generator) * 2 - 1
    with torch.inference_mode():
        actual_encode = ours.vae.encode(image)
        expected_encode = reference.vae.encode(image).latent_dist.mean
    _exact(record, "sdxl_vae_encode_segmind-vega", actual_encode, expected_encode)
    ours_scheduler = EulerDiscreteScheduler(ours.scheduler_config)
    reference_scheduler = diffusers.EulerDiscreteScheduler.from_config(reference.scheduler.config)
    ours_scheduler.set_timesteps(4)
    reference_scheduler.set_timesteps(4)
    _exact(
        record,
        "sdxl_scheduler_segmind-vega",
        ours_scheduler.sigmas,
        reference_scheduler.sigmas,
        steps=[4],
    )


def _e2e(root, reference, record, *, seed, record_name):
    steps = 4
    size = 64
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=5.0,
            height=size,
            width=size,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = (
        pipeline("t2i", backend="segmind-vega", model=root)(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seed=seed,
            steps=steps,
            guidance=5.0,
            height=size,
            width=size,
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
        steps=steps,
        guidance=5.0,
        size=[size, size],
        tolerance=0,
        runtime_note="Four 64x64 steps keep the real-checkpoint comparison short.",
    )
    assert passed


def test_end_to_end_seed_0(real_pipelines, record):
    root, _, reference = real_pipelines
    _e2e(root, reference, record, seed=0, record_name="e2e_segmind-vega")


def test_end_to_end_seed_42(real_pipelines, record):
    root, _, reference = real_pipelines
    _e2e(root, reference, record, seed=42, record_name="seed_e2e_segmind-vega")
