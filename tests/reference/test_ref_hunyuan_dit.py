"""HunyuanDiT parity against the local distilled snapshot on CPU float32."""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.hunyuan_dit import HunyuanDiTComponents, decode_latents, denoise
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.bert import BertModel
from imagehub.models.hunyuan_dit_transformer import HunyuanDiT2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "hunyuan-dit-v1.1-distilled"
# The v1.2 repositories share the text encoders, tokenizers and VAE with v1.1, so those component
# tests run once. End-to-end runs of the v1.2 releases use fewer steps.
MODEL_STEPS = {
    "hunyuan-dit-v1.1-distilled": 25,
    "hunyuan-dit-v1.2": 10,
    "hunyuan-dit-v1.2-distilled": 10,
}
MODEL_IDS = list(MODEL_STEPS)
PROMPT = "a photograph of an astronaut riding a horse"
PROMPTS = [PROMPT, "一个宇航员在骑马", ""]


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is None:
        files = list(directory.glob("*.safetensors"))
    else:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_headers(record, model_id):
    root = snapshot_dir(model_id)
    result = {}
    for component, cls in {
        "text_encoder": BertModel,
        "text_encoder_2": T5EncoderModel,
        "transformer": HunyuanDiT2DModel,
        "vae": AutoencoderKL,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    record("header_hunyuan_dit" if model_id == MODEL_ID else f"header_{model_id}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_mt5_tokens_masks_and_encoder(record):
    root = snapshot_dir(MODEL_ID)
    ours_tokenizer = T5Tokenizer(root / "tokenizer_2")
    reference_tokenizer = transformers.T5Tokenizer.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    input_ids, attention_mask = ours_tokenizer(PROMPTS, 256)
    expected_tokens = reference_tokenizer(
        PROMPTS,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    ids_match = torch.equal(input_ids, expected_tokens.input_ids)
    masks_match = torch.equal(attention_mask, expected_tokens.attention_mask.bool())
    record(
        "hunyuan_dit_mt5_tokenizer",
        ids_match=ids_match,
        masks_match=masks_match,
        prompts=PROMPTS,
        tolerance="exact",
    )
    assert ids_match and masks_match

    ours = T5EncoderModel.from_pretrained(root / "text_encoder_2", device=CPU, dtype=torch.float32)
    reference = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder_2", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(input_ids, attention_mask)[0]
        expected = reference(input_ids, attention_mask=attention_mask).last_hidden_state
    _check(record, "hunyuan_dit_mt5_encoder", actual, expected)
    del ours, reference
    gc.collect()


def test_vae(record):
    root = snapshot_dir(MODEL_ID)
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 4, 128, 128, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    with torch.inference_mode():
        actual_decode = ours.decode(latents)
        expected_decode = reference.decode(latents).sample
        actual_encode = ours.encode(image)
        expected_encode = reference.encode(image).latent_dist.mean
    _check(record, "hunyuan_dit_vae_decode", actual_decode, expected_decode)
    _check(record, "hunyuan_dit_vae_encode", actual_encode, expected_encode)
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_scheduler_timesteps(record, model_id):
    root = snapshot_dir(model_id)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    ours = DDPMScheduler(config)
    reference = diffusers.DDPMScheduler.from_config(config)
    ours.set_timesteps(25)
    reference.set_timesteps(25)
    exact = torch.equal(ours.timesteps, reference.timesteps.cpu())
    name = "hunyuan_dit_scheduler" if model_id == MODEL_ID else f"hunyuan_dit_scheduler_{model_id}"
    record(name, exact=exact, steps=25, tolerance="exact")
    assert exact


def _reference_output(root, steps, *, latents=None, seed=42):
    reference = diffusers.HunyuanDiTPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": steps,
        "guidance_scale": 5.0,
        "height": 1024,
        "width": 1024,
        "generator": torch.Generator("cpu").manual_seed(seed),
    }
    if latents is not None:
        kwargs["latents"] = latents.clone()
    expected_latents = reference(**kwargs, output_type="latent").images
    kwargs["generator"] = torch.Generator("cpu").manual_seed(seed)
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    del reference
    gc.collect()
    return expected_latents, expected_pixels


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end_fixed_latents(record, model_id):
    root = snapshot_dir(model_id)
    steps = MODEL_STEPS[model_id]
    noise = torch.randn(1, 4, 128, 128, generator=torch.Generator("cpu").manual_seed(7))
    expected_latents, expected_pixels = _reference_output(root, steps, latents=noise)
    components = HunyuanDiTComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=steps,
        guidance=5.0,
        guidance_rescale=0.0,
        width=1024,
        height=1024,
        latents=noise.clone(),
        generator=torch.Generator("cpu").manual_seed(42),
    )
    actual_pixels = decode_latents(components.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    record(
        f"e2e_{model_id}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=latent_diff["max_abs"] == 0 and pixel_diff == 0,
        prompt=PROMPT,
        steps=steps,
        guidance=5.0,
        size=[1024, 1024],
        noise_seed=7,
        scheduler_seed=42,
    )
    assert latent_diff["max_abs"] == 0 and pixel_diff == 0
    del components
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(record, model_id):
    root = snapshot_dir(model_id)
    steps = MODEL_STEPS[model_id]
    _, expected = _reference_output(root, steps, seed=42)
    actual = pipeline("t2i", backend=model_id, model=root)(PROMPT, seed=42, steps=steps)
    pixels = torch.stack([image.tensor for image in actual.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=42,
        steps=steps,
    )
    assert passed
