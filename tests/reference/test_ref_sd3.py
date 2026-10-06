"""Stable Diffusion 3.5 parity against local gated snapshots on CPU fp32."""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion_3 import (
    StableDiffusion3Components,
    decode_latents,
    denoise,
)
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.sd3_transformer import SD3Transformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_IDS = ("sd3-medium", "sd3.5-medium", "sd3.5-large", "sd3.5-large-turbo")
PROMPT = "a photograph of an astronaut riding a horse"
PROMPTS = [PROMPT, "Bir kedi, yağlı boya; İSTANBUL'da ĞÜŞİÖÇ ığüşöç", ""]
TEST_STEPS = {"sd3-medium": 12, "sd3.5-medium": 12, "sd3.5-large": 8, "sd3.5-large-turbo": 4}
TOLERANCES = {
    "text_encoder": 0.0,
    "text_encoder_2": 0.0,
    "text_encoder_3": 0.0,
    "transformer": 0.0,
    "vae": 0.0,
    "scheduler": 0.0,
    "e2e_latents": 0.0,
    "e2e_pixels": 0,
}


def _record_name(component: str, model_id: str) -> str:
    return f"sd3_{component}" if model_id == "sd3.5-medium" else f"sd3_{component}_{model_id}"


def _check(record, name, actual, expected, tolerance_key):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=TOLERANCES[tolerance_key])
    torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index_files = list(directory.glob("*.safetensors.index.json"))
    files: list[Path]
    if index_files:
        weight_map = json.loads(index_files[0].read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = list(directory.glob("*.safetensors"))
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_headers(model_id, record):
    root = snapshot_dir(model_id)
    result = {}
    classes = {
        "text_encoder": CLIPTextModelWithProjection,
        "text_encoder_2": CLIPTextModelWithProjection,
        "text_encoder_3": T5EncoderModel,
        "transformer": SD3Transformer2DModel,
        "vae": AutoencoderKL,
    }
    for component, model_class in classes.items():
        with torch.device("meta"):
            model = model_class(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
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
            "tensors": len(shapes),
        }
    record(f"header_{model_id}", **result)
    assert all(
        not value[key] for value in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_tokenizers(model_id, record):
    root = snapshot_dir(model_id)
    mismatches = []
    for index, name in enumerate(("tokenizer", "tokenizer_2"), start=1):
        ours = CLIPTokenizer(root / name)
        reference = transformers.CLIPTokenizer.from_pretrained(root / name, local_files_only=True)
        for prompt in PROMPTS:
            expected = reference(
                prompt, padding="max_length", max_length=77, truncation=True
            ).input_ids
            if ours(prompt) != expected:
                mismatches.append({"tokenizer": index, "prompt": prompt})
    ours_t5 = T5Tokenizer(root / "tokenizer_3")
    reference_t5 = transformers.T5TokenizerFast.from_pretrained(
        root / "tokenizer_3", local_files_only=True
    )
    actual_ids, actual_mask = ours_t5(PROMPTS, 256)
    expected_t5 = reference_t5(
        PROMPTS, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
    )
    if not torch.equal(actual_ids, expected_t5.input_ids):
        mismatches.append({"tokenizer": 3, "field": "input_ids"})
    if not torch.equal(actual_mask, expected_t5.attention_mask.bool()):
        mismatches.append({"tokenizer": 3, "field": "attention_mask"})
    record(
        _record_name("tokenizers", model_id),
        prompts=len(PROMPTS),
        tokenizers=3,
        mismatches=mismatches,
        tolerance="exact",
    )
    assert not mismatches, mismatches


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("component", ("text_encoder", "text_encoder_2"))
def test_clip_text_encoders(model_id, component, record):
    root = snapshot_dir(model_id)
    tokenizer = CLIPTokenizer(root / component.replace("text_encoder", "tokenizer"))
    input_ids = torch.tensor([tokenizer(prompt) for prompt in PROMPTS])
    ours = CLIPTextModelWithProjection.from_pretrained(
        root / component, device=CPU, dtype=torch.float32
    )
    reference = transformers.CLIPTextModelWithProjection.from_pretrained(
        root / component, local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        actual_pooled, _, actual_hidden = ours(input_ids, output_hidden_states=True)
        expected = reference(input_ids, output_hidden_states=True)
    hidden_diff = max_diff(actual_hidden[-2], expected.hidden_states[-2])
    pooled_diff = max_diff(actual_pooled, expected.text_embeds)
    worst = max((hidden_diff, pooled_diff), key=lambda value: value["max_abs"])
    record(
        _record_name(component, model_id),
        **worst,
        hidden_states=hidden_diff,
        pooled=pooled_diff,
        tolerance=TOLERANCES[component],
    )
    torch.testing.assert_close(
        actual_hidden[-2], expected.hidden_states[-2], atol=TOLERANCES[component], rtol=0
    )
    torch.testing.assert_close(
        actual_pooled, expected.text_embeds, atol=TOLERANCES[component], rtol=0
    )
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_t5_text_encoder(model_id, record):
    root = snapshot_dir(model_id)
    tokenizer = T5Tokenizer(root / "tokenizer_3")
    input_ids, _ = tokenizer(PROMPTS, 256)
    ours = T5EncoderModel.from_pretrained(root / "text_encoder_3", device=CPU, dtype=torch.float32)
    reference = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder_3", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(input_ids)[0]
        expected = reference(input_ids).last_hidden_state
    _check(record, _record_name("text_encoder_3", model_id), actual, expected, "text_encoder_3")
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_transformer(model_id, record):
    root = snapshot_dir(model_id)
    ours = SD3Transformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.SD3Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(0)
    sample = torch.randn(1, 16, 128, 128, generator=generator)
    context = torch.randn(1, 333, 4096, generator=generator)
    pooled = torch.randn(1, 2048, generator=generator)
    pairs = []
    with torch.inference_mode():
        for value in (1000.0, 500.0):
            timestep = torch.tensor([value])
            actual = ours(sample, context, pooled, timestep)
            expected = reference(
                sample,
                encoder_hidden_states=context,
                pooled_projections=pooled,
                timestep=timestep,
            ).sample
            pairs.append((actual, expected, max_diff(actual, expected)))
    worst = max((measurement for _, _, measurement in pairs), key=lambda value: value["max_abs"])
    record(
        _record_name("transformer", model_id),
        **worst,
        timesteps=[1000.0, 500.0],
        size=[1024, 1024],
        tolerance=TOLERANCES["transformer"],
    )
    for actual, expected, _ in pairs:
        torch.testing.assert_close(actual, expected, atol=TOLERANCES["transformer"], rtol=0)
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_vae(model_id, record):
    root = snapshot_dir(model_id)
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 16, 128, 128, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    with torch.inference_mode():
        actual_decode, expected_decode = ours.decode(latents), reference.decode(latents).sample
        actual_encode = ours.encode(image)
        expected_encode = reference.encode(image).latent_dist.mean
    _check(record, _record_name("vae_decode", model_id), actual_decode, expected_decode, "vae")
    _check(record, _record_name("vae_encode", model_id), actual_encode, expected_encode, "vae")
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_scheduler(model_id, record):
    root = snapshot_dir(model_id)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    steps = int(MODEL_CATALOG[model_id].config["defaults"]["steps"])
    actual = FlowMatchEulerDiscreteScheduler(config)
    expected = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    actual.set_timesteps(steps)
    expected.set_timesteps(steps)
    record(
        _record_name("scheduler", model_id),
        timesteps=max_diff(actual.timesteps, expected.timesteps),
        sigmas=max_diff(actual.sigmas, expected.sigmas),
        steps=steps,
        tolerance=TOLERANCES["scheduler"],
    )
    torch.testing.assert_close(actual.timesteps, expected.timesteps, atol=0, rtol=0)
    torch.testing.assert_close(actual.sigmas, expected.sigmas, atol=0, rtol=0)


def _reference_output(root, *, steps, guidance, noise, negative_prompt=None):
    reference = diffusers.StableDiffusion3Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        latents = reference(
            PROMPT,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=1024,
            width=1024,
            max_sequence_length=256,
            latents=noise.clone(),
            output_type="latent",
        ).images
        decoded = reference.vae.decode(
            latents / reference.vae.config.scaling_factor + reference.vae.config.shift_factor
        ).sample
        pixels = reference.image_processor.postprocess(decoded, output_type="np")
    pixels = torch.from_numpy((pixels * 255).round().astype("uint8"))
    del reference
    gc.collect()
    return latents, pixels


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(model_id, record):
    root = snapshot_dir(model_id)
    steps = TEST_STEPS[model_id]
    guidance = float(MODEL_CATALOG[model_id].config["defaults"]["guidance"])
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(1234))
    expected_latents, expected_pixels = _reference_output(
        root, steps=steps, guidance=guidance, noise=noise
    )
    ours = StableDiffusion3Components.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=None,
        steps=steps,
        guidance=guidance,
        latents=noise.clone(),
        max_sequence_length=256,
    )
    actual_pixels = decode_latents(ours.vae, actual_latents)
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
        runtime_note=(
            "Model default."
            if steps == MODEL_CATALOG[model_id].config["defaults"]["steps"]
            else (
                f"Reduced from the {MODEL_CATALOG[model_id].config['defaults']['steps']}-step "
                "model default."
            )
        ),
    )
    assert passed, (latent_diff, pixel_diff)
    del ours
    gc.collect()


def test_end_to_end_guided_large_turbo(record):
    model_id = "sd3.5-large-turbo"
    root = snapshot_dir(model_id)
    steps = 4
    guidance = 3.5
    negative_prompt = "blurry"
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(1234))
    expected_latents, expected_pixels = _reference_output(
        root,
        steps=steps,
        guidance=guidance,
        noise=noise,
        negative_prompt=negative_prompt,
    )
    ours = StableDiffusion3Components.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        latents=noise.clone(),
        max_sequence_length=256,
    )
    actual_pixels = decode_latents(ours.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = (
        latent_diff["max_abs"] <= TOLERANCES["e2e_latents"]
        and pixel_diff <= TOLERANCES["e2e_pixels"]
    )
    record(
        f"e2e_guided_{model_id}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        size=[1024, 1024],
        noise_seed=1234,
        runtime_note="Model default step count with classifier-free guidance.",
    )
    assert passed, (latent_diff, pixel_diff)
    del ours
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(model_id, record):
    root = snapshot_dir(model_id)
    steps = TEST_STEPS[model_id]
    guidance = float(MODEL_CATALOG[model_id].config["defaults"]["guidance"])
    reference = diffusers.StableDiffusion3Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=1024,
            width=1024,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    del reference
    gc.collect()
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT,
            seed=42,
            steps=steps,
            guidance=guidance,
            width=1024,
            height=1024,
            max_sequence_length=256,
        )
        .images[0]
        .tensor
    )
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        pixels_max_abs=pixel_diff,
        seed=42,
        steps=steps,
        guidance=guidance,
        size=[1024, 1024],
        passed=pixel_diff <= TOLERANCES["e2e_pixels"],
        runtime_note=(
            "Model default."
            if steps == MODEL_CATALOG[model_id].config["defaults"]["steps"]
            else (
                f"Reduced from the {MODEL_CATALOG[model_id].config['defaults']['steps']}-step "
                "model default."
            )
        ),
    )
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
