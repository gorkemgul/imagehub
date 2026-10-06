"""Parity of native Z-Image components and pipeline on CPU fp32."""

import gc

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.z_image import ZImageComponents, decode_latents, denoise
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.qwen3 import Qwen3Model
from imagehub.models.z_image_transformer import ZImageTransformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, z_image_sigmas
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_IDS = ("z-image-turbo", "z-image")
TEST_STEPS = {"z-image-turbo": 9, "z-image": 8}
PROMPT = "a photograph of an astronaut riding a horse"
PROMPTS = [PROMPT, "Bir kedi, yağlı boya; İSTANBUL'da ĞÜŞİÖÇ ığüşöç", ""]
TOLERANCES = {
    "text_encoder": 0.0,
    "transformer": 0.0,
    "vae": 0.0,
    "scheduler": 0.0,
    "e2e_latents": 0.0,
    "e2e_pixels": 0,
}


def _record_name(key, model_id):
    return key if model_id == "z-image-turbo" else f"{key}_{model_id}"


def _check(record, name, actual, expected, tolerance_key):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=TOLERANCES[tolerance_key])
    torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_tokenizer(model_id, record):
    root = snapshot_dir(model_id)
    ours = Qwen2Tokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    prompts = [ours.chat_prompt(prompt) for prompt in PROMPTS]
    actual_ids, actual_mask = ours(prompts)
    expected = reference(
        prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    ids_match = torch.equal(actual_ids, expected.input_ids)
    masks_match = torch.equal(actual_mask, expected.attention_mask.bool())
    record(
        _record_name("z_image_tokenizer", model_id),
        prompts=len(PROMPTS),
        ids_match=ids_match,
        masks_match=masks_match,
        tolerance="exact",
    )
    assert ids_match and masks_match


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_text_encoder(model_id, record):
    root = snapshot_dir(model_id)
    tokenizer = Qwen2Tokenizer(root / "tokenizer")
    input_ids, mask = tokenizer([tokenizer.chat_prompt(prompt) for prompt in PROMPTS])
    ours = Qwen3Model.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    reference = transformers.Qwen3Model.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(input_ids, mask, output_hidden_states=True)[1][-2]
        expected = reference(
            input_ids=input_ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-2]
    _check(
        record,
        _record_name("z_image_text_encoder", model_id),
        actual,
        expected,
        "text_encoder",
    )
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_transformer(model_id, record):
    root = snapshot_dir(model_id)
    ours = ZImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.ZImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(0)
    sample = torch.randn(1, 16, 1, 128, 128, generator=generator)
    context = [torch.randn(512, 2560, generator=generator)]
    pairs = []
    with torch.inference_mode():
        for value in (0.5, 0.25):
            timesteps = torch.tensor([value])
            actual = ours([sample[0]], timesteps, context)[0]
            expected = reference([sample[0]], timesteps, context, return_dict=False)[0][0]
            pairs.append((actual, expected, max_diff(actual, expected)))
    worst = max((measured for _, _, measured in pairs), key=lambda measured: measured["max_abs"])
    record(
        _record_name("z_image_transformer", model_id),
        **worst,
        timesteps=[0.5, 0.25],
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
    _check(
        record,
        _record_name("z_image_vae_decode", model_id),
        actual_decode,
        expected_decode,
        "vae",
    )
    _check(
        record,
        _record_name("z_image_vae_encode", model_id),
        actual_encode,
        expected_encode,
        "vae",
    )
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_scheduler(model_id, record):
    root = snapshot_dir(model_id)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    steps = TEST_STEPS[model_id]
    actual = FlowMatchEulerDiscreteScheduler(config)
    expected = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    actual.set_timesteps(steps, sigmas=z_image_sigmas(steps))
    expected.set_timesteps(steps, sigmas=z_image_sigmas(steps))
    record(
        _record_name("z_image_scheduler", model_id),
        timesteps=max_diff(actual.timesteps, expected.timesteps),
        sigmas=max_diff(actual.sigmas, expected.sigmas),
        tolerance=TOLERANCES["scheduler"],
    )
    torch.testing.assert_close(actual.timesteps, expected.timesteps, atol=0, rtol=0)
    torch.testing.assert_close(actual.sigmas, expected.sigmas, atol=0, rtol=0)


def _run_end_to_end(model_id, record, *, guidance, negative_prompt, steps, record_name):
    root = snapshot_dir(model_id)
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(1234))
    reference = diffusers.ZImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected_latents = reference(
            PROMPT,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            cfg_normalization=False,
            height=1024,
            width=1024,
            latents=noise.clone(),
            output_type="latent",
        ).images
        expected_pixels = reference(
            PROMPT,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            cfg_normalization=False,
            height=1024,
            width=1024,
            latents=noise.clone(),
            output_type="np",
        ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    del reference
    gc.collect()
    ours = ZImageComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        cfg_normalization=False,
        cfg_truncation=1.0,
        latents=noise.clone(),
        max_sequence_length=512,
    )
    actual_pixels = decode_latents(ours.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = (
        latent_diff["max_abs"] <= TOLERANCES["e2e_latents"]
        and pixel_diff <= TOLERANCES["e2e_pixels"]
    )
    record(
        record_name,
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=guidance,
        size=[1024, 1024],
        noise_seed=1234,
        tolerance={"latents": TOLERANCES["e2e_latents"], "pixels": TOLERANCES["e2e_pixels"]},
    )
    assert passed, (latent_diff, pixel_diff)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(model_id, record):
    guidance = float(MODEL_CATALOG[model_id].config.get("defaults", {}).get("guidance", 0.0))
    _run_end_to_end(
        model_id,
        record,
        guidance=guidance,
        negative_prompt="blurry" if guidance > 0 else None,
        steps=TEST_STEPS[model_id],
        record_name=f"e2e_{model_id}",
    )


def test_guided_end_to_end(record):
    _run_end_to_end(
        "z-image-turbo",
        record,
        guidance=4.0,
        negative_prompt="blurry",
        steps=4,
        record_name="e2e_guided_z-image-turbo",
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(model_id, record):
    root = snapshot_dir(model_id)
    steps = TEST_STEPS[model_id]
    guidance = float(MODEL_CATALOG[model_id].config.get("defaults", {}).get("guidance", 0.0))
    negative_prompt = "blurry" if guidance > 0 else None
    reference = diffusers.ZImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            cfg_normalization=False,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    del reference
    gc.collect()
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT,
            negative_prompt=negative_prompt,
            seed=42,
            steps=steps,
            guidance=guidance,
            cfg_normalization=False,
            width=1024,
            height=1024,
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
    )
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
