"""CogView4 parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.cogview4 import (
    CogView4Components,
    calculate_shift,
    cogview4_sigmas,
    denoise,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.cogview4_transformer import CogView4Transformer2DModel
from imagehub.models.glm import GlmModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_glm import GlmTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "cogview4-6b"
PROMPT = "a photograph of an astronaut riding a horse"


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = [next(directory.glob("*.safetensors"))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _reference_tokens(tokenizer, prompt):
    encoded = tokenizer(
        [prompt],
        padding="longest",
        max_length=1024,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    pad_length = (-encoded.input_ids.shape[1]) % 16
    if pad_length:
        padding = torch.full((1, pad_length), tokenizer.pad_token_id, dtype=torch.long)
        encoded.input_ids = torch.cat([padding, encoded.input_ids], dim=1)
    return encoded.input_ids


def test_header(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": GlmModel,
        "transformer": CogView4Transformer2DModel,
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
    record("header_cogview4-6b", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokens_and_text_encoder(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    reference_tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_ids = _reference_tokens(reference_tokenizer, PROMPT)
    native_tokenizer = GlmTokenizer(root / "tokenizer")
    actual_ids, _ = native_tokenizer([PROMPT], 1024)
    tokens_match = torch.equal(actual_ids, expected_ids)
    record("cogview4_tokens", exact=tokens_match, tolerance="exact")
    assert tokens_match

    reference = transformers.GlmModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(expected_ids, output_hidden_states=True).hidden_states[-2]
    output = tmp_path / "text-encoder.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = GlmModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = native(actual_ids)
    expected = torch.load(output, weights_only=True)
    _check(record, "cogview4_text_encoder", actual, expected)
    del native
    gc.collect()


def test_transformer(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(2, 16, 128, 128, generator=generator)
    context = torch.randn(2, 16, config["text_embed_dim"], generator=generator)
    timestep = torch.tensor([1.0, 753.25])
    original_size = torch.tensor([[1024.0, 1024.0]]).repeat(2, 1)
    target_size = torch.tensor([[1024.0, 1024.0]]).repeat(2, 1)
    crop_coords = torch.zeros(2, 2)
    reference = diffusers.CogView4Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            original_size=original_size,
            target_size=target_size,
            crop_coords=crop_coords,
            return_dict=False,
        )[0]
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = CogView4Transformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(
            hidden,
            context,
            timestep,
            original_size,
            target_size,
            crop_coords,
        )
    expected = torch.load(output, weights_only=True)
    _check(record, "cogview4_transformer", actual, expected)
    del native
    gc.collect()


def test_vae(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 16, 128, 128, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = {
            "decode": reference.decode(latents).sample,
            "encode": reference.encode(image).latent_dist.mode(),
        }
    output = tmp_path / "vae.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_decode = native.decode(latents)
        actual_encode = native.encode(image)
    expected = torch.load(output, weights_only=True)
    _check(record, "cogview4_vae_decode", actual_decode, expected["decode"])
    _check(record, "cogview4_vae_encode", actual_encode, expected["encode"])
    del native
    gc.collect()


def test_scheduler(record):
    root = snapshot_dir(MODEL_ID)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    sigmas = cogview4_sigmas(8, int(config["num_train_timesteps"]))
    mu = calculate_shift(4096, config)
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=sigmas, mu=mu)
    expected_sigmas = reference.sigmas.cpu()
    expected_timesteps = reference.timesteps.cpu()
    native = FlowMatchEulerDiscreteScheduler(config)
    native.set_timesteps(8, sigmas=sigmas, mu=mu)
    exact = torch.equal(native.sigmas, expected_sigmas) and torch.equal(
        native.timesteps, expected_timesteps
    )
    record("cogview4_scheduler", mu=mu, exact=exact, tolerance="exact")
    assert exact


def test_end_to_end(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(42))
    reference = diffusers.CogView4Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": 8,
        "guidance_scale": 5.0,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 1024,
        "latents": noise.clone(),
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    output = tmp_path / "end-to-end.pt"
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, output)
    del reference, expected_latents, expected_pixels
    gc.collect()
    components = CogView4Components.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=8,
        guidance=5.0,
        width=1024,
        height=1024,
        latents=noise.clone(),
        max_sequence_length=1024,
    )
    expected = torch.load(output, weights_only=True)
    _check(record, "cogview4_e2e_latents", actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, steps=8, guidance=5.0, latents=noise
    )
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    record(
        "e2e_cogview4-6b",
        passed=passed,
        max_pixel_diff=int(
            (pixels.to(torch.int16) - expected["pixels"].to(torch.int16)).abs().max()
        ),
        steps=8,
    )
    assert passed


def test_seed_end_to_end(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    reference = diffusers.CogView4Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        prompt=PROMPT,
        num_inference_steps=8,
        guidance_scale=5.0,
        height=1024,
        width=1024,
        max_sequence_length=1024,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=42, steps=8, guidance=5.0)
        .images[0]
        .tensor
    )
    expected = torch.load(output, weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        "seed_e2e_cogview4-6b",
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        seed=42,
        steps=8,
    )
    assert passed
