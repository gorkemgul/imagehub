"""Krea 2 Turbo parity on CPU float32."""

import gc
import json
import struct
from types import SimpleNamespace

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.krea2 import (
    Krea2Components,
    _load_text_encoder,
    calculate_shift,
    denoise,
    encode_prompt,
    pack_latents,
    prepare_position_ids,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.krea2_transformer import Krea2Transformer2DModel
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"
PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
SELECTED = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


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


def test_header(record):
    root = snapshot_dir("krea2-turbo")
    result = {}
    for component, cls in {
        "transformer": Krea2Transformer2DModel,
        "vae": AutoencoderKLQwenImage,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in shapes.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    text_config = read_json(root / "text_encoder/config.json")["text_config"]
    with torch.device("meta"):
        text = Qwen3VLTextModel(text_config)
    published = _checkpoint_shapes(root / "text_encoder")
    prefix = next(
        item
        for item in ("language_model.", "model.language_model.")
        if any(name.startswith(item) for name in published)
    )
    actual_text = {
        name.removeprefix(prefix): shape
        for name, shape in published.items()
        if name.startswith(prefix)
    }
    expected_text = {name: tuple(value.shape) for name, value in text.state_dict().items()}
    result["text_encoder"] = {
        "missing": sorted(expected_text.keys() - actual_text.keys()),
        "extra": sorted(actual_text.keys() - expected_text.keys()),
        "mismatch": sorted(
            name
            for name in expected_text.keys() & actual_text.keys()
            if expected_text[name] != actual_text[name]
        ),
        "tensors": len(published),
    }
    record("header_krea2_turbo", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_prompt_template_tokens_and_text_encoder(record, tmp_path):
    root = snapshot_dir("krea2-turbo")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    text_tokens = tokenizer(
        [PREFIX + PROMPT],
        truncation=True,
        padding="max_length",
        max_length=512 + 34 - 5,
        return_tensors="pt",
    )
    suffix_tokens = tokenizer([SUFFIX], return_tensors="pt")
    expected_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    expected_mask = torch.cat(
        [text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1
    ).bool()
    positions = (expected_mask.long().cumsum(-1) - 1).clamp(min=0)
    position_ids = positions.unsqueeze(0).expand(3, -1, -1)
    reference = transformers.Qwen3VLModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        outputs = reference(
            input_ids=expected_ids,
            attention_mask=expected_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        expected_hidden = torch.stack([outputs.hidden_states[index] for index in SELECTED], dim=2)
        expected_hidden = expected_hidden[:, 34:]
    torch.save(expected_hidden, tmp_path / "text.pt")
    del reference, outputs, expected_hidden
    gc.collect()

    native_tokenizer = Qwen2Tokenizer(root / "tokenizer")
    ids, mask = native_tokenizer(PREFIX + PROMPT, 512 + 34 - 5)
    suffix_ids = torch.tensor([native_tokenizer.tokenize(SUFFIX)])
    ids = torch.cat([ids, suffix_ids], dim=1)
    mask = torch.cat([mask, torch.ones_like(suffix_ids, dtype=torch.bool)], dim=1)
    tokens_match = torch.equal(ids, expected_ids) and torch.equal(mask, expected_mask)
    record("krea2_prompt_tokens", exact=tokens_match, tolerance="exact")
    assert tokens_match
    native_text = _load_text_encoder(root / "text_encoder", device=CPU, dtype=torch.float32)
    components = SimpleNamespace(
        tokenizer=native_tokenizer,
        text_encoder=native_text,
        text_encoder_select_layers=SELECTED,
    )
    actual_hidden, actual_mask = encode_prompt(components, PROMPT, CPU, 512)
    expected_hidden = torch.load(tmp_path / "text.pt", weights_only=True)
    _check(record, "krea2_text_encoder", actual_hidden, expected_hidden)
    assert torch.equal(actual_mask, expected_mask[:, 34:])
    del native_text, components
    gc.collect()


def test_transformer_single_step_1024(record, tmp_path):
    root = snapshot_dir("krea2-turbo")
    config = read_json(root / "transformer/config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 4096, config["in_channels"], generator=generator)
    context = torch.randn(1, 512, 12, config["text_hidden_dim"], generator=generator)
    mask = torch.ones(1, 512, dtype=torch.bool)
    positions = prepare_position_ids(512, 64, 64, CPU)
    timestep = torch.tensor([0.5])
    reference = diffusers.Krea2Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            position_ids=positions,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
    torch.save(expected, tmp_path / "transformer.pt")
    del reference, expected
    gc.collect()
    native = Krea2Transformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(hidden, context, timestep, positions, mask)
    expected = torch.load(tmp_path / "transformer.pt", weights_only=True)
    _check(record, "krea2_transformer", actual, expected)
    del native
    gc.collect()


def test_vae_decode_and_encode(record, tmp_path):
    root = snapshot_dir("krea2-turbo")
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 16, 1, 128, 128, generator=generator)
    image = torch.rand(1, 3, 1, 1024, 1024, generator=generator) * 2 - 1
    reference = diffusers.AutoencoderKLQwenImage.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = {
            "decode": reference.decode(latents).sample,
            "encode": reference.encode(image).latent_dist.mode(),
        }
    torch.save(expected, tmp_path / "vae.pt")
    del reference, expected
    gc.collect()
    native = AutoencoderKLQwenImage.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_decode = native.decode(latents)
        actual_encode = native.encode(image)
    expected = torch.load(tmp_path / "vae.pt", weights_only=True)
    _check(record, "krea2_vae_decode", actual_decode, expected["decode"])
    _check(record, "krea2_vae_encode", actual_encode, expected["encode"])
    del native
    gc.collect()


def test_scheduler_including_mu(record):
    root = snapshot_dir("krea2-turbo")
    config = read_json(root / "scheduler/scheduler_config.json")
    sigmas = [1.0, 0.75, 0.5, 0.25]
    mu = calculate_shift(
        4096,
        int(config["base_image_seq_len"]),
        int(config["max_image_seq_len"]),
        float(config["base_shift"]),
        float(config["max_shift"]),
    )
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=sigmas, mu=mu)
    expected_sigmas = reference.sigmas.cpu()
    expected_timesteps = reference.timesteps.cpu()
    del reference
    gc.collect()
    native = FlowMatchEulerDiscreteScheduler(config)
    native.set_timesteps(4, sigmas=sigmas, mu=mu)
    exact = torch.equal(native.sigmas, expected_sigmas) and torch.equal(
        native.timesteps, expected_timesteps
    )
    record("krea2_scheduler", mu=mu, exact=exact, tolerance="exact")
    assert exact


def test_end_to_end_defaults_with_fixed_noise(record, tmp_path):
    root = snapshot_dir("krea2-turbo")
    raw = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(42))
    noise = pack_latents(raw)
    reference = diffusers.Krea2Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 512,
        "latents": noise.clone(),
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, tmp_path / "end-to-end.pt")
    del reference, expected_latents, expected_pixels
    gc.collect()
    components = Krea2Components.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=8,
        guidance=0.0,
        width=1024,
        height=1024,
        latents=noise.clone(),
        max_sequence_length=512,
    )
    expected = torch.load(tmp_path / "end-to-end.pt", weights_only=True)
    _check(record, "krea2_e2e_latents", actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend="krea2-turbo", model=root)(PROMPT, latents=noise)
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    record(
        "e2e_krea2_turbo",
        passed=passed,
        max_pixel_diff=int((pixels.int() - expected["pixels"].int()).abs().max()),
        steps=8,
    )
    assert passed


def test_seed_end_to_end_42(record, tmp_path):
    root = snapshot_dir("krea2-turbo")
    reference = diffusers.Krea2Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        PROMPT,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    torch.save(expected, tmp_path / "seed.pt")
    del reference, expected
    gc.collect()
    actual = pipeline("t2i", backend="krea2-turbo", model=root)(PROMPT, seed=42).images[0].tensor
    expected = torch.load(tmp_path / "seed.pt", weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        "seed_e2e_krea2_turbo",
        passed=passed,
        max_pixel_diff=int((actual.int() - expected.int()).abs().max()),
        seed=42,
        steps=8,
    )
    assert passed
