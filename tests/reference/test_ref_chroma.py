"""Chroma checkpoint parity against local published snapshots on CPU float32."""

from __future__ import annotations

import gc
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from helpers import FIXTURES, max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.chroma import ChromaComponents, calculate_shift, decode_latents, denoise
from imagehub.checkpoints import read_json, save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.chroma_transformer import ChromaTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, _beta_ppf
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_ID = "chroma1-hd"
MODEL_IDS = ("chroma1-hd", "chroma1-flash", "chroma1-base")
PROMPT = "a photograph of an astronaut riding a horse"
MAX_SEQUENCE_LENGTH = 512


@pytest.fixture(autouse=True)
def _reference_beta_sigmas_without_scipy(monkeypatch):
    """Let Diffusers exercise beta schedules in the dependency-minimal test environment."""
    scheduler_module = diffusers.schedulers.scheduling_flow_match_euler_discrete
    if scheduler_module.is_scipy_available():
        return
    beta = SimpleNamespace(ppf=lambda probability, _alpha, _beta: _beta_ppf(float(probability)))
    scipy = SimpleNamespace(stats=SimpleNamespace(beta=beta))
    monkeypatch.setattr(scheduler_module, "scipy", scipy, raising=False)
    monkeypatch.setattr(scheduler_module, "is_scipy_available", lambda: True)


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


def _fixture_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    """Read shapes from header JSON fixtures (no weight files are present)."""
    shapes = {}
    for path in directory.glob("*_headers.json"):
        header = json.loads(path.read_text(encoding="utf-8"))
        shapes |= {name: tuple(shape) for name, shape in header.items()}
    return shapes


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _tiny_transformer_config() -> dict[str, Any]:
    return {
        "_class_name": "ChromaTransformer2DModel",
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 16,
        "in_channels": 64,
        "approximator_num_channels": 16,
        "approximator_hidden_dim": 32,
        "approximator_layers": 1,
        "axes_dims_rope": [2, 2, 4],
    }


@pytest.mark.parametrize(
    ("model_id", "fixture_name"),
    (
        ("chroma1-hd", "chroma_configs"),
        ("chroma1-flash", "chroma_flash_configs"),
        ("chroma1-base", "chroma_configs"),
    ),
)
def test_headers(record, model_id, fixture_name):
    root = FIXTURES / fixture_name
    result = {}
    for component, cls in {
        "text_encoder": T5EncoderModel,
        "transformer": ChromaTransformer2DModel,
        "vae": AutoencoderKL,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _fixture_shapes(root / component)
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
    record(f"header_{model_id}_fixtures", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_transformer_tiny(record):
    config = _tiny_transformer_config()
    reference = diffusers.ChromaTransformer2DModel(**config)
    state = reference.state_dict()
    ours = ChromaTransformer2DModel(config)
    ours.load_state_dict(state, strict=True, assign=True)
    ours.eval()
    reference.eval()

    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 4, 64, generator=generator)
    context = torch.randn(1, 3, 16, generator=generator)
    img_ids = torch.zeros(4, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(2), torch.arange(2))
    txt_ids = torch.zeros(3, 3)
    timestep = torch.tensor([0.5])
    attention_mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1]], dtype=torch.bool)

    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
        ).sample
        actual = ours(hidden, context, timestep, img_ids, txt_ids, attention_mask)
    _check(record, "chroma_transformer_tiny", actual, expected)
    del reference, ours
    gc.collect()


def test_transformer_published_config_small_size(record):
    root = FIXTURES / "chroma_configs"
    config = read_json(root / "transformer" / "config.json")
    reference = diffusers.ChromaTransformer2DModel.from_config(config)
    state = reference.state_dict()
    ours = ChromaTransformer2DModel(config)
    ours.load_state_dict(state, strict=True, assign=True)
    ours.eval()
    reference.eval()

    generator = torch.Generator("cpu").manual_seed(2)
    hidden = torch.randn(1, 256, 64, generator=generator)
    context = torch.randn(1, 77, 4096, generator=generator)
    img_ids = torch.zeros(256, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(16), torch.arange(16))
    txt_ids = torch.zeros(77, 3)
    timestep = torch.tensor([0.25])
    attention_mask = torch.ones(1, 333, dtype=torch.bool)
    attention_mask[0, 70:] = False

    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
        ).sample
        actual = ours(hidden, context, timestep, img_ids, txt_ids, attention_mask)
    _check(record, "chroma_transformer_published_small", actual, expected)
    del reference, ours
    gc.collect()


@pytest.mark.parametrize(
    ("model_id", "fixture_name"),
    (
        ("chroma1-hd", "chroma_configs"),
        ("chroma1-flash", "chroma_flash_configs"),
        ("chroma1-base", "chroma_base_configs"),
    ),
)
def test_scheduler(record, model_id, fixture_name):
    root = FIXTURES / fixture_name
    config = read_json(root / "scheduler" / "scheduler_config.json")
    sigmas = [1.0, 0.75, 0.5, 0.25]
    mu = calculate_shift(4096, config)
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=sigmas, mu=mu)
    expected_sigmas = reference.sigmas.cpu()
    expected_timesteps = reference.timesteps.cpu()
    del reference
    gc.collect()
    ours = FlowMatchEulerDiscreteScheduler(config)
    ours.set_timesteps(4, sigmas=sigmas, mu=mu)
    exact = torch.equal(ours.sigmas, expected_sigmas) and torch.equal(
        ours.timesteps, expected_timesteps
    )
    record(f"{model_id}_scheduler", mu=mu, exact=exact, tolerance="exact")
    assert exact


def _write_tiny_snapshot(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    transformer_cfg = _tiny_transformer_config()
    transformer_dir = directory / "transformer"
    transformer_dir.mkdir(exist_ok=True)
    (transformer_dir / "config.json").write_text(json.dumps(transformer_cfg), encoding="utf-8")
    reference_transformer = diffusers.ChromaTransformer2DModel(**transformer_cfg)
    save_safetensors(
        reference_transformer.state_dict(),
        transformer_dir / "diffusion_pytorch_model.safetensors",
    )

    vae_cfg = read_json(FIXTURES / "chroma_configs" / "vae" / "config.json")
    vae_dir = directory / "vae"
    vae_dir.mkdir(exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps(vae_cfg), encoding="utf-8")
    reference_vae = diffusers.AutoencoderKL.from_config(vae_cfg)
    save_safetensors(reference_vae.state_dict(), vae_dir / "diffusion_pytorch_model.safetensors")

    text_cfg = {
        "architectures": ["T5EncoderModel"],
        "d_model": 16,
        "d_kv": 4,
        "d_ff": 32,
        "num_layers": 1,
        "num_heads": 4,
        "relative_attention_num_buckets": 4,
        "relative_attention_max_distance": 8,
        "layer_norm_epsilon": 1e-6,
        "feed_forward_proj": "gated-gelu",
        "dense_act_fn": "gelu_new",
        "is_gated_act": True,
        "vocab_size": 32128,
    }
    text_dir = directory / "text_encoder"
    text_dir.mkdir(exist_ok=True)
    (text_dir / "config.json").write_text(json.dumps(text_cfg), encoding="utf-8")
    reference_t5 = transformers.T5EncoderModel(transformers.T5Config(**text_cfg))
    save_safetensors(reference_t5.state_dict(), text_dir / "model.safetensors")

    tokenizer_cfg = {
        "model_type": "t5",
        "vocab_size": 128,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": 77,
        "spiece_model_file": "spiece.model",
        "tokenizer_class": "T5Tokenizer",
    }
    tokenizer_dir = directory / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_cfg), encoding="utf-8"
    )
    (tokenizer_dir / "special_tokens_map.json").write_text(
        json.dumps({"pad_token": "<pad>", "eos_token": "</s>", "unk_token": "<unk>"}),
        encoding="utf-8",
    )
    published_tokenizer = FIXTURES / "chroma_configs" / "tokenizer"
    for item in published_tokenizer.iterdir():
        if item.is_file():
            (tokenizer_dir / item.name).write_bytes(item.read_bytes())

    scheduler_cfg = read_json(FIXTURES / "chroma_configs" / "scheduler" / "scheduler_config.json")
    scheduler_dir = directory / "scheduler"
    scheduler_dir.mkdir(exist_ok=True)
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps(scheduler_cfg), encoding="utf-8"
    )

    model_index = {
        "_class_name": "ChromaPipeline",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5Tokenizer"],
        "transformer": ["diffusers", "ChromaTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (directory / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    return directory


def _reference_pipeline(root: Path):
    reference = diffusers.ChromaPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def test_end_to_end_tiny(record, tmp_path):
    root = _write_tiny_snapshot(tmp_path / "chroma_tiny")
    reference = _reference_pipeline(root)
    noise = torch.randn(1, 16, 16, 16, generator=torch.Generator("cpu").manual_seed(0))
    packed_noise = diffusers.ChromaPipeline._pack_latents(noise.clone(), 1, 16, 16, 16)

    expected_latents = reference(
        PROMPT,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=5.0,
        height=128,
        width=128,
        latents=packed_noise,
        max_sequence_length=77,
        output_type="latent",
    ).images
    expected_pixels = reference(
        PROMPT,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=5.0,
        height=128,
        width=128,
        latents=packed_noise,
        max_sequence_length=77,
    ).images
    expected_pixels = torch.from_numpy(np.stack([np.array(item) for item in expected_pixels]))
    del reference
    gc.collect()

    components = ChromaComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        guidance=5.0,
        steps=2,
        width=128,
        height=128,
        latents=noise,
        max_sequence_length=77,
    )
    actual_pixels = decode_latents(components.vae, actual_latents, 128, 128)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        "chroma_e2e_tiny",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        steps=2,
        guidance=5.0,
    )
    assert passed
    del components
    gc.collect()


def test_end_to_end_tiny_no_guidance(record, tmp_path):
    root = _write_tiny_snapshot(tmp_path / "chroma_tiny_ng")
    reference = _reference_pipeline(root)
    noise = torch.randn(1, 16, 16, 16, generator=torch.Generator("cpu").manual_seed(1))
    packed_noise = diffusers.ChromaPipeline._pack_latents(noise.clone(), 1, 16, 16, 16)

    expected_latents = reference(
        PROMPT,
        negative_prompt=None,
        num_inference_steps=2,
        guidance_scale=1.0,
        height=128,
        width=128,
        latents=packed_noise,
        max_sequence_length=77,
        output_type="latent",
    ).images
    del reference
    gc.collect()

    components = ChromaComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        guidance=1.0,
        steps=2,
        width=128,
        height=128,
        latents=noise,
        max_sequence_length=77,
    )
    _check(record, "chroma_e2e_tiny_no_guidance", actual_latents, expected_latents)
    del components
    gc.collect()


def test_seed_end_to_end_tiny(record, tmp_path):
    root = _write_tiny_snapshot(tmp_path / "chroma_tiny_seed")
    reference = _reference_pipeline(root)
    expected = reference(
        PROMPT,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=5.0,
        height=128,
        width=160,
        generator=torch.Generator("cpu").manual_seed(42),
        max_sequence_length=77,
    ).images
    expected = torch.from_numpy(np.stack([np.array(item) for item in expected]))[0]
    del reference
    gc.collect()

    actual = (
        pipeline("t2i", backend="chroma", model=root)(
            PROMPT,
            seed=42,
            steps=2,
            guidance=5.0,
            width=160,
            height=128,
            max_sequence_length=77,
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        "chroma_seed_e2e_tiny",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=42,
        steps=2,
    )
    assert passed


# Real-checkpoint tests: skip cleanly when the snapshot is not present.


@pytest.fixture(params=MODEL_IDS)
def model_id(request):
    return request.param


def test_real_checkpoint_headers(record, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": T5EncoderModel,
        "transformer": ChromaTransformer2DModel,
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
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_real_checkpoint_tokenizer(record, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    reference_tokenizer = transformers.T5TokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_ids = reference_tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    ours = T5Tokenizer(root / "tokenizer")
    actual_ids, _ = ours([PROMPT], MAX_SEQUENCE_LENGTH)
    tokens_match = torch.equal(actual_ids, expected_ids)
    record(f"{MODEL_ID}_tokenizer", exact=tokens_match, tolerance="exact")
    assert tokens_match


def test_real_checkpoint_text_encoder(record, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    reference_tokenizer = transformers.T5TokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_ids = reference_tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    reference_mask = reference_tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).attention_mask

    reference = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(expected_ids, attention_mask=reference_mask).last_hidden_state
    del reference
    gc.collect()

    ours = T5EncoderModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours(expected_ids, reference_mask.bool())[0]
    _check(record, f"{MODEL_ID}_text_encoder", actual, expected)
    del ours
    gc.collect()


def test_real_checkpoint_transformer(record, tmp_path, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 256, 64, generator=generator)
    context = torch.randn(
        1, MAX_SEQUENCE_LENGTH, config["joint_attention_dim"], generator=generator
    )
    img_ids = torch.zeros(256, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(16), torch.arange(16))
    txt_ids = torch.zeros(MAX_SEQUENCE_LENGTH, 3)
    timestep = torch.tensor([0.5])
    attention_mask = torch.ones(1, MAX_SEQUENCE_LENGTH + 256, dtype=torch.bool)

    reference = diffusers.ChromaTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
        ).sample
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()

    ours = ChromaTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(hidden, context, timestep, img_ids, txt_ids, attention_mask)
    expected = torch.load(output, weights_only=True)
    _check(record, f"{MODEL_ID}_transformer", actual, expected)
    del ours
    gc.collect()


def test_real_checkpoint_vae_decode(record, tmp_path, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 16, 128, 128, generator=generator)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference.decode(latents).sample
    output = tmp_path / "vae.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()

    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours.decode(latents)
    expected = torch.load(output, weights_only=True)
    _check(record, f"{MODEL_ID}_vae_decode", actual, expected)
    del ours
    gc.collect()


def test_real_checkpoint_end_to_end(record, tmp_path, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(42))
    reference = diffusers.ChromaPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    packed_noise = diffusers.ChromaPipeline._pack_latents(noise.clone(), 1, 16, 128, 128)
    expected_latents = reference(
        PROMPT,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=5.0,
        height=1024,
        width=1024,
        latents=packed_noise,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        output_type="latent",
    ).images
    expected_pixels = reference(
        PROMPT,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=5.0,
        height=1024,
        width=1024,
        latents=packed_noise,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    output = tmp_path / "end-to-end.pt"
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, output)
    del reference, expected_latents, expected_pixels
    gc.collect()

    components = ChromaComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        guidance=5.0,
        steps=2,
        width=1024,
        height=1024,
        latents=noise,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    expected = torch.load(output, weights_only=True)
    _check(record, f"e2e_{MODEL_ID}_latents", actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT,
        steps=2,
        guidance=5.0,
        width=1024,
        height=1024,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        latents=noise,
    )
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    pixel_diff = int((pixels.to(torch.int16) - expected["pixels"].to(torch.int16)).abs().max())
    record(
        f"e2e_{MODEL_ID}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        steps=2,
        guidance=5.0,
    )
    assert passed


def test_real_checkpoint_seed(record, tmp_path, model_id):
    MODEL_ID = model_id
    root = snapshot_dir(MODEL_ID)
    reference = diffusers.ChromaPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        PROMPT,
        negative_prompt="",
        num_inference_steps=2,
        guidance_scale=5.0,
        height=1024,
        width=1024,
        generator=torch.Generator("cpu").manual_seed(42),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()

    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root)(
            PROMPT,
            seed=42,
            steps=2,
            guidance=5.0,
            width=1024,
            height=1024,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        .images[0]
        .tensor
    )
    expected = torch.load(output, weights_only=True)
    passed = torch.equal(actual, expected)
    pixel_diff = int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max())
    record(
        f"seed_e2e_{MODEL_ID}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=42,
        steps=2,
    )
    assert passed
