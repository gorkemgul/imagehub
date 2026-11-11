"""Cosmos-Predict2 parity against diffusers 0.40.0 on CPU float32."""

import gc
import json
from pathlib import Path

import pytest
import torch
from fixture_files import fixture_path
from helpers import (
    TINY_COSMOS2_TEXT,
    TINY_COSMOS2_TRANSFORMER,
    TINY_COSMOS2_VAE,
    max_diff,
    reference_in_subprocess,
    snapshot_dir,
    write_tiny_cosmos2,
)

from imagehub import pipeline
from imagehub.backends.cosmos2 import DEFAULT_NEGATIVE_PROMPT
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_wan import AutoencoderKLWan
from imagehub.models.cosmos_transformer import CosmosTransformer3DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")
np = pytest.importorskip("numpy")

MODEL_ID = "cosmos-predict2-2b-t2i"
PROMPT = "a photograph of an astronaut riding a horse"
FIXTURES = fixture_path("cosmos2_configs")
CPU = torch.device("cpu")


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class _PermissiveSafetyChecker:
    def to(self, *_args, **_kwargs):
        return self

    def check_text_safety(self, _prompt):
        return True

    def check_video_safety(self, video):
        return video


def _load_state(model, path):
    result = model.load_state_dict(safetensors.load_file(path), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) <= {"encoder.embed_tokens.weight"}
    return model.eval()


def _reference_pipeline(root: Path):
    text_path = root / "text_encoder"
    text_weights = text_path / "model.safetensors"
    if text_weights.is_file():
        text_config = transformers.T5Config(**read_json(text_path / "config.json"))
        text_encoder = _load_state(transformers.T5EncoderModel(text_config), text_weights)
    else:
        text_encoder = transformers.T5EncoderModel.from_pretrained(
            text_path, local_files_only=True, dtype=torch.float32
        ).eval()
    transformer = _load_state(
        diffusers.CosmosTransformer3DModel(**read_json(root / "transformer" / "config.json")),
        root / "transformer" / "diffusion_pytorch_model.safetensors",
    )
    vae = _load_state(
        diffusers.AutoencoderKLWan(**read_json(root / "vae" / "config.json")),
        root / "vae" / "diffusion_pytorch_model.safetensors",
    )
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    reference = diffusers.Cosmos2TextToImagePipeline(
        text_encoder=text_encoder,
        tokenizer=transformers.T5TokenizerFast.from_pretrained(
            root / "tokenizer", local_files_only=True
        ),
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        safety_checker=_PermissiveSafetyChecker(),
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def test_published_fixture_headers_match_native_models(record):
    result = {}
    for component, cls in {
        "text_encoder": T5EncoderModel,
        "transformer": CosmosTransformer3DModel,
        "vae": AutoencoderKLWan,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(FIXTURES / component / "config.json"))
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        published = {
            name: tuple(shape)
            for name, shape in json.loads(
                (FIXTURES / "headers" / f"{component}.json").read_text(encoding="utf-8")
            ).items()
            if not any(name.startswith(prefix) for prefix in cls.skipped_prefixes)
        }
        result[component] = {
            "missing": sorted(expected.keys() - published.keys()),
            "extra": sorted(published.keys() - expected.keys()),
            "mismatch": sorted(
                name
                for name in expected.keys() & published.keys()
                if expected[name] != published[name]
            ),
            "tensors": len(published),
        }
    record("cosmos2_fixture_header", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_random_weight_components_are_exact(record):
    generator = torch.Generator("cpu").manual_seed(1)

    reference_text = transformers.T5EncoderModel(transformers.T5Config(**TINY_COSMOS2_TEXT))
    native_text = T5EncoderModel(TINY_COSMOS2_TEXT)
    native_text.load_state_dict(
        {
            name: value
            for name, value in reference_text.state_dict().items()
            if not name.startswith("encoder.embed_tokens.")
        }
    )
    ids = torch.randint(0, 100, (1, 9), generator=generator)
    mask = torch.tensor([[True] * 7 + [False] * 2])
    reference_text.eval()
    native_text.eval()
    with torch.inference_mode():
        expected_text = reference_text(ids, attention_mask=mask).last_hidden_state
        actual_text = native_text(ids, mask)[0]
    _check(record, "cosmos2_t5_random", actual_text, expected_text)

    reference_transformer = diffusers.CosmosTransformer3DModel(**TINY_COSMOS2_TRANSFORMER)
    native_transformer = CosmosTransformer3DModel(TINY_COSMOS2_TRANSFORMER)
    native_transformer.load_state_dict(reference_transformer.state_dict())
    hidden = torch.randn(1, 16, 1, 4, 6, generator=generator)
    timestep = torch.rand(1, generator=generator)
    context = torch.randn(1, 7, 24, generator=generator)
    padding = torch.zeros(1, 1, 32, 48)
    reference_transformer.eval()
    native_transformer.eval()
    with torch.inference_mode():
        expected_transformer = reference_transformer(
            hidden, timestep, context, padding_mask=padding, return_dict=False
        )[0]
        actual_transformer = native_transformer(hidden, timestep, context, padding_mask=padding)
    _check(record, "cosmos2_transformer_random", actual_transformer, expected_transformer)

    reference_vae = diffusers.AutoencoderKLWan(**TINY_COSMOS2_VAE)
    native_vae = AutoencoderKLWan(TINY_COSMOS2_VAE)
    native_vae.load_state_dict(reference_vae.state_dict())
    latents = torch.randn(1, 16, 1, 4, 6, generator=generator)
    reference_vae.eval()
    native_vae.eval()
    with torch.inference_mode():
        expected_vae = reference_vae.decode(latents, return_dict=False)[0][:, :, 0]
        actual_vae = native_vae.decode(latents[:, :, 0])
    _check(record, "cosmos2_vae_random", actual_vae, expected_vae)


def test_tokenizer_and_scheduler_are_exact(record):
    ours = T5Tokenizer(FIXTURES / "tokenizer")
    reference = transformers.T5TokenizerFast.from_pretrained(
        FIXTURES / "tokenizer", local_files_only=True
    )
    cases = [PROMPT, "  spaces\tand\nlines  ", "café 世界", DEFAULT_NEGATIVE_PROMPT]
    matches = []
    for text in cases:
        actual_ids, actual_mask = ours([text], 64)
        expected = reference(
            [text], padding="max_length", truncation=True, max_length=64, return_tensors="pt"
        )
        matches.append(
            torch.equal(actual_ids, expected.input_ids)
            and torch.equal(actual_mask, expected.attention_mask.bool())
        )
    record("cosmos2_tokenizer", exact=all(matches), cases=len(matches), tolerance="exact")
    assert all(matches)

    config = read_json(FIXTURES / "scheduler" / "scheduler_config.json")
    sigmas = torch.linspace(0, 1, 4, dtype=torch.float64).tolist()
    reference_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference_scheduler.set_timesteps(sigmas=sigmas)
    native_scheduler = FlowMatchEulerDiscreteScheduler(config)
    native_scheduler.set_timesteps(4, sigmas=sigmas)
    exact = torch.equal(native_scheduler.sigmas, reference_scheduler.sigmas.cpu()) and torch.equal(
        native_scheduler.timesteps, reference_scheduler.timesteps.cpu()
    )
    record("cosmos2_scheduler", exact=exact, tolerance="exact")
    assert exact


@pytest.mark.parametrize(("seed", "guidance"), [(0, 1.0), (42, 2.0)])
def test_tiny_pipeline_is_exact(tmp_path, seed, guidance, record):
    root = write_tiny_cosmos2(tmp_path / f"tiny-{seed}")
    reference = _reference_pipeline(root)
    with torch.inference_mode():
        expected_np = reference(
            PROMPT,
            num_inference_steps=2,
            guidance_scale=guidance,
            height=32,
            width=48,
            max_sequence_length=16,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected = torch.from_numpy((np.asarray(expected_np) * 255).round().astype("uint8"))[0]
    del reference
    gc.collect()
    actual = (
        pipeline("t2i", backend="cosmos2", model=root)(
            PROMPT,
            steps=2,
            guidance=guidance,
            height=32,
            width=48,
            max_sequence_length=16,
            seed=seed,
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    record(f"cosmos2_tiny_e2e_{seed}", passed=passed, tolerance="exact")
    assert passed


def test_real_checkpoint_components(record):
    root = snapshot_dir(MODEL_ID)
    tokenizer = transformers.T5TokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_tokens = tokenizer(
        [PROMPT], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
    )
    actual_ids, actual_mask = T5Tokenizer(root / "tokenizer")([PROMPT], 64)
    tokens_match = torch.equal(actual_ids, expected_tokens.input_ids) and torch.equal(
        actual_mask, expected_tokens.attention_mask.bool()
    )
    record("cosmos2_real_tokenizer", exact=tokens_match, tolerance="exact")
    assert tokens_match

    reference_text = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder", local_files_only=True, dtype=torch.float32
    )
    with torch.inference_mode():
        expected_text = reference_text(
            expected_tokens.input_ids, attention_mask=expected_tokens.attention_mask
        ).last_hidden_state
    del reference_text
    gc.collect()
    native_text = T5EncoderModel.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual_text = native_text(actual_ids, actual_mask)[0]
    _check(record, "cosmos2_real_text_encoder", actual_text, expected_text)
    del native_text, actual_text, expected_text
    gc.collect()

    generator = torch.Generator("cpu").manual_seed(2)
    hidden = torch.randn(1, 16, 1, 4, 6, generator=generator)
    timestep = torch.tensor([0.5])
    context = torch.randn(1, 64, 1024, generator=generator)
    padding = torch.zeros(1, 1, 32, 48)
    reference_transformer = diffusers.CosmosTransformer3DModel(
        **read_json(root / "transformer" / "config.json")
    )
    _load_state(reference_transformer, root / "transformer" / "diffusion_pytorch_model.safetensors")
    with torch.inference_mode():
        expected_transformer = reference_transformer(
            hidden, timestep, context, padding_mask=padding, return_dict=False
        )[0]
    del reference_transformer
    gc.collect()
    native_transformer = CosmosTransformer3DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual_transformer = native_transformer(hidden, timestep, context, padding_mask=padding)
    _check(record, "cosmos2_real_transformer", actual_transformer, expected_transformer)
    del native_transformer, actual_transformer, expected_transformer
    gc.collect()

    latents = torch.randn(1, 16, 1, 4, 6, generator=generator)
    reference_vae = diffusers.AutoencoderKLWan(**read_json(root / "vae" / "config.json"))
    _load_state(reference_vae, root / "vae" / "diffusion_pytorch_model.safetensors")
    with torch.inference_mode():
        expected_vae = reference_vae.decode(latents, return_dict=False)[0][:, :, 0]
    del reference_vae
    gc.collect()
    native_vae = AutoencoderKLWan.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_vae = native_vae.decode(latents[:, :, 0])
    _check(record, "cosmos2_real_vae", actual_vae, expected_vae)


def _reference_end_to_end(root: str, seed: int, out_path: str):
    reference = _reference_pipeline(Path(root))
    with torch.inference_mode():
        images = reference(
            PROMPT,
            num_inference_steps=2,
            guidance_scale=7.0,
            height=32,
            width=48,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    pixels = torch.from_numpy((np.asarray(images) * 255).round().astype("uint8"))[0]
    torch.save(pixels, out_path)


@pytest.mark.parametrize("seed", (0, 42))
def test_real_end_to_end(seed, record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    expected = reference_in_subprocess(
        "reference.test_ref_cosmos2._reference_end_to_end",
        {"root": str(root), "seed": seed},
        tmp_path / f"expected-{seed}.pt",
    )
    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root)(
            PROMPT,
            steps=2,
            guidance=7.0,
            width=48,
            height=32,
            max_sequence_length=512,
            seed=seed,
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    name = f"e2e_{MODEL_ID}" if seed == 0 else f"seed_e2e_{MODEL_ID}"
    record(name, passed=passed, tolerance="exact")
    assert passed
