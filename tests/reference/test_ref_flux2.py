"""FLUX.2 klein 4B parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.flux2 import (
    Flux2Components,
    compute_empirical_mu,
    denoise,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_flux2 import AutoencoderKLFlux2
from imagehub.models.flux2_transformer import Flux2Transformer2DModel
from imagehub.models.qwen3 import Qwen3Model
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "flux2-klein-4b"
PROMPT = "a photograph of an astronaut riding a horse"


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    files = []
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
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": Qwen3Model,
        "transformer": Flux2Transformer2DModel,
        "vae": AutoencoderKLFlux2,
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
    record("header_flux2-klein-4b", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_prompt_tokens_and_text_encoder(record):
    root = snapshot_dir(MODEL_ID)
    native_tokenizer = Qwen2Tokenizer(root / "tokenizer")
    reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_text = reference_tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    text_matches = native_tokenizer.chat_prompt(PROMPT, enable_thinking=False) == expected_text
    native_ids, native_mask = native_tokenizer(expected_text)
    expected_tokens = reference_tokenizer(
        expected_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    token_matches = torch.equal(native_ids, expected_tokens.input_ids)
    mask_matches = torch.equal(native_mask, expected_tokens.attention_mask.bool())
    record(
        "flux2_tokenizer",
        prompt_string_match=text_matches,
        ids_match=token_matches,
        masks_match=mask_matches,
        tolerance="exact",
    )
    assert text_matches and token_matches and mask_matches

    ours = Qwen3Model.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    reference = transformers.Qwen3ForCausalLM.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        actual_states = ours(native_ids, native_mask, output_hidden_states=True)[1]
        expected_states = reference(
            input_ids=native_ids,
            attention_mask=native_mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        actual = torch.stack([actual_states[index] for index in (9, 18, 27)], dim=2).flatten(2)
        expected = torch.stack([expected_states[index] for index in (9, 18, 27)], dim=2).flatten(2)
    _check(record, "flux2_text_encoder", actual, expected)
    del ours, reference
    gc.collect()


def test_transformer(record):
    root = snapshot_dir(MODEL_ID)
    ours = Flux2Transformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.Flux2Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 4096, 128, generator=generator)
    context = torch.randn(1, 512, 7680, generator=generator)
    img_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(64), torch.arange(64), torch.arange(1)
    ).unsqueeze(0)
    txt_ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(512)
    ).unsqueeze(0)
    timestep = torch.tensor([0.5])
    with torch.inference_mode():
        actual = ours(hidden, context, timestep, img_ids, txt_ids)
        expected = reference(
            hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample
    _check(record, "flux2_transformer", actual, expected)
    del ours, reference
    gc.collect()


def test_vae(record):
    root = snapshot_dir(MODEL_ID)
    ours = AutoencoderKLFlux2.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKLFlux2.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 32, 128, 128, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    with torch.inference_mode():
        actual_decode = ours.decode(latents)
        expected_decode = reference.decode(latents).sample
        actual_encode = ours.encode(image)
        expected_encode = reference.encode(image).latent_dist.mode()
    _check(record, "flux2_vae_decode", actual_decode, expected_decode)
    _check(record, "flux2_vae_encode", actual_encode, expected_encode)
    del ours, reference
    gc.collect()


def test_scheduler(record):
    root = snapshot_dir(MODEL_ID)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    sigmas = torch.linspace(1.0, 0.25, 4, dtype=torch.float64).tolist()
    mu = compute_empirical_mu(4096, 4)
    ours = FlowMatchEulerDiscreteScheduler(config)
    ours.set_timesteps(4, sigmas=sigmas, mu=mu)
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=sigmas, mu=mu)
    exact = torch.equal(ours.sigmas, reference.sigmas.cpu()) and torch.equal(
        ours.timesteps, reference.timesteps.cpu()
    )
    record("flux2_scheduler", mu=mu, exact=exact, tolerance="exact")
    assert exact


def test_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    reference = diffusers.Flux2KleinPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 128, 64, 64, generator=torch.Generator("cpu").manual_seed(42))
    expected_latents = reference(
        prompt=PROMPT,
        num_inference_steps=4,
        guidance_scale=1.0,
        height=1024,
        width=1024,
        latents=noise.clone(),
        output_type="latent",
    ).images
    components = Flux2Components.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        steps=4,
        guidance=1.0,
        width=1024,
        height=1024,
        latents=noise.clone(),
        max_sequence_length=512,
    )
    _check(record, "flux2_e2e_latents", actual_latents, expected_latents)
    expected = reference(
        prompt=PROMPT,
        num_inference_steps=4,
        guidance_scale=1.0,
        height=1024,
        width=1024,
        latents=noise.clone(),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    actual = pipeline("t2i", backend="flux2-klein-4b", model=root)(PROMPT, seed=42)
    pixels = torch.stack([image.tensor for image in actual.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.to(torch.int16) - expected.to(torch.int16)).abs().max())
    record("e2e_flux2-klein-4b", passed=passed, max_pixel_diff=pixel_diff)
    assert passed


def test_seed_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    reference = diffusers.Flux2KleinPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = pipeline("t2i", backend="flux2-klein-4b", model=root)(PROMPT, seed=42).images[0].tensor
    passed = torch.equal(actual, expected)
    pixel_diff = int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max())
    record(
        "seed_e2e_flux2-klein-4b",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=42,
    )
    assert passed
