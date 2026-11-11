"""FLUX.1-Lite parity on CPU float32 with bfloat16 published weights."""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.backends.flux import calculate_shift
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.flux_transformer import FluxTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

MODEL_ID = "flux-1-lite-8b"
PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(directory: Path):
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is None:
        files = [next(directory.glob("*.safetensors"))]
    else:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes.update(
            {
                name: tuple(entry["shape"])
                for name, entry in header.items()
                if name != "__metadata__"
            }
        )
    return shapes


def test_header(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": CLIPTextModel,
        "text_encoder_2": T5EncoderModel,
        "transformer": FluxTransformer2DModel,
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


def test_tokenizers_and_text_encoders(record):
    root = snapshot_dir(MODEL_ID)
    clip_reference_tokenizer = transformers.CLIPTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    clip_ids = clip_reference_tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids
    actual_clip_ids = torch.tensor([CLIPTokenizer(root / "tokenizer")(PROMPT)])
    t5_reference_tokenizer = transformers.T5TokenizerFast.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    t5_ids = t5_reference_tokenizer(
        [PROMPT], padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    ).input_ids
    actual_t5_ids, _ = T5Tokenizer(root / "tokenizer_2")([PROMPT], 512)
    tokens_match = torch.equal(clip_ids, actual_clip_ids) and torch.equal(t5_ids, actual_t5_ids)
    record("flux1_tokenizers_flux-1-lite-8b", exact=tokens_match, tolerance="exact")
    assert tokens_match

    reference = transformers.CLIPTextModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_clip = reference(clip_ids).pooler_output
    del reference
    gc.collect()
    native = CLIPTextModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        hidden = native(actual_clip_ids)
        actual_clip = hidden[torch.arange(hidden.shape[0]), actual_clip_ids.int().argmax(dim=-1)]
    _check(record, "flux1_clip_flux-1-lite-8b", actual_clip, expected_clip)
    del native, hidden, actual_clip, expected_clip
    gc.collect()

    reference = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder_2", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_t5 = reference(t5_ids).last_hidden_state
    del reference
    gc.collect()
    native = T5EncoderModel.from_pretrained(
        root / "text_encoder_2", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual_t5 = native(actual_t5_ids)[0]
    _check(record, "flux1_t5_flux-1-lite-8b", actual_t5, expected_t5)


def test_transformer(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 16, 64, generator=generator)
    context = torch.randn(1, 512, 4096, generator=generator)
    pooled = torch.randn(1, 768, generator=generator)
    img_ids = torch.zeros(16, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    txt_ids = torch.zeros(512, 3)
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([3.5])
    reference = diffusers.FluxTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        ).sample
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = FluxTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(hidden, context, pooled, timestep, img_ids, txt_ids, guidance)
    _check(
        record, "flux1_transformer_flux-1-lite-8b", actual, torch.load(output, weights_only=True)
    )


def test_vae_and_scheduler(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 16, 8, 8, generator=generator)
    image = torch.rand(1, 3, 64, 64, generator=generator) * 2 - 1
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
    expected = torch.load(output, weights_only=True)
    with torch.inference_mode():
        _check(
            record, "flux1_vae_decode_flux-1-lite-8b", native.decode(latents), expected["decode"]
        )
        _check(record, "flux1_vae_encode_flux-1-lite-8b", native.encode(image), expected["encode"])

    config = read_json(root / "scheduler" / "scheduler_config.json")
    sigmas = [1.0, 0.75, 0.5, 0.25]
    mu = calculate_shift(4096, config)
    reference_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference_scheduler.set_timesteps(sigmas=sigmas, mu=mu)
    native_scheduler = FlowMatchEulerDiscreteScheduler(config)
    native_scheduler.set_timesteps(4, sigmas=sigmas, mu=mu)
    exact = torch.equal(native_scheduler.sigmas, reference_scheduler.sigmas.cpu()) and torch.equal(
        native_scheduler.timesteps, reference_scheduler.timesteps.cpu()
    )
    record("flux1_scheduler_flux-1-lite-8b", mu=mu, exact=exact, tolerance="exact")
    assert exact


def _reference_end_to_end(root: str, seed: int, out_path: str):
    reference = diffusers.FluxPipeline.from_pretrained(
        Path(root), local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        prompt=PROMPT,
        num_inference_steps=2,
        guidance_scale=3.5,
        height=64,
        width=64,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    ).images
    torch.save(torch.from_numpy((expected * 255).round().astype("uint8"))[0], out_path)


@pytest.mark.parametrize("seed", (0, 42))
def test_real_end_to_end(seed, record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    expected = reference_in_subprocess(
        "reference.test_ref_flux_lite._reference_end_to_end",
        {"root": str(root), "seed": seed},
        tmp_path / f"expected-{seed}.pt",
    )
    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root)(
            PROMPT,
            steps=2,
            guidance=3.5,
            width=64,
            height=64,
            max_sequence_length=512,
            seed=seed,
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    name = f"e2e_{MODEL_ID}" if seed == 0 else f"seed_e2e_{MODEL_ID}"
    record(
        name,
        passed=passed,
        seed=seed,
        steps=2,
        max_pixel_diff=int((actual.int() - expected.int()).abs().max()),
    )
    assert passed
