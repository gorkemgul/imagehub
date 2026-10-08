"""Kandinsky 2.2 parity against the local two-repository snapshot on CPU float32."""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.kandinsky22 import (
    Kandinsky22Components,
    _tokenize,
    decode_latents,
    denoise_decoder,
    get_zero_embed,
    run_prior,
)
from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.prior_transformer import PriorTransformer
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.vq_model import VQModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "kandinsky-2.2"
PROMPT = "a photograph of an astronaut riding a horse"


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for path in directory.glob("*.safetensors"):
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _combined_reference(root: Path):
    prior = diffusers.KandinskyV22PriorPipeline.from_pretrained(
        root / "prior", local_files_only=True, torch_dtype=torch.float32
    )
    decoder = diffusers.KandinskyV22Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    combined = diffusers.KandinskyV22CombinedPipeline(
        unet=decoder.unet,
        scheduler=decoder.scheduler,
        movq=decoder.movq,
        prior_prior=prior.prior,
        prior_image_encoder=prior.image_encoder,
        prior_text_encoder=prior.text_encoder,
        prior_tokenizer=prior.tokenizer,
        prior_scheduler=prior.scheduler,
        prior_image_processor=prior.image_processor,
    )
    combined.set_progress_bar_config(disable=True)
    return combined


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for directory, cls in {
        root / "prior" / "text_encoder": CLIPTextModelWithProjection,
        root / "prior" / "image_encoder": CLIPVisionModelWithProjection,
        root / "prior" / "prior": PriorTransformer,
        root / "unet": UNet2DConditionModel,
        root / "movq": VQModel,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(directory / "config.json"))
        shapes = _checkpoint_shapes(directory)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[str(directory.relative_to(root))] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    record("header_kandinsky-2.2", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokens_text_and_zero_image_embeddings(record):
    root = snapshot_dir(MODEL_ID)
    components = Kandinsky22Components.load(root, device=CPU, dtype=torch.float32)
    reference_tokenizer = transformers.CLIPTokenizer.from_pretrained(
        root / "prior" / "tokenizer", local_files_only=True
    )
    expected_tokens = reference_tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    actual_ids, actual_mask = _tokenize(components.tokenizer, PROMPT, CPU)
    exact_tokens = torch.equal(actual_ids, expected_tokens.input_ids)
    exact_mask = torch.equal(actual_mask, expected_tokens.attention_mask.bool())
    record("kandinsky22_tokens", exact=exact_tokens and exact_mask, tolerance="exact")
    assert exact_tokens and exact_mask
    reference_text = transformers.CLIPTextModelWithProjection.from_pretrained(
        root / "prior" / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    reference_vision = transformers.CLIPVisionModelWithProjection.from_pretrained(
        root / "prior" / "image_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_text = reference_text(actual_ids).text_embeds
        size = reference_vision.config.image_size
        expected_zero = reference_vision(torch.zeros(1, 3, size, size)).image_embeds
        actual_text = components.text_encoder(actual_ids)[0]
        actual_zero = get_zero_embed(components, 1, CPU)
    _check(record, "kandinsky22_text", actual_text, expected_text)
    _check(record, "kandinsky22_zero_image", actual_zero, expected_zero)


def test_prior_step_and_25_step_output(record):
    root = snapshot_dir(MODEL_ID)
    components = Kandinsky22Components.load(root, device=CPU, dtype=torch.float32)
    reference = diffusers.KandinskyV22PriorPipeline.from_pretrained(
        root / "prior", local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    latents = torch.randn(1, 1280, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        expected_step = reference.prior(
            latents,
            500,
            torch.zeros(1, 1280),
            encoder_hidden_states=torch.zeros(1, 77, 1280),
            attention_mask=torch.ones(1, 77, dtype=torch.bool),
        ).predicted_image_embedding
        actual_step = components.prior(
            latents,
            500,
            torch.zeros(1, 1280),
            encoder_hidden_states=torch.zeros(1, 77, 1280),
            attention_mask=torch.ones(1, 77, dtype=torch.bool),
        )
        expected = reference(
            PROMPT,
            num_inference_steps=25,
            guidance_scale=4.0,
            latents=latents.clone(),
            generator=torch.Generator().manual_seed(42),
            output_type="pt",
        ).image_embeds
        actual, _ = run_prior(
            components,
            prompt=PROMPT,
            negative_prompt=None,
            steps=25,
            guidance=4.0,
            latents=latents.clone(),
            generator=torch.Generator().manual_seed(42),
            device=CPU,
        )
    _check(record, "kandinsky22_prior_step", actual_step, expected_step)
    _check(record, "kandinsky22_prior_25_steps", actual, expected)


def test_decoder_unet_and_movq(record):
    root = snapshot_dir(MODEL_ID)
    components = Kandinsky22Components.load(root, device=CPU, dtype=torch.float32)
    reference_unet = diffusers.UNet2DConditionModel.from_pretrained(
        root / "unet", local_files_only=True, torch_dtype=torch.float32
    )
    reference_movq = diffusers.VQModel.from_pretrained(
        root / "movq", local_files_only=True, torch_dtype=torch.float32
    )
    sample = torch.randn(2, 4, 64, 64, generator=torch.Generator().manual_seed(1))
    image_embeds = torch.randn(2, 1280, generator=torch.Generator().manual_seed(2))
    latents = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        expected_unet = reference_unet(
            sample,
            500,
            encoder_hidden_states=None,
            added_cond_kwargs={"image_embeds": image_embeds},
        ).sample
        actual_unet = components.unet(sample, torch.tensor(500), None, image_embeds=image_embeds)
        expected_movq = reference_movq.decode(latents, force_not_quantize=True).sample
        actual_movq = components.movq.decode(latents, force_not_quantize=True)
    _check(record, "kandinsky22_decoder_unet_512", actual_unet, expected_unet)
    _check(record, "kandinsky22_movq_decode", actual_movq, expected_movq)


def test_end_to_end_defaults_and_seed(record):
    root = snapshot_dir(MODEL_ID)
    reference = _combined_reference(root)
    prior_latents = torch.randn(1, 1280, generator=torch.Generator().manual_seed(7))
    expected = reference(
        PROMPT,
        latents=prior_latents.clone(),
        generator=torch.Generator().manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    components = Kandinsky22Components.load(root, device=CPU, dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)
    positive, negative = run_prior(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=25,
        guidance=4.0,
        latents=prior_latents,
        generator=generator,
        device=CPU,
    )
    actual_latents = denoise_decoder(
        components,
        positive,
        negative,
        steps=100,
        guidance=4.0,
        width=512,
        height=512,
        generator=generator,
    )
    actual = decode_latents(components.movq, actual_latents)
    passed = torch.equal(actual, expected)
    record(
        "e2e_kandinsky-2.2",
        passed=passed,
        pixels_max_abs=int((actual.int() - expected.int()).abs().max()),
    )
    assert passed
    seeded = pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=42)
    expected_seeded = reference(
        PROMPT, generator=torch.Generator().manual_seed(42), output_type="np"
    ).images
    expected_seeded = torch.from_numpy((expected_seeded * 255).round().astype("uint8"))
    actual_seeded = torch.stack([image.tensor for image in seeded.images])
    seed_passed = torch.equal(actual_seeded, expected_seeded)
    record("seed_e2e_kandinsky-2.2", passed=seed_passed, seed=42)
    assert seed_passed
