"""Kandinsky 2.1 parity against the local two-repository snapshot on CPU float32."""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.kandinsky21 import (
    Kandinsky21Components,
    _decode_latents,
    _denoise_decoder,
    _run_prior,
)
from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.multilingual_clip import MultilingualCLIP
from imagehub.models.prior_transformer import PriorTransformer
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.vq_model import VQModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "kandinsky-2-1"
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
    prior = diffusers.KandinskyPriorPipeline.from_pretrained(
        root / "prior", local_files_only=True, torch_dtype=torch.float32
    )
    decoder = diffusers.KandinskyPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    combined = diffusers.KandinskyCombinedPipeline(
        text_encoder=decoder.text_encoder,
        tokenizer=decoder.tokenizer,
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
        root / "text_encoder": MultilingualCLIP,
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
    record("header_kandinsky-2-1", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer_and_multilingual_text_encoder(record):
    root = snapshot_dir(MODEL_ID)
    components = Kandinsky21Components.load(root, device=CPU, dtype=torch.float32)
    reference_tokenizer = transformers.XLMRobertaTokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    prompts = (
        PROMPT,
        "İstanbul’da güzel bir gün",
        "Привет, мир! こんにちは世界 👩🏽‍🚀",
        "a" * 1000,
    )
    exact = True
    for prompt in prompts:
        expected = reference_tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        actual_ids, actual_mask = components.tokenizer(prompt, max_length=77)
        exact &= torch.equal(actual_ids, expected.input_ids[0])
        exact &= torch.equal(actual_mask, expected.attention_mask[0].bool())
    record("kandinsky21_tokens", exact=exact, prompts=len(prompts), tolerance="exact")
    assert exact

    reference_text = diffusers.pipelines.kandinsky.text_encoder.MultilingualCLIP.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    ids, mask = components.tokenizer(PROMPT, max_length=77)
    with torch.inference_mode():
        actual_pooled, actual_hidden = components.text_encoder(ids[None], mask[None])
        expected_pooled, expected_hidden = reference_text(ids[None], mask[None])
    _check(record, "kandinsky21_multilingual_clip_pooled", actual_pooled, expected_pooled)
    _check(record, "kandinsky21_multilingual_clip_hidden", actual_hidden, expected_hidden)


def test_prior_unet_and_movq(record):
    root = snapshot_dir(MODEL_ID)
    components = Kandinsky21Components.load(root, device=CPU, dtype=torch.float32)
    reference_prior = diffusers.PriorTransformer.from_pretrained(
        root / "prior" / "prior", local_files_only=True, torch_dtype=torch.float32
    )
    reference_unet = diffusers.UNet2DConditionModel.from_pretrained(
        root / "unet", local_files_only=True, torch_dtype=torch.float32
    )
    reference_movq = diffusers.VQModel.from_pretrained(
        root / "movq", local_files_only=True, torch_dtype=torch.float32
    )
    prior_latents = torch.randn(1, 768, generator=torch.Generator().manual_seed(1))
    prompt_embeds = torch.randn(1, 768, generator=torch.Generator().manual_seed(2))
    prior_hidden = torch.randn(1, 77, 768, generator=torch.Generator().manual_seed(3))
    sample = torch.randn(2, 4, 64, 96, generator=torch.Generator().manual_seed(4))
    decoder_hidden = torch.randn(2, 77, 1024, generator=torch.Generator().manual_seed(5))
    text_embeds = torch.randn(2, 768, generator=torch.Generator().manual_seed(6))
    image_embeds = torch.randn(2, 768, generator=torch.Generator().manual_seed(7))
    movq_latents = torch.randn(1, 4, 64, 96, generator=torch.Generator().manual_seed(8))
    with torch.inference_mode():
        expected_prior = reference_prior(
            prior_latents,
            500,
            prompt_embeds,
            encoder_hidden_states=prior_hidden,
            attention_mask=torch.ones(1, 77, dtype=torch.bool),
        ).predicted_image_embedding
        actual_prior = components.prior(
            prior_latents,
            500,
            prompt_embeds,
            encoder_hidden_states=prior_hidden,
            attention_mask=torch.ones(1, 77, dtype=torch.bool),
        )
        expected_unet = reference_unet(
            sample,
            500,
            encoder_hidden_states=decoder_hidden,
            added_cond_kwargs={"text_embeds": text_embeds, "image_embeds": image_embeds},
        ).sample
        actual_unet = components.unet(
            sample,
            torch.tensor(500),
            decoder_hidden,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
        )
        expected_movq = reference_movq.decode(movq_latents, force_not_quantize=True).sample
        actual_movq = components.movq.decode(movq_latents, force_not_quantize=True)
    _check(record, "kandinsky21_prior_step", actual_prior, expected_prior)
    _check(record, "kandinsky21_decoder_unet", actual_unet, expected_unet)
    _check(record, "kandinsky21_movq_decode", actual_movq, expected_movq)


def test_end_to_end_defaults_and_seed(record):
    root = snapshot_dir(MODEL_ID)
    reference = _combined_reference(root)
    prior_latents = torch.randn(1, 768, generator=torch.Generator().manual_seed(7))
    expected = reference(
        PROMPT,
        latents=prior_latents.clone(),
        generator=torch.Generator().manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    expected_seeded = reference(
        PROMPT, generator=torch.Generator().manual_seed(42), output_type="np"
    ).images
    expected_seeded = torch.from_numpy((expected_seeded * 255).round().astype("uint8"))
    del reference
    gc.collect()

    components = Kandinsky21Components.load(root, device=CPU, dtype=torch.float32)
    generator = torch.Generator().manual_seed(42)
    positive, negative = _run_prior(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=25,
        guidance=4.0,
        latents=prior_latents,
        generator=generator,
        device=CPU,
    )
    actual_latents = _denoise_decoder(
        components,
        positive,
        negative,
        prompt=PROMPT,
        steps=100,
        guidance=4.0,
        width=512,
        height=512,
        generator=generator,
    )
    actual = _decode_latents(components.movq, actual_latents)
    passed = torch.equal(actual, expected)
    record(
        "e2e_kandinsky-2-1",
        passed=passed,
        pixels_max_abs=int((actual.int() - expected.int()).abs().max()),
    )
    assert passed
    del components
    gc.collect()

    seeded = pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=42)
    actual_seeded = torch.stack([image.tensor for image in seeded.images])
    seed_passed = torch.equal(actual_seeded, expected_seeded)
    record("seed_e2e_kandinsky-2-1", passed=seed_passed, seed=42)
    assert seed_passed
