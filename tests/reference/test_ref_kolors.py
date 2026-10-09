"""Kolors parity against a local published snapshot on CPU float32.

Only float16 weights are published; the reference is loaded with ``variant="fp16"`` and
``torch_dtype=torch.float32`` so both sides compute on the same float32 upcast of them.
"""

import gc
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from helpers import FIXTURES, max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.kolors import (
    MAX_SEQUENCE_LENGTH,
    KolorsComponents,
    decode_latents,
    denoise,
    encode_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.chatglm import ChatGLMModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.tokenization_chatglm import ChatGLMTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
PROMPT = "A photo of an astronaut riding a horse on mars, 4k"
PUBLISHED = FIXTURES / "kolors_configs"


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index = next(directory.glob("*.safetensors.index*.json"), None)
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


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _reference_pipeline(root, **modules):
    reference = diffusers.KolorsPipeline.from_pretrained(
        root, variant="fp16", local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _our_conditioning(root):
    """Prompt and negative embeddings from the tokenizer and the text encoder only."""
    components = cast(
        KolorsComponents,
        SimpleNamespace(
            tokenizer=ChatGLMTokenizer(root / "tokenizer" / "tokenizer.model"),
            text_encoder=ChatGLMModel.from_pretrained(
                root / "text_encoder", device=CPU, dtype=torch.float32
            ),
        ),
    )
    with torch.inference_mode():
        result = (
            *encode_prompt(components, PROMPT, CPU),
            *encode_prompt(components, "", CPU),
        )
    del components
    gc.collect()
    return result


def test_headers(record):
    root = snapshot_dir("kolors")
    result = {}
    for component, cls, fixture in (
        ("text_encoder", ChatGLMModel, "text_encoder_tensors.json"),
        ("unet", UNet2DConditionModel, "unet_tensors.json"),
        ("vae", AutoencoderKL, None),
    ):
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        if fixture is not None:
            published = json.loads((PUBLISHED / fixture).read_text(encoding="utf-8"))
            assert shapes == {name: tuple(shape) for name, shape in published.items()}
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
    record("header_kolors", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(record):
    root = snapshot_dir("kolors")
    path = root / "tokenizer" / "tokenizer.model"
    ours = ChatGLMTokenizer(path)
    reference = diffusers.pipelines.kolors.tokenizer.ChatGLMTokenizer(str(path))
    matches = []
    for text in (PROMPT, "", "一张室内的猫和狗的照片，高清"):
        ids, mask, positions = ours([text], max_length=MAX_SEQUENCE_LENGTH)
        expected = reference(
            [text],
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        matches.append(
            torch.equal(ids, expected["input_ids"])
            and torch.equal(mask, expected["attention_mask"].bool())
            and torch.equal(positions, expected["position_ids"])
        )
    record("kolors_tokenizer_snapshot", cases=len(matches), tolerance="exact")
    assert all(matches)


def test_text_encoder(record):
    root = snapshot_dir("kolors")
    reference = _reference_pipeline(root, unet=None, vae=None)
    with torch.inference_mode():
        prompt_embeds, negative_embeds, pooled, negative_pooled = reference.encode_prompt(
            PROMPT,
            device=CPU,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    assert not reference.config.force_zeros_for_empty_prompt
    del reference
    gc.collect()
    embeds, pooled_ours, negative, negative_pooled_ours = _our_conditioning(root)
    _check(record, "kolors_text_encoder_prompt", embeds, prompt_embeds)
    _check(record, "kolors_text_encoder_pooled", pooled_ours, pooled)
    _check(record, "kolors_text_encoder_negative", negative, negative_embeds)
    _check(record, "kolors_text_encoder_negative_pooled", negative_pooled_ours, negative_pooled)


def test_unet_step(record):
    root = snapshot_dir("kolors")
    ours = UNet2DConditionModel.from_pretrained(root / "unet", device=CPU, dtype=torch.float32)
    reference = diffusers.UNet2DConditionModel.from_pretrained(
        root / "unet", variant="fp16", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(1)
    sample = torch.randn(2, 4, 64, 64, generator=generator)
    context = torch.randn(2, MAX_SEQUENCE_LENGTH, 4096, generator=generator)
    text_embeds = torch.randn(2, 4096, generator=generator)
    time_ids = torch.tensor([[1024.0, 1024.0, 0.0, 0.0, 1024.0, 1024.0]]).repeat(2, 1)
    timestep = torch.tensor(881.0)
    with torch.inference_mode():
        actual = ours(sample, timestep, context, text_embeds=text_embeds, time_ids=time_ids)
        expected = reference(
            sample,
            timestep,
            encoder_hidden_states=context,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample
    _check(record, "kolors_unet_step", actual, expected)
    del ours, reference
    gc.collect()


def test_vae_decode(record):
    root = snapshot_dir("kolors")
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", variant="fp16", local_files_only=True, torch_dtype=torch.float32
    )
    assert reference.config.scaling_factor == ours.scaling_factor
    latents = torch.randn(1, 4, 64, 64, generator=torch.Generator("cpu").manual_seed(1))
    scaled = latents / reference.config.scaling_factor
    with torch.inference_mode():
        actual = ours.decode(scaled)
        expected = reference.decode(scaled, return_dict=False)[0]
    _check(record, "kolors_vae_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(root, seed, **kwargs):
    reference = _reference_pipeline(root)
    arguments = {"prompt": PROMPT, **kwargs}
    latents = reference(
        **arguments, generator=torch.Generator("cpu").manual_seed(seed), output_type="latent"
    ).images
    images = reference(
        **arguments, generator=torch.Generator("cpu").manual_seed(seed), output_type="pil"
    ).images
    pixels = torch.from_numpy(np.stack([np.array(image) for image in images]))
    del reference
    gc.collect()
    return latents, pixels


def test_end_to_end(record):
    root = snapshot_dir("kolors")
    expected_latents, expected_pixels = _reference_output(root, 0)
    components = KolorsComponents.load(root, device=CPU, dtype=torch.float32, model_id="kolors")
    noise = torch.randn(1, 4, 128, 128, generator=torch.Generator("cpu").manual_seed(0))
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=50,
        guidance=5.0,
        latents=noise,
        original_size=(1024, 1024),
        crops_coords_top_left=(0, 0),
        target_size=(1024, 1024),
    )
    actual_pixels = decode_latents(components, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        "e2e_kolors",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=50,
        guidance=5.0,
        size=[1024, 1024],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


def test_seed_end_to_end(record):
    root = snapshot_dir("kolors")
    _, expected = _reference_output(root, 1234, height=768, width=1360)
    result = pipeline("t2i", backend="kolors", model=root)(
        PROMPT, seed=1234, height=768, width=1360
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        "seed_e2e_kolors",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        size=[1360, 768],
    )
    assert passed
