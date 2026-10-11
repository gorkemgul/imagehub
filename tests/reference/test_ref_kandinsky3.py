"""Kandinsky 3 parity against a local published snapshot on CPU float32.

Only float16 weights are published; the reference is loaded with ``variant="fp16"`` and
``torch_dtype=torch.float32`` so both sides compute on the same float32 upcast of them.
The text encoder is large (about 35 GB in float32): the reference conditioning is computed,
saved and freed before ours is loaded, and the reference denoiser runs without a text encoder.
The end-to-end runs use shortened step counts.
"""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import FIXTURES, max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.kandinsky3 import (
    MAX_SEQUENCE_LENGTH,
    Kandinsky3Components,
    decode_latents,
    denoise,
    encode_prompt,
    latent_size,
)
from imagehub.checkpoints import read_json
from imagehub.models.kandinsky3_unet import Kandinsky3UNet
from imagehub.models.t5 import T5EncoderModel
from imagehub.models.vq_model import VQModel
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
PROMPT = "A photograph of the inside of a subway train. There are raccoons sitting on the seats."
NEGATIVE = "blurry, low quality"
PUBLISHED = FIXTURES / "kandinsky3_configs"
E2E_STEPS = 4
SEED_STEPS = 3


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
    reference = diffusers.Kandinsky3Pipeline.from_pretrained(
        root, variant="fp16", local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


@pytest.fixture(scope="module")
def conditioning(tmp_path_factory):
    """Reference conditioning for the prompt, saved to disk; the reference encoder is freed."""
    root = snapshot_dir("kandinsky-3")
    path = tmp_path_factory.mktemp("kandinsky3_reference") / "conditioning.pt"
    reference = _reference_pipeline(root, unet=None, movq=None)
    result = {}
    with torch.inference_mode():
        for name, negative in (("none", None), ("text", NEGATIVE)):
            result[name] = reference.encode_prompt(
                PROMPT,
                True,
                device=CPU,
                negative_prompt=negative,
                _cut_context=True,
            )
        result["single"] = reference.encode_prompt(PROMPT, False, device=CPU, _cut_context=True)
    del reference
    gc.collect()
    torch.save(result, path)
    del result
    gc.collect()
    return path


def test_headers(record):
    root = snapshot_dir("kandinsky-3")
    result = {}
    for component, cls, fixture in (
        ("text_encoder", T5EncoderModel, "text_encoder_tensors.json"),
        ("unet", Kandinsky3UNet, "unet_tensors.json"),
        ("movq", VQModel, "movq_tensors.json"),
    ):
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
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
    record("header_kandinsky3", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(record):
    root = snapshot_dir("kandinsky-3")
    ours = T5Tokenizer(root / "tokenizer")
    reference = transformers.T5Tokenizer.from_pretrained(root / "tokenizer")
    prompts = [PROMPT, "", "a cat", "Bir kedi, İstanbul'da", "кошка и собака", "word " * 300]
    matches = []
    for text in prompts:
        ids, mask = ours([text], max_length=MAX_SEQUENCE_LENGTH)
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
        )
    record("kandinsky3_tokenizer_snapshot", cases=len(matches), tolerance="exact")
    assert all(matches)


def test_text_encoder(record, conditioning):
    root = snapshot_dir("kandinsky-3")
    expected = torch.load(conditioning)
    tokenizer = T5Tokenizer(root / "tokenizer")
    encoder = T5EncoderModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    components = Kandinsky3Components(tokenizer, encoder, None, None, {})  # type: ignore[arg-type]
    for name, negative in (("none", None), ("text", NEGATIVE)):
        context, mask = encode_prompt(
            components,
            PROMPT,
            negative,
            do_classifier_free_guidance=True,
            num_images=1,
            device=CPU,
        )
        prompt_embeds, negative_embeds, attention_mask, negative_mask = expected[name]
        _check(
            record,
            f"kandinsky3_text_encoder_{name}",
            context,
            torch.cat([negative_embeds, prompt_embeds]),
        )
        assert torch.equal(mask, torch.cat([negative_mask, attention_mask]).bool())
    context, mask = encode_prompt(
        components, PROMPT, None, do_classifier_free_guidance=False, num_images=1, device=CPU
    )
    _check(record, "kandinsky3_text_encoder_single", context, expected["single"][0])
    del components, encoder
    gc.collect()


def test_unet_step(record):
    root = snapshot_dir("kandinsky-3")
    ours = Kandinsky3UNet.from_pretrained(root / "unet", device=CPU, dtype=torch.float32)
    reference = diffusers.Kandinsky3UNet.from_pretrained(
        root / "unet", variant="fp16", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(1)
    sample = torch.randn(2, 4, 64, 64, generator=generator)
    context = torch.randn(2, 40, 4096, generator=generator)
    mask = torch.ones(2, 40, dtype=torch.bool)
    mask[0, 25:] = False
    mask[1, 30:] = False
    context = context * mask.unsqueeze(2)
    for timestep in (981, 501, 1):
        with torch.inference_mode():
            actual = ours(sample, torch.tensor(timestep), context, mask)
            expected = reference(
                sample,
                torch.tensor(timestep),
                encoder_hidden_states=context,
                encoder_attention_mask=mask,
                return_dict=False,
            )[0]
        _check(record, f"kandinsky3_unet_step_{timestep}", actual, expected)
    del ours, reference
    gc.collect()


def test_movq_decode(record):
    root = snapshot_dir("kandinsky-3")
    ours = VQModel.from_pretrained(root / "movq", device=CPU, dtype=torch.float32)
    reference = diffusers.VQModel.from_pretrained(
        root / "movq", variant="fp16", local_files_only=True, torch_dtype=torch.float32
    )
    latents = torch.randn(1, 4, 64, 64, generator=torch.Generator("cpu").manual_seed(1))
    with torch.inference_mode():
        actual = ours.decode(latents, force_not_quantize=True)
        expected = reference.decode(latents, force_not_quantize=True)["sample"]
    _check(record, "kandinsky3_movq_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(root, seed, steps, *, height, width, embeds, record_name=None):
    """Latents and uint8 pixels of the reference given saved conditioning; encoder not loaded."""
    prompt_embeds, negative_embeds, attention_mask, negative_mask = embeds
    reference = _reference_pipeline(root, text_encoder=None, tokenizer=None)
    captured = {}

    def capture(pipe, step, timestep, kwargs):
        captured["latents"] = kwargs["latents"].clone()
        return kwargs

    images = reference(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        attention_mask=attention_mask,
        negative_attention_mask=negative_mask,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=torch.Generator("cpu").manual_seed(seed),
        callback_on_step_end=capture,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="np",
    ).images
    pixels = torch.from_numpy((images * 255).round().astype("uint8"))
    del reference
    gc.collect()
    return captured["latents"], pixels


def test_end_to_end(record, conditioning, tmp_path):
    root = snapshot_dir("kandinsky-3")
    embeds = torch.load(conditioning)["none"]
    expected_latents, expected_pixels = _reference_output(
        root, 0, E2E_STEPS, height=1024, width=1024, embeds=embeds
    )
    torch.save((expected_latents, expected_pixels), tmp_path / "reference.pt")
    del expected_latents, expected_pixels, embeds
    gc.collect()
    expected_latents, expected_pixels = torch.load(tmp_path / "reference.pt")

    components = Kandinsky3Components.load(
        root, device=CPU, dtype=torch.float32, model_id="kandinsky-3"
    )
    generator = torch.Generator("cpu").manual_seed(0)
    noise = torch.randn((1, 4, 128, 128), generator=generator)
    context, mask = encode_prompt(
        components,
        PROMPT,
        None,
        do_classifier_free_guidance=True,
        num_images=1,
        device=CPU,
    )
    actual_latents = denoise(
        components, context, mask, noise, steps=E2E_STEPS, guidance=3.0, generator=generator
    )
    actual_pixels = decode_latents(components.movq, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        "e2e_kandinsky3",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=E2E_STEPS,
        guidance=3.0,
        size=[1024, 1024],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


def test_seed_end_to_end(record, conditioning, tmp_path):
    root = snapshot_dir("kandinsky-3")
    embeds = torch.load(conditioning)["text"]
    height, width = 512, 768
    _, expected = _reference_output(
        root, 1234, SEED_STEPS, height=height, width=width, embeds=embeds
    )
    torch.save(expected, tmp_path / "seed_reference.pt")
    del expected, embeds
    gc.collect()
    expected = torch.load(tmp_path / "seed_reference.pt")
    result = pipeline("t2i", backend="kandinsky3", model=root)(
        PROMPT,
        negative_prompt=NEGATIVE,
        seed=1234,
        steps=SEED_STEPS,
        height=height,
        width=width,
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        "seed_e2e_kandinsky3",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        steps=SEED_STEPS,
        size=[width, height],
    )
    assert passed
    assert latent_size(height) * 8 == height
