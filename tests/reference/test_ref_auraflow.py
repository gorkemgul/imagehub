"""AuraFlow parity against a local published snapshot on CPU float32."""

import gc
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.auraflow import (
    AuraFlowComponents,
    decode_latents,
    denoise,
    encode_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.auraflow_transformer import AuraFlowTransformer2DModel
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.umt5 import UMT5EncoderModel
from imagehub.tokenization_auraflow import AuraFlowTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_ID = "auraflow-v0.3"
PROMPT = "a photograph of an astronaut riding a horse"
MAX_SEQUENCE_LENGTH = 256
STEPS = 50
GUIDANCE = 3.5


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


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _reference_pipeline(root, **modules):
    reference = diffusers.AuraFlowPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _our_conditioning(root):
    """Our (negative, prompt) embeddings, using only the tokenizer and the text encoder."""
    components = cast(
        AuraFlowComponents,
        SimpleNamespace(
            tokenizer=AuraFlowTokenizer(root / "tokenizer"),
            text_encoder=UMT5EncoderModel.from_pretrained(
                root / "text_encoder", device=CPU, dtype=torch.float32
            ),
        ),
    )
    prompt_embeds, negative_embeds = encode_prompt(
        components, PROMPT, None, MAX_SEQUENCE_LENGTH, CPU, True
    )
    del components
    gc.collect()
    assert negative_embeds is not None
    return prompt_embeds, negative_embeds


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": UMT5EncoderModel,
        "transformer": AuraFlowTransformer2DModel,
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
    record("header_auraflow-v0.3", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(record):
    root = snapshot_dir(MODEL_ID)
    ours = AuraFlowTokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    matches = []
    for text in (PROMPT, ""):
        input_ids, attention_mask = ours([text], max_length=MAX_SEQUENCE_LENGTH)
        expected = reference(
            [text],
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        matches.append(
            {
                "ids_match": torch.equal(input_ids, expected.input_ids),
                "masks_match": torch.equal(attention_mask, expected.attention_mask.bool()),
            }
        )
    record("auraflow_auraflow-v0.3_tokenizer", cases=matches, tolerance="exact")
    assert all(case["ids_match"] and case["masks_match"] for case in matches)


def test_text_encoder(record):
    root = snapshot_dir(MODEL_ID)
    # Only the text encoder and tokenizer are needed; loading goes through the pipeline so
    # the attention implementation is the one the pipeline selects.
    reference = _reference_pipeline(root, transformer=None, vae=None)
    with torch.inference_mode():
        prompt_embeds, _, negative_embeds, _ = reference.encode_prompt(
            PROMPT,
            negative_prompt=None,
            do_classifier_free_guidance=True,
            device=CPU,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    del reference
    gc.collect()
    ours_prompt, ours_negative = _our_conditioning(root)
    _check(record, "auraflow_auraflow-v0.3_text_encoder_prompt", ours_prompt, prompt_embeds)
    _check(record, "auraflow_auraflow-v0.3_text_encoder_negative", ours_negative, negative_embeds)


def test_transformer(record):
    root = snapshot_dir(MODEL_ID)
    prompt_embeds, negative_embeds = _our_conditioning(root)
    embeds = torch.cat([negative_embeds, prompt_embeds])
    generator = torch.Generator("cpu").manual_seed(1)
    hidden_states = torch.randn(2, 4, 32, 48, generator=generator)
    timestep = torch.tensor([0.8, 0.3])
    ours = AuraFlowTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(hidden_states, embeds, timestep)
    del ours
    gc.collect()
    reference = diffusers.AuraFlowTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states, encoder_hidden_states=embeds, timestep=timestep, return_dict=False
        )[0]
    _check(record, "auraflow_auraflow-v0.3_transformer", actual, expected)
    del reference
    gc.collect()


def test_vae_decode(record):
    root = snapshot_dir(MODEL_ID)
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    latents = torch.randn(1, 4, 32, 48, generator=torch.Generator("cpu").manual_seed(1))
    scaled = latents / reference.config.scaling_factor
    assert reference.config.scaling_factor == ours.scaling_factor
    with torch.inference_mode():
        actual = ours.decode(scaled)
        expected = reference.decode(scaled, return_dict=False)[0]
    _check(record, "auraflow_auraflow-v0.3_vae_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(root, seed, **kwargs):
    reference = _reference_pipeline(root)
    arguments = {
        "prompt": PROMPT,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE,
        **kwargs,
    }
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
    root = snapshot_dir(MODEL_ID)
    expected_latents, expected_pixels = _reference_output(root, 0, height=1024, width=1024)
    components = AuraFlowComponents.load(root, device=CPU, dtype=torch.float32, model_id=MODEL_ID)
    noise = torch.randn(1, 4, 128, 128, generator=torch.Generator("cpu").manual_seed(0))
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=STEPS,
        guidance=GUIDANCE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        latents=noise,
    )
    actual_pixels = decode_latents(components.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        "e2e_auraflow-v0.3",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=STEPS,
        guidance=GUIDANCE,
        size=[1024, 1024],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


def test_seed_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    _, expected = _reference_output(root, 1234, height=768, width=1360)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, seed=1234, height=768, width=1360
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        "seed_e2e_auraflow-v0.3",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        size=[1360, 768],
    )
    assert passed
