"""Lumina-Image 2.0 parity against a local published snapshot on CPU float32."""

import gc
import inspect
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.lumina2 import (
    DEFAULT_SYSTEM_PROMPT,
    Lumina2Components,
    decode_latents,
    denoise,
    encode_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.lumina2_transformer import Lumina2Transformer2DModel
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_ID = "lumina-image-2.0"
PROMPT = "a photograph of an astronaut riding a horse"
MAX_SEQUENCE_LENGTH = 256
STEPS = 30
GUIDANCE = 4.0


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
    reference = diffusers.Lumina2Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _our_conditioning(root):
    """Our prompt and negative embeddings, using only the tokenizer and the text encoder."""
    components = cast(
        Lumina2Components,
        SimpleNamespace(
            tokenizer=GemmaTokenizer(root / "tokenizer"),
            text_encoder=Gemma2Model.from_pretrained(
                root / "text_encoder", device=CPU, dtype=torch.float32
            ),
        ),
    )
    result = encode_prompt(
        components, PROMPT, "", DEFAULT_SYSTEM_PROMPT, MAX_SEQUENCE_LENGTH, CPU, True
    )
    del components
    gc.collect()
    return result


def test_system_prompt_matches_pipeline_default():
    source = inspect.getsource(diffusers.Lumina2Pipeline.__init__)
    assert f'self.system_prompt = "{DEFAULT_SYSTEM_PROMPT}"' in source


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": Gemma2Model,
        "transformer": Lumina2Transformer2DModel,
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


def test_tokenizer(record):
    root = snapshot_dir(MODEL_ID)
    ours = GemmaTokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    reference.padding_side = "right"
    long_prompt = " ".join([PROMPT] * 40)
    matches = []
    for prompt in (PROMPT, long_prompt):
        text = DEFAULT_SYSTEM_PROMPT + " <Prompt Start> " + prompt
        for case in (text, ""):
            input_ids, attention_mask = ours(
                [case], max_length=MAX_SEQUENCE_LENGTH, padding_side="right"
            )
            expected = reference(
                [case],
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                truncation=True,
                return_tensors="pt",
            )
            matches.append(
                {
                    "truncated": len(ours.tokenize(case)) > MAX_SEQUENCE_LENGTH,
                    "ids_match": torch.equal(input_ids, expected.input_ids),
                    "masks_match": torch.equal(attention_mask, expected.attention_mask.bool()),
                }
            )
    record("lumina2_tokenizer", cases=matches, tolerance="exact")
    assert all(case["ids_match"] and case["masks_match"] for case in matches)


def test_text_encoder(record):
    root = snapshot_dir(MODEL_ID)
    # Only the text encoder and tokenizer are needed; loading goes through the pipeline so
    # the attention implementation is the one the pipeline selects.
    reference = _reference_pipeline(root, transformer=None, vae=None)
    with torch.inference_mode():
        prompt_embeds, prompt_mask, negative_embeds, negative_mask = reference.encode_prompt(
            PROMPT,
            do_classifier_free_guidance=True,
            negative_prompt="",
            device=CPU,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
    del reference
    gc.collect()
    embeds, mask, negative, negative_mask_ours = _our_conditioning(root)
    assert negative is not None and negative_mask_ours is not None
    masks_match = torch.equal(mask, prompt_mask.bool()) and torch.equal(
        negative_mask_ours, negative_mask.bool()
    )
    record("lumina2_text_encoder_masks", masks_match=masks_match, tolerance="exact")
    assert masks_match
    _check(record, "lumina2_text_encoder_prompt", embeds, prompt_embeds)
    _check(record, "lumina2_text_encoder_negative", negative, negative_embeds)


def test_transformer(record):
    root = snapshot_dir(MODEL_ID)
    embeds, mask, negative, negative_mask = _our_conditioning(root)
    assert negative is not None and negative_mask is not None
    ours = Lumina2Transformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.Lumina2Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    channels = int(ours.config["in_channels"])
    generator = torch.Generator("cpu").manual_seed(1)
    hidden_states = torch.randn(2, channels, 64, 64, generator=generator)
    timestep = torch.tensor([0.2, 0.8])
    text = torch.cat([embeds, negative])
    text_mask = torch.cat([mask, negative_mask]).long()
    with torch.inference_mode():
        actual = ours(hidden_states, timestep, text, text_mask)
        expected = reference(
            hidden_states,
            timestep=timestep,
            encoder_hidden_states=text,
            encoder_attention_mask=text_mask,
            return_dict=False,
        )[0]
    _check(record, "lumina2_transformer", actual, expected)
    del ours, reference
    gc.collect()


def test_vae_decode(record):
    root = snapshot_dir(MODEL_ID)
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    assert reference.config.scaling_factor == ours.scaling_factor
    assert reference.config.shift_factor == ours.shift_factor
    channels = int(ours.config["latent_channels"])
    latents = torch.randn(1, channels, 64, 64, generator=torch.Generator("cpu").manual_seed(1))
    scaled = latents / reference.config.scaling_factor + reference.config.shift_factor
    with torch.inference_mode():
        actual = ours.decode(scaled)
        expected = reference.decode(scaled, return_dict=False)[0]
    _check(record, "lumina2_vae_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(root, seed, **kwargs):
    reference = _reference_pipeline(root)
    latents = reference(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
        **kwargs,
    ).images
    # The pipeline's own decode and post-processing, applied to the latents of the same run.
    scaled = latents / reference.vae.config.scaling_factor + reference.vae.config.shift_factor
    with torch.inference_mode():
        image = reference.vae.decode(scaled, return_dict=False)[0]
    images = reference.image_processor.postprocess(image, output_type="pil")
    pixels = torch.from_numpy(np.stack([np.array(item) for item in images]))
    del reference
    gc.collect()
    return latents, pixels


def test_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    expected_latents, expected_pixels = _reference_output(root, 0, height=1024, width=1024)
    components = Lumina2Components.load(root, device=CPU, dtype=torch.float32, model_id=MODEL_ID)
    channels = int(components.transformer.config["in_channels"])
    noise = torch.randn(1, channels, 128, 128, generator=torch.Generator("cpu").manual_seed(0))
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        steps=STEPS,
        guidance=GUIDANCE,
        cfg_trunc_ratio=1.0,
        cfg_normalization=True,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        latents=noise,
    )
    actual_pixels = decode_latents(components.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        f"e2e_{MODEL_ID}",
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
        f"seed_e2e_{MODEL_ID}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        size=[1360, 768],
    )
    assert passed
