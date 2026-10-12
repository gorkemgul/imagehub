"""Sana parity against local published snapshots on CPU float32."""

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
from imagehub.backends.sana import (
    COMPLEX_HUMAN_INSTRUCTION,
    SanaComponents,
    decode_latents,
    denoise,
    encode_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_dc import AutoencoderDC
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.sana_transformer import SanaTransformer2DModel
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODELS = {
    "sana-600m-1024": {"pipeline": "SanaPipeline", "steps": 20, "size": 1024},
    "sana-1600m-1024": {"pipeline": "SanaPipeline", "steps": 20, "size": 1024},
    "sana-sprint-1.6b-1024": {"pipeline": "SanaSprintPipeline", "steps": 2, "size": 1024},
    "sana-1.5-4.8b-1024": {"pipeline": "SanaPipeline", "steps": 20, "size": 1024},
}
MODEL_IDS = tuple(MODELS)
PROMPT = "a photograph of an astronaut riding a horse"
MAX_SEQUENCE_LENGTH = 300

FIXTURES = Path(__file__).parents[1] / "fixtures" / "sana_configs"
SANA_FIXTURE_MODELS = {
    "sana-sprint-1.6b-1024": {
        "config": "1.6b_sprint",
        "headers": "1.6b_sprint_headers",
        "components": {
            "text_encoder": ("text_encoder_1.json", "text_encoder_2.json"),
            "transformer": ("transformer.json",),
            "vae": ("vae.json",),
        },
    },
    "sana-1.5-4.8b-1024": {
        "config": "4.8b_1.5",
        "headers": "4.8b_1.5_headers",
        "components": {
            "text_encoder": ("text_encoder_1.json", "text_encoder_2.json"),
            "transformer": ("transformer_1.json", "transformer_2.json"),
            "vae": ("vae.json",),
        },
    },
}


def _fixture_shapes(header_dir: Path, files: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for name in files:
        data = json.loads((header_dir / name).read_text(encoding="utf-8"))
        shapes.update({k: tuple(v) for k, v in data.items()})
    return shapes


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


def _reference_instruction() -> list[str]:
    """The pipeline's own default instruction (``encode_prompt`` itself defaults to none)."""
    default = (
        inspect.signature(diffusers.SanaPipeline.__call__)
        .parameters["complex_human_instruction"]
        .default
    )
    return list(default)


def _reference_pipeline(model_id, root, **modules):
    pipeline_cls = getattr(diffusers, MODELS[model_id]["pipeline"])
    reference = pipeline_cls.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _our_conditioning(root):
    """Our prompt and negative embeddings, using only the tokenizer and the text encoder."""
    components = cast(
        SanaComponents,
        SimpleNamespace(
            tokenizer=GemmaTokenizer(root / "tokenizer"),
            text_encoder=Gemma2Model.from_pretrained(
                root / "text_encoder", device=CPU, dtype=torch.float32
            ),
        ),
    )
    embeds, mask = encode_prompt(
        components, PROMPT, "", MAX_SEQUENCE_LENGTH, COMPLEX_HUMAN_INSTRUCTION, CPU, True
    )
    del components
    gc.collect()
    return embeds, mask


def test_instruction_matches_pipeline_default():
    assert list(COMPLEX_HUMAN_INSTRUCTION) == _reference_instruction()
    sprint_default = (
        inspect.signature(diffusers.SanaSprintPipeline.__call__)
        .parameters["complex_human_instruction"]
        .default
    )
    assert list(COMPLEX_HUMAN_INSTRUCTION) == list(sprint_default)


@pytest.mark.parametrize("model_id", sorted(SANA_FIXTURE_MODELS))
def test_headers_from_fixtures(model_id, record):
    """Header check for the Sprint and 1.5 fixtures (no downloaded weights needed)."""
    info = SANA_FIXTURE_MODELS[model_id]
    config_dir = FIXTURES / info["config"]
    header_dir = FIXTURES / info["headers"]
    config_files = {
        "text_encoder": config_dir / "text_encoder_config.json",
        "transformer": config_dir / "transformer_config.json",
        "vae": config_dir / "vae_config.json",
    }
    result = {}
    for component, cls in {
        "text_encoder": Gemma2Model,
        "transformer": SanaTransformer2DModel,
        "vae": AutoencoderDC,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(config_files[component]))
        shapes = _fixture_shapes(header_dir, info["components"][component])
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
    record(f"fixture_header_{model_id}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_headers(model_id, record):
    root = snapshot_dir(model_id)
    result = {}
    for component, cls in {
        "text_encoder": Gemma2Model,
        "transformer": SanaTransformer2DModel,
        "vae": AutoencoderDC,
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
    record(f"header_{model_id}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_tokenizer(model_id, record):
    root = snapshot_dir(model_id)
    ours = GemmaTokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    reference.padding_side = "right"
    header = "\n".join(_reference_instruction())
    instruction_length = len(reference.encode(header))
    cases = [
        (header + PROMPT, instruction_length + MAX_SEQUENCE_LENGTH - 2),
        ("", MAX_SEQUENCE_LENGTH),
    ]
    matches = []
    for text, max_length in cases:
        input_ids, attention_mask = ours([text], max_length=max_length, padding_side="right")
        expected = reference(
            [text],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        matches.append(
            {
                "max_length": max_length,
                "ids_match": torch.equal(input_ids, expected.input_ids),
                "masks_match": torch.equal(attention_mask, expected.attention_mask.bool()),
            }
        )
    assert len(ours.tokenize(header)) == instruction_length
    record(f"sana_{model_id}_tokenizer", cases=matches, tolerance="exact")
    assert all(case["ids_match"] and case["masks_match"] for case in matches)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_text_encoder(model_id, record):
    root = snapshot_dir(model_id)
    # Only the text encoder and tokenizer are needed; loading goes through the pipeline so
    # the attention implementation is the one the pipeline selects.
    reference = _reference_pipeline(model_id, root, transformer=None, vae=None)
    with torch.inference_mode():
        prompt_embeds, prompt_mask, negative_embeds, negative_mask = reference.encode_prompt(
            PROMPT,
            do_classifier_free_guidance=True,
            negative_prompt="",
            device=CPU,
            clean_caption=False,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            complex_human_instruction=_reference_instruction(),
        )
    del reference
    gc.collect()
    embeds, mask = _our_conditioning(root)
    masks_match = torch.equal(mask[1:], prompt_mask.bool()) and torch.equal(
        mask[:1], negative_mask.bool()
    )
    record(f"sana_{model_id}_text_encoder_masks", masks_match=masks_match, tolerance="exact")
    assert masks_match
    _check(record, f"sana_{model_id}_text_encoder_prompt", embeds[1:], prompt_embeds)
    _check(record, f"sana_{model_id}_text_encoder_negative", embeds[:1], negative_embeds)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_transformer(model_id, record):
    root = snapshot_dir(model_id)
    embeds, mask = _our_conditioning(root)
    ours = SanaTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.SanaTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(1)
    hidden_states = torch.randn(2, 32, 32, 32, generator=generator)
    timestep = torch.tensor([800.0, 300.0])
    transformer_kwargs = {}
    reference_kwargs = {}
    if ours.guidance_embeds:
        guidance_scale = 4.5
        guidance_embeds_scale = float(ours.config.get("guidance_embeds_scale", 0.1))
        guidance = (
            torch.full([1], guidance_scale, device=CPU, dtype=torch.float32)
            .expand(hidden_states.shape[0])
            .to(torch.float32)
            * guidance_embeds_scale
        )
        transformer_kwargs["guidance"] = guidance
        reference_kwargs["guidance"] = guidance
    with torch.inference_mode():
        actual = ours(hidden_states, embeds, timestep, mask, **transformer_kwargs)
        expected = reference(
            hidden_states,
            encoder_hidden_states=embeds,
            timestep=timestep,
            encoder_attention_mask=mask,
            **reference_kwargs,
        ).sample
    _check(record, f"sana_{model_id}_transformer", actual, expected)
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_vae_decode(model_id, record):
    root = snapshot_dir(model_id)
    ours = AutoencoderDC.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderDC.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    latents = torch.randn(1, 32, 32, 32, generator=torch.Generator("cpu").manual_seed(1))
    scaled = latents / reference.config.scaling_factor
    assert reference.config.scaling_factor == ours.scaling_factor
    with torch.inference_mode():
        actual = ours.decode(scaled)
        expected = reference.decode(scaled, return_dict=False)[0]
    _check(record, f"sana_{model_id}_vae_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(model_id, root, seed, **kwargs):
    reference = _reference_pipeline(model_id, root)
    arguments = {
        "prompt": PROMPT,
        "num_inference_steps": MODELS[model_id]["steps"],
        "guidance_scale": 4.5,
        "clean_caption": False,
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


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(model_id, record):
    root = snapshot_dir(model_id)
    size = MODELS[model_id]["size"]
    steps = MODELS[model_id]["steps"]
    expected_latents, expected_pixels = _reference_output(
        model_id, root, 0, height=size, width=size
    )
    components = SanaComponents.load(root, device=CPU, dtype=torch.float32, model_id=model_id)
    noise = torch.randn(1, 32, 32, 32, generator=torch.Generator("cpu").manual_seed(0))
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=steps,
        guidance=4.5,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        instruction=COMPLEX_HUMAN_INSTRUCTION,
        latents=noise,
    )
    actual_pixels = decode_latents(
        components.vae, actual_latents, output_width=size, output_height=size, resize=True
    )
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        f"e2e_{model_id}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=4.5,
        size=[size, size],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(model_id, record):
    root = snapshot_dir(model_id)
    _, expected = _reference_output(model_id, root, 1234, height=768, width=1360)
    result = pipeline("t2i", backend=model_id, model=root)(
        PROMPT, seed=1234, height=768, width=1360
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        size=[1360, 768],
    )
    assert passed
