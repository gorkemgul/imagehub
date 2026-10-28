"""Lumina-Next parity against a local published snapshot on CPU float32."""

import gc
import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.lumina_next import (
    MAX_SEQUENCE_LENGTH,
    LuminaNextComponents,
    decode_latents,
    denoise,
    encode_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.gemma import GemmaModel
from imagehub.models.lumina_next_transformer import LuminaNextDiT2DModel, rotary_frequencies
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

from .test_ref_gemma_tokenizer import FIXED_PROMPTS, FUZZ_PIECES  # noqa: E402

CPU = torch.device("cpu")
MODEL_ID = "lumina-next"
PROMPT = "a photograph of an astronaut riding a horse"
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
    reference = diffusers.LuminaPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _our_conditioning(root):
    """Our prompt and negative embeddings, using only the tokenizer and the text encoder."""
    components = cast(
        LuminaNextComponents,
        SimpleNamespace(
            tokenizer=GemmaTokenizer(root / "tokenizer"),
            text_encoder=GemmaModel.from_pretrained(
                root / "text_encoder", device=CPU, dtype=torch.float32
            ),
        ),
    )
    result = encode_prompt(components, PROMPT, "", CPU, True)
    del components
    gc.collect()
    return result


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": GemmaModel,
        "transformer": LuminaNextDiT2DModel,
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


def _tokenizers():
    root = snapshot_dir(MODEL_ID) / "tokenizer"
    return GemmaTokenizer(root), transformers.GemmaTokenizerFast.from_pretrained(
        root, local_files_only=True
    )


def _check_tokenizer(ours, reference, prompts, negative_length):
    for prompt in prompts:
        # The prompt: padded to the longest row and a multiple of eight, right-padded.
        expected = reference(
            [prompt],
            pad_to_multiple_of=8,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        ids = ours.tokenize(prompt, MAX_SEQUENCE_LENGTH)
        length = -(-len(ids) // 8) * 8
        assert expected.input_ids.shape[1] == length
        assert expected.input_ids[0].tolist() == ids + [ours.pad_id] * (length - len(ids))
        assert expected.attention_mask[0].sum().item() == len(ids)
        # The negative prompt: truncated and padded to a given length.
        negative = reference(
            [prompt],
            padding="max_length",
            max_length=negative_length,
            truncation=True,
            return_tensors="pt",
        )
        short = ours.tokenize(prompt, negative_length)
        padded = short + [ours.pad_id] * (negative_length - len(short))
        assert negative.input_ids[0].tolist() == padded
        assert negative.attention_mask[0].sum().item() == len(short)
        assert ours.tokenize(prompt) == reference.encode(prompt)


def test_tokenizer_fixed_prompts(record):
    ours, reference = _tokenizers()
    assert ours.add_bos_token and ours.add_eos_token
    for negative_length in (8, 64):
        _check_tokenizer(ours, reference, FIXED_PROMPTS, negative_length)
    record("lumina_next_tokenizer", prompts=len(FIXED_PROMPTS), mismatches=0, tolerance="exact")


def test_tokenizer_seeded_fuzz(record):
    ours, reference = _tokenizers()
    rng = random.Random(0)
    prompts = [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(600)
    ]
    _check_tokenizer(ours, reference, prompts, 24)
    record("lumina_next_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


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
            clean_caption=False,
        )
    del reference
    gc.collect()
    embeds, mask, negative, negative_mask_ours = _our_conditioning(root)
    assert negative is not None and negative_mask_ours is not None
    masks_match = torch.equal(mask, prompt_mask) and torch.equal(negative_mask_ours, negative_mask)
    record("lumina_next_text_encoder_masks", masks_match=masks_match, tolerance="exact")
    assert masks_match
    _check(record, "lumina_next_text_encoder_prompt", embeds, prompt_embeds)
    _check(record, "lumina_next_text_encoder_negative", negative, negative_embeds)


@pytest.mark.parametrize(
    ("factors", "proportional"), [((1.0, 1.0), True), ((1.3, 1.0), True), ((1.0, 1.5), False)]
)
def test_transformer(record, factors, proportional):
    root = snapshot_dir(MODEL_ID)
    embeds, mask, negative, negative_mask = _our_conditioning(root)
    assert negative is not None and negative_mask is not None
    text = torch.cat([embeds, negative])
    text_mask = torch.cat([mask, negative_mask])
    config = read_json(root / "transformer" / "config.json")
    channels = int(config["in_channels"])
    generator = torch.Generator("cpu").manual_seed(1)
    hidden_states = torch.randn(2, channels, 64, 64, generator=generator)
    timestep = torch.tensor([0.2, 0.8])
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    rotary = diffusers.models.embeddings.get_2d_rotary_pos_embed_lumina(
        head_dim, 384, 384, linear_factor=factors[0], ntk_factor=factors[1]
    )
    base = 4096 if proportional else None
    kwargs = {"base_sequence_length": base} if proportional else {}

    reference = diffusers.LuminaNextDiT2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states,
            timestep=timestep,
            encoder_hidden_states=text,
            encoder_mask=text_mask,
            image_rotary_emb=rotary,
            cross_attention_kwargs=kwargs,
            return_dict=False,
        )[0]
    del reference
    gc.collect()

    ours = LuminaNextDiT2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    ours_rotary = rotary_frequencies(ours.head_dim, linear_factor=factors[0], ntk_factor=factors[1])
    assert torch.equal(ours_rotary, rotary)
    with torch.inference_mode():
        actual = ours(
            hidden_states, timestep, text, text_mask, ours_rotary, base_sequence_length=base
        )
    _check(
        record,
        f"lumina_next_transformer_{factors[0]}_{factors[1]}_{proportional}",
        actual,
        expected,
    )
    del ours
    gc.collect()


def test_vae_decode(record):
    root = snapshot_dir(MODEL_ID)
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    assert reference.config.scaling_factor == ours.scaling_factor
    channels = int(ours.config["latent_channels"])
    latents = torch.randn(1, channels, 64, 64, generator=torch.Generator("cpu").manual_seed(1))
    scaled = latents / reference.config.scaling_factor
    with torch.inference_mode():
        actual = ours.decode(scaled)
        expected = reference.decode(scaled, return_dict=False)[0]
    _check(record, "lumina_next_vae_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(root, seed, steps, **kwargs):
    reference = _reference_pipeline(root)
    latents = reference(
        PROMPT,
        num_inference_steps=steps,
        guidance_scale=GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
        clean_caption=False,
        **kwargs,
    ).images
    # The pipeline's own decode and post-processing, applied to the latents of the same run.
    scaled = latents / reference.vae.config.scaling_factor
    with torch.inference_mode():
        image = reference.vae.decode(scaled, return_dict=False)[0]
    images = reference.image_processor.postprocess(image, output_type="pil")
    pixels = torch.from_numpy(np.stack([np.array(item) for item in images]))
    del reference
    gc.collect()
    return latents, pixels


def _end_to_end(record, name, seed, steps, height, width, **options):
    root = snapshot_dir(MODEL_ID)
    expected_latents, expected_pixels = _reference_output(
        root, seed, steps, height=height, width=width, **options
    )
    components = LuminaNextComponents.load(root, device=CPU, dtype=torch.float32, model_id=MODEL_ID)
    channels = int(components.transformer.config["in_channels"])
    noise = torch.randn(
        1, channels, height // 8, width // 8, generator=torch.Generator("cpu").manual_seed(seed)
    )
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=steps,
        guidance=GUIDANCE,
        scaling_watershed=options.get("scaling_watershed", 1.0),
        proportional_attn=True,
        latents=noise,
    )
    actual_pixels = decode_latents(components.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        name,
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=GUIDANCE,
        size=[width, height],
        seed=seed,
        **options,
    )
    assert passed
    del components
    gc.collect()


def test_end_to_end(record):
    _end_to_end(record, f"e2e_{MODEL_ID}", 0, STEPS, 1024, 1024)


def test_end_to_end_time_aware_scaling(record):
    """A larger and a non-square size with the watershed inside the schedule (shortened)."""
    _end_to_end(record, f"e2e_watershed_{MODEL_ID}", 3, 8, 768, 1280, scaling_watershed=0.3)


def test_seed_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    _, expected = _reference_output(root, 1234, 12, height=768, width=1360)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, seed=1234, steps=12, height=768, width=1360
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{MODEL_ID}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        steps=12,
        size=[1360, 768],
    )
    assert passed
