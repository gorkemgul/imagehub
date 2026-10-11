"""Kandinsky 3 end to end on a tiny random snapshot against the reference pipeline."""

import pytest
import torch
from helpers import write_tiny_kandinsky3

from imagehub import pipeline
from imagehub.backends.kandinsky3 import (
    Kandinsky3Backend,
    denoise,
    encode_prompt,
    latent_size,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("numpy")

CASES = {
    "defaults": dict(prompt="a cat", steps=25, guidance=3.0, hw=(128, 128), seed=0),
    "no-cfg": dict(prompt="a b a", steps=6, guidance=1.0, hw=(64, 64), seed=1),
    "negative-non-square": dict(
        prompt="ab abab", steps=8, guidance=4.5, hw=(96, 160), seed=7, negative_prompt="b a"
    ),
    "empty-negative": dict(
        prompt="a b", steps=5, guidance=2.0, hw=(64, 128), seed=3, negative_prompt=""
    ),
    "two-images": dict(prompt="ab", steps=4, guidance=3.0, hw=(64, 64), seed=5, num_images=2),
    "long-prompt": dict(prompt="ab " * 80, steps=3, guidance=3.0, hw=(64, 64), seed=2),
    "high-guidance-short": dict(prompt="a", steps=1, guidance=7.0, hw=(64, 64), seed=9),
}


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    result = {}
    for seed in (0, 1):
        root = write_tiny_kandinsky3(tmp_path_factory.mktemp(f"kandinsky3_{seed}"), seed=seed)
        ref = diffusers.Kandinsky3Pipeline.from_pretrained(
            root, variant="fp16", torch_dtype=torch.float32, local_files_only=True
        )
        ref.set_progress_bar_config(disable=True)
        result[seed] = (root, ref)
    return result


def _reference_call(ref, case, **extra):
    height, width = case["hw"]
    return ref(
        case["prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        negative_prompt=case.get("negative_prompt"),
        num_images_per_prompt=case.get("num_images", 1),
        height=height,
        width=width,
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        **extra,
    )


def _options(case):
    return {key: case[key] for key in ("negative_prompt", "num_images") if key in case}


@pytest.mark.parametrize("weights_seed", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tiny_snapshot_matches_pipeline(references, name, weights_seed, record):
    case = CASES[name]
    root, ref = references[weights_seed]
    expected = _reference_call(ref, case, output_type="np").images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    height, width = case["hw"]
    result = pipeline("t2i", backend="kandinsky3", model=root)(
        case["prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        height=height,
        width=width,
        seed=case["seed"],
        **_options(case),
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert result.metadata["height"] == expected.shape[1]
    assert result.metadata["width"] == expected.shape[2]
    record(f"kandinsky3_tiny_{name}_{weights_seed}", max_abs_diff=0.0, tolerance="exact")


@pytest.mark.parametrize("name", ["negative-non-square", "no-cfg", "two-images"])
def test_tiny_snapshot_latents_match_pipeline(references, name):
    case = CASES[name]
    root, ref = references[0]
    expected = _reference_call(ref, case, output_type="latent").images
    backend = Kandinsky3Backend(root).load()
    try:
        components = backend.components
        assert components is not None
        height, width = case["hw"]
        count = case.get("num_images", 1)
        generator = torch.Generator("cpu").manual_seed(case["seed"])
        noise = torch.randn(
            (count, 4, latent_size(height), latent_size(width)), generator=generator
        )
        context, mask = encode_prompt(
            components,
            case["prompt"],
            case.get("negative_prompt"),
            do_classifier_free_guidance=case["guidance"] > 1,
            num_images=count,
            device=torch.device("cpu"),
        )
        actual = denoise(
            components,
            context,
            mask,
            noise,
            steps=case["steps"],
            guidance=case["guidance"],
            generator=generator,
        )
    finally:
        backend.unload()
    assert torch.equal(actual, expected)


def test_tiny_snapshot_precomputed_context_matches(references):
    """The reference given our cut context reproduces the same latents (encoder isolation)."""
    case = CASES["negative-non-square"]
    root, ref = references[0]
    backend = Kandinsky3Backend(root).load()
    try:
        components = backend.components
        assert components is not None
        context, mask = encode_prompt(
            components,
            case["prompt"],
            case["negative_prompt"],
            do_classifier_free_guidance=True,
            num_images=1,
            device=torch.device("cpu"),
        )
    finally:
        backend.unload()
    expected = ref.encode_prompt(
        case["prompt"],
        True,
        negative_prompt=case["negative_prompt"],
        device=torch.device("cpu"),
        _cut_context=True,
    )
    prompt_embeds, negative_embeds, attention_mask, negative_mask = expected
    assert torch.equal(context, torch.cat([negative_embeds, prompt_embeds]))
    assert torch.equal(mask, torch.cat([negative_mask, attention_mask]).bool())
