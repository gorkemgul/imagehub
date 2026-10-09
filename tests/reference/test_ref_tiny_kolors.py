"""Kolors end to end on a tiny random snapshot in the published layout, against KolorsPipeline."""

import pytest
import torch
from helpers import write_tiny_kolors

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")
pytest.importorskip("numpy")

CASES = {
    "defaults": dict(prompt="a cat", steps=50, guidance=5.0, hw=(256, 256), seed=0),
    "no-cfg": dict(prompt="a b c", steps=6, guidance=1.0, hw=(64, 64), seed=1),
    "negative-non-square": dict(
        prompt="ab abc", steps=8, guidance=3.5, hw=(48, 80), seed=7, negative_prompt="bc"
    ),
    "micro-conditioning": dict(
        prompt="a cat",
        steps=4,
        guidance=4.0,
        hw=(64, 96),
        seed=11,
        original_size=(128, 192),
        crops_coords_top_left=(8, 16),
        target_size=(64, 96),
    ),
}


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    result = {}
    for seed in (0, 1):
        root = write_tiny_kolors(tmp_path_factory.mktemp(f"kolors{seed}"), seed=seed)
        ref = diffusers.KolorsPipeline.from_pretrained(
            root, variant="fp16", torch_dtype=torch.float32, local_files_only=True
        )
        ref.set_progress_bar_config(disable=True)
        result[seed] = (root, ref)
    return result


def _reference_call(ref, case, **extra):
    options = {
        key: case[key]
        for key in ("negative_prompt", "original_size", "crops_coords_top_left", "target_size")
        if key in case
    }
    height, width = case["hw"]
    return ref(
        case["prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        height=height,
        width=width,
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        **options,
        **extra,
    )


@pytest.mark.parametrize("weights_seed", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tiny_snapshot_matches_kolors_pipeline(references, name, weights_seed, record):
    case = CASES[name]
    root, ref = references[weights_seed]
    expected = _reference_call(ref, case, output_type="np").images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    options = {
        key: case[key]
        for key in ("negative_prompt", "original_size", "crops_coords_top_left", "target_size")
        if key in case
    }
    height, width = case["hw"]
    result = pipeline("t2i", backend="kolors", model=root)(
        case["prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        height=height,
        width=width,
        seed=case["seed"],
        **options,
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)
    record(f"kolors_tiny_{name}_{weights_seed}", max_abs_diff=0.0, tolerance="exact")


def test_tiny_snapshot_latents_match_kolors_pipeline(references):
    from imagehub.backends.kolors import KolorsBackend

    case = CASES["negative-non-square"]
    root, ref = references[0]
    expected = _reference_call(ref, case, output_type="latent").images
    backend = KolorsBackend(root).load()
    try:
        components = backend.components
        assert components is not None
        from imagehub.backends.kolors import denoise

        height, width = case["hw"]
        noise = torch.randn(
            (1, 4, height // 8, width // 8),
            generator=torch.Generator("cpu").manual_seed(case["seed"]),
        )
        actual = denoise(
            components,
            prompt=case["prompt"],
            negative_prompt=case["negative_prompt"],
            steps=case["steps"],
            guidance=case["guidance"],
            latents=noise,
            original_size=case["hw"],
            crops_coords_top_left=(0, 0),
            target_size=case["hw"],
        )
    finally:
        backend.unload()
    assert torch.equal(actual, expected)
