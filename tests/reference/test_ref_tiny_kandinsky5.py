"""Kandinsky 5 end to end on a tiny random snapshot against the reference pipeline."""

import pytest
import torch
from helpers import write_tiny_kandinsky5

from imagehub import pipeline
from imagehub.backends.kandinsky5 import Kandinsky5Backend, _snap_resolution, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("numpy")
import diffusers.models.transformers.transformer_kandinsky as _kandinsky_transformer  # noqa: E402

CASES = {
    "defaults": dict(prompt="a cat", steps=4, guidance=3.5, hw=(64, 64), seed=0),
    "no-cfg": dict(prompt="a b a", steps=3, guidance=1.0, hw=(64, 64), seed=1),
    "negative-non-square": dict(
        prompt="ab abab", steps=5, guidance=4.5, hw=(64, 96), seed=7, negative_prompt="b a"
    ),
    "empty-negative": dict(
        prompt="a b", steps=3, guidance=2.0, hw=(64, 128), seed=3, negative_prompt=""
    ),
    "two-images": dict(prompt="ab", steps=3, guidance=3.5, hw=(64, 64), seed=5, num_images=2),
    "long-prompt": dict(prompt="ab " * 80, steps=2, guidance=3.5, hw=(64, 64), seed=2),
    "high-guidance-short": dict(prompt="a", steps=1, guidance=7.0, hw=(64, 64), seed=9),
}


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    result = {}
    # The reference transformer declares fp32 modules but low_cpu_mem_usage needs
    # accelerate; bypass that requirement for the tiny random snapshot.
    _kandinsky_transformer.Kandinsky5Transformer3DModel._keep_in_fp32_modules = None
    for seed in (0, 1):
        root = write_tiny_kandinsky5(tmp_path_factory.mktemp(f"kandinsky5_{seed}"), seed=seed)
        ref = diffusers.Kandinsky5T2IPipeline.from_pretrained(
            root,
            local_files_only=True,
            torch_dtype=torch.float32,
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
    expected = _reference_call(ref, case, output_type="np").image
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    height, width = case["hw"]
    result = pipeline("t2i", backend="kandinsky5", model=root)(
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
    record(f"kandinsky5_tiny_{name}_{weights_seed}", max_abs_diff=0.0, tolerance="exact")


@pytest.mark.parametrize("name", ["negative-non-square", "no-cfg", "two-images"])
def test_tiny_snapshot_latents_match_pipeline(references, name):
    case = CASES[name]
    root, ref = references[0]
    expected = _reference_call(ref, case, output_type="latent").image
    backend = Kandinsky5Backend(root).load()
    try:
        components = backend.components
        assert components is not None
        requested_height, requested_width = case["hw"]
        width, height = _snap_resolution(requested_width, requested_height)
        count = case.get("num_images", 1)
        generator = torch.Generator("cpu").manual_seed(case["seed"])
        latents = torch.randn(
            (count, 1, height // 8, width // 8, components.transformer.in_visual_dim),
            generator=generator,
        )
        actual = denoise(
            components,
            prompt=case["prompt"],
            negative_prompt=case.get("negative_prompt"),
            steps=case["steps"],
            guidance=case["guidance"],
            width=width,
            height=height,
            latents=latents,
            max_sequence_length=512,
        )
    finally:
        backend.unload()
    assert torch.equal(actual, expected)
