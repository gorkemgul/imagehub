"""Tiny Ovis-Image end-to-end parity against ``OvisImagePipeline``."""

import pytest
import torch
from helpers import write_tiny_ovis_image

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("numpy")

CASES = {
    "defaults": dict(prompt="a cat", steps=2, guidance=5.0, height=32, width=32, seed=0),
    "no-guidance": dict(prompt="a cat", steps=2, guidance=0.0, height=32, width=48, seed=42),
    "non-square-guidance": dict(
        prompt="a blue cat",
        negative_prompt="blurry",
        steps=3,
        guidance=2.5,
        height=48,
        width=32,
        seed=7,
    ),
}


def _reference(root):
    value = diffusers.OvisImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    value.set_progress_bar_config(disable=True)
    return value


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    return {
        seed: (
            root := write_tiny_ovis_image(tmp_path_factory.mktemp(f"ovis{seed}"), seed=seed),
            _reference(root),
        )
        for seed in (0, 1)
    }


@pytest.mark.parametrize("weights_seed", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tiny_snapshot_matches_ovis_image_pipeline(references, name, weights_seed, record):
    case = CASES[name]
    root, reference = references[weights_seed]
    expected = reference(
        case["prompt"],
        negative_prompt=case.get("negative_prompt", ""),
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        height=case["height"],
        width=case["width"],
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    state = torch.get_rng_state()
    result = pipeline("t2i", backend="ovis-image-7b", model=root)(
        case["prompt"],
        negative_prompt=case.get("negative_prompt", ""),
        steps=case["steps"],
        guidance=case["guidance"],
        height=case["height"],
        width=case["width"],
        seed=case["seed"],
    )
    assert torch.equal(torch.get_rng_state(), state), "global random state was touched"
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected), int(
        (actual.to(torch.int16) - expected.to(torch.int16)).abs().max()
    )
    record(f"ovis_image_tiny_{name}_{weights_seed}", max_abs_diff=0.0, tolerance="exact")
