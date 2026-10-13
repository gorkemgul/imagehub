"""LDM end-to-end parity on tiny random snapshots in the published layout."""

import pytest
import torch
from helpers import write_tiny_ldm

from imagehub import pipeline
from imagehub.backends.ldm import LDMBackend, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("numpy")

CASES = {
    "defaults": dict(prompt="a cat", steps=50, guidance=1.0, hw=(64, 64), seed=0),
    "guided-non-square": dict(prompt="a b", steps=3, guidance=2.5, hw=(64, 96), seed=7),
}


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    result = {}
    for weights_seed in (0, 1):
        root = write_tiny_ldm(tmp_path_factory.mktemp(f"ldm_{weights_seed}"), seed=weights_seed)
        reference = diffusers.LDMTextToImagePipeline.from_pretrained(
            root, local_files_only=True, torch_dtype=torch.float32
        )
        reference.set_progress_bar_config(disable=True)
        result[weights_seed] = root, reference
    return result


def _reference_call(reference, case, output_type):
    height, width = case["hw"]
    return reference(
        case["prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        height=height,
        width=width,
        eta=0.0,
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        output_type=output_type,
    ).images


@pytest.mark.parametrize("weights_seed", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tiny_snapshot_matches_ldm_pipeline(references, name, weights_seed, record):
    case = CASES[name]
    root, reference = references[weights_seed]
    expected = _reference_call(reference, case, "np")
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    height, width = case["hw"]
    actual = pipeline("t2i", backend="ldm", model=root)(
        case["prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        height=height,
        width=width,
        seed=case["seed"],
    )
    pixels = torch.stack([image.tensor for image in actual.images])
    assert torch.equal(pixels, expected)
    record(f"ldm_tiny_{name}_{weights_seed}", max_abs_diff=0.0, tolerance="exact")


@pytest.mark.parametrize("name", sorted(CASES))
def test_tiny_snapshot_latents_match_ldm_pipeline(references, name):
    case = CASES[name]
    root, reference = references[0]
    captured = []
    decode = reference.vqvae.decode
    scaling_factor = reference.vqvae.config.scaling_factor

    def capture_latents(latents, *args, **kwargs):
        captured.append(latents.detach().clone())
        return decode(latents, *args, **kwargs)

    reference.vqvae.decode = capture_latents
    reference.vqvae.config.scaling_factor = 1.0
    try:
        _reference_call(reference, case, "np")
    finally:
        reference.vqvae.decode = decode
        reference.vqvae.config.scaling_factor = scaling_factor
    assert len(captured) == 1
    expected = captured[0]
    backend = LDMBackend(root).load()
    try:
        assert backend.components is not None
        height, width = case["hw"]
        latents = torch.randn(
            (1, 4, height // 8, width // 8),
            generator=torch.Generator("cpu").manual_seed(case["seed"]),
        )
        actual = denoise(
            backend.components, case["prompt"], case | {"negative_prompt": None}, latents
        )
    finally:
        backend.unload()
    assert torch.equal(actual, expected)
