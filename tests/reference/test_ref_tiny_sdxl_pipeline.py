import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "case",
    [
        dict(prompt="a cat", negative_prompt=None, steps=4, guidance=5.0, num=1, hw=(32, 32)),
        dict(prompt="a dog", negative_prompt="blurry", steps=3, guidance=5.0, num=2, hw=(32, 48)),
        dict(prompt="no cfg", negative_prompt=None, steps=3, guidance=1.0, num=1, hw=(32, 32)),
        dict(
            prompt="crop",
            negative_prompt=None,
            steps=3,
            guidance=5.0,
            num=1,
            hw=(32, 40),
            original_size=(64, 80),
            crops_coords_top_left=(3, 4),
            target_size=(48, 56),
        ),
    ],
    ids=["zero-negative", "negative-batch-nonsquare", "no-cfg", "micro-conditioning"],
)
def test_tiny_sdxl_generation_matches_diffusers(tmp_path, case):
    root = write_tiny_sdxl(tmp_path / "tiny")
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
    )
    ref.set_progress_bar_config(disable=True)
    h, w = case["hw"]
    common = {
        key: case[key]
        for key in ("original_size", "crops_coords_top_left", "target_size")
        if key in case
    }
    expected = ref(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        num_images_per_prompt=case["num"],
        height=h,
        width=w,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
        **common,
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="stable-diffusion-xl", model=root)(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        num_images=case["num"],
        height=h,
        width=w,
        seed=42,
        **common,
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)
