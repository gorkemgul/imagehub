import pytest
import torch
from helpers import write_tiny_sd

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.mark.parametrize(
    "case",
    [
        dict(
            prompt="a cat",
            negative_prompt=None,
            steps=50,
            guidance=7.5,
            num=1,
            hw=(64, 64),
            seed=42,
        ),
        dict(
            prompt="a dog",
            negative_prompt="blurry",
            steps=20,
            guidance=7.5,
            num=2,
            hw=(48, 72),
            seed=7,
        ),
        dict(
            prompt="no cfg",
            negative_prompt=None,
            steps=10,
            guidance=1.0,
            num=1,
            hw=(64, 64),
            seed=1,
        ),
    ],
    ids=["cfg", "negative-batch", "no-cfg"],
)
def test_tiny_snapshot_generation_matches_diffusers(tmp_path, case):
    root = write_tiny_sd(tmp_path / "tiny")
    ref = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    ref.set_progress_bar_config(disable=True)
    h, w = case["hw"]
    expected = ref(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        num_images_per_prompt=case["num"],
        height=h,
        width=w,
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="stable-diffusion", model=root)(
        case["prompt"],
        negative_prompt=case["negative_prompt"],
        steps=case["steps"],
        guidance=case["guidance"],
        num_images=case["num"],
        height=h,
        width=w,
        seed=case["seed"],
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)
