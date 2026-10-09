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
        dict(
            prompt="turbo no cfg",
            negative_prompt=None,
            steps=1,
            guidance=0.0,
            num=1,
            hw=(64, 64),
            seed=11,
            scheduler="euler-trailing",
        ),
        dict(
            prompt="turbo cfg",
            negative_prompt="blurry",
            steps=4,
            guidance=2.0,
            num=1,
            hw=(64, 64),
            seed=12,
            scheduler="euler-trailing",
        ),
    ],
    ids=["cfg", "negative-batch", "no-cfg", "euler-no-cfg", "euler-cfg"],
)
def test_tiny_snapshot_generation_matches_diffusers(tmp_path, case):
    root = write_tiny_sd(tmp_path / "tiny")
    if case.get("scheduler") == "euler-trailing":
        scheduler_path = root / "scheduler" / "scheduler_config.json"
        scheduler = __import__("json").loads(scheduler_path.read_text(encoding="utf-8"))
        scheduler["_class_name"] = "EulerDiscreteScheduler"
        scheduler["timestep_spacing"] = "trailing"
        scheduler_path.write_text(__import__("json").dumps(scheduler), encoding="utf-8")
        # diffusers picks the scheduler class from model_index.json, not from scheduler_config.json.
        index_path = root / "model_index.json"
        index = __import__("json").loads(index_path.read_text(encoding="utf-8"))
        index["scheduler"] = ["diffusers", "EulerDiscreteScheduler"]
        index_path.write_text(__import__("json").dumps(index), encoding="utf-8")
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


@pytest.mark.parametrize(
    "case",
    [
        dict(prompt="a cat", negative_prompt=None, steps=50, num=1, hw=(64, 64), seed=42),
        dict(prompt="a dog", negative_prompt="blurry", steps=25, num=2, hw=(48, 72), seed=7),
    ],
    ids=["v-pred-50", "v-pred-negative-batch"],
)
def test_tiny_v_prediction_ddim_upcast_attention_matches_diffusers(tmp_path, case):
    """SD 2.1 settings: DDIM, v-prediction, set_alpha_to_one=False and upcast_attention."""
    import json

    root = write_tiny_sd(tmp_path / "tiny")
    scheduler_path = root / "scheduler" / "scheduler_config.json"
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
    scheduler |= {
        "_class_name": "DDIMScheduler",
        "prediction_type": "v_prediction",
        "set_alpha_to_one": False,
        "steps_offset": 1,
        "clip_sample": False,
    }
    scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
    index_path = root / "model_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["scheduler"] = ["diffusers", "DDIMScheduler"]
    index_path.write_text(json.dumps(index), encoding="utf-8")
    unet_path = root / "unet" / "config.json"
    unet = json.loads(unet_path.read_text(encoding="utf-8"))
    unet["upcast_attention"] = True
    unet_path.write_text(json.dumps(unet), encoding="utf-8")
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
        guidance_scale=7.5,
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
        guidance=7.5,
        num_images=case["num"],
        height=h,
        width=w,
        seed=case["seed"],
    )
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)
