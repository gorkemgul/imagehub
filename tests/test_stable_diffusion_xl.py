import json
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline
from imagehub.backends.stable_diffusion import initial_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLBackend
from imagehub.checkpoints import SNAPSHOT_FILE
from imagehub.exceptions import CheckpointError


@pytest.fixture(scope="module")
def tiny_sdxl(tmp_path_factory):
    return write_tiny_sdxl(tmp_path_factory.mktemp("tiny-sdxl"))


@pytest.fixture(scope="module")
def tiny_turbo(tmp_path_factory):
    config = json.loads(
        (
            __import__("pathlib").Path(__file__).parent
            / "fixtures/sdxl_turbo_configs/scheduler_config.json"
        ).read_text()
    )
    return write_tiny_sdxl(
        tmp_path_factory.mktemp("tiny-turbo"), scheduler_config=config, sample_size=4
    )


@pytest.fixture(scope="module")
def tiny_playground(tmp_path_factory):
    config = json.loads(
        (
            Path(__file__).parent / "fixtures/playground_v25_configs/scheduler_config.json"
        ).read_text()
    )
    return write_tiny_sdxl(
        tmp_path_factory.mktemp("tiny-playground"), scheduler_config=config, sample_size=4
    )


def test_generates_images_with_metadata(tiny_sdxl):
    result = pipeline("t2i", backend="stable-diffusion-xl", model=tiny_sdxl)(
        "a cat", seed=3, steps=4, width=64, height=48, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (64, 48)
    assert result.metadata["scheduler"] == "euler"
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 4
    assert result.metadata["device"] == "cpu" and result.metadata["dtype"] == "float32"


def test_seed_and_latents_are_reproducible(tiny_sdxl):
    pipe = pipeline("t2i", backend="sdxl-base-1.0", model=tiny_sdxl)
    a = pipe("a cat", seed=11, steps=3, width=32, height=32)
    b = pipe("a cat", seed=11, steps=3, width=32, height=32)
    latents = initial_latents(11, (1, 4, 4, 4))
    c = pipe("a cat", latents=latents, steps=3, width=32, height=32)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_turbo_uses_model_scheduler_defaults_and_native_size(tiny_turbo):
    result = pipeline("t2i", backend="sdxl-turbo", model=tiny_turbo)("a cat", seed=7)
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.metadata["steps"] == 1
    assert result.metadata["guidance"] == 0.0
    assert result.metadata["scheduler"] == "euler_ancestral"


def test_playground_v2_uses_model_guidance_and_euler_ancestral_scheduler(tmp_path):
    config = json.loads(
        (Path(__file__).parent / "fixtures/playground_v2_configs/scheduler_config.json").read_text()
    )
    root = write_tiny_sdxl(tmp_path / "tiny-playground-v2", scheduler_config=config, sample_size=4)
    result = pipeline("t2i", backend="playground-v2", model=root)("a cat", seed=7, steps=1)
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.metadata["guidance"] == 3.0
    assert result.metadata["scheduler"] == "euler_ancestral"


def test_playground_uses_model_guidance_and_edm_scheduler(tiny_playground):
    result = pipeline("t2i", backend="playground-v2.5", model=tiny_playground)(
        "a cat", seed=7, steps=1
    )
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.metadata["steps"] == 1
    assert result.metadata["guidance"] == 3.0
    assert result.metadata["scheduler"] == "edm_dpmsolver_multistep"


@pytest.mark.parametrize(
    ("model_id", "fixture", "scheduler_name"),
    [
        ("juggernaut-xl-v9", "juggernaut_xl_v9_configs", "ddpm"),
        ("dreamshaper-xl-1.0", "dreamshaper_xl_1_configs", "deis"),
    ],
)
def test_finetunes_use_published_schedulers(tmp_path, model_id, fixture, scheduler_name):
    config = json.loads(
        (Path(__file__).parent / "fixtures" / fixture / "scheduler.json").read_text()
    )
    root = write_tiny_sdxl(tmp_path / f"tiny-{model_id}", scheduler_config=config, sample_size=4)
    result = pipeline("t2i", backend=model_id, model=root)(
        "a cat", seed=7, steps=2, width=32, height=32
    )
    assert (result.images[0].width, result.images[0].height) == (32, 32)
    assert result.metadata["scheduler"] == scheduler_name


def test_lightning_uses_scheduler_overrides_and_model_defaults(tiny_sdxl):
    backend = StableDiffusionXLBackend(tiny_sdxl, model_id="sdxl-lightning-2step").load()
    assert backend.components is not None
    assert backend.components.scheduler_config["timestep_spacing"] == "trailing"
    result = backend.infer(
        __import__("imagehub.types", fromlist=["T2IRequest"]).T2IRequest("cat"),
        seed=0,
        width=32,
        height=32,
    )
    assert result.metadata["steps"] == 2
    assert result.metadata["guidance"] == 0.0
    assert result.metadata["scheduler"] == "euler"
    assert result.metadata["scheduler_overrides"] == {"timestep_spacing": "trailing"}


@pytest.mark.parametrize("model_id", ["koala-lightning-700m", "koala-lightning-1b"])
def test_koala_lightning_uses_published_sampling_defaults(tiny_sdxl, model_id):
    backend = StableDiffusionXLBackend(tiny_sdxl, model_id=model_id).load()
    assert backend.components is not None
    assert backend.components.scheduler_config["timestep_spacing"] == "trailing"
    result = backend.infer(
        __import__("imagehub.types", fromlist=["T2IRequest"]).T2IRequest("cat"),
        seed=0,
        width=32,
        height=32,
    )
    assert result.metadata["steps"] == 10
    assert result.metadata["guidance"] == 3.5
    assert result.metadata["scheduler"] == "euler"
    assert result.metadata["scheduler_overrides"] == {"timestep_spacing": "trailing"}


def test_hyper_sd_uses_lcm_scheduler_and_explicit_timestep_defaults(tiny_sdxl):
    backend = StableDiffusionXLBackend(tiny_sdxl, model_id="hyper-sdxl-1step").load()
    assert backend.components is not None
    assert backend.components.scheduler_config["_class_name"] == "LCMScheduler"
    result = backend.infer(
        __import__("imagehub.types", fromlist=["T2IRequest"]).T2IRequest("cat"),
        seed=0,
        width=32,
        height=32,
    )
    assert result.metadata["steps"] == 1
    assert result.metadata["guidance"] == 0.0
    assert result.metadata["timesteps"] == [800]
    assert result.metadata["scheduler"] == "lcm"
    assert result.metadata["scheduler_overrides"] == {"_class_name": "LCMScheduler"}


def test_explicit_timesteps_set_the_step_count(tiny_sdxl):
    backend = StableDiffusionXLBackend(tiny_sdxl, model_id="hyper-sdxl-1step")
    options = backend.validate_options(
        __import__("imagehub.types", fromlist=["T2IRequest"]).T2IRequest("cat"),
        {"timesteps": [999, 749, 499, 249]},
    )
    assert options["steps"] == 4


def test_unknown_scheduler_fails(tiny_sdxl, tmp_path):
    import shutil

    copy = shutil.copytree(tiny_sdxl, tmp_path / "unknown-scheduler")
    scheduler_path = copy / "scheduler/scheduler_config.json"
    config = json.loads(scheduler_path.read_text())
    scheduler_path.write_text(json.dumps(config | {"_class_name": "UnknownScheduler"}))
    with pytest.raises(ValueError, match="Unsupported SDXL scheduler UnknownScheduler"):
        pipeline("t2i", backend="stable-diffusion-xl", model=copy)(
            "a cat", seed=1, steps=1, width=32, height=32
        )


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 30}, ValueError, "multiple of 8"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 4, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 4, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"original_size": (1,)}, ValueError, "original_size"),
        ({"target_size": [32, 32]}, ValueError, "target_size"),
        ({"crops_coords_top_left": (0, -1)}, ValueError, "crops_coords_top_left"),
        ({"timesteps": (800,)}, ValueError, "timesteps"),
        ({"timesteps": [800.0]}, ValueError, "timesteps"),
        ({"steps": 2, "timesteps": [800]}, ValueError, r"len\(timesteps\)"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_sdxl, options, error, match):
    backend = StableDiffusionXLBackend(tiny_sdxl)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_missing_snapshot_hint_names_selected_model(tmp_path):
    pipe = pipeline("t2i", backend="sdxl-base-1.0", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained sdxl-base-1.0 "):
        pipe("a cat")


def test_snapshot_metadata_and_mismatch_warning(tiny_sdxl, tmp_path):
    import shutil

    copy = shutil.copytree(tiny_sdxl, tmp_path / "copy")
    (copy / SNAPSHOT_FILE).write_text(json.dumps({"model_id": "other-sdxl", "revision": "f" * 40}))
    with pytest.warns(UserWarning, match="other-sdxl.*sdxl-base-1.0"):
        backend = StableDiffusionXLBackend(copy, model_id="sdxl-base-1.0").load()
    result = backend.infer(
        __import__("imagehub.types", fromlist=["T2IRequest"]).T2IRequest("cat"),
        seed=0,
        steps=1,
        width=32,
        height=32,
    )
    assert result.metadata["model_id"] == "other-sdxl"
    assert result.metadata["revision"] == "f" * 40


def _run(pipe, prompt="a cat", **options):
    return pipe(prompt, **({"steps": 2, "width": 32, "height": 32} | options))


def test_requests_are_isolated(tiny_sdxl):
    pipe = pipeline("t2i", backend="stable-diffusion-xl", model=tiny_sdxl)
    first = _run(pipe, seed=1)
    _run(
        pipe,
        "a hi cat",
        seed=2,
        negative_prompt="blurry",
        num_images=2,
        guidance=3.0,
        original_size=(64, 48),
        crops_coords_top_left=(1, 2),
        target_size=(32, 32),
    )
    again = _run(pipe, seed=1)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_sdxl, grad_before):
    pipe = pipeline("t2i", backend="stable-diffusion-xl", model=tiny_sdxl)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = _run(pipe, seed=1)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)
