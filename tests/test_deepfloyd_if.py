import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import imagehub.backends.deepfloyd_if as deepfloyd_if
from imagehub.backends.deepfloyd_if import DeepFloydIFBackend
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "deepfloyd_if_configs"


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
def test_if_unet_matches_published_header(stage: str) -> None:
    root = FIXTURES / stage
    with torch.device("meta"):
        model = UNet2DConditionModel(json.loads((root / "unet_config.json").read_text()))
    header = json.loads((root / "unet.safetensors.header.json").read_text())
    tensors = model.state_dict()
    assert set(tensors) == set(header)
    assert all(list(tensors[name].shape) == shape for name, shape in header.items())


def test_if_scheduler_accepts_dynamic_thresholding_and_image_noise() -> None:
    config = json.loads((FIXTURES / "stage_i" / "scheduler_config.json").read_text())
    scheduler = DDPMScheduler(config)
    scheduler.set_timesteps(2)
    sample = torch.zeros(1, 3, 2, 2)
    output = scheduler.step(torch.zeros(1, 6, 2, 2), scheduler.timesteps[0], sample)
    assert output.shape == sample.shape
    assert torch.equal(scheduler.add_noise(sample, sample, torch.tensor([0])), sample)


def test_if_options_validate_before_loading(tmp_path: Path) -> None:
    backend = DeepFloydIFBackend(tmp_path, model_id="deepfloyd-if-i-xl")
    request = T2IRequest("a fox")
    assert backend.validate_options(request, {})["width"] == 256
    with pytest.raises(TypeError, match="Unknown"):
        backend.validate_options(request, {"unknown": 1})
    with pytest.raises(ValueError, match="fixed"):
        backend.validate_options(request, {"width": 128})


def test_stage_ii_initial_noise_precedes_upscaled_image_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    draws: list[int] = []

    def fake_noise(shape, generator, dtype, device):
        draws.append(len(draws) + 1)
        return torch.full(shape, draws[-1], dtype=dtype, device=device)

    def fake_denoise(unet, scheduler_config, context, latents, **kwargs):
        if tuple(latents.shape[-2:]) == (64, 64):
            return torch.zeros_like(latents)
        assert torch.equal(latents, torch.full_like(latents, 2))
        return torch.zeros_like(latents)

    class FakeNoiser:
        def __init__(self, config):
            pass

        def add_noise(self, sample, noise, timesteps):
            assert torch.equal(noise, torch.full_like(noise, 3))
            return sample

    monkeypatch.setattr(deepfloyd_if, "_noise", fake_noise)
    monkeypatch.setattr(deepfloyd_if, "_denoise", fake_denoise)
    monkeypatch.setattr(deepfloyd_if, "DDPMScheduler", FakeNoiser)
    monkeypatch.setattr(
        deepfloyd_if,
        "encode_prompt",
        lambda *args, **kwargs: torch.zeros(2, 1, 1),
    )

    backend = DeepFloydIFBackend(tmp_path, model_id="deepfloyd-if-i-xl")
    backend.components = cast(
        Any,
        SimpleNamespace(
            stage_i=object(),
            stage_i_scheduler={},
            stage_ii=object(),
            stage_ii_scheduler={},
            image_noising_scheduler={},
        ),
    )
    request = T2IRequest("a fox")
    options = backend.validate_options(request, {"seed": 0, "steps": 1, "superresolution_steps": 1})

    backend._infer(request, **options)

    assert draws == [1, 2, 3]
