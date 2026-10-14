import json
from pathlib import Path

import pytest
import torch

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
