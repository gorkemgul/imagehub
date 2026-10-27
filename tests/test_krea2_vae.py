import json
from pathlib import Path

import torch
from fixture_files import fixture_path

from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage

FIXTURES = fixture_path()
CONFIG_PATH = FIXTURES / "krea2_configs" / "vae_config.json"
TENSORS_PATH = FIXTURES / "krea2_vae_tensors.json"
CHECKPOINT = (
    Path(__file__).parents[1]
    / "checkpoints"
    / "krea2-turbo"
    / "vae"
    / "diffusion_pytorch_model.safetensors"
)


def test_matches_krea2_vae_tensor_layout():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = json.loads(TENSORS_PATH.read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = AutoencoderKLQwenImage(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}

    assert actual == expected
    if CHECKPOINT.exists():
        from imagehub.checkpoints import load_safetensors

        checkpoint = load_safetensors(CHECKPOINT)
        assert sorted(actual) == sorted(checkpoint)
