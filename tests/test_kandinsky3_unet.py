import json
from pathlib import Path

import pytest
import torch

from imagehub.models.kandinsky3_unet import Kandinsky3UNet

FIXTURES = Path(__file__).parent / "fixtures" / "kandinsky3_configs"
TINY = {
    "in_channels": 4,
    "time_embedding_dim": 32,
    "groups": 8,
    "attention_head_dim": 8,
    "layers_per_block": 3,
    "block_out_channels": [32, 64, 64, 128],
    "cross_attention_dim": 32,
    "encoder_hid_dim": 24,
}


def test_matches_published_tensor_header_on_the_meta_device():
    config = json.loads((FIXTURES / "unet_config.json").read_text())
    published = json.loads((FIXTURES / "unet_tensors.json").read_text())
    with torch.device("meta"):
        model = Kandinsky3UNet(config)
    shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert shapes == published
    assert len(published) == 1071


def test_forward_shape_and_context_dependence():
    torch.manual_seed(0)
    model = Kandinsky3UNet(TINY).eval()
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)
    sample = torch.randn(2, 4, 16, 16)
    context = torch.randn(2, 5, 24)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    with torch.inference_mode():
        first = model(sample, torch.tensor(10), context, mask)
        second = model(sample, torch.tensor(10), context + 1, mask)
    assert first.shape == sample.shape
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ({"unknown": 1}, "Unsupported Kandinsky 3 U-Net setting unknown"),
        ({"layers_per_block": [3, 3]}, "layers_per_block"),
        ({"block_out_channels": [32, 64]}, "block_out_channels"),
    ],
)
def test_rejects_unsupported_settings(option, message):
    with pytest.raises(ValueError, match=message):
        Kandinsky3UNet(TINY | option)


def test_loads_on_meta_then_moves_to_device(tmp_path):
    from imagehub.checkpoints import save_safetensors

    model = Kandinsky3UNet(TINY)
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    save_safetensors(
        {name: value.half() for name, value in model.state_dict().items()},
        tmp_path / "diffusion_pytorch_model.fp16.safetensors",
    )
    loaded = Kandinsky3UNet.from_pretrained(tmp_path, dtype=torch.float32)
    assert next(loaded.parameters()).dtype == torch.float32
    assert next(loaded.parameters()).device.type == "cpu"
