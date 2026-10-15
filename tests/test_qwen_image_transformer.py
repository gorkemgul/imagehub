import json
from pathlib import Path

import torch

from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures" / "qwen_image_configs"


def test_published_qwen_image_transformer_headers_match_meta_model():
    config = json.loads((FIXTURES / "transformer/config.json").read_text(encoding="utf-8"))
    published = {}
    for path in sorted((FIXTURES / "headers").glob("transformer_*.json")):
        published.update(json.loads(path.read_text(encoding="utf-8")))
    with torch.device("meta"):
        model = QwenImageTransformer2DModel(config)
    native = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert native == published


def test_meta_model_recreates_non_persistent_buffers():
    config = {
        "patch_size": 2,
        "in_channels": 8,
        "out_channels": 2,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 12,
        "axes_dims_rope": [2, 2, 4],
    }
    with torch.device("meta"):
        model = QwenImageTransformer2DModel(config)
    model.init_non_persistent_buffers()
    assert model.pos_embed.pos_freqs.device.type == "cpu"
    assert model.pos_embed.pos_freqs.is_complex()
    assert model.time_text_embed.time_proj.freqs.device.type == "cpu"
