import json
from pathlib import Path

import torch

from imagehub.models.prx_transformer import PRXTransformer2DModel


def test_prx_transformer_matches_published_layout():
    root = Path(__file__).parent / "fixtures" / "prx_configs"
    config = json.loads((root / "prx-1024-t2i-beta" / "transformer" / "config.json").read_text())
    expected = json.loads(
        (
            root
            / "prx-1024-t2i-beta_headers"
            / "transformer_diffusion_pytorch_model.safetensors.json"
        ).read_text()
    )
    with torch.device("meta"):
        model = PRXTransformer2DModel(config)
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected
