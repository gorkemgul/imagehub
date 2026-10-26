"""Published SDXS-512 configuration and tensor-layout checks."""

import json
from pathlib import Path

import torch

from imagehub.catalog import MODEL_CATALOG
from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel

FIXTURES = Path(__file__).parent / "fixtures/sdxs_configs"
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderTiny,
}


def test_sdxs_catalog_is_pinned_to_published_full_precision_files():
    spec = MODEL_CATALOG["sdxs-512-0.9"]
    assert spec.revision == "c332f05f60eb4b453de513be52c2a18c48d8cfe6"
    assert spec.catalog_id == "sdxs"
    assert spec.license == "CreativeML-OpenRAIL++-M"
    assert spec.config["defaults"] == {"steps": 1, "guidance": 0.0}
    assert "text_encoder/model.safetensors" in spec.files
    assert "unet/diffusion_pytorch_model.safetensors" in spec.files
    assert "vae/diffusion_pytorch_model.safetensors" in spec.files
    assert all("vae_large" not in name and not name.endswith(".bin") for name in spec.files)


def test_sdxs_published_tensor_headers_match_native_components():
    for component, model_class in COMPONENTS.items():
        config = json.loads((FIXTURES / f"{component}_config.json").read_text())
        published = json.loads((FIXTURES / f"{component}_header.json").read_text())
        with torch.device("meta"):
            model = model_class(config)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in published.items()}
        )
        assert {name: list(tensor.shape) for name, tensor in prepared.items()} == {
            name: list(tensor.shape) for name, tensor in model.state_dict().items()
        }
