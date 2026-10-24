import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.bagel import BagelBackend, BagelComponents, generate
from imagehub.models.bagel import BagelAutoencoder, BagelModel
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "bagel_7b_mot_configs"


def _tiny_config():
    return {
        "llm_config": {
            "vocab_size": 160000,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
        },
        "vit_config": {
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "patch_size": 2,
            "image_size": 4,
            "num_channels": 3,
        },
        "vae_config": {"z_channels": 16, "downsample": 8},
        "latent_patch_size": 2,
        "max_latent_size": 64,
        "vit_max_num_patch_per_side": 4,
    }


def test_published_tensor_layouts():
    config = json.loads((FIXTURES / "config.json").read_text())
    config["max_latent_size"] = 64
    expected = json.loads((FIXTURES / "ema_header.json").read_text())
    with torch.device("meta"):
        model = BagelModel(config)
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected

    autoencoder_header = json.loads((FIXTURES / "ae_header.json").read_text())
    with torch.device("meta"):
        autoencoder = BagelAutoencoder()
    prepared = autoencoder.prepare_state_dict(
        {name: torch.empty(shape, device="meta") for name, shape in autoencoder_header.items()}
    )
    assert {name: list(value.shape) for name, value in prepared.items()} == {
        name: list(value.shape) for name, value in autoencoder.state_dict().items()
    }


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unknown": 1}, TypeError, "Unknown BAGEL options"),
        ({"width": 17}, ValueError, "multiple of 16"),
        ({"guidance": float("nan")}, ValueError, "finite"),
        ({"steps": 1}, ValueError, "at least 2"),
        ({"cfg_interval": (0.8, 0.2)}, ValueError, "ordered"),
    ],
)
def test_options_are_rejected_before_loading(tmp_path, options, error, message):
    backend = BagelBackend(tmp_path)
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


class _Tokenizer:
    def tokenize(self, _text):
        return [7, 8]


def test_tiny_generation_is_repeatable_and_restores_grad_mode():
    torch.manual_seed(0)
    model = BagelModel(_tiny_config()).eval()
    vae = BagelAutoencoder({"ch": 32, "ch_mult": [1, 1, 1, 1], "num_res_blocks": 1}).eval()
    components = BagelComponents(_Tokenizer(), model, vae)
    options = dict(
        width=16,
        height=16,
        guidance=1.0,
        steps=2,
        timestep_shift=1.0,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
    )
    with torch.enable_grad():
        first = generate(components, "cat", generator=torch.Generator().manual_seed(5), **options)
        assert torch.is_grad_enabled()
    second = generate(components, "cat", generator=torch.Generator().manual_seed(5), **options)
    assert torch.equal(first, second)
    assert first.shape == (1, 3, 16, 16)
