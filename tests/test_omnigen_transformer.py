import json
from pathlib import Path

import pytest
import torch

from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.omnigen_transformer import OmniGenTransformer2DModel

FIXTURES = Path(__file__).parent / "fixtures" / "omnigen_configs"
TINY = json.loads(
    json.dumps(
        {
            "in_channels": 4,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 48,
            "num_layers": 2,
            "vocab_size": 100,
            "pad_token_id": 0,
            "pos_embed_max_size": 8,
            "time_step_dim": 16,
            "rope_scaling": {
                "short_factor": [1.0] * 4,
                "long_factor": [2.0] * 4,
                "type": "su",
            },
        }
    )
)


def test_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "transformer_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = OmniGenTransformer2DModel(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected


def test_vae_matches_published_tensor_layout():
    config = json.loads((FIXTURES / "vae_config.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURES / "vae_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = AutoencoderKL(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == model.prepare_state_dict(expected)


def _inputs():
    ids = torch.tensor([[1, 5, 6, 0, 0], [1, 7, 8, 9, 9]])
    image_latents = [torch.randn(1, 4, 4, 4)]  # 2x2 patches = 4 tokens
    sizes = {0: [[1, 5]]}
    seq = 5 + 1 + 6
    mask = torch.tril(torch.ones(seq, seq)).expand(2, seq, seq).contiguous()
    position = torch.arange(seq).expand(2, seq).contiguous()
    return ids, image_latents, sizes, mask, position


def test_tiny_forward_shape_and_image_dependence():
    torch.manual_seed(0)
    model = OmniGenTransformer2DModel(TINY).eval()
    model.patch_embedding.pos_embed.normal_()
    ids, image_latents, sizes, mask, position = _inputs()
    hidden = torch.randn(2, 4, 4, 6)
    timestep = torch.tensor([0.25, 0.25])
    with torch.inference_mode():
        out = model(hidden, timestep, ids, image_latents, sizes, mask, position)
        other = model(hidden, timestep, ids, [image_latents[0] * 2], sizes, mask, position)
    assert out.shape == hidden.shape
    assert not torch.equal(out[0], other[0])
    assert torch.equal(out[1], other[1])


def test_loads_on_meta_then_moves(tmp_path):
    from imagehub.checkpoints import save_safetensors

    torch.manual_seed(0)
    model = OmniGenTransformer2DModel(TINY)
    (tmp_path / "config.json").write_text(json.dumps(TINY))
    save_safetensors(dict(model.state_dict()), tmp_path / "diffusion_pytorch_model.safetensors")
    loaded = OmniGenTransformer2DModel.from_pretrained(tmp_path, dtype=torch.float32)
    assert all(value.device.type == "cpu" for value in loaded.state_dict().values())
    assert torch.equal(loaded.embed_tokens.weight, model.embed_tokens.weight)


@pytest.mark.parametrize(
    "override",
    [
        {"num_key_value_heads": 2},
        {"rope_scaling": None},
        {"rope_scaling": {"short_factor": [1.0], "long_factor": [1.0]}},
        {"timestep_activation_fn": "gelu"},
        {"flip_sin_to_cos": False},
        {"downscale_freq_shift": 1},
    ],
)
def test_unsupported_option_rejected(override):
    with pytest.raises(ValueError, match="Unsupported OmniGen transformer setting"):
        OmniGenTransformer2DModel(TINY | override)


def test_unknown_option_rejected():
    with pytest.raises(ValueError, match="Unsupported OmniGen transformer setting bogus"):
        OmniGenTransformer2DModel(TINY | {"bogus": 1})
