import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.hart import HARTBackend, sample_tokens
from imagehub.models.hart import HARTAutoencoder, HARTTransformer
from imagehub.models.qwen2 import Qwen2Model
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "hart_configs"


def _json(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_published_hart_transformer_layout():
    config = _json("llm/config.json") | {"latent_channels": 32, "vocab_size": 4096}
    with torch.device("meta"):
        model = HARTTransformer(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == _json("llm_tensors.json")["tensors"]


def test_published_hart_autoencoder_inference_layout():
    config = _json("tokenizer/config.json")
    published = _json("tokenizer_tensors.json")["tensors"]
    expected = {
        name.removeprefix("vae."): shape
        for name, shape in published.items()
        if name.startswith("vae.") and not name.startswith(("vae.encoder.", "vae.quant_conv."))
    }
    with torch.device("meta"):
        model = HARTAutoencoder(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected


def test_published_qwen2_text_layout():
    expected = {}
    for name in ("qwen_model-00001-header.json", "qwen_model-00002-header.json"):
        expected.update(
            (key, shape)
            for key, shape in _json(name).items()
            if key != "__metadata__" and key.startswith("model.")
        )
    with torch.device("meta"):
        model = Qwen2Model(_json("qwen_config.json"))
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected


def test_tiny_transformer_stage_and_diffusion_shapes():
    config = {
        "embed_dim": 32,
        "num_heads": 4,
        "depth": 2,
        "mlp_ratio": 2.0,
        "context_dim": 24,
        "context_token": 4,
        "patch_nums": (1, 2, 4),
        "latent_channels": 8,
        "vocab_size": 32,
        "diff_width": 32,
        "diff_depth": 2,
    }
    model = HARTTransformer(config).eval()
    context = torch.randn(2, 4, 32)
    positions = torch.arange(4).repeat(2, 1)
    for block in model.blocks:
        block.attn.kv_caching(True)
    with torch.no_grad():
        hidden, logits = model.stage_forward(context, context, 0, positions)
        assert hidden.shape == (2, 4, 32)
        assert logits.shape == (2, 4, 32)
        diffusion = model.diffloss.net(
            torch.randn(3, 8), torch.tensor([0, 100, 999]), torch.randn(3, 32)
        )
        assert diffusion.shape == (3, 16)


def test_tiny_qwen2_shape_and_padding_mask():
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "use_sliding_window": False,
    }
    model = Qwen2Model(config).eval()
    ids = torch.randint(0, 64, (2, 5))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        result = model(ids, mask)
    assert result.shape == (2, 5, 32)
    assert torch.isfinite(result).all()


def test_sampling_is_seeded():
    logits = torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(0))
    first = sample_tokens(
        logits.clone(), top_k=5, top_p=0.8, generator=torch.Generator().manual_seed(1)
    )
    second = sample_tokens(
        logits.clone(), top_k=5, top_p=0.8, generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(first, second)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unknown": 1}, TypeError, "Unknown HART options"),
        ({"width": 512}, ValueError, "width and height must be 1024"),
        ({"cfg_scale": float("nan")}, ValueError, "finite number"),
        ({"more_smooth": True}, ValueError, "not supported"),
    ],
)
def test_options_are_rejected_before_loading(tmp_path, options, error, message):
    backend = HARTBackend(tmp_path)
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded
