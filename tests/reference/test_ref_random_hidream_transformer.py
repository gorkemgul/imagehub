"""HiDream-I1 transformer parity with random weights (exact, CPU float32)."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIGS = Path(__file__).parents[1] / "fixtures" / "hidream_configs"

TINY = {
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "caption_channels": [12, 20],
    "text_emb_dim": 10,
    "num_routed_experts": 4,
    "num_activated_experts": 2,
    "axes_dims_rope": [8, 4, 4],
    "max_resolution": [8, 8],
    "llama_layers": [0, 1, 2, 2],
}
# Published width, head layout and expert counts with fewer layers: the full 16 + 32 layer model
# (~17B parameters, 68 GB in float32) does not fit next to the reference in 96 GB.
PUBLISHED_WIDTH = (
    json.loads((CONFIGS / "transformer_config.json").read_text(encoding="utf-8"))
    | {"num_layers": 2, "num_single_layers": 2, "llama_layers": [0, 5, 31, 31]}
    | {"max_resolution": [16, 16]}
)


def _inputs(config, height, width, *, llama_layers, text_length=7, t5_length=5):
    generator = torch.Generator().manual_seed(1)
    batch = 2
    hidden_states = torch.randn(batch, config["in_channels"], height, width, generator=generator)
    t5 = torch.randn(batch, t5_length, config["caption_channels"][0], generator=generator)
    llama = torch.randn(
        llama_layers, batch, text_length, config["caption_channels"][1], generator=generator
    )
    pooled = torch.randn(batch, config["text_emb_dim"], generator=generator)
    timesteps = torch.tensor([999.0, 500.5])
    return hidden_states, timesteps, t5, llama, pooled


def _compare(config, height, width, *, llama_layers):
    torch.manual_seed(0)
    reference = diffusers.HiDreamImageTransformer2DModel(**config).eval()
    model = HiDreamImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    hidden_states, timesteps, t5, llama, pooled = _inputs(
        config, height, width, llama_layers=llama_layers
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden_states,
            timesteps=timesteps,
            encoder_hidden_states_t5=t5,
            encoder_hidden_states_llama3=llama,
            pooled_embeds=pooled,
            return_dict=False,
        )[0]
        actual = model(hidden_states, timesteps, t5, llama, pooled)
    assert actual.shape == (2, config["in_channels"], height, width)
    assert torch.equal(actual, expected)
    return reference


def test_hidream_published_layout():
    config = json.loads((CONFIGS / "transformer_config.json").read_text(encoding="utf-8"))
    published = json.loads((CONFIGS / "transformer_tensors.json").read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = diffusers.HiDreamImageTransformer2DModel(**config).state_dict()
        actual = HiDreamImageTransformer2DModel(config).state_dict()
    assert {name: list(value.shape) for name, value in expected.items()} == published
    assert {name: list(value.shape) for name, value in actual.items()} == published


@pytest.mark.parametrize(
    ("height", "width"),
    [(8, 8), (4, 8), (8, 6), (6, 4)],
    ids=["square", "wide", "wide-odd", "tall"],
)
def test_hidream_tiny_exact(height, width):
    _compare(TINY, height, width, llama_layers=3)


def test_hidream_tiny_default_max_resolution_exact():
    _compare(TINY | {"max_resolution": [128, 128]}, 6, 10, llama_layers=3)


def test_hidream_tiny_routes_every_expert():
    torch.manual_seed(0)
    model = HiDreamImageTransformer2DModel(TINY).eval()
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    gate = model.double_stream_blocks[0].block.ff_i.gate
    x = torch.randn(64, model.inner_dim)
    topk_idx, _ = gate(x)
    assert set(topk_idx.flatten().tolist()) == {0, 1, 2, 3}


@pytest.mark.parametrize(("height", "width"), [(8, 8), (8, 12)], ids=["square", "wide"])
def test_hidream_published_width_exact(height, width):
    _compare(PUBLISHED_WIDTH, height, width, llama_layers=32)
