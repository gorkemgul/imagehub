"""AuraFlow transformer random-weight parity against diffusers."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.auraflow_transformer import AuraFlowTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = Path(__file__).parents[1] / "fixtures" / "auraflow_configs" / "transformer_config.json"
TINY = {
    "sample_size": 8,
    "patch_size": 2,
    "in_channels": 4,
    "num_mmdit_layers": 2,
    "num_single_dit_layers": 3,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 12,
    "caption_projection_dim": 16,
    "out_channels": 4,
    "pos_embed_max_size": 64,
}


def _randomize(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _case(config, hidden, text, timestep, record, name):
    torch.manual_seed(0)
    reference = diffusers.AuraFlowTransformer2DModel(**config).eval()
    _randomize(reference, 1)
    model = AuraFlowTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    with torch.inference_mode():
        expected = reference(hidden, encoder_hidden_states=text, timestep=timestep).sample
        actual = model(hidden, text, timestep)
    record(name, max_abs=float((actual - expected).abs().max()), shape=list(actual.shape))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("shape", [(6, 10), (8, 8), (4, 14)])
def test_tiny_transformer(record, shape):
    generator = torch.Generator().manual_seed(2)
    hidden = torch.randn(2, 4, *shape, generator=generator)
    text = torch.randn(2, 7, 12, generator=generator)
    timestep = torch.tensor([0.8, 0.123])
    _case(TINY, hidden, text, timestep, record, f"auraflow_tiny_transformer_{shape[0]}x{shape[1]}")


def test_published_transformer_small_latents(record):
    config = {
        k: v
        for k, v in json.loads(CONFIG.read_text(encoding="utf-8")).items()
        if not k.startswith("_")
    }
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(2, 4, 32, 48, generator=generator)
    text = torch.randn(2, 256, config["joint_attention_dim"], generator=generator)
    timestep = torch.tensor([0.999, 0.4])
    _case(config, hidden, text, timestep, record, "auraflow_published_transformer")
