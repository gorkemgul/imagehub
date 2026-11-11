"""Exact random-weight parity for the Nucleus-Image transformer."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.nucleus_image_transformer import NucleusImageTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

PUBLISHED = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/nucleus_image_configs/transformer/config.json"
    ).read_text()
)

TINY = {
    "patch_size": 2,
    "in_channels": 16,
    "out_channels": 4,
    "num_layers": 5,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "joint_attention_dim": 24,
    "axes_dims_rope": [2, 2, 4],
    "mlp_ratio": 3.0,
    "moe_enabled": True,
    "dense_moe_strategy": "leave_first_three_blocks_dense",
    "num_experts": 4,
    "moe_intermediate_dim": 16,
    "capacity_factors": [0.0, 0.0, 0.0, 2.0, 2.0],
    "use_sigmoid": False,
    "route_scale": 2.5,
    "use_grouped_mm": False,
}
PUBLISHED_SMALL = PUBLISHED | {
    "in_channels": 16,
    "out_channels": 4,
    "num_layers": 5,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "joint_attention_dim": 24,
    "axes_dims_rope": [2, 2, 4],
    "num_experts": 4,
    "moe_intermediate_dim": 32,
    "capacity_factors": [0.0, 0.0, 0.0, 2.0, 2.0],
    "use_grouped_mm": False,
}


def _randomise(model):
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)


@pytest.mark.parametrize("config", [TINY, PUBLISHED_SMALL], ids=["tiny", "published-small"])
@pytest.mark.parametrize("masked", [False, True], ids=["unmasked", "masked"])
def test_nucleus_image_transformer_random_weights_exact(config, masked):
    torch.manual_seed(0)
    reference = diffusers.NucleusMoEImageTransformer2DModel(**config).eval()
    _randomise(reference)
    model = NucleusImageTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())

    hidden = torch.randn(2, 6, config["in_channels"])
    context = torch.randn(2, 7, config["joint_attention_dim"])
    timestep = torch.tensor([0.125, 0.75])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]) if masked else None
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            img_shapes=[(1, 2, 3)] * 2,
            encoder_hidden_states=context,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            return_dict=False,
        )[0]
        actual = model(hidden, context, timestep, [(1, 2, 3)] * 2, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_nucleus_image_moe_at_published_widths_and_1024_tokens_is_exact():
    # At 1024x1024 the first two expert blocks take 256 tokens per expert. With several CPU
    # threads, grouped products then differ in the last bit from one product per expert, so the
    # published `use_grouped_mm` must be followed. Small widths do not show the difference.
    config = PUBLISHED | {"num_layers": 4, "capacity_factors": PUBLISHED["capacity_factors"][:4]}
    reference = diffusers.NucleusMoEImageTransformer2DModel(**config).eval()
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    model = NucleusImageTransformer2DModel(config).eval()
    model.load_checkpoint({name: value.clone() for name, value in reference.state_dict().items()})

    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, 64 * 64, config["in_channels"], generator=generator)
    context = torch.randn(1, 24, config["joint_attention_dim"], generator=generator)
    timestep = torch.tensor([0.5])
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            img_shapes=[(1, 64, 64)],
            encoder_hidden_states=context,
            timestep=timestep,
            return_dict=False,
        )[0]
        actual = model(hidden, context, timestep, (1, 64, 64), None)
    assert torch.equal(actual, expected)
