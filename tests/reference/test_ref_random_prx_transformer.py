"""PRX transformer structural parity with random weights."""

import pytest
import torch

from imagehub.models.prx_transformer import PRXTransformer2DModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = {
    "in_channels": 8,
    "patch_size": 2,
    "context_in_dim": 24,
    "hidden_size": 32,
    "mlp_ratio": 2.0,
    "num_heads": 4,
    "depth": 2,
    "axes_dim": [4, 4],
}


def test_prx_transformer_random_weights_exact(record):
    torch.manual_seed(0)
    reference = diffusers.PRXTransformer2DModel(**CONFIG).eval()
    model = PRXTransformer2DModel(CONFIG).eval()
    model.load_checkpoint(reference.state_dict())
    hidden_states = torch.randn(2, 8, 6, 10)
    timestep = torch.tensor([0.125, 0.875])
    text = torch.randn(2, 7, 24)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    with torch.inference_mode():
        expected = reference(hidden_states, timestep, text, attention_mask=mask, return_dict=False)[
            0
        ]
        actual = model(hidden_states, timestep, text, attention_mask=mask)
    difference = float((actual - expected).abs().max())
    record("prx_transformer_random", max_abs_diff=difference, tolerance="exact")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
