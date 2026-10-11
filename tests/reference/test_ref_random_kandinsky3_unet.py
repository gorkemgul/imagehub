"""Kandinsky 3 U-Net structural parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.kandinsky3_unet import Kandinsky3UNet

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

PUBLISHED = json.loads(
    (Path(__file__).parents[1] / "fixtures/kandinsky3_configs/unet_config.json").read_text()
)
TINY = {
    "in_channels": 4,
    "time_embedding_dim": 32,
    "groups": 8,
    "attention_head_dim": 8,
    "layers_per_block": 3,
    "block_out_channels": [32, 64, 64, 128],
    "cross_attention_dim": 32,
    "encoder_hid_dim": 24,
}
# The published widths and depth, with a small context and 64-pixel latents.
PUBLISHED_SMALL = {
    **{key: value for key, value in PUBLISHED.items() if not key.startswith("_")},
    "encoder_hid_dim": 64,
    "cross_attention_dim": 64,
}


def _random_pair(config):
    reference = diffusers.Kandinsky3UNet(**config).eval()
    generator = torch.Generator().manual_seed(0)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.05, generator=generator)
    ours = Kandinsky3UNet(config).eval()
    ours.load_checkpoint(reference.state_dict())
    return reference, ours


def _inputs(config, size, tokens, *, batch=2, masked=True, zero_row=False):
    generator = torch.Generator().manual_seed(1)
    sample = torch.randn(batch, config["in_channels"], size, size, generator=generator)
    context = torch.randn(batch, tokens, config["encoder_hid_dim"], generator=generator)
    mask = torch.ones(batch, tokens, dtype=torch.bool)
    if masked:
        mask[0, tokens // 2 :] = False
        mask[-1, -1] = False
    if zero_row:
        mask[-1] = False
    context = context * mask.unsqueeze(2)
    return sample, context, mask


@pytest.mark.parametrize(
    ("size", "tokens", "masked", "zero_row"),
    [(16, 7, False, False), (16, 7, True, False), (24, 9, True, True), (8, 5, True, False)],
    ids=["unmasked", "masked", "non-square-tokens-zero-row", "small"],
)
@pytest.mark.parametrize("timestep", [999, 1])
def test_tiny_unet_matches_reference(size, tokens, masked, zero_row, timestep):
    reference, ours = _random_pair(TINY)
    sample, context, mask = _inputs(TINY, size, tokens, masked=masked, zero_row=zero_row)
    with torch.inference_mode():
        expected = reference(
            sample,
            torch.tensor(timestep),
            encoder_hidden_states=context,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        actual = ours(sample, torch.tensor(timestep), context, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_tiny_unet_float_mask_and_python_timestep():
    reference, ours = _random_pair(TINY)
    sample, context, mask = _inputs(TINY, 16, 7)
    with torch.inference_mode():
        expected = reference(
            sample,
            17,
            encoder_hidden_states=context,
            encoder_attention_mask=mask.long(),
            return_dict=False,
        )[0]
        actual = ours(sample, 17, context, mask.long())
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_published_depth_unet_matches_reference_at_small_sizes():
    reference, ours = _random_pair(PUBLISHED_SMALL)
    sample, context, mask = _inputs(PUBLISHED_SMALL, 16, 11, batch=2)
    with torch.inference_mode():
        expected = reference(
            sample,
            torch.tensor(881),
            encoder_hidden_states=context,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        actual = ours(sample, torch.tensor(881), context, mask)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
