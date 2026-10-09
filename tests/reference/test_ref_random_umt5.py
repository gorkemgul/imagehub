"""UMT5 encoder parity with random weights."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.umt5 import UMT5EncoderModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

CONFIG = Path(__file__).parents[1] / "fixtures" / "auraflow_configs" / "text_encoder_config.json"
TINY = {
    "vocab_size": 128,
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 3,
    "num_heads": 4,
    "feed_forward_proj": "gated-gelu",
    "dense_act_fn": "gelu_new",
    "dropout_rate": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 2,
}


def _pair(config):
    torch.manual_seed(0)
    reference = transformers.UMT5EncoderModel(transformers.UMT5Config(**config)).eval()
    # Distinct per-layer biases, so a bias shared between layers could not pass.
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.5)
    ours = UMT5EncoderModel(config).eval()
    ours.load_checkpoint(
        {k: v for k, v in reference.state_dict().items() if k != "encoder.embed_tokens.weight"}
    )
    return reference, ours


def test_umt5_published_layout() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with torch.device("meta"):
        expected = transformers.UMT5EncoderModel(transformers.UMT5Config(**config)).state_dict()
        actual = UMT5EncoderModel(config).state_dict()
    expected.pop("encoder.embed_tokens.weight")
    assert {n: tuple(v.shape) for n, v in actual.items()} == {
        n: tuple(v.shape) for n, v in expected.items()
    }


@pytest.mark.parametrize("length", [7, 48])
def test_umt5_padded_batch_matches_reference(record, length) -> None:
    reference, ours = _pair(TINY)
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(3, TINY["vocab_size"], (3, length), generator=generator)
    mask = torch.ones(3, length, dtype=torch.long)
    mask[0, length // 2 :] = 0
    mask[1, -3:] = 0
    input_ids[mask == 0] = 1
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, attention_mask=mask).last_hidden_state
        actual = ours(input_ids, mask)[0]
        unmasked = ours(input_ids)[0]
        expected_unmasked = reference(input_ids=input_ids).last_hidden_state
    assert torch.equal(actual, expected)
    assert torch.equal(unmasked, expected_unmasked)
    assert not torch.equal(actual, unmasked)
    # The pipeline multiplies the embeddings by the mask.
    scaled = actual * mask.unsqueeze(-1).expand(actual.shape)
    assert torch.equal(scaled, expected * mask.unsqueeze(-1).expand(expected.shape))
    record("umt5_random", length=length, max_abs_diff=0.0, tolerance="exact")
