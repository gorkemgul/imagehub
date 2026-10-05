"""T5 encoder structural parity with random weights."""

import pytest
import torch

from imagehub.models.t5 import T5EncoderModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

TINY_CONFIG = {
    "vocab_size": 128,
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "feed_forward_proj": "gated-gelu",
    "dropout_rate": 0.0,
}
MEDIUM_CONFIG = {
    "vocab_size": 128,
    "d_model": 512,
    "d_kv": 64,
    "d_ff": 1024,
    "num_layers": 4,
    "num_heads": 8,
    "feed_forward_proj": "gated-gelu",
    "dropout_rate": 0.0,
}


@pytest.mark.parametrize(
    ("config", "sequence_length", "masked"),
    [
        (TINY_CONFIG, 256, False),
        (TINY_CONFIG, 256, True),
        (MEDIUM_CONFIG, 77, False),
        (MEDIUM_CONFIG, 77, True),
    ],
    ids=["tiny", "tiny-mask", "medium", "medium-mask"],
)
def test_t5_encoder_random_weights(
    config: dict[str, int | float | str], sequence_length: int, masked: bool
) -> None:
    torch.manual_seed(0)
    reference_config = transformers.T5Config(**config)
    reference = transformers.T5EncoderModel(reference_config).eval()
    ours = T5EncoderModel(config).eval()
    state = {
        name: value
        for name, value in reference.state_dict().items()
        if name != "encoder.embed_tokens.weight"
    }
    ours.load_checkpoint(state)

    input_ids = torch.randint(0, int(config["vocab_size"]), (2, sequence_length))
    attention_mask = None
    if masked:
        attention_mask = torch.ones(2, sequence_length, dtype=torch.long)
        attention_mask[0, sequence_length // 2 :] = 0
        attention_mask[1, -3:] = 0

    with torch.inference_mode():
        expected = reference(input_ids, attention_mask=attention_mask).last_hidden_state
        actual = ours(input_ids, attention_mask=attention_mask)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
