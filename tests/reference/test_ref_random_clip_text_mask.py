import pytest
import torch

from imagehub.models.clip_text import CLIPTextModelWithProjection

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

EOS = 63
BASE = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "max_position_embeddings": 12,
    "projection_dim": 16,
    "layer_norm_eps": 1e-5,
    "bos_token_id": 62,
    "eos_token_id": EOS,
}


def _inputs(lengths, total):
    ids = torch.full((len(lengths), total), EOS)
    mask = torch.zeros(len(lengths), total, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    for row, length in enumerate(lengths):
        ids[row, :length] = torch.randint(0, 60, (length,), generator=generator)
        ids[row, 0] = 62
        ids[row, length - 1] = EOS
        mask[row, :length] = 1
    return ids, mask


@pytest.mark.parametrize("activation", ["gelu", "quick_gelu"])
@pytest.mark.parametrize("case", ["padded", "all_ones"])
def test_clip_text_attention_mask_random_weights(activation, case, record):
    config = BASE | {"hidden_act": activation}
    torch.manual_seed(0)
    ref = transformers.CLIPTextModelWithProjection(transformers.CLIPTextConfig(**config)).eval()
    assert ref.config._attn_implementation == "sdpa"
    ours = CLIPTextModelWithProjection(config).eval()
    ours.load_checkpoint(ref.state_dict())
    ids, mask = _inputs([5, 12] if case == "padded" else [12, 12], 12)
    if case == "all_ones":
        mask = torch.ones_like(mask)
    with torch.inference_mode():
        embeds, last, hidden = ours(ids, attention_mask=mask, output_hidden_states=True)
        expected = ref(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    results = {
        "text_embeds": torch.equal(embeds, expected.text_embeds),
        "last_hidden_state": torch.equal(last, expected.last_hidden_state),
        "hidden_states": all(
            torch.equal(a, b) for a, b in zip(hidden, expected.hidden_states, strict=True)
        ),
    }
    record(f"clip_text_mask_{activation}_{case}", tolerance="exact", **results)
    assert all(results.values())
