"""GLM-Image autoregressive model: random-weight sampling parity with transformers."""

import pytest
import torch

from imagehub.models.glm_image_ar import GlmImageARModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

VISION = {
    "depth": 1,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "image_size": 64,
    "patch_size": 16,
}
VQ = {"embed_dim": 16, "num_embeddings": 64, "latent_channels": 32}


def _config(text, **extra):
    return {
        "image_start_token_id": 70,
        "image_end_token_id": 71,
        "image_token_id": 95,
        "text_config": text,
        "vision_config": VISION,
        "vq_config": VQ,
    } | extra


TINY = _config(
    {
        "vocab_size": 100,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "pad_token_id": 90,
        "vision_vocab_size": 80,
        "eos_token_id": 71,
        "rope_parameters": {
            "rope_theta": 10000,
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 0.5,
        },
    }
)
# The published geometry (head size 128, multi-section rotary, 32 query and 2 key/value heads,
# the 16512-token vision head) with two layers and a text vocabulary cut to the image tokens.
PUBLISHED = _config(
    {
        "vocab_size": 16512,
        "hidden_size": 4096,
        "intermediate_size": 13696,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "pad_token_id": 16511,
        "vision_vocab_size": 16512,
        "eos_token_id": 16385,
        "rope_parameters": {
            "rope_theta": 10000,
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "partial_rotary_factor": 0.5,
        },
    }
)


def _reference(config):
    torch.manual_seed(0)
    model = transformers.GlmImageForConditionalGeneration(
        transformers.GlmImageConfig(**config)
    ).eval()
    for parameter in model.parameters():
        if parameter.ndim > 1:
            torch.nn.init.normal_(parameter, std=0.2 if config is TINY else 0.02)
    return model


@pytest.mark.parametrize(
    ("config", "grids", "length", "top_k", "top_p", "temperature"),
    [
        (TINY, [(4, 4), (2, 2)], 11, 80, 0.75, 0.9),
        (TINY, [(2, 3), (1, 2)], 5, 20, 1.0, 1.0),
        (TINY, [(3, 2), (2, 1)], 1, 80, 0.5, 1.3),
        (PUBLISHED, [(3, 2), (2, 1)], 9, 16512, 0.75, 0.9),
    ],
    ids=["tiny", "tiny-top-k-only", "tiny-one-token-prompt", "published-geometry"],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sampled_tokens_match_transformers_exactly(
    config, grids, length, top_k, top_p, temperature, seed
):
    reference = _reference(config)
    ours = GlmImageARModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    text = config["text_config"]
    ids = torch.randint(0, min(text["vision_vocab_size"], 60), (1, length))
    thw = torch.tensor([[1, h, w] for h, w in grids])
    new_tokens = sum(h * w for h, w in grids) + 1
    torch.manual_seed(seed)
    expected = reference.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        image_grid_thw=thw,
        images_per_sample=torch.tensor([len(grids)]),
        max_new_tokens=new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=text["pad_token_id"],
        eos_token_id=text["eos_token_id"],
    )
    actual = ours.generate(
        ids,
        grids,
        max_new_tokens=new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    assert torch.equal(actual, expected)
