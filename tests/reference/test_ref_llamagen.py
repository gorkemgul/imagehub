"""Parity of the LlamaGen GPT and VQ-16 decoder against the official code on CPU float32."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

from imagehub.models.llamagen import LlamaGenGPT, gpt_xl_config
from imagehub.models.llamagen_vq import LlamaGenVQ

pytestmark = pytest.mark.reference

REFERENCE_ROOT = Path("/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/LlamaGen")
_REFERENCE_PACKAGES = ("autoregressive", "tokenizer", "utils")


@pytest.fixture(scope="module")
def llamagen_reference():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"LlamaGen reference checkout not found at {REFERENCE_ROOT}")
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        gpt = importlib.import_module("autoregressive.models.gpt")
        generate = importlib.import_module("autoregressive.models.generate")
        vq = importlib.import_module("tokenizer.tokenizer_image.vq_model")
        yield gpt, generate, vq
    finally:
        sys.path.remove(str(REFERENCE_ROOT))
        for name in list(sys.modules):
            if name.split(".")[0] in _REFERENCE_PACKAGES:
                del sys.modules[name]


def _randomize_gpt(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.02, generator=generator)
        for module in model.modules():
            if type(module).__name__ == "RMSNorm":
                module.weight.uniform_(0.5, 1.5, generator=generator)
    return model.eval()


def _ours_from(reference, config):
    with torch.device("meta"):
        model = LlamaGenGPT(config)
    model.load_checkpoint(reference.state_dict())
    model.init_non_persistent_buffers()
    return model.eval()


def _compare_gpt(gpt, reference_config, config, steps, batch=2):
    torch.manual_seed(0)
    reference = _randomize_gpt(
        gpt.Transformer(gpt.ModelArgs(**reference_config, model_type="t2i")), seed=1
    )
    ours = _ours_from(reference, config)
    cls_tokens = config["cls_token_num"]
    caption_dim = config["caption_dim"]
    vocab = config["vocab_size"]
    generator = torch.Generator().manual_seed(2)
    caption = torch.randn(batch, cls_tokens, caption_dim, generator=generator)
    emb_masks = torch.ones(batch, cls_tokens, dtype=torch.int64)
    emb_masks[0, : cls_tokens // 3] = 0  # left padding as in the sampling script
    caption = caption * emb_masks[:, :, None]
    uncond = torch.zeros_like(caption) + reference.cls_embedding.uncond_embedding
    assert torch.equal(uncond, torch.zeros_like(caption) + ours.cls_embedding.uncond_embedding)
    combined = torch.cat([caption, uncond])
    tokens = torch.randint(0, vocab, (2 * batch, steps), generator=generator)

    total = cls_tokens + steps + 1
    with torch.device("cpu"):
        reference.setup_caches(2 * batch, total, torch.float32)
    ours.setup_caches(2 * batch, total, torch.float32)
    reference.causal_mask[:, :, :cls_tokens] = reference.causal_mask[:, :, :cls_tokens] * torch.cat(
        [emb_masks, emb_masks]
    ).unsqueeze(1)
    eye = torch.eye(reference.causal_mask.size(1), reference.causal_mask.size(2))
    reference.causal_mask[:] = reference.causal_mask * (1 - eye) + eye
    ours.apply_caption_mask(torch.cat([emb_masks, emb_masks]))
    assert torch.equal(reference.causal_mask, ours.causal_mask)

    with torch.no_grad():
        positions = torch.arange(0, cls_tokens)
        expected, _ = reference(None, combined, positions)
        actual = ours(None, combined, positions)
        assert torch.equal(actual, expected)
        input_pos = torch.tensor([cls_tokens], dtype=torch.int32)
        for step in range(steps):
            expected, _ = reference(tokens[:, step : step + 1], None, input_pos)
            actual = ours(tokens[:, step : step + 1], None, input_pos.clone())
            assert torch.equal(actual, expected), f"decode step {step}"
            input_pos += 1


def test_tiny_gpt_prefill_and_cached_decode_are_exact(llamagen_reference):
    gpt, _, _ = llamagen_reference
    # The reference embedder always allocates 120 unconditional caption rows.
    config = {
        "dim": 64,
        "n_layer": 3,
        "n_head": 4,
        "multiple_of": 16,
        "caption_dim": 24,
        "vocab_size": 128,
        "cls_token_num": 120,
        "block_size": 16,
    }
    reference_config = config
    _compare_gpt(gpt, reference_config, config, steps=6)


def test_published_xl_gpt_prefill_and_cached_decode_are_exact(llamagen_reference, record):
    gpt, _, _ = llamagen_reference
    config = gpt_xl_config(16)
    reference_config = {
        key: value
        for key, value in config.items()
        if key not in ("n_kv_head", "ffn_dim_multiplier")
    }
    _compare_gpt(gpt, reference_config, config, steps=3, batch=1)
    record("llamagen_gpt_xl_random_weights", prefill_equal=True, decode_steps=3)


def test_freqs_cis_matches_reference_at_both_sizes(llamagen_reference):
    gpt, _, _ = llamagen_reference
    for block_size in (256, 1024):
        expected = gpt.precompute_freqs_cis_2d(int(block_size**0.5), 64, 10000, 120)
        with torch.device("meta"):
            model = LlamaGenGPT(gpt_xl_config(block_size))
        model.init_non_persistent_buffers()
        assert torch.equal(model.freqs_cis, expected)


def _randomize_vq(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.uniform_(0.5, 1.5, generator=generator)
            elif "norm" in name:
                parameter.normal_(0.0, 0.1, generator=generator)
        model.quantize.embedding.weight.normal_(generator=generator)
    return model.eval()


def _compare_vq(vq, reference_kwargs, config, grid):
    torch.manual_seed(0)
    reference = _randomize_vq(vq.VQ_16(**reference_kwargs), seed=1)
    with torch.device("meta"):
        ours = LlamaGenVQ(config)
    ours.load_checkpoint(reference.state_dict())
    ours.eval()
    embed_dim = ours.config["codebook_embed_dim"]
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, ours.config["codebook_size"], (2, grid * grid), generator=generator)
    shape = (2, embed_dim, grid, grid)
    with torch.no_grad():
        expected = reference.decode_code(ids, shape)
        actual = ours.decode_code(ids, shape)
    assert actual.shape == (2, 3, 16 * grid, 16 * grid)
    assert torch.equal(actual, expected)


def test_tiny_vq_decode_code_is_exact(llamagen_reference):
    _, _, vq = llamagen_reference
    kwargs = {"codebook_size": 64, "codebook_embed_dim": 4, "z_channels": 32}
    _compare_vq(vq, kwargs, {"codebook_size": 64, "codebook_embed_dim": 4, "z_channels": 32}, 4)


def test_published_vq_decode_code_is_exact(llamagen_reference, record):
    _, _, vq = llamagen_reference
    kwargs = {"codebook_size": 16384, "codebook_embed_dim": 8}
    _compare_vq(vq, kwargs, {}, 16)
    record("llamagen_vq16_random_weights", decode_code_equal=True, ids="16x16")
