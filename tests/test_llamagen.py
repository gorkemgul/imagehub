import json
from pathlib import Path

import pytest
import torch

from imagehub.checkpoints import CheckpointError
from imagehub.models.llamagen import LlamaGenGPT, gpt_xl_config
from imagehub.models.llamagen_vq import LlamaGenVQ

FIXTURES = Path(__file__).parent / "fixtures" / "llamagen"

TINY_GPT = {
    "dim": 32,
    "n_layer": 2,
    "n_head": 4,
    "multiple_of": 8,
    "caption_dim": 16,
    "vocab_size": 64,
    "cls_token_num": 4,
    "block_size": 16,
}


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(("size", "block_size"), [("256", 256), ("512", 1024)])
def test_published_gpt_layout(size, block_size):
    fixture = _fixture(f"gpt_xl_{size}_tensors.json")
    assert fixture["block_size"] == block_size
    with torch.device("meta"):
        model = LlamaGenGPT(gpt_xl_config(block_size))
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == fixture["tensors"]
    assert list(actual) == list(fixture["tensors"])


def test_published_vq_layout():
    fixture = _fixture("vq16_decoder_tensors.json")
    with torch.device("meta"):
        model = LlamaGenVQ({})
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == fixture["tensors"]
    published = dict(fixture["tensors"])
    published["encoder.conv_in.weight"] = [128, 3, 3, 3]
    published["quant_conv.weight"] = [8, 256, 1, 1]
    published["quantize.codebook_used"] = [65536]
    assert set(model.prepare_state_dict(published)) == set(fixture["tensors"])


def _tiny_gpt():
    torch.manual_seed(0)
    model = LlamaGenGPT(TINY_GPT)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
        model.cls_embedding.uncond_embedding.normal_()
    return model.eval()


def test_non_persistent_buffers_are_rebuilt_after_meta_load():
    reference = _tiny_gpt()
    with torch.device("meta"):
        model = LlamaGenGPT(TINY_GPT)
    model.load_checkpoint(reference.state_dict())
    assert model.freqs_cis.device.type == "meta"
    model.init_non_persistent_buffers()
    assert torch.equal(model.freqs_cis, reference.freqs_cis)
    assert model.freqs_cis.shape == (4 + 16, 4, 2)
    assert not model.freqs_cis[:4].any()


def test_checkpoint_with_missing_tensor_is_rejected():
    model = _tiny_gpt()
    state = model.state_dict()
    state.pop("norm.weight")
    with pytest.raises(CheckpointError, match="norm.weight"):
        LlamaGenGPT(TINY_GPT).load_checkpoint(state)


def test_prepare_state_dict_drops_cache_tensors():
    model = _tiny_gpt()
    state = model.state_dict()
    state["layers.0.attention.kv_cache.k_cache"] = torch.zeros(1)
    assert set(model.prepare_state_dict(state)) == set(model.state_dict())


def test_cached_decode_matches_full_forward():
    model = _tiny_gpt()
    batch, text, grid = 2, 4, 16
    cond = torch.randn(batch, text, 16)
    tokens = torch.randint(0, 64, (batch, grid))
    with torch.no_grad():
        full = model(tokens, cond)
        assert full.shape == (batch, text + grid, 64)
        model.setup_caches(batch, text + grid, torch.float32)
        prefill = model(None, cond, torch.arange(text))
        assert prefill.shape == (batch, text, 64)
        torch.testing.assert_close(prefill, full[:, :text])
        for step in range(grid - 1):
            position = torch.tensor([text + step], dtype=torch.int32)
            logits = model(tokens[:, step : step + 1], None, position)
            assert logits.shape == (batch, 1, 64)
            torch.testing.assert_close(logits[:, 0], full[:, text + step])


def test_caption_mask_hides_padded_positions():
    model = _tiny_gpt()
    batch, text = 1, 4
    cond = torch.randn(batch, text, 16)
    model.setup_caches(batch, text + 2, torch.float32)
    model.apply_caption_mask(torch.tensor([[0, 0, 1, 1]]))
    assert model.causal_mask is not None
    assert model.max_seq_length == 8
    assert model.causal_mask[0, 3, :4].tolist() == [False, False, True, True]
    assert model.causal_mask[0, 0, 0].item() is True  # diagonal always visible
    with torch.no_grad():
        model(None, cond, torch.arange(text))


def test_decoding_without_caches_is_rejected():
    model = _tiny_gpt()
    with pytest.raises(RuntimeError, match="setup_caches"):
        model(None, torch.randn(1, 4, 16), torch.arange(4))


def test_vq_tiny_decode_code_shapes():
    config = {"codebook_size": 32, "codebook_embed_dim": 4, "z_channels": 32, "ch": 32}
    torch.manual_seed(0)
    model = LlamaGenVQ(config).eval()
    ids = torch.randint(0, 32, (2, 4 * 4))
    with torch.no_grad():
        image = model.decode_code(ids, (2, 4, 4, 4))
    assert image.shape == (2, 3, 64, 64)
    with torch.no_grad():
        channels_last = model.decode_code(ids, (2, 4, 4, 4), channel_first=False)
    assert channels_last.shape == (2, 3, 64, 64)
