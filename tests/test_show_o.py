import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.show_o import ShowOBackend, create_attention_mask, generate_tokens
from imagehub.checkpoints import save_safetensors
from imagehub.models.magvit_v2 import MAGVITv2
from imagehub.models.show_o import ShowOModel
from imagehub.tokenization import bytes_to_unicode
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "show_o_configs"


@pytest.mark.parametrize(
    ("model_cls", "header"),
    [(ShowOModel, "showo/header.json"), (MAGVITv2, "vq/header.json")],
)
def test_published_layouts_match_on_meta(model_cls, header):
    expected = json.loads((FIXTURES / header).read_text(encoding="utf-8"))
    with torch.device("meta"):
        model = model_cls({})
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert actual == expected


def test_omni_attention_is_causal_for_text_and_bidirectional_for_image():
    sequence = torch.tensor([[5, 6, 9, 10, 11]])
    mask = create_attention_mask(sequence, pad_id=0, soi_id=9, eoi_id=11)
    allowed = mask == 0
    assert allowed[0, 0, 1, 3].item() is False
    assert allowed[0, 0, 2, 4].item() is True
    assert allowed[0, 0, 4, 2].item() is True
    assert allowed[0, 0, 2, 0].item() is True


def test_generate_tokens_is_seed_repeatable():
    config = {
        "vocab_size": 24,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32,
    }
    model = ShowOModel(config).eval()
    # Sampling constants are patched to a compact vocabulary for this structural test.
    import imagehub.backends.show_o as backend

    old = backend.LLM_VOCAB_SIZE, backend.NUM_NEW_SPECIAL_TOKENS, backend.MASK_TOKEN_ID
    backend.LLM_VOCAB_SIZE, backend.NUM_NEW_SPECIAL_TOKENS, backend.MASK_TOKEN_ID = 8, 2, 23
    try:
        ids = torch.tensor([[1, 9, 23, 23, 23, 10]])
        mask = torch.zeros(1, 1, 6, 6)
        outputs = []
        for _ in range(2):
            outputs.append(
                generate_tokens(
                    model,
                    ids.clone(),
                    None,
                    mask,
                    guidance=0,
                    temperature=1,
                    steps=2,
                    generator=torch.Generator("cpu").manual_seed(7),
                    image_tokens=3,
                )
            )
        assert torch.equal(*outputs)
    finally:
        backend.LLM_VOCAB_SIZE, backend.NUM_NEW_SPECIAL_TOKENS, backend.MASK_TOKEN_ID = old


def test_options_are_validated_before_loading(tmp_path):
    backend = ShowOBackend(tmp_path)
    with pytest.raises(TypeError, match="Unknown Show-o options"):
        backend.infer(T2IRequest("cat"), unknown=True)
    assert not backend.is_loaded
    with pytest.raises(ValueError, match="width must be 512"):
        backend.infer(T2IRequest("cat"), width=256)
    assert not backend.is_loaded


def _write_tiny_snapshot(root):
    model_config = {
        "vocab_size": 58498,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32,
    }
    vq_config = {
        "ch": 32,
        "encoder_ch_mult": [1, 1, 1, 1, 1],
        "encoder_num_res_blocks": [1, 1, 1, 1, 1],
        "decoder_ch_mult": [1, 1, 1, 1, 1],
        "decoder_num_res_blocks": [1, 1, 1, 1, 1],
        "attn_resolutions": [],
        "resolution": 32,
        "z_channels": 13,
        "out_channels": 3,
    }
    root.mkdir()
    (root / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    save_safetensors(ShowOModel(model_config).state_dict(), root / "pytorch_model.safetensors")
    vq = root / "vq"
    vq.mkdir()
    (vq / "config.json").write_text(json.dumps(vq_config), encoding="utf-8")
    save_safetensors(MAGVITv2(vq_config).state_dict(), vq / "pytorch_model.safetensors")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    vocab = {character: index for index, character in enumerate(bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = 256
    (tokenizer / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (tokenizer / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (tokenizer / "added_tokens.json").write_text("{}", encoding="utf-8")
    return root


def test_tiny_backend_repeatability_and_autograd_restoration(tmp_path, monkeypatch):
    import imagehub.backends.show_o as module

    root = _write_tiny_snapshot(tmp_path / "show-o")
    monkeypatch.setattr(module, "IMAGE_TOKENS", 4)
    monkeypatch.setattr(module, "TEXT_LENGTH", 4)
    backend = ShowOBackend(root)
    enabled = torch.is_grad_enabled()
    first = backend.infer(T2IRequest("cat"), seed=5, steps=2, guidance=0, width=512, height=512)
    second = backend.infer(T2IRequest("cat"), seed=5, steps=2, guidance=0, width=512, height=512)
    different = backend.infer(T2IRequest("cat"), seed=6, steps=2, guidance=0, width=512, height=512)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert not torch.equal(first.images[0].tensor, different.images[0].tensor)
    assert first.images[0].tensor.shape == (32, 32, 3)
    assert first.metadata["seed"] == 5
    assert torch.is_grad_enabled() is enabled
