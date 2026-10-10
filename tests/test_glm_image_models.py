import json
import shutil
from pathlib import Path

import pytest
import torch
from glm_image_helpers import AR_CONFIG, FIXTURES, write_tiny_glm_image

from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.glm_image_ar import GlmImageARModel, decode_position_ids
from imagehub.models.glm_image_transformer import GlmImageTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization_byt5 import ByT5Tokenizer
from imagehub.tokenization_glm_image import GlmImageTokenizer, build_prior_prompt, target_shape


def _config(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _check(model, tensors, dropped=()):
    expected = _config(tensors)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    kept = {k: v for k, v in expected.items() if not k.startswith(tuple(dropped))}
    assert actual == kept


def test_transformer_matches_published_tensors():
    with torch.device("meta"):
        model = GlmImageTransformer2DModel(_config("transformer_config.json"))
    _check(model, "transformer_tensors.json")


def test_text_encoder_matches_published_tensors():
    with torch.device("meta"):
        model = T5EncoderModel(_config("text_encoder_config.json"))
    _check(model, "text_encoder_tensors.json")


def test_vae_matches_published_tensors():
    with torch.device("meta"):
        model = AutoencoderKL(_config("vae_config.json"))
    _check(model, "vae_tensors.json")


def test_autoregressive_model_matches_published_tensors_without_the_image_input_parts():
    with torch.device("meta"):
        model = GlmImageARModel(_config("vision_language_encoder_config.json"))
    published = _config("vision_language_encoder_tensors.json")
    prepared = model.prepare_state_dict({name: torch.empty(0) for name in published})
    assert sorted(prepared) == sorted(model.state_dict())
    _check(model, "vision_language_encoder_tensors.json", dropped=("model.visual.", "model.vqmodel."))


def test_autoregressive_model_loads_on_meta_then_moves(tmp_path):
    root = write_tiny_glm_image(tmp_path)
    model = GlmImageARModel.from_pretrained(root / "vision_language_encoder")
    inv_freq = model.model.language_model.rotary_emb.inv_freq
    assert inv_freq.device.type == "cpu" and inv_freq.shape == (4,)


def test_decode_positions_follow_the_preview_then_the_full_grid():
    positions = decode_position_ids(10, [(2, 3), (1, 2)])
    assert positions.shape == (3, 2 + 6 + 1)
    assert positions[:, :2].tolist() == [[10, 10], [10, 10], [10, 11]]
    assert positions[:, 2].tolist() == [12, 12, 12]
    assert positions[:, -1].tolist() == [15, 15, 15]


def test_target_shape_rounds_to_the_patch_grid():
    assert target_shape(1024, 1024) == (32, 32, 16, 16)
    assert target_shape(1152, 768) == (36, 24, 19, 13)
    assert target_shape(1000, 1000) == (31, 31, 16, 16)


def test_prior_prompt_appends_the_target_shape(tmp_path):
    root = write_tiny_glm_image(tmp_path)
    tokenizer = GlmImageTokenizer(root / "processor")
    ids, grids = build_prior_prompt(tokenizer, "hi", 64, 96)
    assert grids == [(2, 3), (13, 19)]
    assert ids[0, 0].item() == tokenizer.encoder["h"]
    assert ids[0, -1].item() == 260
    assert ids[0].tolist().count(257) == 2 and ids[0].tolist().count(258) == 2
    assert AR_CONFIG["text_config"]["vocab_size"] > ids.max().item()


@pytest.fixture
def byt5(tmp_path):
    for name in ("tokenizer_config.json",):
        shutil.copy(FIXTURES / f"tokenizer_{name}", tmp_path / name)
    return ByT5Tokenizer(tmp_path)


def test_byt5_encodes_bytes_with_an_end_token(byt5):
    assert (byt5.pad_token_id, byt5.eos_token_id, byt5.unk_token_id) == (0, 1, 2)
    assert byt5.tokenize("") == [1]
    assert byt5.tokenize("aé") == [97 + 3, 195 + 3, 169 + 3, 1]
    assert byt5.tokenize("a</s>") == [100, 1]
    assert byt5.tokenize("a<extra_id_0>") == [100, 259, 1]
    assert byt5.tokenize("abcdef", 4) == [100, 101, 102, 1]
