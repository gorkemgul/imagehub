import glob
import json
from pathlib import Path

import pytest
import torch

from imagehub.backends.nextstep import NextStepBackend, NextStepComponents, generate
from imagehub.models.nextstep import NextStepAutoencoder, NextStepModel
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "nextstep_1_configs"


def _config():
    return json.loads((FIXTURES / "config.json").read_text())


def _tiny_config():
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "latent_channels": 2,
        "latent_patch_size": 2,
        "latent_size": 2,
        "fm_head_dim": 16,
        "fm_head_layers": 2,
        "bos_token_id": 1,
        "boi": 2,
        "image_placeholder_id": 3,
        "use_gen_pos_embed": False,
    }


def test_published_model_layout():
    expected = {}
    for filename in glob.glob(str(FIXTURES / "model-*-header.json")):
        header = json.loads(Path(filename).read_text())
        expected.update({name: shape for name, shape in header.items() if name != "__metadata__"})
    with torch.device("meta"):
        model = NextStepModel(_config())
    assert {name: list(value.shape) for name, value in model.state_dict().items()} == expected


def test_patch_round_trip_and_flow_shape():
    model = NextStepModel(_tiny_config()).eval()
    latent = torch.randn(2, 2, 4, 6)
    assert torch.equal(model.unpatchify(model.patchify(latent), 2, 3), latent)
    result = model.image_head.sample(
        torch.randn(2, 32),
        torch.randn(1, 8),
        guidance=3,
        steps=2,
        shift=1,
        sde=False,
    )
    assert result.shape == (1, 8)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unknown": 1}, TypeError, "Unknown NextStep-1 options"),
        ({"width": 17}, ValueError, "multiple of 16"),
        ({"guidance": float("nan")}, ValueError, "finite"),
        ({"steps": 0}, ValueError, "positive int"),
        ({"sde": 1}, ValueError, "bool"),
    ],
)
def test_options_are_rejected_before_loading(tmp_path, options, error, message):
    backend = NextStepBackend(tmp_path)
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


class _Tokenizer:
    pad_id = 0

    def tokenize(self, text):
        return [4 + ord(character) % 40 for character in text]


def test_tiny_generation_is_repeatable():
    torch.manual_seed(0)
    model = NextStepModel(_tiny_config()).eval()
    vae = NextStepAutoencoder(
        {"ch": 32, "ch_mult": [1, 1, 1, 1], "num_res_blocks": 1, "z_channels": 2}
    ).eval()
    components = NextStepComponents(_Tokenizer(), model, vae)
    options = dict(
        width=16,
        height=16,
        guidance=1.0,
        negative_prompt="",
        steps=1,
        timestep_shift=1.0,
        sde=False,
    )
    first = generate(components, "cat", generator=torch.Generator().manual_seed(5), **options)
    second = generate(components, "cat", generator=torch.Generator().manual_seed(5), **options)
    assert torch.equal(first, second)
    assert first.shape == (1, 3, 16, 16)

