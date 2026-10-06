import json
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_flux2

from imagehub import pipeline
from imagehub.backends.flux2 import Flux2Backend, compute_empirical_mu
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.models.flux2_transformer import Flux2Transformer2DModel
from imagehub.models.qwen3 import Qwen3Model

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tiny_flux2(tmp_path_factory):
    return write_tiny_flux2(tmp_path_factory.mktemp("tiny-flux2"))


def test_generates_images_with_metadata(tiny_flux2):
    result = pipeline("t2i", backend="flux2", model=tiny_flux2)(
        "a cat", seed=3, steps=2, width=16, height=16, num_images=2
    )
    assert len(result.images) == 2
    assert (result.images[0].width, result.images[0].height) == (16, 16)
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["scheduler"] == "flow_match_euler"


def test_seed_and_latents_are_reproducible(tiny_flux2):
    pipe = pipeline("t2i", backend="flux2-klein-4b", model=tiny_flux2)
    a = pipe("a cat", seed=11, steps=2, width=16, height=16)
    b = pipe("a cat", seed=11, steps=2, width=16, height=16)
    latents = torch.randn(1, 16, 4, 4, generator=torch.Generator().manual_seed(11))
    c = pipe("a cat", latents=latents, steps=2, width=16, height=16)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert torch.equal(a.images[0].tensor, c.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_flux2, grad_before):
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        pipe = pipeline("t2i", backend="flux2", model=tiny_flux2)
        pipe("a cat", seed=1, steps=1, width=16, height=16)
        assert torch.is_grad_enabled() is grad_before
    finally:
        torch.set_grad_enabled(previous)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 15}, ValueError, "multiple of 16"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1, "latents": torch.zeros(1, 16, 4, 4)}, ValueError, "either"),
        ({"latents": torch.zeros(1, 16, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": "blurry"}, TypeError, "negative_prompt"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_flux2, options, error, match):
    backend = Flux2Backend(tiny_flux2)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_empirical_mu_matches_reference_formula():
    assert compute_empirical_mu(4096, 4) == pytest.approx(2.291179894115571)
    assert compute_empirical_mu(4301, 4) == pytest.approx(1.18469693)


def test_catalog_entry_and_snapshot_hint(tmp_path):
    spec = model_info("flux2-klein-4b")
    assert spec.family == "flux2" and spec.catalog_id == "flux-2"
    assert spec.revision == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert len(spec.files) == 18
    pipe = pipeline("t2i", backend="flux2-klein-4b", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained flux2-klein-4b "):
        pipe("a cat")


def test_klein_9b_full_config_layout():
    root = FIXTURES / "flux2_klein_9b_configs"
    text_config = json.loads((root / "text_encoder_config.json").read_text(encoding="utf-8"))
    transformer_config = json.loads((root / "transformer_config.json").read_text(encoding="utf-8"))
    assert text_config["num_hidden_layers"] == 36
    assert text_config["hidden_size"] == 4096
    assert transformer_config["joint_attention_dim"] == 3 * text_config["hidden_size"]
    with torch.device("meta"):
        text_model = Qwen3Model(text_config)
        model = Flux2Transformer2DModel(transformer_config)
    assert len(text_model.model.layers) == 36
    assert tuple(text_model.model.embed_tokens.weight.shape) == (151936, 4096)
    assert tuple(text_model.model.layers[0].self_attn.q_proj.weight.shape) == (4096, 4096)
    assert tuple(text_model.model.layers[0].mlp.gate_proj.weight.shape) == (12288, 4096)
    assert len(model.transformer_blocks) == 8
    assert len(model.single_transformer_blocks) == 24
    assert model.inner_dim == 32 * 128
    assert model.context_embedder.in_features == 12288
