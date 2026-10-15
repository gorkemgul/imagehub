"""Unit tests for the Chroma backend and transformer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from imagehub import pipeline
from imagehub.backends.chroma import ChromaBackend, calculate_shift
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json
from imagehub.image import Image
from imagehub.models.chroma_transformer import ChromaTransformer2DModel
from imagehub.registry import backend_registry
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "chroma_configs"


def _tiny_config() -> dict[str, Any]:
    return {
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 16,
        "in_channels": 16,
        "approximator_num_channels": 16,
        "approximator_hidden_dim": 32,
        "approximator_layers": 1,
        "axes_dims_rope": [2, 2, 4],
    }


@pytest.mark.parametrize(
    ("model_id", "revision", "defaults"),
    (
        ("chroma1-hd", "0e0c60ece1e82b17cb7f77342d765ba5024c40c0", (35, 5.0)),
        ("chroma1-flash", "093c4f63b60507234aa53a073f1f9f565b3f4336", (4, 1.0)),
    ),
)
def test_catalog_entry_exists(model_id, revision, defaults):
    spec = MODEL_CATALOG[model_id]
    assert spec.family == "chroma"
    assert spec.revision == revision
    assert spec.license == "Apache-2.0"
    assert "transformer/diffusion_pytorch_model.safetensors.index.json" in spec.files
    assert (spec.config["defaults"]["steps"], spec.config["defaults"]["guidance"]) == defaults


def test_backend_registered():
    assert "chroma" in backend_registry.names("t2i")
    assert "chroma1-hd" in backend_registry.names("t2i")
    assert "chroma1-flash" in backend_registry.names("t2i")


def test_transformer_meta_construction():
    config = read_json(FIXTURES / "transformer" / "config.json")
    with torch.device("meta"):
        model = ChromaTransformer2DModel(config)
    assert model.inner_dim == 3072
    assert len(model.state_dict()) == 1023


def test_transformer_forward_shape():
    model = ChromaTransformer2DModel(_tiny_config())
    hidden = torch.randn(1, 4, 16)
    context = torch.randn(1, 3, 16)
    img_ids = torch.zeros(4, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(2), torch.arange(2))
    txt_ids = torch.zeros(3, 3)
    timestep = torch.tensor([0.5])
    with torch.inference_mode():
        output = model(hidden, context, timestep, img_ids, txt_ids)
    assert output.shape == (1, 4, 16)


def test_transformer_attention_mask():
    model = ChromaTransformer2DModel(_tiny_config())
    hidden = torch.randn(1, 4, 16)
    context = torch.randn(1, 3, 16)
    img_ids = torch.zeros(4, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(2), torch.arange(2))
    txt_ids = torch.zeros(3, 3)
    timestep = torch.tensor([0.5])
    attention_mask = torch.tensor([[1, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)
    with torch.inference_mode():
        output = model(hidden, context, timestep, img_ids, txt_ids, attention_mask)
    assert output.shape == (1, 4, 16)


def test_transformer_rejects_unknown_options():
    model = ChromaTransformer2DModel(_tiny_config())
    with pytest.raises(ValueError, match="Unsupported Chroma transformer option"):
        model(
            torch.randn(1, 4, 16),
            torch.randn(1, 3, 16),
            torch.tensor([0.5]),
            torch.zeros(4, 3),
            torch.zeros(3, 3),
            joint_attention_kwargs={"foo": "bar"},
        )


def test_calculate_shift():
    config = read_json(FIXTURES / "scheduler" / "scheduler_config.json")
    mu = calculate_shift(4096, config)
    base_seq_len = config["base_image_seq_len"]
    max_seq_len = config["max_image_seq_len"]
    base_shift = config["base_shift"]
    max_shift = config["max_shift"]
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    assert math.isclose(mu, 4096 * slope + intercept)


def _backend(tmp_path: Path) -> ChromaBackend:
    root = tmp_path / "chroma_snapshot"
    root.mkdir(parents=True)
    for component in ("tokenizer", "text_encoder", "transformer", "vae", "scheduler"):
        (root / component).mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        read_text(FIXTURES / "transformer" / "config.json")
    )
    (root / "vae" / "config.json").write_text(read_text(FIXTURES / "vae" / "config.json"))
    (root / "scheduler" / "scheduler_config.json").write_text(
        read_text(FIXTURES / "scheduler" / "scheduler_config.json")
    )
    return ChromaBackend(root)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_backend_validates_options(tmp_path: Path):
    backend = _backend(tmp_path)
    request = T2IRequest(prompt="a cat")
    validated = backend.validate_options(request, {"steps": 10, "width": 1024, "height": 1024})
    assert validated["steps"] == 10
    assert validated["latent_shape"] == (1, 16, 128, 128)


def test_backend_rejects_unknown_option(tmp_path: Path):
    backend = _backend(tmp_path)
    request = T2IRequest(prompt="a cat")
    with pytest.raises(TypeError, match="Unknown Chroma options"):
        backend.validate_options(request, {"unknown_option": 1})


def test_backend_rejects_invalid_dimensions(tmp_path: Path):
    backend = _backend(tmp_path)
    request = T2IRequest(prompt="a cat")
    with pytest.raises(ValueError, match="width must be a multiple of 16"):
        backend.validate_options(request, {"width": 100, "height": 1024})


def test_backend_rejects_invalid_guidance(tmp_path: Path):
    backend = _backend(tmp_path)
    request = T2IRequest(prompt="a cat")
    with pytest.raises(ValueError, match="guidance must be a finite number"):
        backend.validate_options(request, {"guidance": -1})


def test_backend_rejects_latents_seed_together(tmp_path: Path):
    backend = _backend(tmp_path)
    request = T2IRequest(prompt="a cat")
    latents = torch.randn(1, 16, 128, 128)
    with pytest.raises(ValueError, match="Pass either seed or latents"):
        backend.validate_options(request, {"seed": 0, "latents": latents})


def test_backend_metadata(tmp_path: Path):
    backend = _backend(tmp_path)
    request = T2IRequest(prompt="a cat")
    validated = backend.validate_options(request, {})
    assert validated["steps"] == 35
    assert validated["guidance"] == 5.0
    assert validated["width"] == 1024
    assert validated["height"] == 1024


def test_pipeline_factory_returns_backend(tmp_path: Path):
    root = tmp_path / "chroma_snapshot"
    root.mkdir(parents=True)
    for component in ("tokenizer", "text_encoder", "transformer", "vae", "scheduler"):
        (root / component).mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        read_text(FIXTURES / "transformer" / "config.json")
    )
    (root / "vae" / "config.json").write_text(read_text(FIXTURES / "vae" / "config.json"))
    (root / "scheduler" / "scheduler_config.json").write_text(
        read_text(FIXTURES / "scheduler" / "scheduler_config.json")
    )
    pipe = pipeline("t2i", backend="chroma", model=root)
    assert isinstance(pipe.backend, ChromaBackend)


def test_image_result_tensor():
    pixel = torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8)
    image = Image(pixel)
    assert torch.equal(image.tensor, pixel)
