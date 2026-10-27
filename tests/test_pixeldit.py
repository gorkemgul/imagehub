# Derivative of NVlabs PixelDiT under NSCLv1; non-commercial. See licenses/pixeldit.

import json
from pathlib import Path

import pytest
import torch
from fixture_files import fixture_path
from helpers import TINY_GEMMA_CONFIG, write_tiny_gemma_tokenizer

from imagehub.backends.pixeldit import PixelDiTBackend, sample_flow_dpm
from imagehub.checkpoints import save_safetensors
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.pixeldit import PixelDiT
from imagehub.types import T2IRequest

FIXTURES = fixture_path("pixeldit_configs")
TINY_CONFIG = {
    "in_channels": 3,
    "patch_size": 2,
    "num_groups": 4,
    "hidden_size": 32,
    "pixel_hidden_size": 8,
    "pixel_attn_hidden_size": 32,
    "pixel_num_groups": 4,
    "patch_depth": 2,
    "pixel_depth": 1,
    "num_text_blocks": 1,
    "txt_embed_dim": 32,
    "txt_max_length": 5,
    "use_text_rope": True,
    "text_rope_theta": 10000.0,
    "repa_encoder_index": -1,
    "use_pixel_abs_pos": True,
}


def _write_tiny_snapshot(directory: Path) -> Path:
    torch.manual_seed(0)
    directory.mkdir()
    tokenizer_root = write_tiny_gemma_tokenizer(directory / "text_encoder")
    tokenizer_data = json.loads((tokenizer_root / "tokenizer.json").read_text())
    text_config = TINY_GEMMA_CONFIG | {"vocab_size": len(tokenizer_data["model"]["vocab"])}
    text_encoder = Gemma2Model(text_config)
    transformer = PixelDiT(TINY_CONFIG)
    for model in (text_encoder, transformer):
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.02)
    (directory / "config.json").write_text(json.dumps(TINY_CONFIG))
    (directory / "text_encoder" / "config.json").write_text(json.dumps(text_config))
    text_state = {f"model.{name}": value for name, value in text_encoder.state_dict().items()}
    text_state["lm_head.weight"] = text_encoder.embed_tokens.weight
    save_safetensors(text_state, directory / "text_encoder" / "model.safetensors")
    torch.save({"state_dict": transformer.state_dict()}, directory / "pixeldit_t2i_v1.pth")
    (directory / "imagehub_snapshot.json").write_text(
        json.dumps({"model_id": "pixeldit", "revision": "tiny"})
    )
    return directory


def test_published_header_layout() -> None:
    config = json.loads((FIXTURES / "model_config.json").read_text())
    headers = json.loads((FIXTURES / "model_headers" / "pixeldit_t2i_v1.json").read_text())
    with torch.device("meta"):
        model = PixelDiT(config)
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    expected = {name: value["shape"] for name, value in headers.items()}
    assert actual == expected


def test_checkpoint_tensors_are_copied_from_pickle_storage() -> None:
    model = PixelDiT(TINY_CONFIG)
    state = model.state_dict()
    prepared = model.prepare_state_dict(state)
    assert prepared.keys() == state.keys()
    for name, tensor in state.items():
        assert torch.equal(prepared[name], tensor)
        assert prepared[name].data_ptr() != tensor.data_ptr()


def test_tiny_transformer_non_square_shape() -> None:
    model = PixelDiT(TINY_CONFIG).eval()
    output = model(
        torch.randn(1, 3, 4, 6),
        torch.tensor([0.5]),
        torch.randn(1, 5, 32),
    )
    assert output.shape == (1, 3, 4, 6)
    assert torch.isfinite(output).all()


class _ToyDenoiser:
    def __call__(self, x, timestep, text, mask=None):
        return (
            x * 0.1
            + timestep[:, None, None, None] * 0.0002
            + text.mean((1, 2))[:, None, None, None] * 0.03
        )


def test_flow_sampler_repeatable_and_guided() -> None:
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(1, 3, 4, 4, generator=generator)
    condition = torch.randn(1, 5, 6, generator=generator)
    unconditional = torch.randn(1, 5, 6, generator=generator)
    first = sample_flow_dpm(_ToyDenoiser(), noise, condition, unconditional, steps=4, guidance=2.75)
    second = sample_flow_dpm(
        _ToyDenoiser(), noise, condition, unconditional, steps=4, guidance=2.75
    )
    unguided = sample_flow_dpm(
        _ToyDenoiser(), noise, condition, unconditional, steps=4, guidance=1.0
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, unguided)


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"unknown": 1}, TypeError, "Unknown PixelDiT options"),
        ({"width": 15}, ValueError, "divisible by 16"),
        ({"steps": 0}, ValueError, "positive int"),
        ({"guidance": float("nan")}, ValueError, "finite number"),
        ({"interval_guidance": (0.8, 0.2)}, ValueError, "in increasing order"),
    ],
)
def test_options_rejected_before_loading(tmp_path, options, error, message) -> None:
    backend = PixelDiTBackend(tmp_path)
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_tiny_snapshot_end_to_end_repeatability(tmp_path) -> None:
    root = _write_tiny_snapshot(tmp_path / "pixeldit")
    backend = PixelDiTBackend(root, model_id="pixeldit")
    previous = torch.is_grad_enabled()
    first = backend.infer(
        T2IRequest("a cat"), steps=2, width=16, height=16, seed=7, use_chi_prompt=False
    )
    second = backend.infer(
        T2IRequest("a cat"), steps=2, width=16, height=16, seed=7, use_chi_prompt=False
    )
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert first.images[0].tensor.shape == (16, 16, 3)
    assert first.metadata["sampler"] == "flow_dpm-solver"
    assert torch.is_grad_enabled() is previous
