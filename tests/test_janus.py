import json

import pytest
import torch
from helpers import write_tiny_janus

from imagehub import pipeline
from imagehub.backends.janus import JanusBackend, JanusComponents, generate_image_tokens
from imagehub.checkpoints import load_safetensors, save_safetensors
from imagehub.exceptions import CheckpointError
from imagehub.models.janus import JanusModel


@pytest.fixture(scope="module")
def tiny_janus(tmp_path_factory):
    return write_tiny_janus(tmp_path_factory.mktemp("tiny-janus"))


def test_loads_published_layout_and_casts_bfloat16_weights(tmp_path):
    root = write_tiny_janus(tmp_path / "bfloat16")
    state = load_safetensors(root / "model.safetensors")
    state = {
        name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
        for name, tensor in state.items()
    }
    save_safetensors(state, root / "model.safetensors")
    model = JanusModel.from_pretrained(root, dtype=torch.float32)
    assert model.language_model.embed_tokens.weight.dtype == torch.float32
    assert model.generation_embeddings.weight.dtype == torch.float32


def test_loads_sharded_published_layout(tmp_path):
    root = write_tiny_janus(tmp_path / "sharded")
    state = load_safetensors(root / "model.safetensors")
    names = sorted(state)
    weight_map = {}
    for index, shard_names in enumerate((names[::3], names[1::3], names[2::3]), start=1):
        shard = f"model-{index:05d}-of-00003.safetensors"
        save_safetensors({name: state[name] for name in shard_names}, root / shard)
        weight_map.update(dict.fromkeys(shard_names, shard))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    (root / "model.safetensors").unlink()

    components = JanusComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)

    assert components.model.language_model.embed_tokens.weight.dtype == torch.float32


def test_unknown_published_tensor_is_rejected(tiny_janus, tmp_path):
    root = write_tiny_janus(tmp_path / "bad")
    state = load_safetensors(root / "model.safetensors")
    state["model.not_understanding_only"] = torch.zeros(1)
    save_safetensors(state, root / "model.safetensors")
    with pytest.raises(CheckpointError, match="not_understanding_only"):
        JanusModel.from_pretrained(root)


@pytest.mark.parametrize("guidance", [1.0, 5.0])
def test_sampling_is_repeatable_and_does_not_change_global_rng(tiny_janus, guidance):
    components = JanusComponents.load(tiny_janus, device=torch.device("cpu"), dtype=torch.float32)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    first = generate_image_tokens(
        components,
        prompt="hi",
        guidance=guidance,
        temperature=1.0,
        do_sample=True,
        num_images=2,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    after = torch.random.get_rng_state()
    second = generate_image_tokens(
        components,
        prompt="hi",
        guidance=guidance,
        temperature=1.0,
        do_sample=True,
        num_images=2,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    assert torch.equal(first, second)
    assert torch.equal(before, after)
    assert first.shape == (2, 4)


def test_backend_generates_images_and_metadata(tiny_janus):
    result = pipeline("t2i", backend="janus", model=tiny_janus)(
        "hi", seed=7, num_images=2, guidance=5.0
    )
    assert len(result.images) == 2
    assert result.metadata["seed"] == 7
    assert result.metadata["guidance"] == 5.0
    assert result.metadata["temperature"] == 1.0
    assert result.metadata["do_sample"] is True
    assert result.metadata["width"] == result.metadata["height"] == 384


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 512}, ValueError, "384"),
        ({"height": 512}, ValueError, "384"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"guidance": 0}, ValueError, "guidance"),
        ({"temperature": 0}, ValueError, "temperature"),
        ({"do_sample": 1}, TypeError, "do_sample"),
        ({"num_images": 0}, ValueError, "num_images"),
        ({"seed": -1}, ValueError, "seed"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_janus, options, error, match):
    backend = JanusBackend(tiny_janus)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("hi", **options)
    assert not backend.is_loaded
