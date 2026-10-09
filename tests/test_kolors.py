import json

import pytest
import torch
from helpers import write_tiny_kolors

from imagehub import pipeline
from imagehub.backends.kolors import KolorsBackend
from imagehub.catalog import model_info
from imagehub.checkpoints import load_weights
from imagehub.exceptions import CheckpointError
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.types import T2IRequest


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_tiny_kolors(tmp_path_factory.mktemp("kolors"))


def test_variant_weight_names_load(snapshot):
    state = load_weights(snapshot / "text_encoder")
    assert state["embedding.word_embeddings.weight"].dtype == torch.float16
    unet = UNet2DConditionModel.from_pretrained(snapshot / "unet", dtype=torch.float32)
    assert unet.encoder_hid_proj.weight.shape == (32, 64)


def test_unet_text_proj_projects_context(snapshot):
    unet = UNet2DConditionModel.from_pretrained(snapshot / "unet", dtype=torch.float32)
    sample = torch.randn(1, 4, 8, 8)
    kwargs = {"text_embeds": torch.randn(1, 64), "time_ids": torch.zeros(1, 6)}
    context = torch.randn(1, 5, 64)
    direct = unet(sample, torch.tensor(10.0), context, **kwargs)
    again = unet(sample, torch.tensor(10.0), context + 1, **kwargs)
    assert direct.shape == sample.shape
    assert not torch.equal(direct, again)


def test_text_proj_requires_dimensions():
    config = {"encoder_hid_dim_type": "text_proj", "cross_attention_dim": 32}
    with pytest.raises(ValueError, match="text projection"):
        UNet2DConditionModel(config)


def test_generation_is_deterministic_and_metadata_complete(snapshot):
    with pipeline("t2i", backend="kolors", model=snapshot) as pipe:
        first = pipe("a cat", steps=2, guidance=3.0, width=64, height=48, seed=3)
        second = pipe("a cat", steps=2, guidance=3.0, width=64, height=48, seed=3)
        other = pipe("a cat", steps=2, guidance=3.0, width=64, height=48, seed=4)
    assert first.images[0].tensor.shape == (48, 64, 3)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert not torch.equal(first.images[0].tensor, other.images[0].tensor)
    assert first.metadata["seed"] == 3
    assert first.metadata["scheduler"] == "euler"
    assert first.metadata["dtype"] == "float32"


def test_catalog_defaults_apply_to_model_selector(snapshot):
    backend = KolorsBackend(snapshot, model_id="kolors")
    assert backend.option_defaults["width"] == 1024
    assert backend.option_defaults["guidance"] == 5.0
    assert backend.option_defaults["steps"] == 50


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1101}, ValueError, "at most 1100"),
        ({"width": 60}, ValueError, "multiple of 8"),
        ({"guidance": -1}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"original_size": (0, 8)}, ValueError, "original_size"),
        ({"crops_coords_top_left": (-1, 0)}, ValueError, "crops_coords_top_left"),
        ({"timesteps": [1]}, TypeError, "Unknown Kolors options"),
        ({"latents": torch.zeros(1, 4, 2, 2)}, ValueError, "latents"),
        ({"latents": torch.zeros(1, 4, 4, 4), "seed": 1}, ValueError, "either seed or latents"),
    ],
)
def test_options_are_validated_before_loading(snapshot, options, error, message):
    backend = KolorsBackend(snapshot)
    options = {"width": 32, "height": 32} | options
    with pytest.raises(error, match=message):
        backend.infer(T2IRequest("a cat"), **options)
    assert not backend.is_loaded


def test_missing_components_name_the_download_command(tmp_path):
    backend = KolorsBackend(tmp_path, model_id="kolors")
    with pytest.raises(CheckpointError, match="python -m imagehub.pretrained kolors"):
        backend.load()


def test_snapshot_must_use_euler(tmp_path):
    root = write_tiny_kolors(tmp_path / "kolors")
    path = root / "scheduler" / "scheduler_config.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | {"_class_name": "DDIMScheduler"}))
    with pytest.raises(ValueError, match="Unsupported Kolors scheduler"):
        KolorsBackend(root).load()


def test_catalog_entry_lists_variant_files():
    spec = model_info("kolors")
    assert spec.revision == "7e091c75199e910a26cd1b51ed52c28de5db3711"
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in spec.files
    assert not any(name.endswith("model.safetensors") for name in spec.files)


def test_float16_vae_decodes_in_float32_and_is_restored(snapshot):
    from imagehub.backends.kolors import KolorsComponents, decode_latents

    components = KolorsComponents.load(snapshot, device=torch.device("cpu"), dtype=torch.float16)
    latents = torch.randn(1, 4, 4, 4).half()
    seen = []
    components.vae.decoder.register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    pixels = decode_latents(components, latents)
    assert seen == [torch.float32]
    assert pixels.dtype == torch.uint8
    assert next(components.vae.parameters()).dtype == torch.float16


def test_unet_matches_published_tensor_header():
    from pathlib import Path

    directory = Path(__file__).parent / "fixtures" / "kolors_configs"
    config = json.loads((directory / "unet_config.json").read_text())
    published = json.loads((directory / "unet_tensors.json").read_text())
    with torch.device("meta"):
        model = UNet2DConditionModel(config)
    shapes = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert shapes == published
    assert len(published) == 1682
