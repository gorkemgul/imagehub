import json

import pytest
import torch
from helpers import TINY_VOCAB_SIZE, write_tiny_tokenizer

from imagehub import pipeline
from imagehub.backends.stable_cascade import StableCascadeBackend
from imagehub.catalog import model_info
from imagehub.checkpoints import save_safetensors
from imagehub.exceptions import CheckpointError
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.stable_cascade_unet import StableCascadeUNet
from imagehub.types import T2IRequest

TEXT_CONFIG = {
    "vocab_size": TINY_VOCAB_SIZE,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 77,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 16,
    "bos_token_id": TINY_VOCAB_SIZE - 2,
    "eos_token_id": TINY_VOCAB_SIZE - 1,
    "pad_token_id": TINY_VOCAB_SIZE - 1,
}
PRIOR_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "conditioning_dim": 24,
    "block_out_channels": [32, 32],
    "num_attention_heads": [4, 4],
    "down_num_layers_per_block": [1, 2],
    "up_num_layers_per_block": [2, 1],
    "down_blocks_repeat_mappers": [1, 1],
    "up_blocks_repeat_mappers": [1, 1],
    "block_types_per_layer": [
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
    ],
    "clip_text_in_channels": 16,
    "clip_text_pooled_in_channels": 16,
    "clip_image_in_channels": 12,
    "clip_seq": 4,
    "dropout": [0.1, 0.1],
    "self_attn": True,
    "timestep_conditioning_type": ["sca", "crp"],
    "switch_level": [False],
}
DECODER_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "patch_size": 2,
    "conditioning_dim": 24,
    "block_out_channels": [8, 16, 32, 32],
    "num_attention_heads": [0, 0, 4, 4],
    "down_num_layers_per_block": [1, 1, 2, 1],
    "up_num_layers_per_block": [1, 2, 1, 1],
    "down_blocks_repeat_mappers": [1, 1, 1, 1],
    "up_blocks_repeat_mappers": [3, 3, 2, 2],
    "block_types_per_layer": [
        ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
        ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
    ],
    "clip_text_pooled_in_channels": 16,
    "effnet_in_channels": 4,
    "pixel_mapper_in_channels": 3,
    "dropout": [0, 0, 0.1, 0.1],
    "timestep_conditioning_type": ["sca"],
    "switch_level": None,
}
VQGAN_CONFIG = {
    "levels": 2,
    "bottleneck_blocks": 2,
    "embed_dim": 16,
    "latent_channels": 4,
    "num_vq_embeddings": 8,
    "scale_factor": 0.3764,
}
SCHEDULER_CONFIG = {"s": 0.008, "scaler": 1.0}
PRIOR_INDEX = {"_class_name": "StableCascadePriorPipeline", "resolution_multiple": 42.67}
DECODER_INDEX = {"_class_name": "StableCascadeDecoderPipeline", "latent_dim_scale": 10.67}


def write_component(root, name, state, config, weights):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_safetensors(state, directory / weights)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_tiny_snapshot(root, *, prior="prior", decoder="decoder"):
    """Write random native weights in the published two-repository layout."""
    torch.manual_seed(0)
    prior_root = root / "prior"
    text = CLIPTextModelWithProjection(TEXT_CONFIG)
    write_component(prior_root, "text_encoder", text.state_dict(), TEXT_CONFIG, "model.safetensors")
    write_component(
        prior_root,
        prior,
        StableCascadeUNet(PRIOR_CONFIG).state_dict(),
        PRIOR_CONFIG,
        "diffusion_pytorch_model.safetensors",
    )
    write_tiny_tokenizer(prior_root / "tokenizer")
    write_json(prior_root / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    write_json(prior_root / "model_index.json", PRIOR_INDEX)
    write_component(
        root,
        decoder,
        StableCascadeUNet(DECODER_CONFIG).state_dict(),
        DECODER_CONFIG,
        "diffusion_pytorch_model.safetensors",
    )
    write_component(
        root,
        "vqgan",
        PaellaVQModel(VQGAN_CONFIG).state_dict(),
        VQGAN_CONFIG,
        "diffusion_pytorch_model.safetensors",
    )
    write_json(root / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    write_json(root / "model_index.json", DECODER_INDEX)
    return root


@pytest.fixture(scope="module")
def tiny_cascade(tmp_path_factory):
    return write_tiny_snapshot(tmp_path_factory.mktemp("tiny-cascade"))


def test_generates_images_with_metadata(tiny_cascade):
    result = pipeline("t2i", backend="stable-cascade", model=tiny_cascade)(
        "a cat", seed=3, steps=2, prior_steps=2, width=128, height=128
    )
    assert len(result.images) == 1
    assert (result.images[0].width, result.images[0].height) == (128, 128)
    metadata = result.metadata
    assert metadata["seed"] == 3
    assert (metadata["steps"], metadata["prior_steps"]) == (2, 2)
    assert (metadata["guidance"], metadata["prior_guidance"]) == (0.0, 4.0)
    assert metadata["scheduler"] == "ddpm_wuerstchen"
    assert metadata["device"] == "cpu" and metadata["dtype"] == "float32"


def test_guided_decoder_with_negative_prompt_and_other_size(tiny_cascade):
    pipe = pipeline("t2i", backend="stable-cascade", model=tiny_cascade)
    result = pipe(
        "a hi cat",
        seed=4,
        steps=2,
        prior_steps=2,
        guidance=3.0,
        negative_prompt="cat",
        width=128,
        height=256,
    )
    assert result.images[0].tensor.dtype == torch.uint8
    assert result.images[0].width != result.images[0].height


def test_lite_layout_is_found_without_a_catalog_id(tmp_path):
    root = write_tiny_snapshot(tmp_path / "lite", prior="prior_lite", decoder="decoder_lite")
    result = pipeline("t2i", backend="stable_cascade", model=root)(
        "a cat", seed=1, steps=1, prior_steps=1, width=128, height=128
    )
    assert result.images[0].width == 128
    lite = pipeline("t2i", backend="stable-cascade-lite", model=root)
    assert lite("a cat", seed=1, steps=1, prior_steps=1, width=128, height=128).images


def test_seed_is_reproducible(tiny_cascade):
    pipe = pipeline("t2i", backend="stable-cascade", model=tiny_cascade)
    options = {"steps": 2, "prior_steps": 2, "width": 128, "height": 128}
    a = pipe("a cat", seed=11, **options)
    b = pipe("a cat", seed=11, **options)
    c = pipe("a cat", seed=12, **options)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_cascade):
    pipe = pipeline("t2i", backend="stable-cascade", model=tiny_cascade)
    options = {"width": 128, "height": 128}
    first = pipe("a cat", seed=1, steps=2, prior_steps=2, **options)
    pipe(
        "a hi cat",
        seed=2,
        steps=3,
        prior_steps=3,
        guidance=4.0,
        negative_prompt="blurry",
        **options,
    )
    again = pipe("a cat", seed=1, steps=2, prior_steps=2, **options)
    assert torch.equal(first.images[0].tensor, again.images[0].tensor)


@pytest.mark.parametrize("grad_before", [True, False])
def test_callers_autograd_mode_is_restored(tiny_cascade, grad_before):
    pipe = pipeline("t2i", backend="stable-cascade", model=tiny_cascade)
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(grad_before)
        result = pipe("a cat", seed=1, steps=1, prior_steps=1, width=128, height=128)
        assert torch.is_grad_enabled() is grad_before
        assert not torch.is_inference_mode_enabled()
        assert not result.images[0].tensor.requires_grad
    finally:
        torch.set_grad_enabled(previous)


def test_options_are_validated_before_loading(tmp_path):
    backend = StableCascadeBackend(tmp_path)
    request = T2IRequest("a cat")
    with pytest.raises(TypeError, match="Unknown Stable Cascade options"):
        backend.infer(request, unknown=True)
    with pytest.raises(ValueError, match="steps"):
        backend.infer(request, steps=0)
    with pytest.raises(ValueError, match="width"):
        backend.infer(request, width=511)
    with pytest.raises(ValueError, match="prior_guidance"):
        backend.infer(request, prior_guidance=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        backend.infer(request, seed=-1)
    with pytest.raises(TypeError, match="negative_prompt"):
        backend.infer(request, negative_prompt=1)
    with pytest.raises(ValueError, match="prior_guidance must be above 1"):
        backend.infer(request, guidance=2.0, prior_guidance=1.0)
    assert not backend.is_loaded


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a Stable Cascade snapshot"):
        StableCascadeBackend(tmp_path, model_id="stable-cascade").load()


@pytest.mark.parametrize("model_id", ["stable-cascade", "stable-cascade-lite"])
def test_catalog_defaults_are_used(tmp_path, model_id):
    backend = StableCascadeBackend(tmp_path, model_id=model_id)
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert options["steps"] == 12
    assert options["guidance"] == 0.0
    assert options["prior_steps"] == 60
    assert options["prior_guidance"] == 4.0
    assert options["width"] == options["height"] == 512
    assert model_info(model_id).config["decoder_component"] in ("decoder", "decoder_lite")
