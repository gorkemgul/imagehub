import json

import pytest
import torch
from helpers import TINY_VOCAB_SIZE, write_tiny_tokenizer

from imagehub import pipeline
from imagehub.backends.wuerstchen import WuerstchenBackend
from imagehub.catalog import model_info
from imagehub.checkpoints import save_safetensors
from imagehub.exceptions import CheckpointError
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.wuerstchen import WuerstchenDiffNeXt, WuerstchenPrior
from imagehub.types import T2IRequest

# The two text encoders differ in width, as the published CLIP-H and CLIP-bigG do.
DECODER_TEXT_CONFIG = {
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
PRIOR_TEXT_CONFIG = DECODER_TEXT_CONFIG | {
    "hidden_size": 20,
    "intermediate_size": 40,
    "num_hidden_layers": 3,
    "projection_dim": 8,
}
PRIOR_CONFIG = {"c_in": 6, "c": 32, "c_cond": 20, "c_r": 16, "depth": 2, "nhead": 4, "dropout": 0.1}
DECODER_CONFIG = {
    "c_in": 4,
    "c_out": 4,
    "c_r": 16,
    "patch_size": 2,
    "c_cond": 24,
    "c_hidden": [8, 16, 32, 32],
    "nhead": [-1, 2, 4, 4],
    "blocks": [1, 1, 2, 1],
    "level_config": ["CT", "CTA", "CTA", "CTA"],
    "inject_effnet": [False, True, True, True],
    "effnet_embd": 6,
    "clip_embd": 16,
    "kernel_size": 3,
    "dropout": 0.1,
}
VQGAN_CONFIG = {
    "levels": 2,
    "bottleneck_blocks": 2,
    "embed_dim": 16,
    "latent_channels": 4,
    "num_vq_embeddings": 8,
    "scale_factor": 0.3764,
    "up_down_scale_factor": 2,
}
SCHEDULER_CONFIG = {"s": 0.008, "scaler": 1.0}
PRIOR_INDEX = {
    "_class_name": "WuerstchenPriorPipeline",
    "latent_mean": 42.0,
    "latent_std": 1.0,
    "resolution_multiple": 42.67,
}
DECODER_INDEX = {"_class_name": "WuerstchenDecoderPipeline", "latent_dim_scale": 10.67}


def write_component(root, name, state, config, weights):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_safetensors(state, directory / weights)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_tiny_snapshot(root, states=None):
    """Write weights in the published two-repository layout (random native ones by default)."""
    if states is None:
        torch.manual_seed(0)
        states = {
            "prior_text": CLIPTextModel(PRIOR_TEXT_CONFIG).state_dict(),
            "prior": WuerstchenPrior(PRIOR_CONFIG).state_dict(),
            "text": CLIPTextModel(DECODER_TEXT_CONFIG).state_dict(),
            "decoder": WuerstchenDiffNeXt(DECODER_CONFIG).state_dict(),
            "vqgan": PaellaVQModel(VQGAN_CONFIG).state_dict(),
        }
    prior_root = root / "prior"
    write_component(
        prior_root, "text_encoder", states["prior_text"], PRIOR_TEXT_CONFIG, "model.safetensors"
    )
    write_component(
        prior_root, "prior", states["prior"], PRIOR_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    write_tiny_tokenizer(prior_root / "tokenizer")
    write_json(prior_root / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    write_json(prior_root / "model_index.json", PRIOR_INDEX)
    write_component(root, "text_encoder", states["text"], DECODER_TEXT_CONFIG, "model.safetensors")
    write_tiny_tokenizer(root / "tokenizer")
    write_component(
        root, "decoder", states["decoder"], DECODER_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    write_component(
        root, "vqgan", states["vqgan"], VQGAN_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    write_json(root / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    write_json(root / "model_index.json", DECODER_INDEX)
    return root


@pytest.fixture(scope="module")
def tiny_wuerstchen(tmp_path_factory):
    return write_tiny_snapshot(tmp_path_factory.mktemp("tiny-wuerstchen"))


def test_generates_images_with_metadata(tiny_wuerstchen):
    result = pipeline("t2i", backend="wuerstchen", model=tiny_wuerstchen)(
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


def test_guided_decoder_with_negative_prompt_and_other_size(tiny_wuerstchen):
    result = pipeline("t2i", backend="wuerstchen", model=tiny_wuerstchen)(
        "a hi cat",
        seed=4,
        steps=2,
        prior_steps=2,
        guidance=3.0,
        negative_prompt="cat",
        width=128,
        height=256,
    )
    image = result.images[0]
    assert image.tensor.dtype == torch.uint8
    assert (image.width, image.height) == (128, 256)


def test_decoder_guidance_does_not_need_prior_guidance(tiny_wuerstchen):
    result = pipeline("t2i", backend="wuerstchen", model=tiny_wuerstchen)(
        "a cat", seed=4, steps=2, prior_steps=2, guidance=3.0, prior_guidance=1.0, width=128
    )
    assert result.images[0].width == 128


def test_seed_is_reproducible(tiny_wuerstchen):
    pipe = pipeline("t2i", backend="wuerstchen", model=tiny_wuerstchen)
    options = {"steps": 2, "prior_steps": 2, "width": 128, "height": 128}
    a = pipe("a cat", seed=11, **options)
    b = pipe("a cat", seed=11, **options)
    c = pipe("a cat", seed=12, **options)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)


def test_requests_are_isolated(tiny_wuerstchen):
    pipe = pipeline("t2i", backend="wuerstchen", model=tiny_wuerstchen)
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
def test_callers_autograd_mode_is_restored(tiny_wuerstchen, grad_before):
    pipe = pipeline("t2i", backend="wuerstchen", model=tiny_wuerstchen)
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
    backend = WuerstchenBackend(tmp_path)
    request = T2IRequest("a cat")
    with pytest.raises(TypeError, match="Unknown Würstchen options"):
        backend.infer(request, unknown=True)
    with pytest.raises(ValueError, match="steps"):
        backend.infer(request, steps=0)
    with pytest.raises(ValueError, match="width"):
        backend.infer(request, width=0)
    with pytest.raises(ValueError, match="multiple of 16"):
        backend.infer(request, width=512, height=597)
    with pytest.raises(ValueError, match="prior_guidance"):
        backend.infer(request, prior_guidance=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        backend.infer(request, seed=-1)
    with pytest.raises(TypeError, match="negative_prompt"):
        backend.infer(request, negative_prompt=1)
    assert not backend.is_loaded


def test_incomplete_snapshot_is_reported(tmp_path):
    with pytest.raises(CheckpointError, match="not a Würstchen snapshot"):
        WuerstchenBackend(tmp_path, model_id="wuerstchen-v2").load()


def test_catalog_defaults_are_used(tmp_path):
    backend = WuerstchenBackend(tmp_path, model_id="wuerstchen-v2")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert options["steps"] == 12
    assert options["guidance"] == 0.0
    assert options["prior_steps"] == 60
    assert options["prior_guidance"] == 4.0
    assert options["width"] == options["height"] == 512
    assert model_info("wuerstchen-v2").family == "wuerstchen"


def test_model_id_selector_resolves_to_the_backend(tiny_wuerstchen):
    pipe = pipeline("t2i", backend="wuerstchen-v2", model=tiny_wuerstchen)
    assert pipe("a cat", seed=1, steps=1, prior_steps=1, width=128, height=128).images
