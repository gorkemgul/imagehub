import json

import pytest
import torch
from helpers import bytes_to_unicode

from imagehub import pipeline
from imagehub.backends.lumina_dimoo import (
    CODEBOOK_SIZE,
    IMAGE_TOKEN_OFFSET,
    MASK_TOKEN_ID,
    LuminaDiMOOBackend,
    generate_tokens,
)
from imagehub.checkpoints import save_safetensors
from imagehub.models.llada import LLaDAForMultiModalGeneration
from imagehub.models.vq_model import VQModel
from imagehub.types import T2IRequest

TINY_MODEL = {
    "d_model": 16,
    "embedding_size": IMAGE_TOKEN_OFFSET + CODEBOOK_SIZE,
    "vocab_size": IMAGE_TOKEN_OFFSET + CODEBOOK_SIZE,
    "n_heads": 4,
    "n_kv_heads": 4,
    "n_layers": 1,
    "mlp_hidden_size": 24,
    "max_sequence_length": 256,
}
TINY_VQ = {
    "block_out_channels": [32, 32, 32, 32, 32],
    "down_block_types": ["DownEncoderBlock2D"] * 5,
    "up_block_types": ["UpDecoderBlock2D"] * 5,
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 32,
    "num_vq_embeddings": CODEBOOK_SIZE,
    "lookup_from_codebook": True,
    "mid_block_add_attention": False,
    "force_upcast": False,
}


@pytest.fixture(scope="module")
def tiny_snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-lumina-dimoo")
    (root / "vqvae").mkdir()
    (root / "config.json").write_text(json.dumps(TINY_MODEL), encoding="utf-8")
    (root / "vqvae/config.json").write_text(json.dumps(TINY_VQ), encoding="utf-8")
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "LuminaDiMOOPipeline"}), encoding="utf-8"
    )
    model = LLaDAForMultiModalGeneration(TINY_MODEL).eval()
    vqvae = VQModel(TINY_VQ).eval()
    save_safetensors(model.state_dict(), root / "model.safetensors")
    save_safetensors(vqvae.state_dict(), root / "vqvae/diffusion_pytorch_model.safetensors")
    weight_map = {name: "model.safetensors" for name in model.state_dict()}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )
    vocab = {value: index for index, value in enumerate(bytes_to_unicode().values())}
    tokenizer = {
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
        "added_tokens": [
            {
                "id": 126351,
                "content": "<uncondition>",
                "normalized": False,
                "special": True,
            }
        ],
    }
    (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    (root / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": {"content": "<uncondition>"}}), encoding="utf-8"
    )
    (root / "imagehub_snapshot.json").write_text(
        json.dumps({"model_id": "lumina-dimoo", "revision": "test"}), encoding="utf-8"
    )
    return root


def test_tiny_snapshot_generates_and_is_repeatable(tiny_snapshot):
    pipe = pipeline("t2i", backend="lumina-dimoo", model=tiny_snapshot)
    first = pipe("cat", width=32, height=16, steps=2, guidance=0, seed=3)
    second = pipe("cat", width=32, height=16, steps=2, guidance=0, seed=3)
    assert first.images[0].tensor.shape == (16, 32, 3)
    assert torch.equal(first.images[0].tensor, second.images[0].tensor)
    assert first.metadata["scheduler"] == "cosine-maskgit"
    assert first.metadata["seed"] == 3


def test_requests_are_isolated(tiny_snapshot):
    pipe = pipeline("t2i", backend="lumina-dimoo", model=tiny_snapshot)
    options = {"width": 32, "height": 16, "steps": 1, "guidance": 0, "seed": 7}
    first = pipe("cat", **options).images[0].tensor
    pipe("a different prompt", **options)
    again = pipe("cat", **options).images[0].tensor
    assert torch.equal(first, again)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_callers_autograd_mode_is_restored(tiny_snapshot, grad_enabled):
    pipe = pipeline("t2i", backend="lumina-dimoo", model=tiny_snapshot)
    with torch.set_grad_enabled(grad_enabled):
        pipe("cat", width=32, height=16, steps=1, guidance=0, seed=1)
        assert torch.is_grad_enabled() is grad_enabled


def test_generation_two_seeds_and_guidance_paths():
    torch.manual_seed(0)
    model = LLaDAForMultiModalGeneration(TINY_MODEL).eval()
    conditional = torch.tensor([[1, 2, MASK_TOKEN_ID, MASK_TOKEN_ID, 3, 4]])
    unconditional = torch.tensor([[5, 6]])
    arguments = dict(
        model=model,
        conditional_ids=conditional,
        unconditional_ids=unconditional,
        code_start=2,
        image_tokens=2,
        steps=2,
        temperature=1.0,
    )
    first = generate_tokens(**arguments, guidance=0, generator=torch.Generator().manual_seed(0))
    again = generate_tokens(**arguments, guidance=0, generator=torch.Generator().manual_seed(0))
    guided = generate_tokens(**arguments, guidance=2, generator=torch.Generator().manual_seed(42))
    assert torch.equal(first, again)
    assert first.shape == guided.shape == (1, 2)
    assert bool(((0 <= first) & (first < CODEBOOK_SIZE)).all())


def test_options_are_validated_before_loading(tmp_path):
    backend = LuminaDiMOOBackend(tmp_path / "missing")
    with pytest.raises(TypeError, match="Unknown Lumina-DiMOO options"):
        backend.infer(T2IRequest("cat"), surprise=True)
    with pytest.raises(ValueError, match="multiple of 16"):
        backend.infer(T2IRequest("cat"), width=17)
    assert not backend.is_loaded
