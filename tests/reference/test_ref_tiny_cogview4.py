"""Tiny-snapshot CogView4 end-to-end parity."""

import json
from pathlib import Path

import pytest
import torch

from imagehub import pipeline
from imagehub.backends.cogview4 import CogView4Components, denoise
from imagehub.checkpoints import save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.cogview4_transformer import CogView4Transformer2DModel
from imagehub.models.glm import GlmModel
from imagehub.tokenization import bytes_to_unicode

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
Tokenizer = tokenizers.Tokenizer
decoders, models, pre_tokenizers, processors = (
    tokenizers.decoders,
    tokenizers.models,
    tokenizers.pre_tokenizers,
    tokenizers.processors,
)

TEXT_CONFIG = {
    "vocab_size": 259,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1.5625e-7,
    "rope_theta": 10_000.0,
    "partial_rotary_factor": 0.5,
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "pad_token_id": 256,
}
TRANSFORMER_CONFIG = {
    "patch_size": 2,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "text_embed_dim": 32,
    "time_embed_dim": 16,
    "condition_dim": 4,
    "pos_embed_max_size": 8,
    "sample_size": 8,
    "rope_axes_dim": [8, 8],
}
VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [32, 32, 64, 64],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "layers_per_block": 3,
    "norm_num_groups": 8,
    "mid_block_add_attention": False,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "scaling_factor": 1.0,
    "shift_factor": 0.0,
    "sample_size": 32,
}
SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.25,
    "max_shift": 0.75,
    "base_image_seq_len": 4,
    "max_image_seq_len": 16,
}


def _write_tokenizer(directory: Path) -> None:
    directory.mkdir()
    vocabulary = {value: key for key, value in bytes_to_unicode().items()}
    vocabulary |= {"<|endoftext|>": 256, "[gMASK]": 257, "<sop>": 258}
    tokenizer = Tokenizer(models.BPE(vocabulary, [], ignore_merges=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<|endoftext|>", "[gMASK]", "<sop>"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[gMASK] <sop> $A",
        special_tokens=[("[gMASK]", 257), ("<sop>", 258)],
    )
    tokenizer.save(str(directory / "tokenizer.json"))
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "padding_side": "left",
        "model_max_length": 1024,
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "pad_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )


def _write_tiny_cogview4(directory: Path) -> Path:
    directory.mkdir()
    _write_tokenizer(directory / "tokenizer")
    (directory / "scheduler").mkdir()
    (directory / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(SCHEDULER_CONFIG), encoding="utf-8"
    )
    (directory / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "CogView4Pipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "GlmModel"],
                "tokenizer": ["transformers", "PreTrainedTokenizerFast"],
                "transformer": ["diffusers", "CogView4Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    torch.manual_seed(0)
    for name, cls, config, weights in (
        ("text_encoder", GlmModel, TEXT_CONFIG, "model.safetensors"),
        (
            "transformer",
            CogView4Transformer2DModel,
            TRANSFORMER_CONFIG,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, VAE_CONFIG, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (directory / name).mkdir()
        (directory / name / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_safetensors(model.state_dict(), directory / name / weights)
    return directory


@pytest.mark.parametrize("use_seed", [False, True], ids=["latents", "seed"])
@pytest.mark.parametrize("negative_prompt", [None, "blurry"], ids=["empty-negative", "negative"])
def test_tiny_cogview4_matches_diffusers_exactly(tmp_path, use_seed, negative_prompt):
    root = _write_tiny_cogview4(tmp_path / "tiny")
    reference = diffusers.CogView4Pipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    common = {
        "prompt": "a cat",
        "negative_prompt": negative_prompt,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
        "height": 32,
        "width": 32,
        "max_sequence_length": 16,
    }
    source = (
        {"generator": torch.Generator("cpu").manual_seed(42)}
        if use_seed
        else {"latents": noise.clone()}
    )
    expected_latents = reference(**common, **source, output_type="latent").images
    if use_seed:
        source = {"generator": torch.Generator("cpu").manual_seed(42)}
    expected_pixels = reference(**common, **source, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = CogView4Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_noise = (
        torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
        if use_seed
        else noise.clone()
    )
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=5.0,
        width=32,
        height=32,
        latents=actual_noise,
        max_sequence_length=16,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    call = {
        "negative_prompt": negative_prompt,
        "steps": 2,
        "guidance": 5.0,
        "width": 32,
        "height": 32,
        "max_sequence_length": 16,
    }
    if use_seed:
        call["seed"] = 42
    else:
        call["latents"] = noise
    result = pipeline("t2i", backend="cogview4", model=root)("a cat", **call)
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
