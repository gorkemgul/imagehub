"""Shared test helpers: tiny tokenizer, checkpoint locations and difference metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from imagehub.tokenization import BOS, EOS, bytes_to_unicode

TINY_MERGES = (("h", "i</w>"), ("c", "a"), ("ca", "t</w>"))
TINY_VOCAB_SIZE = 517


def write_tiny_tokenizer(directory: str | os.PathLike[str], *, pad_token: str = EOS) -> Path:
    """Byte-level vocab (512 symbols) + 3 merges + BOS/EOS; ids: hi</w>=512, ca=513, cat</w>=514."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    chars = list(bytes_to_unicode().values())
    vocab = {c: i for i, c in enumerate(chars)} | {c + "</w>": 256 + i for i, c in enumerate(chars)}
    for a, b in TINY_MERGES:
        vocab[a + b] = len(vocab)
    vocab[BOS] = len(vocab)
    vocab[EOS] = len(vocab)
    assert len(vocab) == TINY_VOCAB_SIZE
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    merges = "#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in TINY_MERGES)
    (path / "merges.txt").write_text(merges, encoding="utf-8")
    special = {"bos_token": BOS, "eos_token": EOS, "pad_token": pad_token, "unk_token": EOS}
    (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(special | {"model_max_length": 77, "do_lower_case": True}), encoding="utf-8"
    )
    return path


def checkpoint_dir(version: int) -> Path:
    """Local sd-v1-<version> snapshot from IMAGEHUB_SD_V1_<version>, or skip the test."""
    value = os.environ.get(f"IMAGEHUB_SD_V1_{version}")
    if not value:
        pytest.skip(f"Set IMAGEHUB_SD_V1_{version} to a local sd-v1-{version} snapshot")
    return Path(value)


def max_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.detach().cpu().double()
    e = expected.detach().cpu().double()
    d = (a - e).abs()
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / e.abs().clamp_min(1e-12)).max()),
    }


FIXTURES = Path(__file__).parent / "fixtures"

TINY_TEXT_CONFIG = {
    "vocab_size": TINY_VOCAB_SIZE,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 77,
}


def published_shapes(component: str) -> dict[str, list[int]]:
    data = json.loads((FIXTURES / "sd1_tensor_shapes.json").read_text(encoding="utf-8"))
    return {name: entry["shape"] for name, entry in data[component].items()}


def sdxl_published_shapes(component: str) -> dict[str, list[int]]:
    data = json.loads((FIXTURES / "sdxl_tensor_shapes.json").read_text(encoding="utf-8"))
    return {name: entry["shape"] for name, entry in data[component].items()}


def meta_shapes(model_cls, config) -> dict[str, list[int]]:
    with torch.device("meta"):
        model = model_cls(config)
    return {name: list(t.shape) for name, t in model.state_dict().items()}


def text_state_for_ours(ref_state: dict) -> dict:
    """transformers 5 CLIPTextModel state -> published-file names (``text_model.`` prefix)."""
    return {
        (
            name if name.startswith(("text_model.", "text_projection.")) else "text_model." + name
        ): value
        for name, value in ref_state.items()
    }


TINY_VAE_CONFIG = {"block_out_channels": [32, 32, 64, 64], "norm_num_groups": 8}

SD1_SCHEDULER = {
    "_class_name": "PNDMScheduler",
    "_diffusers_version": "0.6.0",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "num_train_timesteps": 1000,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "trained_betas": None,
    "clip_sample": False,
}

SDXL_SCHEDULER = {
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "num_train_timesteps": 1000,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "prediction_type": "epsilon",
    "interpolation_type": "linear",
    "use_karras_sigmas": False,
}


def legacy_vae_state(ref_state: dict) -> dict:
    """diffusers AutoencoderKL state -> the legacy attention names used by published files."""
    renamed = {}
    for name, value in ref_state.items():
        if ".attentions." in name:
            for new, old in (
                ("to_q", "query"),
                ("to_k", "key"),
                ("to_v", "value"),
                ("to_out.0", "proj_attn"),
            ):
                name = name.replace(f".{new}.", f".{old}.")
        renamed[name] = value
    return renamed


SD1_UNET = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280, 1280],
    "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
    "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    "layers_per_block": 2,
    "attention_head_dim": 8,
    "cross_attention_dim": 768,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "sample_size": 64,
}
SDXL_UNET = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "transformer_layers_per_block": [1, 2, 10],
    "attention_head_dim": [5, 10, 20],
    "cross_attention_dim": 2048,
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "sample_size": 128,
}
TINY_UNET_CONFIG = SD1_UNET | {
    "block_out_channels": [32, 64, 64, 64],
    "cross_attention_dim": 32,
    "attention_head_dim": 2,
    "norm_num_groups": 8,
}


TINY_TEXT_FILE_CONFIG = TINY_TEXT_CONFIG | {
    "architectures": ["CLIPTextModel"],
    "model_type": "clip_text_model",
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "bos_token_id": 515,
    "eos_token_id": 516,
    "pad_token_id": 516,
    "projection_dim": 32,
}
TINY_VAE_FILE_CONFIG = TINY_VAE_CONFIG | {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 2,
    "act_fn": "silu",
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "sample_size": 512,
}
TINY_UNET_FILE_CONFIG = TINY_UNET_CONFIG | {"_class_name": "UNet2DConditionModel"}
MODEL_INDEX = {
    "_class_name": "StableDiffusionPipeline",
    "feature_extractor": [None, None],
    "safety_checker": [None, None],
    "scheduler": ["diffusers", "PNDMScheduler"],
    "text_encoder": ["transformers", "CLIPTextModel"],
    "tokenizer": ["transformers", "CLIPTokenizer"],
    "unet": ["diffusers", "UNet2DConditionModel"],
    "vae": ["diffusers", "AutoencoderKL"],
}


def write_tiny_sd(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight SD 1.x snapshot in the published diffusers layout (diffusers loads it)."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(SD1_SCHEDULER))
    (root / "model_index.json").write_text(json.dumps(MODEL_INDEX))
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        (
            "unet",
            UNet2DConditionModel,
            TINY_UNET_FILE_CONFIG,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_VAE_FILE_CONFIG, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


def write_tiny_sdxl(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight SDXL snapshot in the published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    write_tiny_tokenizer(root / "tokenizer_2", pad_token="!")

    text_1 = TINY_TEXT_FILE_CONFIG | {"hidden_act": "quick_gelu"}
    text_2 = TINY_TEXT_FILE_CONFIG | {
        "architectures": ["CLIPTextModelWithProjection"],
        "hidden_act": "gelu",
        "projection_dim": 32,
    }
    unet = {
        "_class_name": "UNet2DConditionModel",
        "in_channels": 4,
        "out_channels": 4,
        "block_out_channels": [32, 64, 64],
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        "transformer_layers_per_block": [1, 1, 1],
        "attention_head_dim": [2, 4, 4],
        "cross_attention_dim": 64,
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "projection_class_embeddings_input_dim": 80,
        "layers_per_block": 1,
        "norm_num_groups": 8,
        "sample_size": 32,
    }
    vae = TINY_VAE_FILE_CONFIG | {"scaling_factor": 0.13025, "force_upcast": True}
    model_index = {
        "_class_name": "StableDiffusionXLPipeline",
        "_diffusers_version": "0.40.0",
        "force_zeros_for_empty_prompt": True,
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(SDXL_SCHEDULER))
    (root / "model_index.json").write_text(json.dumps(model_index))

    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, text_1, "model.safetensors"),
        ("text_encoder_2", CLIPTextModelWithProjection, text_2, "model.safetensors"),
        ("unet", UNet2DConditionModel, unet, "diffusion_pytorch_model.safetensors"),
        ("vae", AutoencoderKL, vae, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name.startswith("text_encoder"):
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root
