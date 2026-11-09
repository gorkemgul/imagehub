"""Shared test helpers: tiny tokenizer, checkpoint locations and difference metrics."""

from __future__ import annotations

import base64
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from fixture_files import fixture_path

from imagehub.tokenization import BOS, EOS, bytes_to_unicode

TINY_MERGES = (("h", "i</w>"), ("c", "a"), ("ca", "t</w>"))
TINY_VOCAB_SIZE = 517


def reference_in_subprocess(
    function_path: str, kwargs: dict[str, object], out_path: str | os.PathLike[str]
) -> Any:
    """Run a reference function in a new interpreter and load its saved result.

    The function must be importable by ``function_path`` and accept ``out_path`` as a keyword
    argument. It is responsible for saving its result with ``torch.save``.
    """
    output = str(out_path)
    payload = base64.b64encode(
        pickle.dumps(
            (function_path, kwargs, output, sys.path, torch.get_num_threads()),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    ).decode("ascii")
    code = "\n".join(
        (
            "import base64, importlib, pickle, sys, torch",
            "payload = pickle.loads(base64.b64decode(sys.argv[1]))",
            "function_path, kwargs, output, paths, threads = payload",
            "sys.path[:0] = paths",
            "torch.set_num_threads(threads)",
            "module_path, function_name = function_path.rsplit('.', 1)",
            "function = getattr(importlib.import_module(module_path), function_name)",
            "parameters = __import__('inspect').signature(function).parameters",
            "output_name = 'out_path' if 'out_path' in parameters else 'output'",
            "function(**kwargs, **{output_name: output})",
        )
    )
    subprocess.run([sys.executable, "-c", code, payload], check=True, env=os.environ.copy())
    return torch.load(output, weights_only=True)


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


def snapshot_dir(model_id: str) -> Path:
    """Local model snapshot from IMAGEHUB_SNAPSHOT_<MODEL_ID>, or skip the test."""
    key = "".join(character if character.isalnum() else "_" for character in model_id.upper())
    variable = f"IMAGEHUB_SNAPSHOT_{key}"
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"Set {variable} to a local {model_id} snapshot")
    return Path(value)


def max_diff(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    a = actual.detach().cpu().double()
    e = expected.detach().cpu().double()
    d = (a - e).abs()
    return {
        "max_abs": float(d.max()),
        "max_rel": float((d / e.abs().clamp_min(1e-12)).max()),
    }


FIXTURES = fixture_path()

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


def ssd1b_published_shapes(component: str) -> dict[str, list[int]]:
    data = json.loads((FIXTURES / "ssd1b_tensor_shapes.json").read_text(encoding="utf-8"))
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

LCM_SCHEDULER = {
    "_class_name": "LCMScheduler",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "dynamic_thresholding_ratio": 0.995,
    "num_train_timesteps": 1000,
    "original_inference_steps": 50,
    "prediction_type": "epsilon",
    "sample_max_value": 1.0,
    "set_alpha_to_one": True,
    "steps_offset": 0,
    "thresholding": False,
    "timestep_scaling": 10.0,
    "timestep_spacing": "leading",
    "trained_betas": None,
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
SSD1B_UNET = SDXL_UNET | {
    "transformer_layers_per_block": [[1], [2, 2], [4, 4]],
    "reverse_transformer_layers_per_block": [[4, 4, 10], [2, 1, 1], 1],
    "mid_block_type": "UNetMidBlock2D",
}
TINY_SSD1B_UNET = SSD1B_UNET | {
    "block_out_channels": [32, 64, 64],
    "attention_head_dim": [2, 4, 4],
    "cross_attention_dim": 32,
    "addition_time_embed_dim": 8,
    "projection_class_embeddings_input_dim": 80,
    "norm_num_groups": 8,
    "sample_size": 32,
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
TINY_AUTOENCODER_CONFIG = {
    "_class_name": "AutoencoderTiny",
    "in_channels": 3,
    "out_channels": 3,
    "encoder_block_out_channels": [8, 8, 8, 8],
    "decoder_block_out_channels": [8, 8, 8, 8],
    "latent_channels": 4,
    "upsampling_scaling_factor": 2,
    "num_encoder_blocks": [1, 1, 1, 1],
    "num_decoder_blocks": [1, 1, 1, 1],
    "latent_magnitude": 3,
    "latent_shift": 0.5,
    "scaling_factor": 1.0,
    "act_fn": "relu",
    "upsample_fn": "nearest",
    "force_upcast": False,
    "sample_size": 32,
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


def write_tiny_sd(
    directory: str | os.PathLike[str],
    *,
    seed: int = 0,
    unet_config: dict[str, Any] | None = None,
    scheduler_config: dict | None = None,
    tiny_vae: bool = False,
) -> Path:
    """A random-weight SD 1.x snapshot in the published diffusers layout (diffusers loads it)."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.autoencoder_tiny import AutoencoderTiny
    from imagehub.models.clip_text import CLIPTextModel
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    (root / "scheduler").mkdir()
    selected_scheduler = SD1_SCHEDULER | (scheduler_config or {})
    selected_unet = TINY_UNET_FILE_CONFIG | (unet_config or {})
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(selected_scheduler))
    vae_type = AutoencoderTiny if tiny_vae else AutoencoderKL
    vae_config = TINY_AUTOENCODER_CONFIG if tiny_vae else TINY_VAE_FILE_CONFIG
    model_index = MODEL_INDEX | {
        "scheduler": ["diffusers", selected_scheduler["_class_name"]],
        "vae": ["diffusers", vae_config["_class_name"]],
    }
    (root / "model_index.json").write_text(json.dumps(model_index))
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        (
            "unet",
            UNet2DConditionModel,
            selected_unet,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", vae_type, vae_config, "diffusion_pytorch_model.safetensors"),
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


def write_tiny_ldm(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight LDM snapshot in the original published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.ldm_bert import LDMBertModel
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    tokens = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "cat", "b")
    (tokenizer / "vocab.txt").write_text("\n".join(tokens) + "\n", encoding="utf-8")
    tokenizer_config = {
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "do_lower_case": True,
        "tokenizer_class": "BertTokenizer",
    }
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (tokenizer / "special_tokens_map.json").write_text(json.dumps(tokenizer_config))
    bert = {
        "_class_name": "LDMBertModel",
        "model_type": "ldmbert",
        "vocab_size": len(tokens),
        "max_position_embeddings": 77,
        "d_model": 1280,
        "encoder_layers": 1,
        "encoder_ffn_dim": 32,
        "encoder_attention_heads": 2,
        "head_dim": 16,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "activation_dropout": 0.0,
        "activation_function": "gelu",
    }
    unet = TINY_UNET_FILE_CONFIG | {"cross_attention_dim": 1280, "sample_size": 8}
    vae = TINY_VAE_FILE_CONFIG | {"sample_size": 64}
    scheduler = {
        "_class_name": "DDIMScheduler",
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "num_train_timesteps": 1000,
        "clip_sample": False,
    }
    index = {
        "_class_name": "LDMTextToImagePipeline",
        "bert": ["latent_diffusion", "LDMBertModel"],
        "scheduler": ["diffusers", "DDIMScheduler"],
        "tokenizer": ["transformers", "BertTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vqvae": ["diffusers", "AutoencoderKL"],
    }
    (root / "model_index.json").write_text(json.dumps(index))
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler))
    for name, cls, config, weights in (
        ("bert", LDMBertModel, bert, "model.safetensors"),
        ("unet", UNet2DConditionModel, unet, "diffusion_pytorch_model.safetensors"),
        ("vqvae", AutoencoderKL, vae, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.02)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


def write_tiny_lcm(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight LCM snapshot in the published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(LCM_SCHEDULER))
    model_index = MODEL_INDEX | {
        "_class_name": "LatentConsistencyModelPipeline",
        "scheduler": ["diffusers", "LCMScheduler"],
    }
    (root / "model_index.json").write_text(json.dumps(model_index))
    unet_config = TINY_UNET_FILE_CONFIG | {"sample_size": 4, "time_cond_proj_dim": 16}
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        (
            "unet",
            UNet2DConditionModel,
            unet_config,
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


def write_tiny_sdxl(
    directory: str | os.PathLike[str],
    *,
    seed: int = 0,
    scheduler_config: dict | None = None,
    vae_config: dict | None = None,
    unet_config: dict | None = None,
    sample_size: int = 32,
    refiner: bool = False,
) -> Path:
    """A random-weight SDXL snapshot in the published diffusers layout.

    With ``refiner`` the snapshot has the refiner layout: only the second text encoder, five
    time ids per sample and the image-to-image pipeline class.
    """
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if not refiner:
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
        "cross_attention_dim": 32 if refiner else 64,
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "projection_class_embeddings_input_dim": 72 if refiner else 80,
        "layers_per_block": 1,
        "norm_num_groups": 8,
        "sample_size": sample_size,
    }
    if unet_config is not None:
        unet |= unet_config
    vae = TINY_VAE_FILE_CONFIG | {"scaling_factor": 0.13025, "force_upcast": True}
    if vae_config is not None:
        vae |= vae_config
    model_index = {
        "_class_name": "StableDiffusionXLPipeline",
        "_diffusers_version": "0.40.0",
        "force_zeros_for_empty_prompt": True,
        "scheduler": [
            "diffusers",
            (scheduler_config or SDXL_SCHEDULER).get("_class_name", "EulerDiscreteScheduler"),
        ],
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(scheduler_config or SDXL_SCHEDULER)
    )
    if refiner:
        model_index |= {
            "_class_name": "StableDiffusionXLImg2ImgPipeline",
            "force_zeros_for_empty_prompt": False,
            "requires_aesthetics_score": True,
            "text_encoder": [None, None],
            "tokenizer": [None, None],
        }
    (root / "model_index.json").write_text(json.dumps(model_index))

    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, text_1, "model.safetensors"),
        ("text_encoder_2", CLIPTextModelWithProjection, text_2, "model.safetensors"),
        ("unet", UNet2DConditionModel, unet, "diffusion_pytorch_model.safetensors"),
        ("vae", AutoencoderKL, vae, "diffusion_pytorch_model.safetensors"),
    ):
        if refiner and name == "text_encoder":
            continue
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


TINY_CHATGLM = {
    "_name_or_path": "tiny",
    "architectures": ["ChatGLMModel"],
    "model_type": "chatglm",
    "num_layers": 3,
    "padded_vocab_size": 320,
    "hidden_size": 64,
    "ffn_hidden_size": 96,
    "kv_channels": 16,
    "num_attention_heads": 4,
    "seq_length": 512,
    "layernorm_epsilon": 1e-5,
    "rmsnorm": True,
    "add_bias_linear": False,
    "add_qkv_bias": True,
    "multi_query_attention": True,
    "multi_query_group_num": 2,
    "original_rope": True,
    "post_layer_norm": True,
    "pad_token_id": 0,
    "eos_token_id": 2,
}
KOLORS_SCHEDULER = {
    "_class_name": "EulerDiscreteScheduler",
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "beta_end": 0.014,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1100,
    "prediction_type": "epsilon",
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "use_karras_sigmas": False,
}


def write_tiny_kolors(
    directory: str | os.PathLike[str], *, seed: int = 0, sample_size: int = 32
) -> Path:
    """A random-weight Kolors snapshot in the published layout (float16 variant file names)."""
    from test_tokenization_chatglm import write_model

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.chatglm import ChatGLMModel
    from imagehub.models.unet_2d import UNet2DConditionModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    (root / "tokenizer").mkdir()
    write_model(root / "tokenizer" / "tokenizer.model")
    (root / "tokenizer" / "tokenizer_config.json").write_text(
        (FIXTURES / "kolors_configs" / "tokenizer_tokenizer_config.json").read_text()
    )
    (root / "tokenizer" / "vocab.txt").write_text("")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(KOLORS_SCHEDULER))
    (root / "model_index.json").write_text(
        (FIXTURES / "kolors_configs" / "model_index.json").read_text()
    )
    unet = {
        "_class_name": "UNet2DConditionModel",
        "in_channels": 4,
        "out_channels": 4,
        "block_out_channels": [32, 64, 64],
        "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
        "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        "transformer_layers_per_block": [1, 1, 1],
        "attention_head_dim": [2, 4, 4],
        "cross_attention_dim": 32,
        "encoder_hid_dim": 64,
        "encoder_hid_dim_type": "text_proj",
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "projection_class_embeddings_input_dim": 112,
        "layers_per_block": 1,
        "norm_num_groups": 8,
        "sample_size": sample_size,
    }
    vae = TINY_VAE_FILE_CONFIG | {"scaling_factor": 0.13025}
    for name, cls, config in (
        ("text_encoder", ChatGLMModel, TINY_CHATGLM),
        ("unet", UNet2DConditionModel, unet),
        ("vae", AutoencoderKL, vae),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = {key: value.half() for key, value in model.state_dict().items()}
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        if name != "text_encoder":
            save_safetensors(state, root / name / "diffusion_pytorch_model.fp16.safetensors")
            continue
        shards = [{key: state[key] for key in list(state)[index::3]} for index in range(3)]
        weight_map = {}
        for index, shard in enumerate(shards, start=1):
            file = f"model.fp16-0000{index}-of-00003.safetensors"
            save_safetensors(shard, root / name / file)
            weight_map |= dict.fromkeys(shard, file)
        (root / name / "model.safetensors.index.fp16.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
    return root


TINY_Z_IMAGE_TEXT = {
    "vocab_size": 34,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "sliding_window": None,
    "model_type": "qwen3",
    "architectures": ["Qwen3Model"],
}
TINY_Z_IMAGE_TRANSFORMER = {
    "_class_name": "ZImageTransformer2DModel",
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "in_channels": 16,
    "dim": 32,
    "n_layers": 2,
    "n_refiner_layers": 1,
    "n_heads": 2,
    "n_kv_heads": 2,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 32,
    "siglip_feat_dim": None,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [4, 4, 8],
    "axes_lens": [64, 32, 32],
}
TINY_Z_IMAGE_VAE = {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": [32, 32, 64, 64],
    "layers_per_block": 2,
    "norm_num_groups": 8,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "mid_block_add_attention": True,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "sample_size": 32,
}
TINY_Z_IMAGE_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": False,
    "shift": 3.0,
}


def write_tiny_z_image(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight Z-Image snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.qwen3 import Qwen3Model
    from imagehub.models.z_image_transformer import ZImageTransformer2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = root / "tokenizer"
    shutil.copytree(FIXTURES / "qwen_tokenizer", tokenizer)
    tokenizer_config = json.loads((tokenizer / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(TINY_Z_IMAGE_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ZImagePipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "ZImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    for name, cls, config, weights in (
        ("text_encoder", Qwen3Model, TINY_Z_IMAGE_TEXT, "model.safetensors"),
        (
            "transformer",
            ZImageTransformer2DModel,
            TINY_Z_IMAGE_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_Z_IMAGE_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state = {key.removeprefix("model."): value for key, value in state.items()}
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


TINY_OVIS_IMAGE_TEXT = TINY_Z_IMAGE_TEXT | {"vocab_size": 151936, "max_position_embeddings": 256}
TINY_OVIS_IMAGE_TRANSFORMER = {
    "_class_name": "OvisImageTransformer2DModel",
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "axes_dims_rope": [2, 2, 4],
}
TINY_OVIS_IMAGE_VAE = TINY_Z_IMAGE_VAE | {"sample_size": 32}
TINY_OVIS_IMAGE_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": True,
    "base_image_seq_len": 1,
    "max_image_seq_len": 16,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "shift": 3.0,
}


def write_tiny_ovis_image(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """Write a random tiny Ovis-Image snapshot in the published layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.ovis_image_transformer import OvisImageTransformer2DModel
    from imagehub.models.qwen3 import Qwen3Model

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    source = FIXTURES / "ovis_image_configs"
    for name in (
        "vocab.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        shutil.copyfile(source / f"tokenizer_{name}", tokenizer / name)
    shutil.copyfile(source / "tokenizer_added_tokens.json", tokenizer / "added_tokens.json")
    tokenizer_config = json.loads((tokenizer / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n{% endif %}"
    )
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(TINY_OVIS_IMAGE_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "OvisImagePipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3Model"],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "transformer": ["diffusers", "OvisImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    for name, cls, config, weights in (
        ("text_encoder", Qwen3Model, TINY_OVIS_IMAGE_TEXT, "model.safetensors"),
        (
            "transformer",
            OvisImageTransformer2DModel,
            TINY_OVIS_IMAGE_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_OVIS_IMAGE_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state = {key.removeprefix("model."): value for key, value in state.items()}
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


TINY_FLUX2_TEXT = TINY_Z_IMAGE_TEXT | {
    "vocab_size": 36,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 27,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "architectures": ["Qwen3ForCausalLM"],
    "tie_word_embeddings": False,
}
TINY_FLUX2_TRANSFORMER = {
    "_class_name": "Flux2Transformer2DModel",
    "patch_size": 1,
    "in_channels": 16,
    "out_channels": 16,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 24,
    "timestep_guidance_channels": 16,
    "mlp_ratio": 3.0,
    "axes_dims_rope": [2, 2, 2, 2],
    "rope_theta": 2000,
    "eps": 1e-6,
    "guidance_embeds": False,
}
TINY_FLUX2_VAE = {
    "_class_name": "AutoencoderKLFlux2",
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
    "block_out_channels": [8, 16],
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 4,
    "sample_size": 16,
    "force_upcast": True,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "mid_block_add_attention": True,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
    "patch_size": [2, 2],
}
TINY_FLUX2_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}


def write_tiny_flux2(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight FLUX.2 klein snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from imagehub.models.flux2_transformer import Flux2Transformer2DModel
    from imagehub.models.qwen3 import Qwen3Model

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = root / "tokenizer"
    shutil.copytree(FIXTURES / "qwen_tokenizer", tokenizer)
    tokenizer_data = json.loads((tokenizer / "tokenizer.json").read_text())
    for token_id, content in ((34, "<think>"), (35, "</think>")):
        tokenizer_data["model"]["vocab"][content] = token_id
        tokenizer_data["added_tokens"].append(
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": False,
            }
        )
    (tokenizer / "tokenizer.json").write_text(json.dumps(tokenizer_data))
    tokenizer_config = json.loads((tokenizer / "tokenizer_config.json").read_text())
    tokenizer_config["chat_template"] = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{{ message['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% endif %}{% endif %}"
    )
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(TINY_FLUX2_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "is_distilled": True,
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3ForCausalLM"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "Flux2Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        )
    )
    for name, cls, config, weights in (
        ("text_encoder", Qwen3Model, TINY_FLUX2_TEXT, "model.safetensors"),
        (
            "transformer",
            Flux2Transformer2DModel,
            TINY_FLUX2_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        (
            "vae",
            AutoencoderKLFlux2,
            TINY_FLUX2_VAE,
            "diffusion_pytorch_model.safetensors",
        ),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        if name == "vae":
            model.bn.running_mean.normal_(0, 0.05)
            model.bn.running_var.uniform_(0.5, 1.5)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["lm_head.weight"] = torch.randn(config["vocab_size"], config["hidden_size"])
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


TINY_QWEN_IMAGE21_TEXT = {
    "vocab_size": 262,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000.0,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "rope_type": "default",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
}
TINY_QWEN_IMAGE21_TRANSFORMER = {
    "_class_name": "QwenImage21Transformer2DModel",
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "context_in_dim": 16,
    "mlp_ratio": 3,
    "axes_dims_rope": [2, 2, 4],
    "eps": 1e-6,
    "causal_condition": True,
}
TINY_QWEN_IMAGE21_VAE = {
    "_class_name": "AutoencoderKLQwenImage21",
    "attn_scales": [],
    "base_dim": 4,
    "decoder_base_dim": 4,
    "z_dim": 64,
    "dim_mult": [1, 1, 1, 1, 1],
    "num_res_blocks": 1,
    "temperal_downsample": [False, False, False, False],
    "dropout": 0.0,
    "in_channels": 4,
    "out_channels": 4,
    "is_residual": True,
    "latents_mean": [0.1, -0.2, 0.3, -0.4] * 16,
    "latents_std": [1.1, 0.9, 1.2, 0.8] * 16,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 8,
}
TINY_QWEN_IMAGE21_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len": 1,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 16,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def write_tiny_qwen_image21(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """Write a random tiny Qwen-Image 2.1 snapshot in the published layout."""
    import shutil

    import transformers

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl_qwen_image21 import AutoencoderKLQwenImage21
    from imagehub.models.qwen_image21_transformer import QwenImage21Transformer2DModel
    from imagehub.tokenization_qwen import bytes_to_unicode as qwen_bytes_to_unicode

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    processor_dir = root / "processor"
    processor_dir.mkdir()

    tokenizer_data = json.loads((FIXTURES / "qwen_tokenizer/tokenizer.json").read_text())
    vocab = {character: index for index, character in enumerate(qwen_bytes_to_unicode().values())}
    special_tokens = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>")
    for token in special_tokens:
        vocab[token] = len(vocab)
    tokenizer_data["model"]["vocab"] = vocab
    tokenizer_data["model"]["merges"] = []
    tokenizer_data["added_tokens"] = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in special_tokens
    ]
    (processor_dir / "tokenizer.json").write_text(json.dumps(tokenizer_data))
    shutil.copyfile(FIXTURES / "qwen_tokenizer/merges.txt", processor_dir / "merges.txt")
    (processor_dir / "vocab.json").write_text(json.dumps(vocab))
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 256,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": list(special_tokens[1:]),
    }
    (processor_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (processor_dir / "special_tokens_map.json").write_text(json.dumps(tokenizer_config))
    chat_template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{% for content in message['content'] %}{{ content['text'] }}{% endfor %}"
        "<|im_end|>\n{% endfor %}"
    )
    (processor_dir / "chat_template.jinja").write_text(chat_template)
    (processor_dir / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "Qwen3VLImageProcessor"})
    )
    (processor_dir / "video_preprocessor_config.json").write_text(
        json.dumps({"video_processor_type": "Qwen3VLVideoProcessor"})
    )

    text_config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": TINY_QWEN_IMAGE21_TEXT,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 16,
            "in_channels": 3,
            "intermediate_size": 32,
            "model_type": "qwen3_vl",
            "num_heads": 2,
            "num_position_embeddings": 64,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "image_token_id": vocab["<|image_pad|>"],
        "video_token_id": vocab["<|image_pad|>"],
        "vision_start_token_id": vocab["<|im_start|>"],
        "vision_end_token_id": vocab["<|im_end|>"],
        "tie_word_embeddings": False,
    }
    torch.manual_seed(seed)
    reference_text = transformers.Qwen3VLForConditionalGeneration(
        transformers.Qwen3VLConfig(**text_config)
    )
    (root / "text_encoder").mkdir()
    (root / "text_encoder/config.json").write_text(json.dumps(text_config))
    save_safetensors(reference_text.state_dict(), root / "text_encoder/model.safetensors")

    for name, cls, config in (
        ("transformer", QwenImage21Transformer2DModel, TINY_QWEN_IMAGE21_TRANSFORMER),
        ("vae", AutoencoderKLQwenImage21, TINY_QWEN_IMAGE21_VAE),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(model.state_dict(), root / name / "diffusion_pytorch_model.safetensors")
    (root / "scheduler").mkdir()
    (root / "scheduler/scheduler_config.json").write_text(json.dumps(TINY_QWEN_IMAGE21_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImage21Pipeline",
                "processor": ["transformers", "Qwen3VLProcessor"],
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
                "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage21"],
            }
        )
    )
    return root


TINY_QWEN_IMAGE_TEXT = {
    "vocab_size": 262,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_scaling": {"type": "default", "mrope_section": [2, 1, 1]},
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "tie_word_embeddings": False,
}
TINY_QWEN_IMAGE_TRANSFORMER = {
    "_class_name": "QwenImageTransformer2DModel",
    "patch_size": 2,
    "in_channels": 64,
    "out_channels": 16,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 16,
    "guidance_embeds": False,
    "axes_dims_rope": [2, 2, 4],
}
TINY_QWEN_IMAGE_VAE = {
    "_class_name": "AutoencoderKLQwenImage",
    "base_dim": 4,
    "z_dim": 16,
    "dim_mult": [1, 2, 2, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "input_channels": 3,
    "latents_mean": [0.0] * 16,
    "latents_std": [1.0] * 16,
}
TINY_QWEN_IMAGE_SCHEDULER = TINY_QWEN_IMAGE21_SCHEDULER | {"shift_terminal": None}


def write_tiny_qwen_image(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """Write a random tiny original Qwen-Image snapshot in the published layout."""
    import shutil

    import transformers

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
    from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
    from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel
    from imagehub.tokenization_qwen import bytes_to_unicode as qwen_bytes_to_unicode

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    tokenizer_data = json.loads((FIXTURES / "qwen_tokenizer/tokenizer.json").read_text())
    vocab = {character: index for index, character in enumerate(qwen_bytes_to_unicode().values())}
    special_tokens = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>")
    for token in special_tokens:
        vocab[token] = len(vocab)
    tokenizer_data["model"]["vocab"] = vocab
    tokenizer_data["model"]["merges"] = []
    tokenizer_data["added_tokens"] = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in special_tokens
    ]
    (tokenizer / "tokenizer.json").write_text(json.dumps(tokenizer_data))
    shutil.copyfile(FIXTURES / "qwen_tokenizer/merges.txt", tokenizer / "merges.txt")
    (tokenizer / "vocab.json").write_text(json.dumps(vocab))
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 1058,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": list(special_tokens[1:]),
    }
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (tokenizer / "special_tokens_map.json").write_text(json.dumps(tokenizer_config))

    text_config = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "text_config": TINY_QWEN_IMAGE_TEXT,
        "vision_config": {
            "depth": 1,
            "hidden_act": "silu",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "in_chans": 3,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "spatial_patch_size": 2,
            "window_size": 4,
            "fullatt_block_indexes": [],
            "tokens_per_second": 2,
            "temporal_patch_size": 2,
        },
        "bos_token_id": vocab["<|endoftext|>"],
        "eos_token_id": vocab["<|im_end|>"],
        "vision_start_token_id": vocab["<|im_start|>"],
        "vision_end_token_id": vocab["<|im_end|>"],
        "vision_token_id": vocab["<|image_pad|>"],
        "image_token_id": vocab["<|image_pad|>"],
        "video_token_id": vocab["<|image_pad|>"],
        "tie_word_embeddings": False,
    }
    torch.manual_seed(seed)
    native_text = Qwen25VLTextModel(text_config)
    full_text = transformers.Qwen2_5_VLForConditionalGeneration(
        transformers.Qwen2_5_VLConfig.from_dict(text_config)
    )
    for parameter in full_text.parameters():
        parameter.data.normal_(0, 0.05)
    text_state = dict(full_text.state_dict())
    text_state.update(native_text.state_dict())
    (root / "text_encoder").mkdir()
    (root / "text_encoder/config.json").write_text(json.dumps(text_config))
    save_safetensors(text_state, root / "text_encoder/model.safetensors")

    for name, cls, config in (
        ("transformer", QwenImageTransformer2DModel, TINY_QWEN_IMAGE_TRANSFORMER),
        ("vae", AutoencoderKLQwenImage, TINY_QWEN_IMAGE_VAE),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(model.state_dict(), root / name / "diffusion_pytorch_model.safetensors")
    (root / "scheduler").mkdir()
    (root / "scheduler/scheduler_config.json").write_text(json.dumps(TINY_QWEN_IMAGE_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "QwenImagePipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "QwenImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage"],
            }
        )
    )
    return root


TINY_KREA2_TEXT = {
    "vocab_size": 262,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 13,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000.0,
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
        "rope_type": "default",
    },
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
}
TINY_KREA2_TRANSFORMER = {
    "_class_name": "Krea2Transformer2DModel",
    "in_channels": 16,
    "num_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "intermediate_size": 32,
    "timestep_embed_dim": 16,
    "text_hidden_dim": 16,
    "num_text_layers": 12,
    "text_num_attention_heads": 2,
    "text_num_key_value_heads": 1,
    "text_intermediate_size": 32,
    "num_layerwise_text_blocks": 1,
    "num_refiner_text_blocks": 1,
    "axes_dims_rope": [2, 2, 4],
    "rope_theta": 1000.0,
    "norm_eps": 1e-5,
}
TINY_KREA2_VAE = {
    "_class_name": "AutoencoderKLQwenImage",
    "base_dim": 4,
    "z_dim": 4,
    "dim_mult": [1, 2, 2, 2],
    "num_res_blocks": 1,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "input_channels": 3,
    "latents_mean": [0.1, -0.2, 0.3, -0.4],
    "latents_std": [1.1, 0.9, 1.2, 0.8],
}
TINY_KREA2_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "base_image_seq_len": 1,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 16,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def write_tiny_krea2(
    directory: str | os.PathLike[str], *, seed: int = 0, distilled: bool = True
) -> Path:
    """Write a random tiny Krea 2 snapshot in the published layout."""
    import shutil

    import transformers

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
    from imagehub.models.krea2_transformer import Krea2Transformer2DModel
    from imagehub.tokenization_qwen import bytes_to_unicode as qwen_bytes_to_unicode

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    tokenizer_data = json.loads((FIXTURES / "qwen_tokenizer/tokenizer.json").read_text())
    vocab = {character: index for index, character in enumerate(qwen_bytes_to_unicode().values())}
    special_tokens = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>")
    for token in special_tokens:
        vocab[token] = len(vocab)
    tokenizer_data["model"]["vocab"] = vocab
    tokenizer_data["model"]["merges"] = []
    tokenizer_data["added_tokens"] = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in special_tokens
    ]
    (tokenizer / "tokenizer.json").write_text(json.dumps(tokenizer_data))
    shutil.copyfile(FIXTURES / "qwen_tokenizer/merges.txt", tokenizer / "merges.txt")
    (tokenizer / "vocab.json").write_text(json.dumps(vocab))
    tokenizer_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 256,
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "additional_special_tokens": list(special_tokens[1:]),
    }
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (tokenizer / "special_tokens_map.json").write_text(json.dumps(tokenizer_config))
    (tokenizer / "chat_template.jinja").write_text("")

    text_config = {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "text_config": TINY_KREA2_TEXT,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 16,
            "in_channels": 3,
            "intermediate_size": 32,
            "model_type": "qwen3_vl",
            "num_heads": 2,
            "num_position_embeddings": 64,
            "out_hidden_size": 16,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "image_token_id": vocab["<|image_pad|>"],
        "video_token_id": vocab["<|image_pad|>"],
        "vision_start_token_id": vocab["<|im_start|>"],
        "vision_end_token_id": vocab["<|im_end|>"],
        "tie_word_embeddings": False,
    }
    torch.manual_seed(seed)
    text_encoder = transformers.Qwen3VLModel(transformers.Qwen3VLConfig(**text_config))
    (root / "text_encoder").mkdir()
    (root / "text_encoder/config.json").write_text(json.dumps(text_config))
    save_safetensors(text_encoder.state_dict(), root / "text_encoder/model.safetensors")

    for name, cls, config in (
        ("transformer", Krea2Transformer2DModel, TINY_KREA2_TRANSFORMER),
        ("vae", AutoencoderKLQwenImage, TINY_KREA2_VAE),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(model.state_dict(), root / name / "diffusion_pytorch_model.safetensors")
    (root / "scheduler").mkdir()
    (root / "scheduler/scheduler_config.json").write_text(json.dumps(TINY_KREA2_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Krea2Pipeline",
                "is_distilled": distilled,
                "patch_size": 2,
                "text_encoder_select_layers": list(range(1, 13)),
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen3VLModel"],
                "tokenizer": ["transformers", "Qwen2Tokenizer"],
                "transformer": ["diffusers", "Krea2Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLQwenImage"],
            }
        )
    )
    return root


TINY_SD3_T5 = {
    "architectures": ["T5EncoderModel"],
    "vocab_size": 11,
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "dense_act_fn": "gelu_new",
    "is_gated_act": True,
    "is_encoder_decoder": False,
    "use_cache": False,
}
TINY_SD3_TRANSFORMER = {
    "_class_name": "SD3Transformer2DModel",
    "sample_size": 4,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "caption_projection_dim": 16,
    "pooled_projection_dim": 32,
    "out_channels": 4,
    "pos_embed_max_size": 4,
    "qk_norm": "rms_norm",
    "dual_attention_layers": [],
}
TINY_SD3_VAE = TINY_VAE_FILE_CONFIG | {
    "latent_channels": 4,
    "scaling_factor": 1.5305,
    "shift_factor": 0.0609,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "sample_size": 32,
}
TINY_SD3_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 3.0,
}


def write_tiny_sd3(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight SD3 snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModelWithProjection
    from imagehub.models.sd3_transformer import SD3Transformer2DModel
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    write_tiny_tokenizer(root / "tokenizer_2", pad_token="!")
    tokenizer_3 = root / "tokenizer_3"
    tokenizer_3.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer_3 / "tokenizer.json")
    tokenizer_3_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (tokenizer_3 / "tokenizer_config.json").write_text(json.dumps(tokenizer_3_config))
    (tokenizer_3 / "special_tokens_map.json").write_text(
        json.dumps(
            {key: tokenizer_3_config[key] for key in ("pad_token", "eos_token", "unk_token")}
        )
    )

    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(TINY_SD3_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusion3Pipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
                "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
                "text_encoder_3": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "CLIPTokenizer"],
                "tokenizer_3": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "SD3Transformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    clip_config = TINY_TEXT_FILE_CONFIG | {
        "architectures": ["CLIPTextModelWithProjection"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "projection_dim": 16,
    }
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModelWithProjection, clip_config, "model.safetensors"),
        ("text_encoder_2", CLIPTextModelWithProjection, clip_config, "model.safetensors"),
        ("text_encoder_3", T5EncoderModel, TINY_SD3_T5, "model.safetensors"),
        (
            "transformer",
            SD3Transformer2DModel,
            TINY_SD3_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_SD3_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name.startswith("text_encoder") and name != "text_encoder_3":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


def write_tiny_flux1(
    directory: str | os.PathLike[str],
    *,
    guidance_embeds: bool = True,
    seed: int = 0,
    in_channels: int = 16,
    out_channels: int = 16,
    latent_channels: int = 4,
    scheduler_config: dict[str, Any] | None = None,
) -> Path:
    """A random-weight FLUX.1 snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel
    from imagehub.models.flux_transformer import FluxTransformer2DModel
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    tokenizer_2 = root / "tokenizer_2"
    tokenizer_2.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer_2 / "tokenizer.json")
    tokenizer_2_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (tokenizer_2 / "tokenizer_config.json").write_text(json.dumps(tokenizer_2_config))
    (tokenizer_2 / "special_tokens_map.json").write_text(
        json.dumps(
            {key: tokenizer_2_config[key] for key in ("pad_token", "eos_token", "unk_token")}
        )
    )
    scheduler = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": guidance_embeds,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 4,
        "max_image_seq_len": 16,
    }
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(scheduler | (scheduler_config or {}))
    )
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    transformer_config = {
        "_class_name": "FluxTransformer2DModel",
        "patch_size": 1,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "guidance_embeds": guidance_embeds,
        "axes_dims_rope": [2, 2, 4],
    }
    vae_config = TINY_SD3_VAE | {
        "latent_channels": latent_channels,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "sample_size": 16,
    }
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        ("text_encoder_2", T5EncoderModel, TINY_SD3_T5, "model.safetensors"),
        (
            "transformer",
            FluxTransformer2DModel,
            transformer_config,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, vae_config, "diffusion_pytorch_model.safetensors"),
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


TINY_HIDREAM_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": 262,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "bos_token_id": 260,
    "eos_token_id": 261,
}
TINY_HIDREAM_TRANSFORMER = {
    "_class_name": "HiDreamImageTransformer2DModel",
    "patch_size": 2,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "caption_channels": [32, 32],
    "text_emb_dim": 32,
    "num_routed_experts": 4,
    "num_activated_experts": 2,
    "axes_dims_rope": [8, 4, 4],
    "max_resolution": [128, 128],
    "llama_layers": [0, 1, 2, 2],
}
TINY_HIDREAM_VAE = TINY_VAE_FILE_CONFIG | {
    "block_out_channels": [16, 16],
    "down_block_types": ["DownEncoderBlock2D"] * 2,
    "up_block_types": ["UpDecoderBlock2D"] * 2,
    "latent_channels": 4,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "sample_size": 32,
}
TINY_LLAMA3_MERGES = (("h", "i"), ("c", "a"), ("ca", "t"), ("\u0120", "c"))
LLAMA3_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def write_tiny_llama3_tokenizer(directory: str | os.PathLike[str]) -> Path:
    """A byte-level BPE tokenizer.json in the Llama 3 layout: 256 bytes, 4 merges, BOS and EOT."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    vocab = {char: index for index, char in enumerate(bytes_to_unicode().values())}
    for left, right in TINY_LLAMA3_MERGES:
        vocab[left + right] = len(vocab)
    assert len(vocab) == 260
    bos, eot = "<|begin_of_text|>", "<|eot_id|>"
    added = []
    for token in (bos, eot):
        added.append(
            {
                "id": len(vocab),
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
        vocab[token] = len(vocab)
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": LLAMA3_SPLIT_PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "Sequence",
            "processors": [
                {
                    "type": "ByteLevel",
                    "add_prefix_space": True,
                    "trim_offsets": False,
                    "use_regex": True,
                },
                {
                    "type": "TemplateProcessing",
                    "single": [
                        {"SpecialToken": {"id": bos, "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                    ],
                    "pair": [
                        {"SpecialToken": {"id": bos, "type_id": 0}},
                        {"Sequence": {"id": "A", "type_id": 0}},
                        {"SpecialToken": {"id": bos, "type_id": 1}},
                        {"Sequence": {"id": "B", "type_id": 1}},
                    ],
                    "special_tokens": {bos: {"id": bos, "ids": [vocab[bos]], "tokens": [bos]}},
                },
            ],
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [list(pair) for pair in TINY_LLAMA3_MERGES],
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "bos_token": bos,
                "eos_token": eot,
                "model_max_length": 131072,
                "tokenizer_class": "PreTrainedTokenizerFast",
                "clean_up_tokenization_spaces": True,
            }
        ),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": bos, "eos_token": eot}), encoding="utf-8"
    )
    return path


def write_tiny_hidream(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight HiDream-I1 snapshot in the published layout, Llama in ``text_encoder_4``."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModelWithProjection
    from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel
    from imagehub.models.llama_text import LlamaTextModel
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer")
    write_tiny_tokenizer(root / "tokenizer_2", pad_token="!")
    tokenizer_3 = root / "tokenizer_3"
    tokenizer_3.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer_3 / "tokenizer.json")
    tokenizer_3_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (tokenizer_3 / "tokenizer_config.json").write_text(json.dumps(tokenizer_3_config))
    (tokenizer_3 / "special_tokens_map.json").write_text(
        json.dumps(
            {key: tokenizer_3_config[key] for key in ("pad_token", "eos_token", "unk_token")}
        )
    )
    write_tiny_llama3_tokenizer(root / "text_encoder_4")

    (root / "scheduler").mkdir()
    shutil.copy2(
        FIXTURES / "hidream_configs" / "scheduler_scheduler_config.json",
        root / "scheduler" / "scheduler_config.json",
    )
    (root / "model_index.json").write_text(
        (FIXTURES / "hidream_configs" / "model_index.json").read_text(encoding="utf-8")
    )
    clip_config = TINY_TEXT_FILE_CONFIG | {
        "architectures": ["CLIPTextModelWithProjection"],
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "projection_dim": 16,
        "max_position_embeddings": 128,
    }
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModelWithProjection, clip_config, "model.safetensors"),
        ("text_encoder_2", CLIPTextModelWithProjection, clip_config, "model.safetensors"),
        ("text_encoder_3", T5EncoderModel, TINY_SD3_T5, "model.safetensors"),
        ("text_encoder_4", LlamaTextModel, TINY_HIDREAM_LLAMA, "model.safetensors"),
        (
            "transformer",
            HiDreamImageTransformer2DModel,
            TINY_HIDREAM_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_HIDREAM_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name.startswith("text_encoder") and name not in ("text_encoder_3", "text_encoder_4"):
            state["text_model.embeddings.position_ids"] = torch.arange(128)[None]
        if name == "text_encoder_4":
            state = {f"model.{key}": value for key, value in state.items()}
            state["lm_head.weight"] = torch.randn(262, 32) * 0.05
        (root / name).mkdir(exist_ok=True)
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


TINY_JANUS_TEXT_CONFIG = {
    "model_type": "llama",
    "vocab_size": 32,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "mlp_bias": False,
    "hidden_act": "silu",
    "pad_token_id": 22,
    "bos_token_id": 20,
    "eos_token_id": 21,
    "use_cache": True,
}
TINY_JANUS_VISION_CONFIG = {
    "model_type": "janus_vision",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 32,
    "patch_size": 16,
    "num_image_tokens": 4,
    "projection_dim": 32,
}
TINY_JANUS_VQ_CONFIG = {
    "model_type": "janus_vqgan",
    "embed_dim": 8,
    "num_embeddings": 16,
    "double_latent": False,
    "latent_channels": 32,
    "in_channels": 3,
    "base_channels": 32,
    "channel_multiplier": [1],
    "num_res_blocks": 1,
    "dropout": 0.0,
    "initializer_range": 0.02,
    "num_patches": 2,
    "out_channels": 3,
    "projection_dim": 32,
    "num_hidden_layers": 2,
    "hidden_act": "gelu",
    "image_token_embed_dim": 32,
}
TINY_JANUS_CONFIG = {
    "model_type": "janus",
    "text_config": TINY_JANUS_TEXT_CONFIG,
    "vision_config": TINY_JANUS_VISION_CONFIG,
    "vq_config": TINY_JANUS_VQ_CONFIG,
    "image_token_id": 26,
}


def write_tiny_janus(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """Write a random tiny Janus checkpoint in the published transformers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.janus import JanusModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = FIXTURES / "deepseek_tokenizer"
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tokenizer / name, root / name)
    tokenizer_data = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_data["model"]["vocab"][":"] = 26
    (root / "tokenizer.json").write_text(json.dumps(tokenizer_data), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(TINY_JANUS_CONFIG), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 20,
                "eos_token_id": 21,
                "pad_token_id": 22,
                "do_sample": True,
                "guidance_scale": 5.0,
                "max_length": 20,
                "num_return_sequences": 1,
                "generation_kwargs": {"boi_token_id": 23},
            }
        ),
        encoding="utf-8",
    )
    torch.manual_seed(seed)
    model = JanusModel(TINY_JANUS_CONFIG)
    for parameter in model.parameters():
        parameter.data.normal_(0, 0.05)
    state = {f"model.{name}": value for name, value in model.state_dict().items()}
    state["lm_head.weight"] = torch.zeros(32, 32)
    state["model.vision_model.placeholder"] = torch.zeros(1)
    state["model.aligner.placeholder"] = torch.zeros(1)
    state["model.vqmodel.encoder.placeholder"] = torch.zeros(1)
    state["model.vqmodel.quant_conv.placeholder"] = torch.zeros(1)
    save_safetensors(state, root / "model.safetensors")
    return root


SANA_FIXTURES = Path(__file__).parent / "fixtures" / "sana_configs"
TINY_GEMMA_MERGES = (
    ("t", "h"),
    ("th", "e"),
    ("▁", "a"),
    ("▁a", "n"),
    ("▁", "th"),
    ("e", "r"),
    ("i", "n"),
    ("▁", "c"),
    ("▁c", "a"),
    ("▁ca", "t"),
)
TINY_GEMMA_CONFIG = {
    "architectures": ["Gemma2Model"],
    "model_type": "gemma2",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attn_logit_softcapping": 50.0,
    "final_logit_softcapping": 30.0,
    "query_pre_attn_scalar": 8,
    "sliding_window": 16,
    "hidden_activation": "gelu_pytorch_tanh",
    "pad_token_id": 0,
    "bos_token_id": 2,
    "eos_token_id": 1,
}
TINY_SANA_TRANSFORMER = {
    "_class_name": "SanaTransformer2DModel",
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "num_layers": 2,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 8,
    "cross_attention_dim": 16,
    "caption_channels": 32,
    "in_channels": 8,
    "out_channels": 8,
    "patch_size": 1,
    "mlp_ratio": 2.5,
    "sample_size": 16,
}
TINY_SANA_VAE = json.loads(
    (SANA_FIXTURES / "1600m" / "vae_config.json").read_text(encoding="utf-8")
) | {
    "latent_channels": 8,
    "attention_head_dim": 8,
    "encoder_block_out_channels": [8, 8, 16, 16, 16, 16],
    "decoder_block_out_channels": [8, 8, 16, 16, 16, 16],
    "encoder_layers_per_block": [1, 1, 1, 1, 1, 1],
    "decoder_layers_per_block": [1, 1, 1, 1, 1, 1],
}


def write_tiny_gemma_tokenizer(directory: str | os.PathLike[str]) -> Path:
    """A small BPE tokenizer with byte fallback in the layout of the published Gemma files."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
    symbols = ["▁", *"abcdefghijklmnopqrstuvwxyz", ",", ".", "-", ":"]
    symbols += [a + b for a, b in TINY_GEMMA_MERGES]
    vocab = {token: index for index, token in enumerate([*specials, *symbols])}
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    added = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in specials
    ]
    bos = {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": None,
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<bos>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<bos>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {"<bos>": bos},
        },
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [list(pair) for pair in TINY_GEMMA_MERGES],
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "model_max_length": 1000000,
        "tokenizer_class": "GemmaTokenizer",
        "use_default_system_prompt": False,
    }
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    special = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    return path


def write_tiny_prx(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight PRX snapshot in the published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.prx_transformer import PRXTransformer2DModel
    from imagehub.models.t5gemma import T5GemmaEncoder

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = write_tiny_gemma_tokenizer(root / "tokenizer")
    vocab_size = len(json.loads((tokenizer / "tokenizer.json").read_text())["model"]["vocab"])
    text_config = {
        "vocab_size": vocab_size,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0,
        "query_pre_attn_scalar": 8,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "hidden_activation": "gelu_pytorch_tanh",
        "dropout_rate": 0.0,
        "pad_token_id": 0,
    }
    transformer_config = {
        "in_channels": 4,
        "patch_size": 2,
        "context_in_dim": 32,
        "hidden_size": 32,
        "mlp_ratio": 2.0,
        "num_heads": 4,
        "depth": 2,
        "axes_dim": [4, 4],
        "theta": 10000,
        "time_factor": 1000.0,
        "time_max_period": 10000,
    }
    vae_config = TINY_VAE_FILE_CONFIG | {
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "force_upcast": True,
        "sample_size": 64,
        "latent_channels": 4,
    }
    for name, cls, config, weights in (
        ("text_encoder", T5GemmaEncoder, text_config, "model.safetensors"),
        (
            "transformer",
            PRXTransformer2DModel,
            transformer_config,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, vae_config, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "num_train_timesteps": 1000,
                "shift": 3.0,
            }
        )
    )
    (root / "model_index.json").write_text(
        json.dumps({"_class_name": "PRXPipeline", "default_sample_size": 64})
    )
    return root


def write_tiny_sana(
    directory: str | os.PathLike[str],
    *,
    seed: int = 0,
    scheduler: dict | None = None,
    sample_size: int = 16,
) -> Path:
    """A random-weight Sana snapshot in the published diffusers layout (diffusers loads it)."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_dc import AutoencoderDC
    from imagehub.models.gemma2 import Gemma2Model
    from imagehub.models.sana_transformer import SanaTransformer2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    vocab_size = len(
        json.loads(
            write_tiny_gemma_tokenizer(root / "tokenizer").joinpath("tokenizer.json").read_text()
        )["model"]["vocab"]
    )
    (root / "scheduler").mkdir()
    scheduler_config = scheduler or json.loads(
        (SANA_FIXTURES / "1600m" / "scheduler_scheduler_config.json").read_text(encoding="utf-8")
    )
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler_config))
    (root / "model_index.json").write_text(
        (SANA_FIXTURES / "1600m" / "model_index.json").read_text(encoding="utf-8")
    )
    for name, cls, config, weights in (
        (
            "text_encoder",
            Gemma2Model,
            TINY_GEMMA_CONFIG | {"vocab_size": vocab_size},
            "model.safetensors",
        ),
        (
            "transformer",
            SanaTransformer2DModel,
            TINY_SANA_TRANSFORMER | {"sample_size": sample_size},
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderDC, TINY_SANA_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.1)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


TINY_SANA_SPRINT_SCHEDULER = {
    "_class_name": "SCMScheduler",
    "num_train_timesteps": 1000,
    "prediction_type": "trigflow",
    "sigma_data": 0.5,
}
TINY_SANA_SPRINT_TRANSFORMER = TINY_SANA_TRANSFORMER | {
    "guidance_embeds": True,
    "qk_norm": "rms_norm_across_heads",
}


def write_tiny_sana_sprint(
    directory: str | os.PathLike[str],
    *,
    seed: int = 0,
    sample_size: int = 16,
) -> Path:
    """A random-weight Sana-Sprint snapshot (SCM scheduler + guidance embeddings)."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_dc import AutoencoderDC
    from imagehub.models.gemma2 import Gemma2Model
    from imagehub.models.sana_transformer import SanaTransformer2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    vocab_size = len(
        json.loads(
            write_tiny_gemma_tokenizer(root / "tokenizer").joinpath("tokenizer.json").read_text()
        )["model"]["vocab"]
    )
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(TINY_SANA_SPRINT_SCHEDULER)
    )
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "SanaSprintPipeline",
                "scheduler": ["diffusers", "SCMScheduler"],
                "text_encoder": ["transformers", "Gemma2Model"],
                "tokenizer": ["transformers", "GemmaTokenizerFast"],
                "transformer": ["diffusers", "SanaTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderDC"],
            }
        )
    )
    for name, cls, config, weights in (
        (
            "text_encoder",
            Gemma2Model,
            TINY_GEMMA_CONFIG | {"vocab_size": vocab_size},
            "model.safetensors",
        ),
        (
            "transformer",
            SanaTransformer2DModel,
            TINY_SANA_SPRINT_TRANSFORMER | {"sample_size": sample_size},
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderDC, TINY_SANA_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.1)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


LUMINA2_FIXTURES = Path(__file__).parent / "fixtures" / "lumina2_configs"
TINY_LUMINA2_TRANSFORMER = {
    "_class_name": "Lumina2Transformer2DModel",
    "in_channels": 4,
    "hidden_size": 32,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "multiple_of": 16,
    "axes_dim_rope": [2, 2, 4],
    "axes_lens": [300, 64, 64],
    "cap_feat_dim": 32,
    "sample_size": 16,
}
TINY_LUMINA2_VAE = {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [8, 8, 16, 16],
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "mid_block_add_attention": True,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "sample_size": 32,
}


def write_tiny_lumina2(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight Lumina-Image 2.0 snapshot in the published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.gemma2 import Gemma2Model
    from imagehub.models.lumina2_transformer import Lumina2Transformer2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    vocab_size = len(
        json.loads(
            write_tiny_gemma_tokenizer(root / "tokenizer").joinpath("tokenizer.json").read_text()
        )["model"]["vocab"]
    )
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        (LUMINA2_FIXTURES / "scheduler_scheduler_config.json").read_text(encoding="utf-8")
    )
    (root / "model_index.json").write_text(
        (LUMINA2_FIXTURES / "model_index.json").read_text(encoding="utf-8")
    )
    for name, cls, config, weights, std in (
        (
            "text_encoder",
            Gemma2Model,
            TINY_GEMMA_CONFIG | {"vocab_size": vocab_size},
            "model.safetensors",
            0.1,
        ),
        (
            "transformer",
            Lumina2Transformer2DModel,
            TINY_LUMINA2_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
            0.05,
        ),
        ("vae", AutoencoderKL, TINY_LUMINA2_VAE, "diffusion_pytorch_model.safetensors", 0.05),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, std)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


AMUSED_FIXTURES = Path(__file__).parent / "fixtures" / "amused_configs"
TINY_AMUSED_TRANSFORMER = {
    "_class_name": "UVit2DModel",
    "hidden_size": 32,
    "cond_embed_dim": 24,
    "micro_cond_encode_dim": 8,
    "micro_cond_embed_dim": 40,
    "encoder_hidden_size": 32,
    "vocab_size": 24,
    "codebook_size": 20,
    "in_channels": 16,
    "block_out_channels": 16,
    "num_res_blocks": 2,
    "downsample": False,
    "upsample": False,
    "block_num_heads": 2,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 48,
    "sample_size": 2,
}
TINY_AMUSED_VQVAE = {
    "_class_name": "VQModel",
    "act_fn": "silu",
    "block_out_channels": [16, 16, 16, 32, 32],
    "down_block_types": ["DownEncoderBlock2D"] * 5,
    "in_channels": 3,
    "latent_channels": 8,
    "layers_per_block": 1,
    "lookup_from_codebook": True,
    "mid_block_add_attention": False,
    "norm_num_groups": 8,
    "norm_type": "group",
    "num_vq_embeddings": 20,
    "out_channels": 3,
    "sample_size": 32,
    "scaling_factor": 0.18215,
    "up_block_types": ["UpDecoderBlock2D"] * 5,
    "vq_embed_dim": None,
    "force_upcast": True,
}


def write_tiny_amused(
    directory: str | os.PathLike[str], *, seed: int = 0, downsample: bool = False
) -> Path:
    """A random-weight aMUSEd snapshot in the published diffusers layout (diffusers loads it).

    With ``downsample`` the transformer halves and re-doubles its token grid, as the 512 model does.
    """
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.clip_text import CLIPTextModelWithProjection
    from imagehub.models.uvit_2d import UVit2DModel
    from imagehub.models.vq_model import VQModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer", pad_token="!")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            json.loads(
                (AMUSED_FIXTURES / "256" / "scheduler_scheduler_config.json").read_text(
                    encoding="utf-8"
                )
            )
            | {"mask_token_id": 23}
        )
    )
    (root / "model_index.json").write_text(
        (AMUSED_FIXTURES / "256" / "model_index.json").read_text(encoding="utf-8")
    )
    text = TINY_TEXT_FILE_CONFIG | {
        "architectures": ["CLIPTextModelWithProjection"],
        "projection_dim": 24,
    }
    transformer = TINY_AMUSED_TRANSFORMER | (
        {"downsample": True, "upsample": True} if downsample else {}
    )
    for name, cls, config, weights, std in (
        ("text_encoder", CLIPTextModelWithProjection, text, "model.safetensors", 0.1),
        ("transformer", UVit2DModel, transformer, "diffusion_pytorch_model.safetensors", 0.4),
        ("vqvae", VQModel, TINY_AMUSED_VQVAE, "diffusion_pytorch_model.safetensors", 0.1),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, std)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


TINY_MEISSONIC_TRANSFORMER = {
    "_class_name": "Transformer2DModel",
    "patch_size": 1,
    "in_channels": 8,
    "num_layers": 1,
    "num_single_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "pooled_projection_dim": 16,
    "guidance_embeds": False,
    "axes_dims_rope": [2, 2, 4],
    "vocab_size": 24,
    "codebook_size": 20,
    "downsample": False,
    "upsample": False,
}


def write_tiny_meissonic(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """Write a random tiny Meissonic snapshot in the published layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.clip_text import CLIPTextModelWithProjection
    from imagehub.models.meissonic_transformer import MeissonicTransformer2DModel
    from imagehub.models.vq_model import VQModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_tokenizer(root / "tokenizer", pad_token="!")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "Scheduler", "mask_token_id": 23, "masking_schedule": "cosine"})
    )
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Pipeline",
                "scheduler": ["scheduler", "Scheduler"],
                "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "transformer": ["transformer", "Transformer2DModel"],
                "vqvae": ["diffusers", "VQModel"],
            }
        )
    )
    text = TINY_TEXT_FILE_CONFIG | {
        "architectures": ["CLIPTextModelWithProjection"],
        "projection_dim": 16,
    }
    for name, cls, config, weights, std in (
        ("text_encoder", CLIPTextModelWithProjection, text, "model.safetensors", 0.1),
        (
            "transformer",
            MeissonicTransformer2DModel,
            TINY_MEISSONIC_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
            0.2,
        ),
        ("vqvae", VQModel, TINY_AMUSED_VQVAE, "diffusion_pytorch_model.safetensors", 0.1),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, std)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


AURAFLOW_FIXTURES = Path(__file__).parent / "fixtures" / "auraflow_configs"
TINY_AURAFLOW_MERGES = (
    ("▁", "a"),
    ("a", "t"),
    ("▁", "c"),
    ("▁c", "at"),
    ("▁", "t"),
    ("▁t", "h"),
    ("▁th", "e"),
    ("i", "n"),
    ("e", "r"),
)
TINY_AURAFLOW_TRANSFORMER = {
    "_class_name": "AuraFlowTransformer2DModel",
    "sample_size": 16,
    "patch_size": 2,
    "in_channels": 4,
    "num_mmdit_layers": 1,
    "num_single_dit_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "caption_projection_dim": 16,
    "out_channels": 4,
    "pos_embed_max_size": 64,
}
TINY_AURAFLOW_TEXT = {
    "architectures": ["UMT5EncoderModel"],
    "model_type": "umt5",
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "dense_act_fn": "gelu_new",
    "is_gated_act": True,
    "is_encoder_decoder": False,
    "use_cache": False,
    "pad_token_id": 1,
    "eos_token_id": 2,
    "decoder_start_token_id": 0,
    "tie_word_embeddings": False,
}


def write_tiny_auraflow_tokenizer(directory: str | os.PathLike[str]) -> Path:
    """A small Llama-style BPE (byte fallback, ``</s>`` appended) in the published layout."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    pieces = ["<unk>", "<s>", "</s>"]
    symbols = ["▁", *"abcdefghijklmnopqrstuvwxyz", ",", ".", "-", ":"]
    symbols += [a + b for a, b in TINY_AURAFLOW_MERGES]
    vocab = {token: index for index, token in enumerate([*pieces, *symbols])}
    for byte in range(256):
        vocab[f"<0x{byte:02X}>"] = len(vocab)
    added = [
        {
            "id": vocab[token],
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for token in pieces
    ]
    eos = {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "first",
            "split": False,
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
            ],
            "pair": [
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "</s>", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "</s>", "type_id": 1}},
            ],
            "special_tokens": {"</s>": eos},
        },
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "first",
            "split": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [list(pair) for pair in TINY_AURAFLOW_MERGES],
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    config = {
        "add_bos_token": False,
        "add_eos_token": True,
        "add_prefix_space": True,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<s>",
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "legacy": False,
        "model_max_length": 512,
        "padding_side": "right",
        "sp_model_kwargs": {},
        "tokenizer_class": "LlamaTokenizer",
        "use_default_system_prompt": False,
    }
    (path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    special = {"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<s>", "unk_token": "<unk>"}
    (path / "special_tokens_map.json").write_text(json.dumps(special), encoding="utf-8")
    return path


def write_tiny_auraflow(
    directory: str | os.PathLike[str], *, seed: int = 0, scheduler: dict | None = None
) -> Path:
    """A random-weight AuraFlow snapshot in the published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.auraflow_transformer import AuraFlowTransformer2DModel
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.umt5 import UMT5EncoderModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    vocab_size = len(
        json.loads(
            write_tiny_auraflow_tokenizer(root / "tokenizer")
            .joinpath("tokenizer.json")
            .read_text(encoding="utf-8")
        )["model"]["vocab"]
    )
    (root / "scheduler").mkdir()
    scheduler_config = scheduler or json.loads(
        (AURAFLOW_FIXTURES / "scheduler_scheduler_config.json").read_text(encoding="utf-8")
    )
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler_config))
    (root / "model_index.json").write_text(
        (AURAFLOW_FIXTURES / "model_index.json").read_text(encoding="utf-8")
    )
    vae_config = TINY_VAE_FILE_CONFIG | {
        "scaling_factor": 0.13025,
        "force_upcast": True,
        "sample_size": 64,
    }
    for name, cls, config, weights in (
        (
            "text_encoder",
            UMT5EncoderModel,
            TINY_AURAFLOW_TEXT | {"vocab_size": vocab_size},
            "model.safetensors",
        ),
        (
            "transformer",
            AuraFlowTransformer2DModel,
            TINY_AURAFLOW_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, vae_config, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.1)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


COSMOS2_FIXTURES = fixture_path("cosmos2_configs")
TINY_COSMOS2_TEXT = {
    "architectures": ["T5EncoderModel"],
    "d_ff": 32,
    "d_kv": 8,
    "d_model": 24,
    "dense_act_fn": "relu",
    "dropout_rate": 0.0,
    "feed_forward_proj": "relu",
    "is_gated_act": False,
    "layer_norm_epsilon": 1e-6,
    "num_heads": 3,
    "num_layers": 2,
    "relative_attention_max_distance": 128,
    "relative_attention_num_buckets": 32,
    "vocab_size": 32128,
}
TINY_COSMOS2_TRANSFORMER = {
    "_class_name": "CosmosTransformer3DModel",
    "adaln_lora_dim": 8,
    "attention_head_dim": 12,
    "concat_padding_mask": True,
    "extra_pos_embed_type": None,
    "in_channels": 16,
    "max_size": [2, 8, 8],
    "mlp_ratio": 2.0,
    "num_attention_heads": 2,
    "num_layers": 2,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "rope_scale": [1.0, 4.0, 4.0],
    "text_embed_dim": 24,
}
TINY_COSMOS2_VAE = {
    "_class_name": "AutoencoderKLWan",
    "attn_scales": [],
    "base_dim": 8,
    "dim_mult": [1, 2, 2, 2],
    "dropout": 0.0,
    "latents_mean": [0.0] * 16,
    "latents_std": [1.0] * 16,
    "num_res_blocks": 1,
    "temperal_downsample": [False, False, False],
    "z_dim": 16,
}


def write_tiny_cosmos2(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight Cosmos-Predict2 snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl_wan import AutoencoderKLWan
    from imagehub.models.cosmos_transformer import CosmosTransformer3DModel
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    shutil.copytree(COSMOS2_FIXTURES / "tokenizer", root / "tokenizer")
    shutil.copytree(COSMOS2_FIXTURES / "scheduler", root / "scheduler")
    (root / "model_index.json").write_text(
        (COSMOS2_FIXTURES / "model_index.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name, cls, config, weights in (
        ("text_encoder", T5EncoderModel, TINY_COSMOS2_TEXT, "model.safetensors"),
        (
            "transformer",
            CosmosTransformer3DModel,
            TINY_COSMOS2_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKLWan, TINY_COSMOS2_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


LUMINA_NEXT_FIXTURES = Path(__file__).parent / "fixtures" / "lumina_next_configs"
TINY_LUMINA_NEXT_TEXT_ENCODER = {
    "architectures": ["GemmaModel"],
    "model_type": "gemma",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "hidden_act": "gelu",
    "hidden_activation": "gelu_pytorch_tanh",
    "pad_token_id": 0,
    "bos_token_id": 2,
    "eos_token_id": 1,
}
TINY_LUMINA_NEXT_TRANSFORMER = {
    "_class_name": "LuminaNextDiT2DModel",
    "in_channels": 4,
    "hidden_size": 32,
    "num_layers": 2,
    "num_attention_heads": 4,
    "num_kv_heads": 2,
    "multiple_of": 16,
    "cross_attention_dim": 32,
    "patch_size": 2,
    "qk_norm": True,
    "learn_sigma": True,
    "sample_size": 16,
}
TINY_LUMINA_NEXT_VAE = {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [8, 8, 16, 16],
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "scaling_factor": 0.13025,
    "shift_factor": None,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "mid_block_add_attention": True,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "sample_size": 32,
}


def write_tiny_lumina_next(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight Lumina-Next snapshot in the published diffusers layout."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.gemma import GemmaModel
    from imagehub.models.lumina_next_transformer import LuminaNextDiT2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = write_tiny_gemma_tokenizer(root / "tokenizer")
    vocab_size = len(json.loads((tokenizer / "tokenizer.json").read_text())["model"]["vocab"])
    # The published tokenizer appends the end token (its post-processor template says so) and
    # pads on the right.
    tokenizer_path = tokenizer / "tokenizer.json"
    data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    processor = data["post_processor"]
    end = {"SpecialToken": {"id": "<eos>", "type_id": 0}}
    processor["single"].append(end)
    processor["special_tokens"]["<eos>"] = {"id": "<eos>", "ids": [1], "tokens": ["<eos>"]}
    del processor["pair"]
    processor["pair"] = [*processor["single"], {"SpecialToken": {"id": "<bos>", "type_id": 1}}]
    processor["pair"] += [{"Sequence": {"id": "B", "type_id": 1}}, end]
    tokenizer_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    config_path = tokenizer / "tokenizer_config.json"
    tokenizer_config = json.loads(config_path.read_text())
    config_path.write_text(
        json.dumps(tokenizer_config | {"add_eos_token": True, "padding_side": "right"})
    )
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        (LUMINA_NEXT_FIXTURES / "scheduler_scheduler_config.json").read_text(encoding="utf-8")
    )
    (root / "model_index.json").write_text(
        (LUMINA_NEXT_FIXTURES / "model_index.json").read_text(encoding="utf-8")
    )
    for name, cls, config, weights, std in (
        (
            "text_encoder",
            GemmaModel,
            TINY_LUMINA_NEXT_TEXT_ENCODER | {"vocab_size": vocab_size},
            "model.safetensors",
            0.1,
        ),
        (
            "transformer",
            LuminaNextDiT2DModel,
            TINY_LUMINA_NEXT_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
            0.05,
        ),
        ("vae", AutoencoderKL, TINY_LUMINA_NEXT_VAE, "diffusion_pytorch_model.safetensors", 0.05),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, std)
        if isinstance(model, LuminaNextDiT2DModel):
            for layer in model.layers:
                layer.gate.data.normal_(0, 1.0)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


TINY_LLAMAGEN_GPT = {
    "dim": 64,
    "n_layer": 2,
    "n_head": 4,
    "multiple_of": 16,
    "caption_dim": 32,
    "vocab_size": 64,
    "cls_token_num": 120,
}
TINY_LLAMAGEN_VQ = {"codebook_size": 64, "codebook_embed_dim": 4, "z_channels": 32}
TINY_LLAMAGEN_T5 = {
    "architectures": ["T5ForConditionalGeneration"],
    "model_type": "t5",
    "vocab_size": 164,
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "is_encoder_decoder": True,
    "tie_word_embeddings": False,
}


def write_tiny_llamagen(
    directory: str | os.PathLike[str], *, seed: int = 0, image_size: int = 64
) -> Path:
    """A random-weight LlamaGen snapshot in the published layout (GPT and VQ pickles, T5 folder).

    A ``config.json`` at the root holds the tiny architecture; the published snapshots have none.
    ``image_size`` is a multiple of 16 and is part of the GPT file name, as for the real files.
    """
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.llamagen import LlamaGenGPT, gpt_xl_config
    from imagehub.models.llamagen_vq import LlamaGenVQ
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    (root / "text_encoder").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    gpt_config = gpt_xl_config((image_size // 16) ** 2) | TINY_LLAMAGEN_GPT
    (root / "config.json").write_text(
        json.dumps({"gpt": TINY_LLAMAGEN_GPT, "vq": TINY_LLAMAGEN_VQ})
    )
    for model in (LlamaGenGPT(gpt_config), LlamaGenVQ(TINY_LLAMAGEN_VQ)):
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.1)
        if isinstance(model, LlamaGenGPT):
            model.cls_embedding.uncond_embedding.normal_(0, 0.5)
            # Peaked token distributions, so that sampling options visibly change the draws.
            model.output.weight.data.normal_(0, 0.6)
            name = f"t2i_XL_stage1_{image_size}.pt"
        else:
            name = "vq_ds16_t2i.pt"
        torch.save({"model": dict(model.state_dict())}, root / name)

    text_dir = root / "text_encoder"
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", text_dir / "tokenizer.json")
    tokenizer_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (text_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (text_dir / "special_tokens_map.json").write_text(
        json.dumps({key: tokenizer_config[key] for key in ("pad_token", "eos_token", "unk_token")})
    )
    text = T5EncoderModel(TINY_LLAMAGEN_T5)
    for parameter in text.parameters():
        parameter.data.normal_(0, 0.3)
    state = dict(text.state_dict())
    # A sequence-to-sequence checkpoint also holds the decoder and the output head.
    state["encoder.embed_tokens.weight"] = state["shared.weight"].clone()
    state["lm_head.weight"] = torch.randn(164, 32)
    state["decoder.embed_tokens.weight"] = state["shared.weight"].clone()
    (text_dir / "config.json").write_text(json.dumps(TINY_LLAMAGEN_T5))
    save_safetensors(state, text_dir / "model.safetensors")
    return root


TINY_INFINITY_TRANSFORMER = {
    "depth": 2,
    "embed_dim": 64,
    "num_heads": 4,
    "mlp_ratio": 4.0,
    "text_channels": 32,
    "text_max_length": 512,
    "bit_channels": 4,
    "block_chunks": 1,
    "level_count": 15,
}
TINY_INFINITY_VAE = {
    "base_channels": 32,
    "channel_multipliers": (1, 1, 1, 1, 1),
    "num_res_blocks": 1,
    "latent_channels": 4,
    "patch_size": 16,
}


def write_tiny_infinity(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """Random Infinity snapshot in the two published pickle containers plus Flan-T5 layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.infinity import InfinityTransformer
    from imagehub.models.infinity_vae import InfinityVAE
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    text_dir = root / "text_encoder"
    text_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    transformer = InfinityTransformer(TINY_INFINITY_TRANSFORMER)
    vae = InfinityVAE(TINY_INFINITY_VAE)
    for model in (transformer, vae):
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.08)
    torch.save(dict(transformer.state_dict()), root / "infinity_2b_reg.pth")
    torch.save({"vae": dict(vae.state_dict())}, root / "infinity_vae_d32reg.pth")
    (root / "infinity_config.json").write_text(
        json.dumps({"transformer": TINY_INFINITY_TRANSFORMER, "vae": TINY_INFINITY_VAE})
    )

    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", text_dir / "tokenizer.json")
    tokenizer_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (text_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (text_dir / "special_tokens_map.json").write_text(
        json.dumps({key: tokenizer_config[key] for key in ("pad_token", "eos_token", "unk_token")})
    )
    text = T5EncoderModel(TINY_LLAMAGEN_T5)
    for parameter in text.parameters():
        parameter.data.normal_(0, 0.08)
    state = dict(text.state_dict())
    state["encoder.embed_tokens.weight"] = state["shared.weight"].clone()
    state["lm_head.weight"] = torch.randn(164, 32)
    state["decoder.embed_tokens.weight"] = state["shared.weight"].clone()
    (text_dir / "config.json").write_text(json.dumps(TINY_LLAMAGEN_T5))
    save_safetensors(state, text_dir / "model.safetensors")
    return root


ERNIE_TINY_PRETOKENIZER_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
ERNIE_TINY_MERGES = (("Ġ", "t"), ("h", "e"), ("c", "a"), ("ca", "t"), ("Ġ", "a"), ("Ġt", "he"))
ERNIE_TINY_SPECIALS = ("<unk>", "<s>", "</s>", "<pad>", "[INST]", "[/INST]")
TINY_ERNIE_TEXT = {
    "model_type": "ministral3",
    "vocab_size": 300,
    "hidden_size": 24,
    "intermediate_size": 40,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "sliding_window": None,
    "pad_token_id": 3,
    "rope_parameters": {
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "factor": 16.0,
        "llama_4_scaling_beta": 0.1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 16,
        "rope_theta": 1000000.0,
        "rope_type": "yarn",
        "type": "yarn",
    },
}
TINY_ERNIE_VISION = {
    "model_type": "pixtral",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "head_dim": 8,
    "image_size": 28,
    "patch_size": 14,
}
TINY_ERNIE_TRANSFORMER = {
    "_class_name": "ErnieImageTransformer2DModel",
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_layers": 2,
    "ffn_hidden_size": 48,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": 1,
    "text_in_dim": 24,
    "rope_theta": 256,
    "rope_axes_dim": [2, 2, 4],
    "eps": 1e-6,
    "qk_layernorm": True,
}
TINY_ERNIE_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 4.0,
    "use_dynamic_shifting": False,
}


def write_tiny_ernie_tokenizer(directory: str | os.PathLike[str]) -> Path:
    """A byte-level BPE tokenizer.json in the Mistral layout with a handful of merges."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    vocab = {token: index for index, token in enumerate(ERNIE_TINY_SPECIALS)}
    for character in bytes_to_unicode().values():
        vocab[character] = len(vocab)
    for left, right in ERNIE_TINY_MERGES:
        vocab[left + right] = len(vocab)
    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": vocab[token],
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token in ERNIE_TINY_SPECIALS
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": ERNIE_TINY_PRETOKENIZER_REGEX},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}},
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": vocab,
            "merges": [list(pair) for pair in ERNIE_TINY_MERGES],
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "backend": "tokenizers",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "model_max_length": 64,
                "tokenizer_class": "TokenizersBackend",
            }
        ),
        encoding="utf-8",
    )
    return path


def write_tiny_ernie_image(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight ERNIE-Image snapshot in the published diffusers layout."""
    import transformers

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl_flux2 import AutoencoderKLFlux2
    from imagehub.models.ernie_image_transformer import ErnieImageTransformer2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    write_tiny_ernie_tokenizer(root / "tokenizer")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(TINY_ERNIE_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "ErnieImagePipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Mistral3Model"],
                "tokenizer": ["transformers", "TokenizersBackend"],
                "transformer": ["diffusers", "ErnieImageTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKLFlux2"],
            }
        )
    )

    text_config = {
        "architectures": ["Mistral3Model"],
        "model_type": "mistral3",
        "image_token_index": 10,
        "text_config": TINY_ERNIE_TEXT,
        "vision_config": TINY_ERNIE_VISION,
    }
    encoder = transformers.Mistral3Model(transformers.Mistral3Config(**text_config))
    for parameter in encoder.parameters():
        parameter.data.normal_(0, 0.1)
    state = {}
    for name, value in encoder.state_dict().items():
        if name.startswith("language_model."):
            name = "language_model.model." + name.removeprefix("language_model.")
        state[name] = value
    (root / "text_encoder").mkdir()
    (root / "text_encoder" / "config.json").write_text(json.dumps(text_config))
    save_safetensors(state, root / "text_encoder" / "model.safetensors")

    for name, cls, config in (
        ("transformer", ErnieImageTransformer2DModel, TINY_ERNIE_TRANSFORMER),
        ("vae", AutoencoderKLFlux2, TINY_FLUX2_VAE),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        if name == "vae":
            model.bn.running_mean.normal_(0, 0.05)
            model.bn.running_var.uniform_(0.5, 1.5)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(
            dict(model.state_dict()), root / name / "diffusion_pytorch_model.safetensors"
        )
    return root


COGVIEW3_PLUS_FIXTURES = FIXTURES / "cogview3_plus_configs"
TINY_COGVIEW3_PLUS_T5 = TINY_SD3_T5 | {"is_encoder_decoder": True}
TINY_COGVIEW3_PLUS_TRANSFORMER = {
    "_class_name": "CogView3PlusTransformer2DModel",
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
}
TINY_COGVIEW3_PLUS_VAE = TINY_VAE_FILE_CONFIG | {
    "latent_channels": 4,
    "scaling_factor": 1.0,
    "shift_factor": None,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "mid_block_add_attention": False,
    "layers_per_block": 3,
    "sample_size": 64,
}


def write_tiny_cogview3_plus(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight CogView3-Plus snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.cogview3_plus_transformer import CogView3PlusTransformer2DModel
    from imagehub.models.t5 import T5EncoderModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer / "tokenizer.json")
    tokenizer_config = {
        "tokenizer_class": "T5Tokenizer",
        "model_max_length": 224,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (root / "scheduler").mkdir()
    shutil.copy2(
        COGVIEW3_PLUS_FIXTURES / "scheduler_scheduler_config.json",
        root / "scheduler" / "scheduler_config.json",
    )
    shutil.copy2(COGVIEW3_PLUS_FIXTURES / "model_index.json", root / "model_index.json")
    for name, cls, config, weights in (
        ("text_encoder", T5EncoderModel, TINY_COGVIEW3_PLUS_T5, "model.safetensors"),
        (
            "transformer",
            CogView3PlusTransformer2DModel,
            TINY_COGVIEW3_PLUS_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_COGVIEW3_PLUS_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(dict(model.state_dict()), root / name / weights)
    return root


OMNIGEN_FIXTURES = Path(__file__).parent / "fixtures" / "omnigen_configs"
OMNIGEN_TOKENIZER = Path(__file__).parent / "fixtures" / "omnigen_tokenizer"
TINY_OMNIGEN_TRANSFORMER = {
    "_class_name": "OmniGenTransformer2DModel",
    "in_channels": 4,
    "patch_size": 2,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "intermediate_size": 48,
    "num_layers": 2,
    "pad_token_id": 32000,
    "vocab_size": 32064,
    "max_position_embeddings": 131072,
    "original_max_position_embeddings": 4096,
    "rope_scaling": {
        "short_factor": [1.05, 1.1, 1.2, 1.4],
        "long_factor": [1.5, 2.5, 4.0, 8.0],
        "type": "su",
    },
    "pos_embed_max_size": 16,
    "time_step_dim": 16,
}
TINY_OMNIGEN_VAE = {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [8, 8, 16, 16],
    "layers_per_block": 1,
    "norm_num_groups": 4,
    "scaling_factor": 0.13025,
    "use_quant_conv": True,
    "use_post_quant_conv": True,
    "mid_block_add_attention": True,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "sample_size": 64,
}


def write_tiny_omnigen(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight OmniGen snapshot in the published diffusers layout (real tokenizer files)."""
    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.omnigen_transformer import OmniGenTransformer2DModel

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    (root / "tokenizer").mkdir()
    for source in OMNIGEN_TOKENIZER.iterdir():
        (root / "tokenizer" / source.name).write_bytes(source.read_bytes())
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        (OMNIGEN_FIXTURES / "scheduler_scheduler_config.json").read_text(encoding="utf-8")
    )
    (root / "model_index.json").write_text(
        (OMNIGEN_FIXTURES / "model_index.json").read_text(encoding="utf-8")
    )
    for name, cls, config, std in (
        ("transformer", OmniGenTransformer2DModel, TINY_OMNIGEN_TRANSFORMER, 0.05),
        ("vae", AutoencoderKL, TINY_OMNIGEN_VAE, 0.05),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, std)
        if name == "transformer":
            model.patch_embedding.pos_embed.normal_(0, 0.5)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(
            dict(model.state_dict()), root / name / "diffusion_pytorch_model.safetensors"
        )
    return root


TINY_KANDINSKY3_TEXT = {
    "vocab_size": 16,
    "d_model": 32,
    "d_kv": 16,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 2,
    "feed_forward_proj": "gated-silu",
    "dense_act_fn": "silu",
    "is_gated_act": True,
    "layer_norm_epsilon": 1e-6,
    "dropout_rate": 0.0,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "model_type": "t5",
    "architectures": ["T5EncoderModel"],
}
TINY_KANDINSKY3_UNET = {
    "_class_name": "Kandinsky3UNet",
    "in_channels": 4,
    "time_embedding_dim": 32,
    "groups": 8,
    "attention_head_dim": 8,
    "layers_per_block": 3,
    "block_out_channels": [32, 64, 64, 128],
    "cross_attention_dim": 32,
    "encoder_hid_dim": 32,
}
TINY_KANDINSKY3_MOVQ = {
    "_class_name": "VQModel",
    "act_fn": "silu",
    "block_out_channels": [32, 32, 32, 32],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "AttnDownEncoderBlock2D",
    ],
    "in_channels": 3,
    "latent_channels": 4,
    "layers_per_block": 1,
    "norm_num_groups": 32,
    "norm_type": "spatial",
    "num_vq_embeddings": 16,
    "out_channels": 3,
    "sample_size": 32,
    "scaling_factor": 0.18215,
    "up_block_types": [
        "AttnUpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ],
    "vq_embed_dim": 4,
}


def write_tiny_kandinsky3(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight Kandinsky 3 snapshot in the published layout (float16 variant file names)."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.kandinsky3_unet import Kandinsky3UNet
    from imagehub.models.t5 import T5EncoderModel
    from imagehub.models.vq_model import VQModel

    source = FIXTURES / "kandinsky3_configs"
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer / "tokenizer.json")
    tokenizer_config = {
        "tokenizer_class": "T5Tokenizer",
        "model_max_length": 2048,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
        "legacy": True,
    }
    (tokenizer / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (tokenizer / "special_tokens_map.json").write_text(
        json.dumps({key: tokenizer_config[key] for key in ("pad_token", "eos_token", "unk_token")})
    )
    (root / "scheduler").mkdir()
    shutil.copy2(source / "scheduler_config.json", root / "scheduler" / "scheduler_config.json")
    shutil.copy2(source / "model_index.json", root / "model_index.json")
    for name, cls, config in (
        ("text_encoder", T5EncoderModel, TINY_KANDINSKY3_TEXT),
        ("unet", Kandinsky3UNet, TINY_KANDINSKY3_UNET),
        ("movq", VQModel, TINY_KANDINSKY3_MOVQ),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = {key: value.half() for key, value in model.state_dict().items()}
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        if name != "text_encoder":
            save_safetensors(state, root / name / "diffusion_pytorch_model.fp16.safetensors")
            continue
        shards = [{key: state[key] for key in list(state)[index::2]} for index in range(2)]
        weight_map = {}
        for index, shard in enumerate(shards, start=1):
            file = f"model.fp16-0000{index}-of-00002.safetensors"
            save_safetensors(shard, root / name / file)
            weight_map |= dict.fromkeys(shard, file)
        (root / name / "model.safetensors.index.fp16.json").write_text(
            json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
        )
    return root


TINY_KANDINSKY5_TEXT = {
    "vocab_size": 152064,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "max_position_embeddings": 128000,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]},
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "tie_word_embeddings": False,
}
TINY_KANDINSKY5_CLIP = {
    "architectures": ["CLIPModel"],
    "model_type": "clip",
    "projection_dim": 32,
    "text_config": {
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "projection_dim": 32,
        "hidden_act": "quick_gelu",
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": 77,
        "vocab_size": 49408,
        "bos_token_id": 0,
        "eos_token_id": 2,
        "pad_token_id": 1,
    },
}
TINY_KANDINSKY5_TRANSFORMER = {
    "_class_name": "Kandinsky5Transformer3DModel",
    "in_visual_dim": 4,
    "out_visual_dim": 4,
    "time_dim": 64,
    "patch_size": [1, 2, 2],
    "model_dim": 128,
    "ff_dim": 256,
    "num_text_blocks": 2,
    "num_visual_blocks": 2,
    "axes_dims": [16, 24, 24],
    "visual_cond": False,
    "in_text_dim": 32,
    "in_text_dim2": 32,
    "attention_type": "regular",
}
TINY_KANDINSKY5_VAE = {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
    "block_out_channels": [32, 32, 64, 64],
    "layers_per_block": 2,
    "norm_num_groups": 8,
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "mid_block_add_attention": True,
    "sample_size": 32,
    "force_upcast": True,
}
TINY_KANDINSKY5_SCHEDULER = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 5.0,
}


def write_tiny_kandinsky5(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
    """A random-weight Kandinsky 5 T2I snapshot in the published diffusers layout."""
    import shutil

    from imagehub.checkpoints import save_safetensors
    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.clip_text import CLIPTextModel
    from imagehub.models.kandinsky5_transformer import Kandinsky5Transformer3DModel
    from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel

    source = FIXTURES / "kandinsky5_configs"
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    shutil.copytree(source / "tokenizer", root / "tokenizer", dirs_exist_ok=True)
    shutil.copytree(source / "tokenizer_2", root / "tokenizer_2", dirs_exist_ok=True)

    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(TINY_KANDINSKY5_SCHEDULER))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Kandinsky5T2IPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
                "tokenizer": ["transformers", "Qwen2VLProcessor"],
                "text_encoder_2": ["transformers", "CLIPTextModel"],
                "tokenizer_2": ["transformers", "CLIPTokenizer"],
                "transformer": ["diffusers", "Kandinsky5Transformer3DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )

    text_config = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "text_config": TINY_KANDINSKY5_TEXT,
        "vision_config": {
            "depth": 1,
            "hidden_act": "silu",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "in_chans": 3,
            "out_hidden_size": 32,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "spatial_patch_size": 2,
            "window_size": 4,
            "fullatt_block_indexes": [],
            "tokens_per_second": 2,
            "temporal_patch_size": 2,
        },
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "vision_token_id": 151654,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "tie_word_embeddings": False,
    }
    qwen = Qwen25VLTextModel(text_config)
    for parameter in qwen.parameters():
        parameter.data.normal_(0, 0.05)
    qwen_state = dict(qwen.state_dict())
    # The optional reference pipeline loads the full conditional-generation model, so when
    # transformers is installed add its tiny vision-tower weights under the published names.
    try:
        import transformers
    except ModuleNotFoundError:
        full_state = qwen_state
    else:
        full_config = transformers.Qwen2_5_VLConfig.from_dict(text_config)
        full_qwen = transformers.Qwen2_5_VLForConditionalGeneration(full_config)
        for parameter in full_qwen.parameters():
            parameter.data.normal_(0, 0.05)
        full_state = dict(full_qwen.state_dict())
        full_state.update(qwen_state)
    (root / "text_encoder").mkdir()
    (root / "text_encoder" / "config.json").write_text(json.dumps(text_config))
    save_safetensors(full_state, root / "text_encoder" / "model.safetensors")

    clip = CLIPTextModel(TINY_KANDINSKY5_CLIP["text_config"])
    for parameter in clip.parameters():
        parameter.data.normal_(0, 0.05)
    clip_state = dict(clip.state_dict())
    clip_state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
    (root / "text_encoder_2").mkdir()
    (root / "text_encoder_2" / "config.json").write_text(json.dumps(TINY_KANDINSKY5_CLIP))
    save_safetensors(clip_state, root / "text_encoder_2" / "model.safetensors")

    for name, cls, config, weights in (
        (
            "transformer",
            Kandinsky5Transformer3DModel,
            TINY_KANDINSKY5_TRANSFORMER,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, TINY_KANDINSKY5_VAE, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(model.state_dict(), root / name / weights)
    return root
