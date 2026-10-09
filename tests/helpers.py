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
    sample_size: int = 32,
) -> Path:
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
        "sample_size": sample_size,
    }
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
    "vocab_size": 260,
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


TINY_KREA2_TEXT = {
    "vocab_size": 260,
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


def write_tiny_krea2(directory: str | os.PathLike[str], *, seed: int = 0) -> Path:
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
                "is_distilled": True,
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
