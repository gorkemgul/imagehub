"""A tiny random GLM-Image snapshot in the published layout."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from imagehub.checkpoints import save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.glm_image_ar import GlmImageARModel
from imagehub.models.glm_image_transformer import GlmImageTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization import bytes_to_unicode

FIXTURES = Path(__file__).parent / "fixtures" / "glm_image_configs"
SCHEDULER = json.loads((FIXTURES / "scheduler_scheduler_config.json").read_text(encoding="utf-8"))
VISION_VOCAB = 80
AR_CONFIG = {
    "image_start_token_id": 260,
    "image_end_token_id": 259,
    "image_token_id": 261,
    "text_config": {
        "vocab_size": 300,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "pad_token_id": 259,
        "eos_token_id": 259,
        "vision_vocab_size": VISION_VOCAB,
        "rope_parameters": {
            "rope_theta": 10000,
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
            "partial_rotary_factor": 0.5,
        },
    },
}
GENERATION = {"do_sample": True, "temperature": 0.9, "top_k": VISION_VOCAB, "top_p": 0.75}
T5_CONFIG = {
    "vocab_size": 384,
    "d_model": 16,
    "d_kv": 4,
    "d_ff": 32,
    "num_layers": 2,
    "num_heads": 2,
}
TRANSFORMER_CONFIG = {
    "patch_size": 2,
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 2,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "text_embed_dim": 16,
    "time_embed_dim": 16,
    "condition_dim": 4,
    "prior_vq_quantizer_codebook_size": VISION_VOCAB,
}
VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": [8, 8, 8, 8],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "in_channels": 3,
    "latent_channels": 4,
    "latents_mean": [0.1, -0.2, 0.3, 0.0],
    "latents_std": [1.5, 2.0, 1.0, 2.5],
    "layers_per_block": 1,
    "mid_block_add_attention": False,
    "norm_num_groups": 4,
    "out_channels": 3,
    "sample_size": 64,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "scaling_factor": 0.18215,
    "shift_factor": None,
}
PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {
                "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
                "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            },
            "behavior": "Isolated",
            "invert": False,
        },
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}
PROCESSOR_TOKENS = {"[gMASK]": 256, "<sop>": 257, "<eop>": 258, "<end>": 259, "<begin>": 260}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _component(root: Path, name: str, state: dict, config: dict, weights: str) -> None:
    _write_json(root / name / "config.json", config)
    save_safetensors(state, root / name / weights)


def write_tiny_glm_image(root: Path, states: dict[str, dict] | None = None) -> Path:
    """Write the snapshot; ``states`` may hold ``ar``, ``text_encoder``, ``transformer``, ``vae``."""
    torch.manual_seed(0)
    states = states or {}
    chars = list(bytes_to_unicode().values())
    tokenizer = {
        "model": {"type": "BPE", "vocab": {c: i for i, c in enumerate(chars)}, "merges": []},
        "normalizer": None,
        "pre_tokenizer": PRE_TOKENIZER,
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "added_tokens": [
            {
                "id": index,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for content, index in PROCESSOR_TOKENS.items()
        ],
    }
    _write_json(root / "processor" / "tokenizer.json", tokenizer)
    _write_json(
        root / "processor" / "tokenizer_config.json",
        {
            "pad_token": "<end>",
            "bos_token": "<begin>",
            "grid_bos_token": "<sop>",
            "grid_eos_token": "<eop>",
            "added_tokens_decoder": {"261": {"content": "<image>"}},
        },
    )
    _write_json(
        root / "tokenizer" / "tokenizer_config.json",
        {
            "added_tokens_decoder": {
                str(index): {"content": content}
                for index, content in enumerate(("<pad>", "</s>", "<unk>"))
            },
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        },
    )
    _write_json(root / "scheduler" / "scheduler_config.json", SCHEDULER)
    _write_json(root / "model_index.json", {"_class_name": "GlmImagePipeline"})
    _component(
        root,
        "vision_language_encoder",
        states.get("ar") or GlmImageARModel(AR_CONFIG).state_dict(),
        AR_CONFIG,
        "model.safetensors",
    )
    _write_json(root / "vision_language_encoder" / "generation_config.json", GENERATION)
    _component(
        root,
        "text_encoder",
        states.get("text_encoder") or T5EncoderModel(T5_CONFIG).state_dict(),
        T5_CONFIG,
        "model.safetensors",
    )
    _component(
        root,
        "transformer",
        states.get("transformer") or GlmImageTransformer2DModel(TRANSFORMER_CONFIG).state_dict(),
        TRANSFORMER_CONFIG,
        "diffusion_pytorch_model.safetensors",
    )
    _component(
        root,
        "vae",
        states.get("vae") or AutoencoderKL(VAE_CONFIG).state_dict(),
        VAE_CONFIG,
        "diffusion_pytorch_model.safetensors",
    )
    return root
