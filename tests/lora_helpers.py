"""Write random LoRA files for native modules in the published layouts."""

import json
import re

import torch
from torch import nn

from imagehub.checkpoints import save_safetensors

KOHYA_PREFIX = {"unet": "lora_unet_", "text_encoder": "lora_te_", "text_encoder_2": "lora_te2_"}
UNET_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
)
TEXT_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")


def random_lora(components, targets, *, rank=2, seed=0, std=0.1):
    """``{(component, module): (down, up)}`` for Linear/Conv2d layers named by ``targets``."""
    g = torch.Generator().manual_seed(seed)
    result = {}
    for component, model in components.items():
        for name, module in model.named_modules():
            if not any(name == t or name.endswith("." + t) for t in targets):
                continue
            if isinstance(module, nn.Linear):
                down = torch.randn(rank, module.in_features, generator=g) * std
                up = torch.randn(module.out_features, rank, generator=g) * std
            elif isinstance(module, nn.Conv2d):
                kernel = module.kernel_size
                down = torch.randn(rank, module.in_channels, *kernel, generator=g) * std
                up = torch.randn(module.out_channels, rank, 1, 1, generator=g) * std
            else:
                continue
            result[(component, name)] = (down, up)
    return result


def write_peft(path, weights, *, alpha=None):
    tensors = {}
    for (component, module), (down, up) in weights.items():
        tensors[f"{component}.{module}.lora_A.weight"] = down
        tensors[f"{component}.{module}.lora_B.weight"] = up
    metadata = None
    if alpha is not None:
        packed = {}
        for component in sorted({c for c, _ in weights}):
            modules = sorted({m for c, m in weights if c == component})
            packed[f"{component}.r"] = int(weights[(component, modules[0])][0].shape[0])
            packed[f"{component}.lora_alpha"] = alpha
            packed[f"{component}.target_modules"] = modules
        metadata = {
            "format": "pt",
            "lora_adapter_metadata": json.dumps(packed, indent=2, sort_keys=True),
        }
    save_safetensors(tensors, path, metadata=metadata)
    return path


def sgm_name(module, layers_per_block):
    """The original SDXL (SGM) spelling of a diffusers U-Net attention module name."""
    width = layers_per_block + 1
    match = re.fullmatch(r"down_blocks\.(\d+)\.attentions\.(\d+)\.(.*)", module)
    if match:
        index = 1 + int(match.group(1)) * width + int(match.group(2))
        return f"input_blocks.{index}.1.{match.group(3)}"
    match = re.fullmatch(r"mid_block\.attentions\.0\.(.*)", module)
    if match:
        return f"middle_block.1.{match.group(1)}"
    match = re.fullmatch(r"up_blocks\.(\d+)\.attentions\.(\d+)\.(.*)", module)
    if match:
        index = int(match.group(1)) * width + int(match.group(2))
        return f"output_blocks.{index}.1.{match.group(3)}"
    return module


def write_kohya(path, weights, *, alpha=None, layers_per_block=None, text_prefixes=None):
    """With ``layers_per_block`` U-Net names use the SGM spelling (kohya SDXL files)."""
    prefixes = KOHYA_PREFIX | (text_prefixes or {})
    tensors = {}
    for (component, module), (down, up) in weights.items():
        name = module
        if component == "unet" and layers_per_block is not None:
            name = sgm_name(module, layers_per_block)
        key = prefixes[component] + name.replace(".", "_")
        tensors[f"{key}.lora_down.weight"] = down
        tensors[f"{key}.lora_up.weight"] = up
        if alpha is not None:
            tensors[f"{key}.alpha"] = torch.tensor(float(alpha))
    save_safetensors(tensors, path)
    return path


def write_legacy(path, weights):
    tensors = {}
    for (component, module), (down, up) in weights.items():
        parent, _, leaf = module.rpartition(".")
        if component == "unet" and leaf in ("to_q", "to_k", "to_v"):
            stem = f"unet.{parent}.processor.{leaf}_lora"
        elif component == "unet" and module.endswith("to_out.0"):
            stem = f"unet.{module[: -len('.to_out.0')]}.processor.to_out_lora"
        elif component == "unet":
            stem = f"unet.{module}.lora"
        else:
            stem = f"{component}.{module}.lora_linear_layer"
        tensors[f"{stem}.down.weight"] = down
        tensors[f"{stem}.up.weight"] = up
    save_safetensors(tensors, path)
    return path
