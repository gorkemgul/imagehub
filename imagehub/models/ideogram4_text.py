"""Ideogram 4 prompt packing and Qwen3-VL feature extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch
from torch import Tensor

from imagehub.models.fp8 import load_fp8_state_dict, swap_linears_to_fp8
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel

SEQUENCE_PADDING_INDICATOR = -1
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
IMAGE_POSITION_OFFSET = 65536
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


class Ideogram4Tokenizer(Protocol):
    def chat_prompt(self, prompt: str, *, enable_thinking: bool = True) -> str: ...

    def tokenize(self, text: str) -> list[int]: ...


def tokenize_prompt(
    tokenizer: Ideogram4Tokenizer, prompt: str, max_text_tokens: int = 2048
) -> tuple[Tensor, int]:
    """Build chat-formatted token ids for a single prompt."""
    text = tokenizer.chat_prompt(prompt)
    token_ids = torch.tensor(tokenizer.tokenize(text), dtype=torch.long)
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > max_text_tokens:
        raise ValueError(
            f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}"
        )
    return token_ids, num_text_tokens


def build_text_inputs(
    tokenizer: Ideogram4Tokenizer,
    prompts: list[str],
    height: int,
    width: int,
    *,
    device: str | torch.device = "cpu",
    patch_size: int = 2,
    ae_scale_factor: int = 8,
    max_text_tokens: int = 2048,
) -> dict[str, Tensor | int]:
    """Build the packed sequence of text and image token slots."""
    tokenized = [tokenize_prompt(tokenizer, prompt, max_text_tokens) for prompt in prompts]
    batch_size = len(prompts)

    patch = patch_size * ae_scale_factor
    if height % patch != 0 or width % patch != 0:
        raise ValueError(f"height/width must be divisible by patch_size*ae_scale_factor={patch}")
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    max_prompt_tokens = max(num_text for _, num_text in tokenized)
    total_seq_len = max_prompt_tokens + num_image_tokens

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
    text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.full(
        (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
    )
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for batch_index, (tokens, num_text) in enumerate(tokenized):
        pad_len = max_prompt_tokens - num_text
        total_unpadded = num_text + num_image_tokens
        offset = pad_len
        token_ids[batch_index, offset : offset + num_text] = tokens

        text_pos = torch.arange(num_text)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        text_position_ids[batch_index, offset : offset + num_text] = text_pos_3d
        position_ids[batch_index, offset : offset + num_text] = text_pos_3d
        position_ids[batch_index, offset + num_text :] = image_pos

        indicator[batch_index, offset : offset + num_text] = LLM_TOKEN_INDICATOR
        indicator[batch_index, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
        segment_ids[batch_index, offset : offset + total_unpadded] = 1

    target = torch.device(device)
    return {
        "token_ids": token_ids.to(target),
        "text_position_ids": text_position_ids.to(target),
        "position_ids": position_ids.to(target),
        "segment_ids": segment_ids.to(target),
        "indicator": indicator.to(target),
        "num_image_tokens": num_image_tokens,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "max_text_tokens": max_prompt_tokens,
    }


def encode_text(
    model: Qwen3VLTextModel,
    token_ids: Tensor,
    text_position_ids: Tensor,
    indicator: Tensor,
    *,
    activation_layers: Sequence[int] = QWEN3_VL_ACTIVATION_LAYERS,
) -> Tensor:
    """Concatenate the selected post-layer Qwen3-VL hidden states."""
    batch_size, seq_len = token_ids.shape
    attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
    pos_2d = text_position_ids[..., 0].contiguous()
    position_ids = pos_2d.unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        hidden_states = model(
            token_ids,
            attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
    selected = [hidden_states[index + 1] for index in activation_layers]
    stacked = torch.stack(selected, dim=0)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)
    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    return (stacked * text_mask).to(torch.float32)


def load_fp8_text_encoder(
    config: Mapping[str, Any],
    state_dict: Mapping[str, Tensor],
    device: str | torch.device,
    dtype: torch.dtype,
) -> Qwen3VLTextModel:
    """Build and load the weight-only FP8 Qwen3-VL language model."""
    text_config = config.get("text_config", config)
    model = Qwen3VLTextModel(text_config)
    prepared = model.prepare_state_dict(dict(state_dict))
    model.to(dtype)
    swap_linears_to_fp8(model, prepared, compute_dtype=dtype)
    load_fp8_state_dict(model, prepared, torch.device(device), dtype, assign=True)
    return model.eval()
