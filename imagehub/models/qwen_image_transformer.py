"""Native original-release Qwen-Image transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.qwen_image21_transformer import apply_rotary_emb_qwen


class QwenImageRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states * self.weight


class QwenImageTimesteps(nn.Module):
    freqs: Tensor

    def __init__(self, channels: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.register_buffer("freqs", self._make_freqs(), persistent=False)

    def _make_freqs(self) -> Tensor:
        half = self.channels // 2
        exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32)
        return torch.exp(exponent / half)

    def forward(self, timestep: Tensor) -> Tensor:
        frequencies = self.freqs.to(device=timestep.device, dtype=timestep.dtype)
        args = timestep[:, None].float() * frequencies[None]
        args = 1000 * args
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class QwenImageTimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(hidden_states)))


class QwenImageTimestepProjEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.time_proj = QwenImageTimesteps()
        self.timestep_embedder = QwenImageTimestepEmbedding(dim)

    def forward(self, timestep: Tensor, hidden_states: Tensor) -> Tensor:
        projected = self.time_proj(timestep)
        return self.timestep_embedder(projected.to(hidden_states.dtype))


class QwenImageRope(nn.Module):
    pos_freqs: Tensor
    neg_freqs: Tensor

    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.pos_freqs, self.neg_freqs = self._make_freqs()

    @staticmethod
    def _rope_params(index: Tensor, dim: int, theta: int) -> Tensor:
        frequencies = torch.outer(
            index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim))
        )
        return torch.polar(torch.ones_like(frequencies), frequencies)

    def _make_freqs(self) -> tuple[Tensor, Tensor]:
        positive = torch.arange(4096)
        negative = torch.arange(4096).flip(0) * -1 - 1
        return (
            torch.cat(
                [self._rope_params(positive, dim, self.theta) for dim in self.axes_dim], dim=1
            ),
            torch.cat(
                [self._rope_params(negative, dim, self.theta) for dim in self.axes_dim], dim=1
            ),
        )

    def forward(
        self,
        img_shapes: list[list[tuple[int, int, int]]],
        text_length: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        shapes = img_shapes[0]
        pos_freqs = self.pos_freqs.to(device)
        neg_freqs = self.neg_freqs.to(device)
        positive = pos_freqs.split([dim // 2 for dim in self.axes_dim], dim=1)
        negative = neg_freqs.split([dim // 2 for dim in self.axes_dim], dim=1)
        image_frequencies = []
        max_image_index = 0
        for index, (frames, height, width) in enumerate(shapes):
            frame = positive[0][index : index + frames].view(frames, 1, 1, -1)
            frame = frame.expand(frames, height, width, -1)
            vertical = torch.cat(
                [negative[1][-(height - height // 2) :], positive[1][: height // 2]], dim=0
            ).view(1, height, 1, -1)
            horizontal = torch.cat(
                [negative[2][-(width - width // 2) :], positive[2][: width // 2]], dim=0
            ).view(1, 1, width, -1)
            image_frequencies.append(
                torch.cat(
                    [
                        frame,
                        vertical.expand(frames, height, width, -1),
                        horizontal.expand(frames, height, width, -1),
                    ],
                    dim=-1,
                )
                .reshape(frames * height * width, -1)
                .clone()
                .contiguous()
            )
            max_image_index = max(max_image_index, height // 2, width // 2)
        text_frequencies = pos_freqs[max_image_index : max_image_index + text_length]
        return torch.cat(image_frequencies), text_frequencies


class QwenImageGELU(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim * 4)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.gelu(self.proj(hidden_states), approximate="tanh")


class QwenImageFeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([QwenImageGELU(dim), nn.Dropout(0.0), nn.Linear(dim * 4, dim)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
    query, key, value = (item.permute(0, 2, 1, 3) for item in (query, key, value))
    output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    return output.permute(0, 2, 1, 3)


class QwenImageAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = nn.Linear(dim, heads * head_dim)
        self.to_k = nn.Linear(dim, heads * head_dim)
        self.to_v = nn.Linear(dim, heads * head_dim)
        self.add_k_proj = nn.Linear(dim, heads * head_dim)
        self.add_v_proj = nn.Linear(dim, heads * head_dim)
        self.add_q_proj = nn.Linear(dim, heads * head_dim)
        self.to_out = nn.ModuleList([nn.Linear(heads * head_dim, dim), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(heads * head_dim, dim)
        self.norm_q = QwenImageRMSNorm(head_dim, eps)
        self.norm_k = QwenImageRMSNorm(head_dim, eps)
        self.norm_added_q = QwenImageRMSNorm(head_dim, eps)
        self.norm_added_k = QwenImageRMSNorm(head_dim, eps)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        encoder_hidden_states_mask: Tensor | None,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        image_length = hidden_states.shape[1]
        text_length = encoder_hidden_states.shape[1]
        image_q = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, self.head_dim)))
        image_k = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, self.head_dim)))
        image_v = self.to_v(hidden_states).unflatten(-1, (self.heads, self.head_dim))
        text_q = self.norm_added_q(
            self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.head_dim))
        )
        text_k = self.norm_added_k(
            self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.head_dim))
        )
        text_v = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.head_dim))
        image_freqs, text_freqs = rotary_emb
        image_q = apply_rotary_emb_qwen(image_q, image_freqs)
        image_k = apply_rotary_emb_qwen(image_k, image_freqs)
        text_q = apply_rotary_emb_qwen(text_q, text_freqs)
        text_k = apply_rotary_emb_qwen(text_k, text_freqs)
        query = torch.cat([text_q, image_q], dim=1)
        key = torch.cat([text_k, image_k], dim=1)
        value = torch.cat([text_v, image_v], dim=1)
        mask = None
        if encoder_hidden_states_mask is not None:
            image_mask = torch.ones(
                hidden_states.shape[0],
                image_length,
                dtype=torch.bool,
                device=hidden_states.device,
            )
            mask = torch.cat([encoder_hidden_states_mask.bool(), image_mask], dim=1)[:, None, None]
        output = _attention(query, key, value, mask).flatten(2, 3).to(query.dtype)
        text_output = self.to_add_out(output[:, :text_length].contiguous())
        image_output = self.to_out[1](self.to_out[0](output[:, text_length:].contiguous()))
        return image_output, text_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn = QwenImageAttention(dim, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.img_mlp = QwenImageFeedForward(dim)
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.txt_norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.txt_mlp = QwenImageFeedForward(dim)

    @staticmethod
    def _modulate(hidden_states: Tensor, params: Tensor) -> tuple[Tensor, Tensor]:
        shift, scale, gate = params.chunk(3, dim=-1)
        return hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        encoder_hidden_states_mask: Tensor | None,
        temb: Tensor,
        rotary_emb: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        image_mod1, image_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        text_mod1, text_mod2 = self.txt_mod(temb).chunk(2, dim=-1)
        image_input, image_gate1 = self._modulate(self.img_norm1(hidden_states), image_mod1)
        text_input, text_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), text_mod1)
        image_output, text_output = self.attn(
            image_input, text_input, encoder_hidden_states_mask, rotary_emb
        )
        hidden_states = hidden_states + image_gate1 * image_output
        encoder_hidden_states = encoder_hidden_states + text_gate1 * text_output
        image_input, image_gate2 = self._modulate(self.img_norm2(hidden_states), image_mod2)
        text_input, text_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), text_mod2)
        hidden_states = hidden_states + image_gate2 * self.img_mlp(image_input)
        encoder_hidden_states = encoder_hidden_states + text_gate2 * self.txt_mlp(text_input)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class QwenImageAdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        scale, shift = self.linear(self.silu(temb).to(hidden_states.dtype)).chunk(2, dim=1)
        return self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None]


class QwenImageTransformer2DModel(NativeModel):
    # Diffusers keeps safetensor storage at its mapped offsets when accelerate is absent.
    clone_shards = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = {name: value for name, value in config.items() if not name.startswith("_")}
        super().__init__(cfg)
        patch_size = int(cfg.get("patch_size", 2))
        in_channels = int(cfg.get("in_channels", 64))
        out_channels = int(cfg.get("out_channels") or in_channels)
        num_layers = int(cfg.get("num_layers", 60))
        head_dim = int(cfg.get("attention_head_dim", 128))
        heads = int(cfg.get("num_attention_heads", 24))
        text_dim = int(cfg.get("joint_attention_dim", 3584))
        axes_dims = list(cfg.get("axes_dims_rope", (16, 56, 56)))
        self.inner_dim = heads * head_dim
        self.pos_embed = QwenImageRope(10000, axes_dims)
        self.time_text_embed = QwenImageTimestepProjEmbeddings(self.inner_dim)
        self.txt_norm = QwenImageRMSNorm(text_dim)
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(text_dim, self.inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [QwenImageTransformerBlock(self.inner_dim, heads, head_dim) for _ in range(num_layers)]
        )
        self.norm_out = QwenImageAdaLayerNormContinuous(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * out_channels)

    def init_non_persistent_buffers(self) -> None:
        self.time_text_embed.time_proj.freqs = self.time_text_embed.time_proj._make_freqs()
        positive, negative = self.pos_embed._make_freqs()
        self.pos_embed.pos_freqs = positive
        self.pos_embed.neg_freqs = negative

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        encoder_hidden_states_mask: Tensor | None = None,
        guidance: Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool | None = None,
    ) -> Tensor:
        del attention_kwargs, return_dict
        if guidance is not None:
            raise ValueError("The original Qwen-Image transformer has no guidance embedding")
        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))
        timestep = timestep.to(hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)
        rotary_emb = self.pos_embed(
            img_shapes, encoder_hidden_states.shape[1], hidden_states.device
        )
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                rotary_emb,
            )
        return self.proj_out(self.norm_out(hidden_states, temb))
