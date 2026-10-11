"""Native HiDream-I1 sparse mixture-of-experts diffusion transformer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.flux_transformer import TimestepEmbedding
from imagehub.models.layers import timestep_embedding

HIDREAM_TRANSFORMER_DEFAULTS: dict[str, Any] = {
    "in_channels": 64,
    "out_channels": None,
    "num_layers": 16,
    "num_single_layers": 32,
    "attention_head_dim": 128,
    "num_attention_heads": 20,
    "text_emb_dim": 2048,
    "num_routed_experts": 4,
    "num_activated_experts": 2,
    "max_resolution": (128, 128),
    "force_inference_output": False,
}
_REQUIRED = ("patch_size", "caption_channels", "llama_layers", "axes_dims_rope")


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PooledEmbed(nn.Module):
    def __init__(self, text_emb_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(text_emb_dim, hidden_size)

    def forward(self, pooled_embed: Tensor) -> Tensor:
        return self.pooled_embedder(pooled_embed)


class TimestepEmbed(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, hidden_size)

    def forward(self, timesteps: Tensor, dtype: torch.dtype) -> Tensor:
        projected = timestep_embedding(timesteps, 256, flip_sin_to_cos=True, shift=0)
        return self.timestep_embedder(projected.to(dtype=dtype))


class OutEmbed(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, hidden_states: Tensor, temb: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(
            1
        )
        return self.linear(hidden_states)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels * patch_size * patch_size, out_channels, bias=True)

    def forward(self, latent: Tensor) -> Tensor:
        return self.proj(latent)


def _rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    dtype = torch.float32 if pos.device.type == "mps" else torch.float64
    scale = torch.arange(0, dim, 2, dtype=dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    batch_size, _ = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    stacked = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    return stacked.view(batch_size, -1, dim // 2, 2, 2).float()


class RopeEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
            dim=-3,
        )
        return emb.unsqueeze(2)


def _apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class HiDreamAttention(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, *, single: bool) -> None:
        super().__init__()
        inner_dim = heads * head_dim
        self.heads = heads
        self.single = single
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(inner_dim, inner_dim)
        self.to_v = nn.Linear(inner_dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.q_rms_norm = nn.RMSNorm(inner_dim, 1e-5)
        self.k_rms_norm = nn.RMSNorm(inner_dim, 1e-5)
        if not single:
            self.to_q_t = nn.Linear(dim, inner_dim)
            self.to_k_t = nn.Linear(inner_dim, inner_dim)
            self.to_v_t = nn.Linear(inner_dim, inner_dim)
            self.to_out_t = nn.Linear(inner_dim, dim)
            self.q_rms_norm_t = nn.RMSNorm(inner_dim, 1e-5)
            self.k_rms_norm_t = nn.RMSNorm(inner_dim, 1e-5)

    def forward(
        self,
        hidden_states: Tensor,
        hidden_states_masks: Tensor | None,
        encoder_hidden_states: Tensor | None,
        image_rotary_emb: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]
        query_i = self.q_rms_norm(self.to_q(hidden_states)).to(dtype=dtype)
        key_i = self.k_rms_norm(self.to_k(hidden_states)).to(dtype=dtype)
        value_i = self.to_v(hidden_states)
        head_dim = key_i.shape[-1] // self.heads
        query_i = query_i.view(batch_size, -1, self.heads, head_dim)
        key_i = key_i.view(batch_size, -1, self.heads, head_dim)
        value_i = value_i.view(batch_size, -1, self.heads, head_dim)
        if hidden_states_masks is not None:
            key_i = key_i * hidden_states_masks.view(batch_size, -1, 1, 1)

        if not self.single:
            assert encoder_hidden_states is not None
            query_t = self.q_rms_norm_t(self.to_q_t(encoder_hidden_states)).to(dtype=dtype)
            key_t = self.k_rms_norm_t(self.to_k_t(encoder_hidden_states)).to(dtype=dtype)
            value_t = self.to_v_t(encoder_hidden_states)
            query_t = query_t.view(batch_size, -1, self.heads, head_dim)
            key_t = key_t.view(batch_size, -1, self.heads, head_dim)
            value_t = value_t.view(batch_size, -1, self.heads, head_dim)
            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query, key, value = query_i, key_i, value_i

        query, key = _apply_rope(query, key, image_rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if self.single:
            return self.to_out(hidden_states)
        hidden_states_i, hidden_states_t = torch.split(
            hidden_states, [num_image_tokens, num_text_tokens], dim=1
        )
        return self.to_out(hidden_states_i), self.to_out_t(hidden_states_t)


class MoEGate(nn.Module):
    def __init__(self, embed_dim: int, num_routed_experts: int, num_activated_experts: int) -> None:
        super().__init__()
        self.top_k = num_activated_experts
        self.weight = nn.Parameter(torch.empty(num_routed_experts, embed_dim))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        return topk_idx, topk_weight


class MoEFeedForward(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, num_routed_experts: int, num_activated_experts: int
    ) -> None:
        super().__init__()
        self.shared_experts = SwiGLUFeedForward(dim, hidden_dim // 2)
        self.experts = nn.ModuleList(
            SwiGLUFeedForward(dim, hidden_dim) for _ in range(num_routed_experts)
        )
        self.gate = MoEGate(dim, num_routed_experts, num_activated_experts)
        self.num_activated_experts = num_activated_experts

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight = self.gate(x)
        x = x.view(-1, x.shape[-1])
        y = self.moe_infer(x, topk_idx.view(-1), topk_weight.view(-1, 1)).view(*orig_shape)
        return y + self.shared_experts(identity)

    @torch.no_grad()
    def moe_infer(
        self, x: Tensor, flat_expert_indices: Tensor, flat_expert_weights: Tensor
    ) -> Tensor:
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cumsum(0).tolist()
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_out = self.experts[i](x[exp_token_idx])
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum"
            )
        return expert_cache


class TextProjection(nn.Module):
    def __init__(self, in_features: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_size, bias=False)

    def forward(self, caption: Tensor) -> Tensor:
        return self.linear(caption)


def _image_feed_forward(dim: int, routed: int, activated: int) -> nn.Module:
    if routed > 0:
        return MoEFeedForward(dim, 4 * dim, routed, activated)
    return SwiGLUFeedForward(dim, 4 * dim)


class SingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, routed: int, activated: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(dim, heads, head_dim, single=True)
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_i = _image_feed_forward(dim, routed, activated)

    def forward(
        self,
        hidden_states: Tensor,
        hidden_states_masks: Tensor | None,
        temb: Tensor,
        image_rotary_emb: Tensor,
    ) -> Tensor:
        wtype = hidden_states.dtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            temb
        )[:, None].chunk(6, dim=-1)
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        attn_output = self.attn1(norm_hidden_states, hidden_states_masks, None, image_rotary_emb)
        hidden_states = gate_msa * attn_output + hidden_states
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        ff_output = gate_mlp * self.ff_i(norm_hidden_states.to(dtype=wtype))
        return ff_output + hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, routed: int, activated: int) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim))
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.norm1_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(dim, heads, head_dim, single=False)
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_i = _image_feed_forward(dim, routed, activated)
        self.norm3_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_t = SwiGLUFeedForward(dim, 4 * dim)

    def forward(
        self,
        hidden_states: Tensor,
        hidden_states_masks: Tensor | None,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wtype = hidden_states.dtype
        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp_t,
            scale_mlp_t,
            gate_mlp_t,
        ) = self.adaLN_modulation(temb)[:, None].chunk(12, dim=-1)

        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        norm_encoder = self.norm1_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder = norm_encoder * (1 + scale_msa_t) + shift_msa_t
        attn_output_i, attn_output_t = self.attn1(
            norm_hidden_states, hidden_states_masks, norm_encoder, image_rotary_emb
        )
        hidden_states = gate_msa_i * attn_output_i + hidden_states
        encoder_hidden_states = gate_msa_t * attn_output_t + encoder_hidden_states

        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        norm_encoder = self.norm3_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder = norm_encoder * (1 + scale_mlp_t) + shift_mlp_t
        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states)
        ff_output_t = gate_mlp_t * self.ff_t(norm_encoder)
        return ff_output_i + hidden_states, ff_output_t + encoder_hidden_states


class BlockWrapper(nn.Module):
    """Holds a block under ``.block`` to match the published checkpoint names."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.block(*args, **kwargs)


class HiDreamImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        user = {name: value for name, value in config.items() if not name.startswith("_")}
        unknown = sorted(set(user) - set(HIDREAM_TRANSFORMER_DEFAULTS) - set(_REQUIRED))
        if unknown:
            raise ValueError(f"Unsupported HiDream transformer setting {unknown[0]}")
        missing = [name for name in _REQUIRED if name not in user]
        if missing:
            raise ValueError(f"Missing HiDream transformer setting {missing[0]}")
        cfg = HIDREAM_TRANSFORMER_DEFAULTS | user
        patch_size = cfg["patch_size"]
        heads, head_dim = cfg["num_attention_heads"], cfg["attention_head_dim"]
        layers, single_layers = cfg["num_layers"], cfg["num_single_layers"]
        routed, activated = cfg["num_routed_experts"], cfg["num_activated_experts"]
        axes = [int(value) for value in cfg["axes_dims_rope"]]
        if (
            type(patch_size) is not int
            or patch_size < 1
            or layers < 1
            or single_layers < 0
            or len(cfg["caption_channels"]) != 2
            or len(cfg["max_resolution"]) != 2
            or sum(axes) != head_dim
            or any(value < 2 or value % 2 for value in axes)
            or routed < 0
            or (routed > 0 and not 0 < activated <= routed)
            or len(cfg["llama_layers"]) < layers + single_layers
            or cfg["out_channels"] not in (None, cfg["in_channels"])
        ):
            raise ValueError("Unsupported HiDream transformer configuration")
        super().__init__(cfg)
        self.patch_size = patch_size
        self.llama_layers = [int(index) for index in cfg["llama_layers"]]
        self.out_channels = cfg["in_channels"]
        self.inner_dim = heads * head_dim
        self.max_seq = cfg["max_resolution"][0] * cfg["max_resolution"][1] // (patch_size**2)

        self.t_embedder = TimestepEmbed(self.inner_dim)
        self.p_embedder = PooledEmbed(cfg["text_emb_dim"], self.inner_dim)
        self.x_embedder = PatchEmbed(patch_size, cfg["in_channels"], self.inner_dim)
        self.pe_embedder = RopeEmbed(10000, axes)
        self.double_stream_blocks = nn.ModuleList(
            BlockWrapper(TransformerBlock(self.inner_dim, heads, head_dim, routed, activated))
            for _ in range(layers)
        )
        self.single_stream_blocks = nn.ModuleList(
            BlockWrapper(SingleTransformerBlock(self.inner_dim, heads, head_dim, routed, activated))
            for _ in range(single_layers)
        )
        self.final_layer = OutEmbed(self.inner_dim, patch_size, self.out_channels)
        channels = [cfg["caption_channels"][1]] * (layers + single_layers) + [
            cfg["caption_channels"][0]
        ]
        self.caption_projection = nn.ModuleList(
            TextProjection(channel, self.inner_dim) for channel in channels
        )

    def unpatchify(self, x: Tensor, img_sizes: Tensor | list[tuple[int, int]]) -> Tensor:
        p = self.patch_size
        outputs = []
        for i, img_size in enumerate(img_sizes):
            patch_height, patch_width = (int(value) for value in img_size)
            t = x[i, : patch_height * patch_width].reshape(1, patch_height, patch_width, -1)
            channels = t.shape[-1] // (p * p)
            t = t.reshape(1, patch_height, patch_width, p, p, channels)
            t = t.permute(0, 5, 1, 3, 2, 4)
            outputs.append(t.reshape(1, channels, patch_height * p, patch_width * p))
        return torch.cat(outputs, dim=0)

    def patchify(self, hidden_states: Tensor) -> tuple[Tensor, Tensor | None, Tensor, Tensor]:
        batch_size, channels, height, width = hidden_states.shape
        p = self.patch_size
        patch_height, patch_width = height // p, width // p
        device = hidden_states.device
        dtype = hidden_states.dtype
        img_sizes = torch.tensor([patch_height, patch_width], dtype=torch.int64, device=device)
        img_sizes = img_sizes.reshape(-1).unsqueeze(0).repeat(batch_size, 1)
        square = height == width
        if patch_height * patch_width > self.max_seq and not square:
            raise ValueError("The latent grid exceeds max_resolution")

        masks: Tensor | None = None
        if not square:
            masks = torch.zeros((batch_size, self.max_seq), dtype=dtype, device=device)
            masks[:, : patch_height * patch_width] = 1.0

        img_ids = torch.zeros(patch_height, patch_width, 3, device=device)
        row_indices = torch.arange(patch_height, device=device)[:, None]
        col_indices = torch.arange(patch_width, device=device)[None, :]
        img_ids[..., 1] = img_ids[..., 1] + row_indices
        img_ids[..., 2] = img_ids[..., 2] + col_indices
        img_ids = img_ids.reshape(patch_height * patch_width, -1)
        if not square:
            img_ids_pad = torch.zeros(self.max_seq, 3, device=device)
            img_ids_pad[: patch_height * patch_width, :] = img_ids
            img_ids = img_ids_pad.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            img_ids = img_ids.unsqueeze(0).repeat(batch_size, 1, 1)

        if not square:
            out = torch.zeros(
                (batch_size, channels, self.max_seq, p * p), dtype=dtype, device=device
            )
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, p, patch_width, p
            )
            hidden_states = hidden_states.permute(0, 1, 2, 4, 3, 5)
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height * patch_width, p * p
            )
            out[:, :, 0 : patch_height * patch_width] = hidden_states
            hidden_states = out.permute(0, 2, 3, 1).reshape(
                batch_size, self.max_seq, p * p * channels
            )
        else:
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, p, patch_width, p
            )
            hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1)
            hidden_states = hidden_states.reshape(
                batch_size, patch_height * patch_width, p * p * channels
            )
        return hidden_states, masks, img_sizes, img_ids

    def forward(
        self,
        hidden_states: Tensor,
        timesteps: Tensor,
        encoder_hidden_states_t5: Tensor,
        encoder_hidden_states_llama3: Tensor,
        pooled_embeds: Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool | None = None,
    ) -> Tensor:
        del attention_kwargs, return_dict
        if hidden_states.ndim != 4:
            raise ValueError("hidden_states must have shape (batch, channels, height, width)")
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype
        hidden_states, hidden_states_masks, img_sizes, img_ids = self.patchify(hidden_states)
        hidden_states = self.x_embedder(hidden_states)

        temb = self.t_embedder(timesteps, hidden_states_type) + self.p_embedder(pooled_embeds)

        if max(self.llama_layers) >= len(encoder_hidden_states_llama3):
            raise ValueError("encoder_hidden_states_llama3 has fewer layers than llama_layers uses")
        encoder_states = [encoder_hidden_states_llama3[k] for k in self.llama_layers]
        projected = []
        for i, state in enumerate(encoder_states):
            state = self.caption_projection[i](state)
            projected.append(state.view(batch_size, -1, hidden_states.shape[-1]))
        encoder_states = projected
        t5 = self.caption_projection[-1](encoder_hidden_states_t5)
        encoder_states.append(t5.view(batch_size, -1, hidden_states.shape[-1]))

        txt_ids = torch.zeros(
            batch_size,
            encoder_states[-1].shape[1] + encoder_states[-2].shape[1] + encoder_states[0].shape[1],
            3,
            device=img_ids.device,
            dtype=img_ids.dtype,
        )
        image_rotary_emb = self.pe_embedder(torch.cat((img_ids, txt_ids), dim=1))

        block_id = 0
        initial_encoder_hidden_states = torch.cat(
            [
                encoder_states[-1].to(hidden_states.device),
                encoder_states[-2].to(hidden_states.device),
            ],
            dim=1,
        )
        initial_seq_len = initial_encoder_hidden_states.shape[1]
        for block in self.double_stream_blocks:
            current_llama = encoder_states[block_id].to(hidden_states.device)
            current_encoder = torch.cat([initial_encoder_hidden_states, current_llama], dim=1)
            hidden_states, initial_encoder_hidden_states = block(
                hidden_states, hidden_states_masks, current_encoder, temb, image_rotary_emb
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if hidden_states_masks is not None:
            ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + current_llama.shape[1]),
                device=hidden_states_masks.device,
                dtype=hidden_states_masks.dtype,
            )
            hidden_states_masks = torch.cat([hidden_states_masks, ones], dim=1)

        for block in self.single_stream_blocks:
            current_llama = encoder_states[block_id].to(hidden_states.device)
            hidden_states = torch.cat([hidden_states, current_llama], dim=1)
            hidden_states = block(hidden_states, hidden_states_masks, temb, image_rotary_emb)
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, temb)
        return self.unpatchify(output, img_sizes)
