"""Native Ideogram 4 transformer backbone."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2

_DEFAULT_CONFIG: dict[str, Any] = {
    "num_layers": 34,
    "num_heads": 18,
    "attention_head_dim": 256,
    "intermediate_size": 12288,
    "adanln_dim": 512,
    "in_channels": 128,
    "llm_features_dim": 53248,
    "rope_theta": 5_000_000,
    "mrope_section": [24, 20, 20],
    "norm_eps": 1e-5,
}


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    inv_freq: Tensor

    def __init__(self, head_dim: int, base: int, mrope_section: tuple[int, ...]) -> None:
        super().__init__()
        self.base = base
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self) -> Tensor:
        return 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, _, _ = position_ids.shape
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(
            3, batch_size, -1, 1
        )
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)

        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: Tensor,
        segment_ids: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        attn_mask = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)
        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        x: Tensor,
        segment_ids: Tensor,
        cos: Tensor,
        sin: Tensor,
        adaln_input: Tensor,
    ) -> Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(t: Tensor, dim: int, scale: float = 1e4) -> Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)  # type: ignore[assignment]
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(NativeModel):
    """Ideogram 4 single-stream flow-matching transformer."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = _DEFAULT_CONFIG | {
            name: value for name, value in config.items() if not name.startswith("_")
        }
        if "adaln_dim" in cfg:
            cfg["adanln_dim"] = cfg.pop("adaln_dim")
        emb_dim = int(cfg.get("emb_dim") or int(cfg["num_heads"]) * int(cfg["attention_head_dim"]))
        cfg["emb_dim"] = emb_dim
        cfg["mrope_section"] = list(cfg["mrope_section"])
        super().__init__(cfg)

        num_heads = int(cfg["num_heads"])
        head_dim = emb_dim // num_heads
        in_channels = int(cfg["in_channels"])
        llm_features_dim = int(cfg["llm_features_dim"])
        adanln_dim = int(cfg["adanln_dim"])
        self.in_channels = in_channels
        self.input_proj = nn.Linear(in_channels, emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(llm_features_dim, emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(emb_dim, adanln_dim, bias=True)
        self.embed_image_indicator = nn.Embedding(2, emb_dim)
        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=int(cfg["rope_theta"]),
            mrope_section=tuple(cfg["mrope_section"]),
        )
        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=emb_dim,
                    intermediate_size=int(cfg["intermediate_size"]),
                    num_heads=num_heads,
                    norm_eps=float(cfg["norm_eps"]),
                    adanln_dim=adanln_dim,
                )
                for _ in range(int(cfg["num_layers"]))
            ]
        )
        self.final_layer = Ideogram4FinalLayer(
            hidden_size=emb_dim,
            out_channels=in_channels,
            adanln_dim=adanln_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def init_non_persistent_buffers(self) -> None:
        self.rotary_emb.inv_freq = self.rotary_emb._make_inv_freq()

    def forward(
        self,
        *,
        llm_features: Tensor,
        x: Tensor,
        t: Tensor,
        position_ids: Tensor,
        segment_ids: Tensor,
        indicator: Tensor,
    ) -> Tensor:
        _, _, in_channels = x.shape
        assert in_channels == self.in_channels
        param_dtype = (
            getattr(self.input_proj, "compute_dtype", None) or self.input_proj.weight.dtype
        )
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)
        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)
        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask
        x = self.input_proj(x) * output_image_mask
        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))
        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask
        h = x + llm_features
        image_indicator_embedding = self.embed_image_indicator(
            (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long)
        )
        h = h + image_indicator_embedding
        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)
        for layer in self.layers:
            h = layer(
                h,
                segment_ids=segment_ids,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
            )
        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
