"""LlamaGen text-conditional autoregressive GPT (image tokens from caption embeddings)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

# GPT-XL as published for the text-to-image checkpoints; ``block_size`` is the token count
# (256 for 256x256 images, 1024 for 512x512 images).
LLAMAGEN_GPT_XL: dict[str, Any] = {
    "dim": 1280,
    "n_layer": 36,
    "n_head": 20,
    "n_kv_head": None,
    "multiple_of": 256,
    "ffn_dim_multiplier": None,
    "rope_base": 10000,
    "norm_eps": 1e-5,
    "caption_dim": 2048,
    "vocab_size": 16384,
    "cls_token_num": 120,
    "block_size": 256,
}


def gpt_xl_config(block_size: int) -> dict[str, Any]:
    return LLAMAGEN_GPT_XL | {"block_size": block_size}


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: float = 10000, cls_token_num: int = 120
) -> Tensor:
    """Rotation table ``(cls_token_num + grid_size**2, n_elem // 2, 2)``; caption rows are zero."""
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)
    cache = cache_grid.flatten(0, 1)
    return torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class LlamaGenMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LlamaGenCaptionEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, token_num: int) -> None:
        super().__init__()
        self.cap_proj = LlamaGenMLP(in_channels, hidden_size, hidden_size)
        # The learned "no caption" embedding used for the unconditional half of guidance.
        self.register_buffer("uncond_embedding", torch.zeros(token_num, in_channels))

    def forward(self, caption: Tensor) -> Tensor:
        return self.cap_proj(caption)


class LlamaGenRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaGenFeedForward(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden_dim = 4 * config["dim"]
        hidden_dim = int(2 * hidden_dim / 3)
        if config["ffn_dim_multiplier"] is not None:
            hidden_dim = int(config["ffn_dim_multiplier"] * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config["multiple_of"])
        self.w1 = nn.Linear(config["dim"], hidden_dim, bias=False)
        self.w3 = nn.Linear(config["dim"], hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config["dim"], bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaGenAttention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        assert config["dim"] % config["n_head"] == 0
        self.dim = config["dim"]
        self.head_dim = config["dim"] // config["n_head"]
        self.n_head = config["n_head"]
        self.n_kv_head = config["n_kv_head"] if config["n_kv_head"] is not None else self.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.wqkv = nn.Linear(config["dim"], total_kv_dim, bias=False)
        self.wo = nn.Linear(config["dim"], config["dim"], bias=False)
        self.k_cache: Tensor | None = None
        self.v_cache: Tensor | None = None

    def setup_cache(
        self, batch_size: int, max_seq_length: int, dtype: torch.dtype, device: torch.device | str
    ) -> None:
        shape = (batch_size, self.n_head, max_seq_length, self.head_dim)
        self.k_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache = torch.zeros(shape, dtype=dtype, device=device)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        input_pos: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        xq, xk, xv = (value.transpose(1, 2) for value in (xq, xk, xv))
        if mask is not None:
            if self.k_cache is None or self.v_cache is None:
                raise RuntimeError("Call setup_caches before decoding with a cache")
            assert input_pos is not None
            self.k_cache[:, :, input_pos] = xk
            self.v_cache[:, :, input_pos] = xv
            keys, values = self.k_cache, self.v_cache
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        output = F.scaled_dot_product_attention(
            xq, keys, values, attn_mask=mask, is_causal=mask is None
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.wo(output)


class LlamaGenTransformerBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.attention = LlamaGenAttention(config)
        self.feed_forward = LlamaGenFeedForward(config)
        self.attention_norm = LlamaGenRMSNorm(config["dim"], config["norm_eps"])
        self.ffn_norm = LlamaGenRMSNorm(config["dim"], config["norm_eps"])

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        input_pos: Tensor | None,
        mask: Tensor | None,
    ) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, input_pos, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class LlamaGenGPT(NativeModel):
    """Text-to-image GPT; tensor names are the published ones.

    Inference runs in two cached modes, both through ``forward``: prefill (``cond_embed`` with the
    caption embeddings, ``input_pos = arange(cls_token_num)``) and one-token decode (``idx`` of
    shape ``(batch, 1)``, ``input_pos`` of shape ``(1,)``). ``setup_caches`` must be called first.
    With both ``idx`` and ``cond_embed`` the full sequence is run without a cache.
    """

    freqs_cis: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(LLAMAGEN_GPT_XL | dict(config))
        cfg = self.config
        self.cls_token_num: int = cfg["cls_token_num"]
        self.block_size: int = cfg["block_size"]
        self.cls_embedding = LlamaGenCaptionEmbedder(
            cfg["caption_dim"], cfg["dim"], self.cls_token_num
        )
        self.tok_embeddings = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.layers = nn.ModuleList(LlamaGenTransformerBlock(cfg) for _ in range(cfg["n_layer"]))
        self.norm = LlamaGenRMSNorm(cfg["dim"], cfg["norm_eps"])
        self.output = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size
        self.register_buffer("freqs_cis", self._freqs_cis(), persistent=False)
        self.causal_mask: Tensor | None = None
        self.max_seq_length = -1

    def _freqs_cis(self) -> Tensor:
        cfg = self.config
        return precompute_freqs_cis_2d(
            int(self.block_size**0.5),
            cfg["dim"] // cfg["n_head"],
            cfg["rope_base"],
            self.cls_token_num,
        )

    def init_non_persistent_buffers(self) -> None:
        self.freqs_cis = self._freqs_cis()

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Drop attention-cache buffers should a checkpoint have been saved after sampling.

        The released checkpoints are plain weight dictionaries; the rotation table and the caches
        are not stored.
        """
        return {
            name: tensor
            for name, tensor in state.items()
            if not name.endswith((".k_cache", ".v_cache")) and name != "freqs_cis"
        }

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        device = self.freqs_cis.device if device is None else device
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        for layer in self.layers:
            cast(LlamaGenTransformerBlock, layer).attention.setup_cache(
                max_batch_size, max_seq_length, dtype, device
            )
        causal_mask = torch.tril(
            torch.ones(max_seq_length, max_seq_length, dtype=torch.bool, device=device)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(max_batch_size, 1, 1)

    def apply_caption_mask(self, emb_masks: Tensor) -> None:
        """Hide padded caption positions; ``emb_masks`` is ``(cache batch, cls_token_num)``."""
        assert self.causal_mask is not None
        caption_length = emb_masks.shape[-1]
        self.causal_mask[:, :, :caption_length] = self.causal_mask[
            :, :, :caption_length
        ] * emb_masks.unsqueeze(1)
        eye_matrix = torch.eye(
            self.causal_mask.size(1), self.causal_mask.size(2), device=self.causal_mask.device
        )
        self.causal_mask[:] = self.causal_mask * (1 - eye_matrix) + eye_matrix

    def forward(
        self,
        idx: Tensor | None,
        cond_embed: Tensor | None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        mask: Tensor | None = None
        if idx is not None and cond_embed is not None:
            cond_embeddings = self.cls_embedding(cond_embed)[:, : self.cls_token_num]
            h = torch.cat((cond_embeddings, self.tok_embeddings(idx)), dim=1)
            freqs_cis = self.freqs_cis[: h.shape[1]]
        else:
            if cond_embed is not None:
                h = self.cls_embedding(cond_embed)[:, : self.cls_token_num]
            else:
                assert idx is not None
                h = self.tok_embeddings(idx)
            if self.causal_mask is None or input_pos is None:
                raise RuntimeError("Call setup_caches and pass input_pos for cached decoding")
            mask = self.causal_mask[: h.shape[0], None, input_pos]
            freqs_cis = self.freqs_cis[input_pos]
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        h = self.norm(h)
        return self.output(h).float()
