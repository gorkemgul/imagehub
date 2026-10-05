"""Native Z-Image single-stream transformer, limited to text-to-image inputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from imagehub.checkpoints import NativeModel

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

Z_IMAGE_DEFAULTS: dict[str, Any] = {
    "all_patch_size": [2],
    "all_f_patch_size": [1],
    "in_channels": 16,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner_layers": 2,
    "n_heads": 30,
    "n_kv_heads": 30,
    "norm_eps": 1e-5,
    "qk_norm": True,
    "cap_feat_dim": 2560,
    "siglip_feat_dim": None,
    "rope_theta": 256.0,
    "t_scale": 1000.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1024, 512, 512],
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps) * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(
        self, out_size: int, mid_size: int = 1024, frequency_embedding_size: int = 256
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return (
            torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            if dim % 2
            else embedding
        )

    def forward(self, t: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = cast(torch.dtype, self.mlp[0].weight.dtype)
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float) -> None:
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim // heads, eps)
        self.norm_k = RMSNorm(dim // heads, eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])

    @staticmethod
    def _rotary(x_in: Tensor, freqs_cis: Tensor) -> Tensor:
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        x_out = torch.view_as_real(x * freqs_cis.unsqueeze(2)).flatten(3)
        return x_out.type_as(x_in)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None, freqs_cis: Tensor
    ) -> Tensor:
        query = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))
        query, key = self._rotary(query, freqs_cis), self._rotary(key, freqs_cis)
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        hidden_states = (
            F.scaled_dot_product_attention(
                query.permute(0, 2, 1, 3),
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            .permute(0, 2, 1, 3)
            .flatten(2, 3)
        )
        return self.to_out[0](hidden_states)


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden_dim = int(dim / 3 * 8)
        self.w1, self.w2, self.w3 = (
            nn.Linear(dim, hidden_dim, bias=False),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Linear(dim, hidden_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, norm_eps: float, modulation: bool = True) -> None:
        super().__init__()
        self.attention = Attention(dim, heads, 1e-5)
        self.feed_forward = FeedForward(dim)
        self.attention_norm1, self.ffn_norm1 = RMSNorm(dim, norm_eps), RMSNorm(dim, norm_eps)
        self.attention_norm2, self.ffn_norm2 = RMSNorm(dim, norm_eps), RMSNorm(dim, norm_eps)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
            )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        freqs_cis: Tensor,
        adaln_input: Tensor | None = None,
    ) -> Tensor:
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask, freqs_cis)
            x = x + gate_msa * self.attention_norm2(attn_out)
            return x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        attn_out = self.attention(self.attention_norm1(x), attn_mask, freqs_cis)
        x = x + self.attention_norm2(attn_out)
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        return self.linear(self.norm_final(x) * (1.0 + self.adaLN_modulation(c)).unsqueeze(1))


class RopeEmbedder:
    def __init__(self, theta: float, axes_dims: list[int], axes_lens: list[int]) -> None:
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis: list[Tensor] | None = None

    def __call__(self, ids: Tensor) -> Tensor:
        if self.freqs_cis is None or self.freqs_cis[0].device != ids.device:
            self.freqs_cis = []
            for dim, end in zip(self.axes_dims, self.axes_lens, strict=True):
                freqs = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
                freqs = torch.outer(torch.arange(end, dtype=torch.float64), freqs).float()
                self.freqs_cis.append(
                    torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64).to(ids.device)
                )
        assert self.freqs_cis is not None
        return torch.cat(
            [freqs[ids[:, index]] for index, freqs in enumerate(self.freqs_cis)], dim=-1
        )


class ZImageTransformer2DModel(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        cfg = Z_IMAGE_DEFAULTS | dict(config)
        unsupported = (
            cfg["in_channels"] != 16
            and cfg["in_channels"] != 8
            or not cfg["qk_norm"]
            or cfg["siglip_feat_dim"] is not None
        )
        if unsupported:
            raise ValueError("Unsupported Z-Image transformer option")
        if len(cfg["all_patch_size"]) != len(cfg["all_f_patch_size"]) or cfg["dim"] // cfg[
            "n_heads"
        ] != sum(cfg["axes_dims"]):
            raise ValueError("Unsupported Z-Image transformer configuration")
        super().__init__(cfg)
        self.in_channels, self.out_channels = cfg["in_channels"], cfg["in_channels"]
        self.all_patch_size, self.all_f_patch_size, self.t_scale = (
            cfg["all_patch_size"],
            cfg["all_f_patch_size"],
            cfg["t_scale"],
        )
        self.all_x_embedder = nn.ModuleDict(
            {
                f"{p}-{f}": nn.Linear(f * p * p * self.in_channels, cfg["dim"], bias=True)
                for p, f in zip(cfg["all_patch_size"], cfg["all_f_patch_size"], strict=True)
            }
        )
        self.all_final_layer = nn.ModuleDict(
            {
                f"{p}-{f}": FinalLayer(cfg["dim"], p * p * f * self.out_channels)
                for p, f in zip(cfg["all_patch_size"], cfg["all_f_patch_size"], strict=True)
            }
        )

        def make_block(modulation: bool = True) -> ZImageTransformerBlock:
            return ZImageTransformerBlock(cfg["dim"], cfg["n_heads"], cfg["norm_eps"], modulation)

        self.noise_refiner = nn.ModuleList(make_block() for _ in range(cfg["n_refiner_layers"]))
        self.context_refiner = nn.ModuleList(
            make_block(False) for _ in range(cfg["n_refiner_layers"])
        )
        self.t_embedder = TimestepEmbedder(min(cfg["dim"], ADALN_EMBED_DIM))
        self.cap_embedder = nn.Sequential(
            RMSNorm(cfg["cap_feat_dim"], cfg["norm_eps"]),
            nn.Linear(cfg["cap_feat_dim"], cfg["dim"], bias=True),
        )
        self.x_pad_token, self.cap_pad_token = (
            nn.Parameter(torch.zeros((1, cfg["dim"]))),
            nn.Parameter(torch.zeros((1, cfg["dim"]))),
        )
        self.layers = nn.ModuleList(make_block() for _ in range(cfg["n_layers"]))
        self.rope_embedder = RopeEmbedder(cfg["rope_theta"], cfg["axes_dims"], cfg["axes_lens"])

    @staticmethod
    def _grid(
        size: tuple[int, int, int], start: tuple[int, int, int], device: torch.device
    ) -> Tensor:
        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size, strict=True)
        ]
        return torch.stack(torch.meshgrid(axes, indexing="ij"), dim=-1)

    def _pad(
        self, feat: Tensor, grid_size: tuple[int, int, int], start: tuple[int, int, int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        length, pad = len(feat), (-len(feat)) % SEQ_MULTI_OF
        position = self._grid(grid_size, start, feat.device).flatten(0, 2)
        if not pad:
            return feat, position, torch.zeros(length, dtype=torch.bool, device=feat.device)
        position = torch.cat(
            [position, self._grid((1, 1, 1), (0, 0, 0), feat.device).flatten(0, 2).repeat(pad, 1)]
        )
        return (
            torch.cat([feat, feat[-1:].repeat(pad, 1)]),
            position,
            torch.cat(
                [
                    torch.zeros(length, dtype=torch.bool, device=feat.device),
                    torch.ones(pad, dtype=torch.bool, device=feat.device),
                ]
            ),
        )

    def _prepare(
        self, feats: list[Tensor], positions: list[Tensor], masks: list[Tensor], pad_token: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | None, list[int]]:
        lengths = [len(feat) for feat in feats]
        values_tensor = torch.cat(feats)
        values_tensor = torch.where(torch.cat(masks).unsqueeze(-1), pad_token, values_tensor)
        value_items = list(values_tensor.split(lengths))
        freq_items = list(
            self.rope_embedder(torch.cat(positions)).split(
                [len(position) for position in positions]
            )
        )
        values = pad_sequence(value_items, batch_first=True, padding_value=0.0)
        freqs = pad_sequence(freq_items, batch_first=True, padding_value=0.0)
        max_length = max(lengths)
        mask = (
            None
            if all(length == max_length for length in lengths)
            else torch.tensor(
                [[index < length for index in range(max_length)] for length in lengths],
                dtype=torch.bool,
                device=values.device,
            )
        )
        return values, freqs[:, : values.shape[1]], mask, lengths

    def forward(
        self, x: list[Tensor], t: Tensor, cap_feats: list[Tensor], **options: Any
    ) -> list[Tensor]:
        if any(value is not None for value in options.values()):
            raise ValueError("Only the text-to-image path is supported")
        patch_size, f_patch_size = 2, 1
        if patch_size not in self.all_patch_size or f_patch_size not in self.all_f_patch_size:
            raise ValueError("Unsupported Z-Image transformer patch size")
        adaln_input = self.t_embedder(t * self.t_scale).type_as(x[0])
        (
            images,
            captions,
            image_sizes,
            image_positions,
            caption_positions,
            image_masks,
            caption_masks,
        ) = [], [], [], [], [], [], []
        for image, caption in zip(x, cap_feats, strict=True):
            cap, cap_pos, cap_mask = self._pad(
                caption, (len(caption) + (-len(caption)) % SEQ_MULTI_OF, 1, 1), (1, 0, 0)
            )
            channels, frames, height, width = image.shape
            patches = (
                image.view(channels, frames, 1, height // 2, 2, width // 2, 2)
                .permute(1, 3, 5, 2, 4, 6, 0)
                .reshape(-1, 4 * channels)
            )
            img, img_pos, img_mask = self._pad(
                patches, (frames, height // 2, width // 2), (len(cap) + 1, 0, 0)
            )
            images.append(img)
            captions.append(cap)
            image_sizes.append((frames, height, width))
            image_positions.append(img_pos)
            caption_positions.append(cap_pos)
            image_masks.append(img_mask)
            caption_masks.append(cap_mask)
        image_lengths = [len(item) for item in images]
        image_tokens = self.all_x_embedder["2-1"](torch.cat(images))
        image_tokens, image_freqs, image_mask, image_lengths = self._prepare(
            list(image_tokens.split(image_lengths)), image_positions, image_masks, self.x_pad_token
        )
        for layer in self.noise_refiner:
            image_tokens = layer(image_tokens, image_mask, image_freqs, adaln_input)
        caption_lengths = [len(item) for item in captions]
        caption_tokens = self.cap_embedder(torch.cat(captions))
        caption_tokens, caption_freqs, caption_mask, caption_lengths = self._prepare(
            list(caption_tokens.split(caption_lengths)),
            caption_positions,
            caption_masks,
            self.cap_pad_token,
        )
        for layer in self.context_refiner:
            caption_tokens = layer(caption_tokens, caption_mask, caption_freqs)
        unified_items = [
            torch.cat(
                [image_tokens[i, : image_lengths[i]], caption_tokens[i, : caption_lengths[i]]]
            )
            for i in range(len(x))
        ]
        frequency_items = [
            torch.cat([image_freqs[i, : image_lengths[i]], caption_freqs[i, : caption_lengths[i]]])
            for i in range(len(x))
        ]
        lengths = [len(item) for item in unified_items]
        unified = pad_sequence(unified_items, batch_first=True, padding_value=0.0)
        frequencies = pad_sequence(frequency_items, batch_first=True, padding_value=0.0)
        max_length = max(lengths)
        unified_mask = (
            None
            if all(length == max_length for length in lengths)
            else torch.tensor(
                [[index < length for index in range(max_length)] for length in lengths],
                dtype=torch.bool,
                device=unified.device,
            )
        )
        for layer in self.layers:
            unified = layer(unified, unified_mask, frequencies, adaln_input)
        unified = self.all_final_layer["2-1"](unified, adaln_input)
        result = []
        for item, (frames, height, width) in zip(unified.unbind(), image_sizes, strict=True):
            length = frames * (height // 2) * (width // 2)
            result.append(
                item[:length]
                .view(frames, height // 2, width // 2, 1, 2, 2, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, frames, height, width)
            )
        return result
