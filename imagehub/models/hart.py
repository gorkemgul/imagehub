"""Native inference modules for the HART hybrid autoregressive transformer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel

HART_DEFAULTS: dict[str, Any] = {
    "vocab_size": 4096,
    "latent_channels": 32,
    "embed_dim": 1536,
    "num_heads": 24,
    "depth": 24,
    "mlp_ratio": 4.0,
    "norm_eps": 1e-6,
    "context_dim": 1536,
    "context_token": 300,
    "patch_nums": (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64),
    "diff_width": 1024,
    "diff_depth": 6,
}


class HARTRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-6, *, scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size) * scale)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        y = x.float()
        return self.weight * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(
            dtype
        )


class HARTTimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        batched = timesteps.ndim == 2
        if batched:
            timesteps = timesteps[0]
        half = 128
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device).float() / half
        )
        args = timesteps[:, None].float() * frequencies[None]
        result = self.mlp(
            torch.cat((args.cos(), args.sin()), dim=-1).to(
                self.mlp[0].weight.dtype  # type: ignore[arg-type]
            )
        )
        return result.unsqueeze(0) if batched else result


def _rotate_half(x: Tensor) -> Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _position_ids(
    batch: int, patch_nums: Sequence[int], device: torch.device, stage: int, selected: Any = None
) -> Tensor:
    stages = patch_nums if stage < 0 else patch_nums[stage : stage + 1]
    largest = patch_nums[-1]
    grids = []
    for size in stages:
        axis = torch.arange(size, device=device)
        grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)
        grids.append(grid.reshape(-1, 2) / size * largest)
    positions = torch.cat(grids).unsqueeze(0).repeat(batch, 1, 1)
    if selected is not None:
        positions = positions[selected].reshape(batch, -1, 2)
    return positions


class HARTAttention(nn.Module):
    inv_freq_1d: Tensor
    inv_freq_2d: Tensor

    def __init__(self, config: Mapping[str, Any], block_index: int) -> None:
        super().__init__()
        hidden, heads = config["embed_dim"], config["num_heads"]
        self.block_index = block_index
        self.heads = heads
        self.head_dim = hidden // heads
        self.patch_nums = tuple(config["patch_nums"])
        self.context_token = config["context_token"]
        self.scale = 0.25 / math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(hidden, hidden * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(hidden))
        self.v_bias = nn.Parameter(torch.zeros(hidden))
        self.register_buffer("zero_k_bias", torch.zeros(hidden))
        self.proj = nn.Linear(hidden, hidden)
        self.register_buffer("inv_freq_1d", self._freqs(2), persistent=False)
        self.register_buffer("inv_freq_2d", self._freqs(4), persistent=False)
        self.cached_k: Tensor | None = None
        self.cached_v: Tensor | None = None
        self.caching = False

    def _freqs(self, stride: int) -> Tensor:
        return 1.0 / (10000 ** (torch.arange(0, self.head_dim, stride).float() / self.head_dim))

    def kv_caching(self, enabled: bool) -> None:
        self.caching, self.cached_k, self.cached_v = enabled, None, None

    def _rope(self, x: Tensor, positions: Tensor, *, two_dimensional: bool) -> Tensor:
        inv = self.inv_freq_2d if two_dimensional else self.inv_freq_1d
        if two_dimensional:
            freqs = positions[..., None] * inv
            emb = torch.cat((freqs, freqs), dim=-1).flatten(-2)
        else:
            freqs = positions[..., None] * inv
            emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.to(x.dtype).unsqueeze(1)
        return x * emb.cos() + _rotate_half(x) * emb.sin()

    def forward(
        self,
        x: Tensor,
        *,
        stage: int,
        context_position_ids: Tensor,
        selected: Any = None,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        batch, length, channels = x.shape
        qkv = F.linear(
            x,
            self.qkv_proj.weight,
            torch.cat(
                (self.q_bias, self.zero_k_bias, self.v_bias)  # type: ignore[arg-type]
            ),
        ).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if stage == 0:
            q = self._rope(q, context_position_ids, two_dimensional=False)
            k = self._rope(k, context_position_ids, two_dimensional=False)
        elif stage > 0:
            positions = (
                _position_ids(batch, self.patch_nums[1:], x.device, stage - 1, selected=selected)
                + context_position_ids[:, -1, None, None]
            )
            q = self._rope(q, positions, two_dimensional=True)
            k = self._rope(k, positions, two_dimensional=True)
        else:
            positions = _position_ids(batch, self.patch_nums[1:], x.device, -1, selected=selected)
            positions = positions + context_position_ids[:, -1, None, None]
            visual = positions.shape[1]
            q_context, q_visual = q[..., :-visual, :], q[..., -visual:, :]
            k_context, k_visual = k[..., :-visual, :], k[..., -visual:, :]
            q = torch.cat(
                (
                    self._rope(q_context, context_position_ids, two_dimensional=False),
                    self._rope(q_visual, positions, two_dimensional=True),
                ),
                dim=2,
            )
            k = torch.cat(
                (
                    self._rope(k_context, context_position_ids, two_dimensional=False),
                    self._rope(k_visual, positions, two_dimensional=True),
                ),
                dim=2,
            )
        if self.caching:
            if self.cached_k is not None:
                k = torch.cat((self.cached_k, k), dim=2)
                v = torch.cat((self.cached_v, v), dim=2)  # type: ignore[arg-type]
            self.cached_k, self.cached_v = k, v
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias, scale=self.scale)
        return self.proj(output.transpose(1, 2).contiguous().view(batch, length, channels))


class HARTMLP(nn.Module):
    def __init__(self, hidden: int, ratio: float) -> None:
        super().__init__()
        intermediate = int(hidden * ratio * 2 / 3 + 255) // 256 * 256
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HARTBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any], index: int) -> None:
        super().__init__()
        self.attn = HARTAttention(config, index)
        self.ffn = HARTMLP(config["embed_dim"], config["mlp_ratio"])
        self.ln_wo_grad = HARTRMSNorm(config["embed_dim"], config["norm_eps"])

    def forward(
        self,
        x: Tensor,
        *,
        stage: int,
        context_position_ids: Tensor,
        selected: Any = None,
    ) -> Tensor:
        x = x + self.attn(
            self.ln_wo_grad(x),
            stage=stage,
            context_position_ids=context_position_ids,
            selected=selected,
        )
        return x + self.ffn(self.ln_wo_grad(x))


class HARTDiffusionTimestep(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t: Tensor) -> Tensor:
        half = 128
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs
        return self.mlp(
            torch.cat((args.cos(), args.sin()), dim=-1).to(
                self.mlp[0].weight.dtype  # type: ignore[arg-type]
            )
        )


class HARTDiffusionResBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, -1)
        value = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(value)


class HARTDiffusionFinal(nn.Module):
    def __init__(self, width: int, output: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(width, output)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, -1)
        return self.linear(self.norm_final(x) * (1 + scale) + shift)


class HARTDiffusionMLP(nn.Module):
    def __init__(self, latent: int, condition: int, width: int, depth: int) -> None:
        super().__init__()
        self.in_channels = latent
        self.time_embed = HARTDiffusionTimestep(width)
        self.cond_embed = nn.Linear(condition, width)
        self.input_proj = nn.Linear(latent, width)
        self.res_blocks = nn.ModuleList(HARTDiffusionResBlock(width) for _ in range(depth))
        self.final_layer = HARTDiffusionFinal(width, latent * 2)

    def forward(self, x: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        x = self.input_proj(x)
        cond = self.time_embed(timestep) + self.cond_embed(condition)
        for block in self.res_blocks:
            x = block(x, cond)
        return self.final_layer(x, cond)


class _DiffLoss(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.net = HARTDiffusionMLP(
            config["latent_channels"],
            config["embed_dim"],
            config["diff_width"],
            config["diff_depth"],
        )


class HARTHeadNorm(nn.Module):
    def __init__(self, hidden: int, eps: float) -> None:
        super().__init__()
        self.ln_wo_grad = HARTRMSNorm(hidden, eps)
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden * 2))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        condition = condition.max(1, keepdim=True).values
        scale, shift = self.ada_lin(condition).view(-1, 1, 2, x.shape[-1]).unbind(2)
        return self.ln_wo_grad(x) * (1 + scale) + shift


class HARTTransformer(NativeModel):
    """Published HART transformer and residual diffusion head."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(HART_DEFAULTS | dict(config))
        cfg = self.config
        hidden = cfg["embed_dim"]
        self.context_embed = nn.Linear(cfg["context_dim"], hidden)
        self.context_norm = HARTRMSNorm(
            cfg["context_dim"], cfg["norm_eps"], scale=cfg.get("context_norm_scale", 1.0)
        )
        self.word_embed = nn.Linear(cfg["latent_channels"], hidden)
        self.lvl_embed = HARTTimestepEmbedder(hidden)
        self.shared_ada_lin = nn.Identity()
        self.blocks = nn.ModuleList(HARTBlock(cfg, index) for index in range(cfg["depth"]))
        self.head_nm = HARTHeadNorm(hidden, cfg["norm_eps"])
        self.head = nn.Linear(hidden, cfg["vocab_size"])
        self.decoder_norm = HARTRMSNorm(hidden, cfg["norm_eps"])
        self.diffloss = _DiffLoss(cfg)
        levels = torch.cat(
            [torch.zeros(cfg["context_token"], dtype=torch.long)]
            + [
                torch.full((size * size,), index + 1)
                for index, size in enumerate(cfg["patch_nums"][1:])
            ]
        ).unsqueeze(0)
        self.register_buffer("lvl_1L", levels)
        stage = levels.unsqueeze(-1)
        bias = torch.where(stage >= stage.transpose(1, 2), 0.0, -torch.inf).unsqueeze(1)
        self.register_buffer("attn_bias_for_masking", bias)

    def init_non_persistent_buffers(self) -> None:
        for block in self.blocks:
            block.attn.inv_freq_1d = block.attn._freqs(2)  # type: ignore[union-attr,operator]
            block.attn.inv_freq_2d = block.attn._freqs(4)  # type: ignore[union-attr,operator]

    def get_logits(self, x: Tensor, condition: Tensor) -> Tensor:
        return self.head(self.head_nm(x.float(), condition).float()).float()

    def stage_forward(
        self,
        x: Tensor,
        condition: Tensor,
        stage: int,
        context_position_ids: Tensor,
        selected: Any = None,
    ) -> tuple[Tensor, Tensor]:
        for block in self.blocks:
            x = block(
                x,
                stage=stage,
                context_position_ids=context_position_ids,
                selected=selected,
            )
        return x, self.get_logits(x, condition)


class HARTPhi(nn.Conv2d):
    def __init__(self, channels: int, ratio: float = 0.5) -> None:
        super().__init__(channels, channels, 3, padding=1)
        self.ratio = abs(ratio)

    def forward(self, x: Tensor) -> Tensor:
        return x * (1 - self.ratio) + super().forward(x) * self.ratio


class HARTQuantizer(nn.Module):
    def __init__(self, vocab_size: int, channels: int) -> None:
        super().__init__()
        self.quant_resi = nn.Module()
        self.quant_resi.qresi_ls = nn.ModuleList(HARTPhi(channels) for _ in range(4))
        self.register_buffer("ema_vocab_hit_SV", torch.zeros(14, vocab_size))
        self.embedding = nn.Embedding(vocab_size, channels)

    def residual(self, stage: int, stages: int, x: Tensor) -> Tensor:
        ticks = (1 / 12, 13 / 36, 23 / 36, 11 / 12)
        ratio = stage / (stages - 1)
        index = min(range(4), key=lambda item: abs(ticks[item] - ratio))
        return self.quant_resi.qresi_ls[index](x)  # type: ignore[index,operator]

    def next_input(
        self, stage: int, patch_nums: Sequence[int], f_hat: Tensor, tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        largest = patch_nums[-1]
        value = F.interpolate(tokens, size=(largest, largest), mode="bicubic")
        f_hat = f_hat + self.residual(stage, len(patch_nums), value)
        next_size = patch_nums[stage + 1]
        return f_hat, F.interpolate(f_hat, size=(next_size, next_size), mode="area")


class HARTResnetBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(32, incoming, eps=1e-6)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, outgoing, eps=1e-6)
        self.dropout = nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        value = self.conv1(F.silu(self.norm1(x), inplace=True))
        value = self.conv2(self.dropout(F.silu(self.norm2(value), inplace=True)))
        return self.nin_shortcut(x) + value


class HARTVAEAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(batch, 3, self.channels, height * width).unbind(1)
        weights = torch.bmm(q.transpose(1, 2), k) * (self.channels**-0.5)
        value = torch.bmm(v, weights.softmax(dim=2).transpose(1, 2))
        return x + self.proj_out(value.reshape(batch, self.channels, height, width))


class HARTUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class HARTDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        channels, latent = config["ch"], config["z_channels"]
        multipliers = tuple(config["ch_mult"])
        block_channels = channels * multipliers[-1]
        self.conv_in = nn.Conv2d(latent, block_channels, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = HARTResnetBlock(block_channels, block_channels, config["dropout"])
        self.mid.attn_1 = HARTVAEAttention(block_channels)
        self.mid.block_2 = HARTResnetBlock(block_channels, block_channels, config["dropout"])
        self.up = nn.ModuleList()
        for level in reversed(range(len(multipliers))):
            outgoing = channels * multipliers[level]
            up = nn.Module()
            up.block = nn.ModuleList()
            up.attn = nn.ModuleList()
            for _ in range(3):
                up.block.append(HARTResnetBlock(block_channels, outgoing, config["dropout"]))
                block_channels = outgoing
                if level == len(multipliers) - 1:
                    up.attn.append(HARTVAEAttention(block_channels))
            if level != 0:
                up.upsample = HARTUpsample(block_channels)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(32, block_channels, eps=1e-6)
        self.conv_out = nn.Conv2d(block_channels, 3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        x = self.mid.block_2(self.mid.attn_1(self.mid.block_1(x)))  # type: ignore[operator]
        for level in reversed(range(len(self.up))):
            for index, block in enumerate(self.up[level].block):  # type: ignore[arg-type]
                x = block(x)
                if self.up[level].attn:
                    x = self.up[level].attn[index](x)  # type: ignore[index,operator]
            if level != 0:
                x = self.up[level].upsample(x)  # type: ignore[operator]
        return self.conv_out(F.silu(self.norm_out(x), inplace=True))


class HARTAutoencoder(NativeModel):
    """Inference subset of the published `HARTAutoEncoderWithDisc` checkpoint."""

    skipped_prefixes = ("disc.", "vae.encoder.", "vae.quant_conv.")

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.decoder = HARTDecoder(self.config)
        channels = self.config["z_channels"]
        size = self.config["quant_conv_ks"]
        self.post_quant_conv = nn.Conv2d(channels, channels, size, padding=size // 2)
        self.quantize = HARTQuantizer(self.config["vocab_size"], channels)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        prefix = "vae."
        return {
            name.removeprefix(prefix): tensor
            for name, tensor in state.items()
            if name.startswith(prefix) and not name.startswith(("vae.encoder.", "vae.quant_conv."))
        }

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(self.post_quant_conv(latents)).clamp(-1, 1)
