"""Native NextStep-1 causal transformer and continuous-token flow head."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel
from imagehub.models.autoencoder_kl import AutoencoderKL


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.variance_epsilon = eps

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * normalized.to(dtype)


def _rotate_half(value: Tensor) -> Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", hidden // self.heads)
        self.groups = self.heads // self.kv_heads
        self.q_proj = nn.Linear(hidden, self.heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, self.kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(
            self.heads * self.head_dim, hidden, bias=config.get("o_attention_bias", False)
        )

    def forward(
        self,
        value: Tensor,
        cos: Tensor,
        sin: Tensor,
        attention_mask: Tensor,
        past: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = value.shape
        query = self.q_proj(value).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(value).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        values = (
            self.v_proj(value).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        )
        query = query * cos[:, None] + _rotate_half(query) * sin[:, None]
        key = key * cos[:, None] + _rotate_half(key) * sin[:, None]
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            values = torch.cat((past[1], values), dim=2)
        present = (key, values)
        key = (
            key[:, :, None]
            .expand(-1, -1, self.groups, -1, -1)
            .reshape(batch, self.heads, key.shape[2], self.head_dim)
        )
        values = (
            values[:, :, None]
            .expand(-1, -1, self.groups, -1, -1)
            .reshape(batch, self.heads, values.shape[2], self.head_dim)
        )
        output = F.scaled_dot_product_attention(query, key, values, attn_mask=attention_mask)
        output = output.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.o_proj(output), present


class MLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        hidden, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(hidden, intermediate, bias=config.get("mlp_bias", False))
        self.up_proj = nn.Linear(hidden, intermediate, bias=config.get("mlp_bias", False))
        self.down_proj = nn.Linear(intermediate, hidden, bias=config.get("mlp_bias", False))

    def forward(self, value: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class DecoderLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])

    def forward(self, value: Tensor, cos: Tensor, sin: Tensor, mask: Tensor, past: Any) -> Any:
        attention, present = self.self_attn(self.input_layernorm(value), cos, sin, mask, past)
        value = value + attention
        return value + self.mlp(self.post_attention_layernorm(value)), present


class Transformer(nn.Module):
    inv_freq: Tensor

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config["num_hidden_layers"])
        )
        self.norm = RMSNorm(config["hidden_size"], config["rms_norm_eps"])
        dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
        self.register_buffer(
            "inv_freq",
            1 / (config["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )

    def forward(
        self,
        value: Tensor,
        attention_mask: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        positions = torch.arange(past_length, past_length + value.shape[1], device=value.device)
        frequencies = torch.outer(positions.float(), self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), -1)
        cos, sin = embedding.cos().to(value.dtype)[None], embedding.sin().to(value.dtype)[None]
        query_positions = positions[:, None]
        key_positions = torch.arange(past_length + value.shape[1], device=value.device)[None]
        mask = (key_positions <= query_positions)[None, None]
        mask = mask & attention_mask[:, None, None, :].bool()
        presents = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            value, present = layer(value, cos, sin, mask, past)
            presents.append(present)
        return self.norm(value), presents


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, timestep: Tensor) -> Tensor:
        half = 128
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device).float() / half
        )
        args = timestep[:, None].float() * frequencies[None]
        value = torch.cat((args.cos(), args.sin()), -1)
        return self.mlp(value.to(self.mlp[0].weight.dtype))


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, -1)
        hidden = self.in_ln(value) * (1 + scale) + shift
        return value + gate * self.mlp(hidden)


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, output: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, output)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, value: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, -1)
        return self.linear(self.norm_final(value) * (1 + scale) + shift)


class FlowNetwork(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.time_embed = TimestepEmbedder(hidden)
        self.cond_embed = nn.Linear(cond_dim, hidden)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.res_blocks = nn.ModuleList(ResBlock(hidden) for _ in range(layers))
        self.final_layer = FinalLayer(hidden, input_dim)

    def forward(self, value: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        value = self.input_proj(value)
        combined = self.time_embed(timestep) + self.cond_embed(condition)
        for block in self.res_blocks:
            value = block(value, combined)
        return self.final_layer(value, combined)


class FlowMatchingHead(nn.Module):
    def __init__(self, input_dim: int, cond_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.net = FlowNetwork(input_dim, cond_dim, hidden, layers)

    @staticmethod
    def guided(velocity: Tensor, guidance: float) -> Tensor:
        if guidance == 1:
            return velocity
        conditional, unconditional = velocity.chunk(2)
        return unconditional + guidance * (conditional - unconditional)

    def sample(
        self,
        condition: Tensor,
        noise: Tensor,
        *,
        guidance: float,
        steps: int,
        shift: float,
        generator: torch.Generator | None = None,
        sde: bool = True,
    ) -> Tensor:
        value = noise
        times = torch.linspace(0, 1, steps + 1, device=value.device)
        times = times / (shift - (shift - 1) * times)
        for current, following in zip(times[:-1], times[1:], strict=True):
            combined = torch.cat((value, value)) if guidance != 1 else value
            velocity = self.net(
                combined.to(condition.dtype),
                current.expand(condition.shape[0]).to(condition),
                condition,
            ).float()
            velocity = self.guided(velocity, guidance)
            current_view = current.to(value.device)
            x0 = value - velocity * current_view
            x1 = value + velocity * (1 - current_view)
            value = x0 * (1 - following) + x1 * following
            if sde:
                sigma = 1 - current
                sigma_max = times[-2]
                denominator = 1 - torch.where(sigma == 1, sigma_max, sigma)
                std = torch.sqrt(sigma / denominator) * 0.8
                delta = following - current
                value = value - std**2 * delta / (2 * sigma) * x0
                draw_device = generator.device if generator is not None else value.device
                random = torch.randn(
                    value.shape, generator=generator, device=draw_device, dtype=value.dtype
                ).to(value.device)
                value = value + std * torch.sqrt(delta) * random
        return value.to(condition.dtype)


class NextStepModel(NativeModel):
    """NextStep-1 with the published top-level tensor layout."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        core = Transformer(self.config)
        self.embed_tokens = core.embed_tokens
        self.layers = core.layers
        self.norm = core.norm
        self.register_buffer("inv_freq", core.inv_freq, persistent=False)
        hidden = config["hidden_size"]
        token_dim = config["latent_channels"] * config["latent_patch_size"] ** 2
        self.lm_head = nn.Linear(hidden, config["vocab_size"], bias=False)
        self.image_in_projector = nn.Linear(token_dim, hidden)
        self.image_out_projector = nn.Linear(hidden, hidden)
        self.image_head = FlowMatchingHead(
            token_dim, hidden, config["fm_head_dim"], config["fm_head_layers"]
        )
        if config.get("use_gen_pos_embed", False):
            size = config.get("base_image_grid_size", config["latent_size"] // 2)
            self.register_buffer("gen_pos_embed", torch.empty(1, size * size, hidden))

    def init_non_persistent_buffers(self) -> None:
        dim = self.config.get(
            "head_dim", self.config["hidden_size"] // self.config["num_attention_heads"]
        )
        self.inv_freq = 1 / (self.config["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim))

    def patchify(self, latent: Tensor) -> Tensor:
        batch, channels, height, width = latent.shape
        patch = self.config["latent_patch_size"]
        value = latent.reshape(batch, channels, height // patch, patch, width // patch, patch)
        return torch.einsum("nchpwq->nhwcpq", value).reshape(
            batch, height * width // patch**2, channels * patch**2
        )

    def unpatchify(self, tokens: Tensor, height: int, width: int) -> Tensor:
        batch = tokens.shape[0]
        patch, channels = self.config["latent_patch_size"], self.config["latent_channels"]
        value = tokens.reshape(batch, height, width, channels, patch, patch)
        return torch.einsum("nhwcpq->nchpwq", value).reshape(
            batch, channels, height * patch, width * patch
        )

    def embed(self, ids: Tensor, latents: Tensor | None = None) -> Tensor:
        if latents is None:
            return self.embed_tokens(ids)
        image = ids == self.config["image_placeholder_id"]
        result = self.embed_tokens(ids.clamp_max(self.config["vocab_size"] - 1))
        result[image] = self.image_in_projector(self.patchify(latents)).reshape(
            -1, result.shape[-1]
        )
        return result

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        past_key_values: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        positions = torch.arange(
            past_length, past_length + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        frequencies = torch.outer(positions.float(), self.inv_freq.float())
        embedding = torch.cat((frequencies, frequencies), -1)
        cos = embedding.cos().to(inputs_embeds.dtype)[None]
        sin = embedding.sin().to(inputs_embeds.dtype)[None]
        query_positions = positions[:, None]
        key_positions = torch.arange(
            past_length + inputs_embeds.shape[1], device=inputs_embeds.device
        )[None]
        mask = (key_positions <= query_positions)[None, None]
        mask = mask & attention_mask[:, None, None, :].bool()
        presents = []
        hidden = inputs_embeds
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present = layer(hidden, cos, sin, mask, past)
            presents.append(present)
        return self.norm(hidden), presents

    def generation_positions(self, height: int, width: int) -> Tensor:
        value = self.gen_pos_embed
        side = int(value.shape[1] ** 0.5)
        return value.reshape(1, side, side, -1)[:, :height, :width].reshape(1, height * width, -1)


class NextStepAutoencoder(AutoencoderKL):
    """Deterministic 16-channel VAE with the published legacy tensor names."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        native = {
            "in_channels": config.get("in_channels", 3),
            "out_channels": config.get("out_ch", 3),
            "latent_channels": config.get("z_channels", 16),
            "block_out_channels": [
                config.get("ch", 128) * value for value in config.get("ch_mult", [1, 2, 4, 4])
            ],
            "layers_per_block": config.get("num_res_blocks", 2),
            "norm_num_groups": 32,
            "scaling_factor": config.get("scaling_factor", 1.0),
            "shift_factor": config.get("shift_factor", 0.0),
            "use_quant_conv": False,
            "use_post_quant_conv": False,
            "mid_block_add_attention": True,
        }
        super().__init__(native)

    def prepare_state_dict(self, state: dict[str, Tensor]) -> dict[str, Tensor]:
        renamed: dict[str, Tensor] = {}
        resolutions = len(self.config["block_out_channels"])
        for original, value in state.items():
            name = original.replace(".nin_shortcut.", ".conv_shortcut.")
            name = name.replace(".norm_out.", ".conv_norm_out.")
            if name.startswith("encoder."):
                name = name.replace(".down.", ".down_blocks.")
                name = name.replace(".block.", ".resnets.")
                name = name.replace(".downsample.", ".downsamplers.0.")
            elif name.startswith("decoder."):
                for index in range(resolutions):
                    name = name.replace(f".up.{index}.", f".up_blocks.{resolutions - 1 - index}.")
                name = name.replace(".block.", ".resnets.")
                name = name.replace(".upsample.", ".upsamplers.0.")
            name = name.replace(".mid.block_1.", ".mid_block.resnets.0.")
            name = name.replace(".mid.block_2.", ".mid_block.resnets.1.")
            name = name.replace(".mid.attn_1.norm.", ".mid_block.attentions.0.group_norm.")
            name = name.replace(".mid.attn_1.q.", ".mid_block.attentions.0.query.")
            name = name.replace(".mid.attn_1.k.", ".mid_block.attentions.0.key.")
            name = name.replace(".mid.attn_1.v.", ".mid_block.attentions.0.value.")
            name = name.replace(".mid.attn_1.proj_out.", ".mid_block.attentions.0.proj_attn.")
            if ".attentions." in name and value.ndim == 4 and value.shape[-2:] == (1, 1):
                value = value.squeeze(-1).squeeze(-1)
            renamed[name] = value
        return renamed
