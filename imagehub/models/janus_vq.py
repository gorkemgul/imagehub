"""Janus-Pro VQ decoder and image-generation projection modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.checkpoints import NativeModel


class JanusVQVectorQuantizer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config["num_embeddings"], config["embed_dim"])
        self.quant_state_dims = [config["num_patches"]] * 2

    def get_codebook_entry(self, image_tokens: Tensor) -> Tensor:
        batch_size = image_tokens.shape[0]
        embedding_dim = self.embedding.weight.shape[-1]
        hidden_states = self.embedding(image_tokens)
        hidden_states = F.normalize(hidden_states, p=2, dim=-1)
        hidden_states = hidden_states.view((batch_size, *self.quant_state_dims, embedding_dim))
        return hidden_states.permute(0, 3, 1, 2).contiguous()


class JanusVQResnetBlock(nn.Module):
    def __init__(
        self,
        config: Mapping[str, Any],
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, self.out_channels, eps=1e-6)
        self.dropout = nn.Dropout(config["dropout"])
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return residual + hidden_states


class JanusVQAttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)
        batch_size, channels, height, width = query_states.shape
        query_states = query_states.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        key_states = key_states.reshape(batch_size, channels, height * width)
        attention_weights = torch.bmm(query_states, key_states)
        attention_weights = attention_weights * (int(channels) ** (-0.5))
        attention_weights = F.softmax(attention_weights, dim=2)
        value_states = value_states.reshape(batch_size, channels, height * width)
        attention_weights = attention_weights.permute(0, 2, 1)
        attention_output = torch.bmm(value_states, attention_weights).reshape(
            batch_size, channels, height, width
        )
        return residual + self.proj_out(attention_output)


class JanusVQUpsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class JanusVQMidBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any], channels: int) -> None:
        super().__init__()
        self.block_1 = JanusVQResnetBlock(config, channels, channels)
        self.attn_1 = JanusVQAttentionBlock(channels)
        self.block_2 = JanusVQResnetBlock(config, channels, channels)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.block_1(hidden_states)
        hidden_states = self.attn_1(hidden_states)
        return self.block_2(hidden_states)


class JanusVQUpBlock(nn.Module):
    def __init__(
        self,
        block: nn.ModuleList,
        attention: nn.ModuleList,
        upsample: JanusVQUpsample | None,
    ) -> None:
        super().__init__()
        self.block = block
        self.attn = attention
        self.upsample = upsample


class JanusVQDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.num_resolutions = len(config["channel_multiplier"])
        self.num_res_blocks = config["num_res_blocks"]
        base_channels = config["base_channels"]
        block_in = base_channels * config["channel_multiplier"][-1]
        self.conv_in = nn.Conv2d(config["latent_channels"], block_in, 3, padding=1)
        self.mid = JanusVQMidBlock(config, block_in)
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attention = nn.ModuleList()
            block_out = base_channels * config["channel_multiplier"][level]
            for _ in range(self.num_res_blocks + 1):
                block.append(JanusVQResnetBlock(config, block_in, block_out))
                block_in = block_out
                if level == self.num_resolutions - 1:
                    attention.append(JanusVQAttentionBlock(block_in))
            upsample = JanusVQUpsample(block_in) if level != 0 else None
            self.up.append(JanusVQUpBlock(block, attention, upsample))
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, config["out_channels"], 3, padding=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv_in(hidden_states)
        hidden_states = self.mid(hidden_states)
        for level in range(self.num_resolutions):
            up = cast(JanusVQUpBlock, self.up[level])
            for block in range(self.num_res_blocks + 1):
                hidden_states = up.block[block](hidden_states)
                if len(up.attn) > 0:
                    hidden_states = up.attn[block](hidden_states)
            if level != self.num_resolutions - 1:
                upsample = up.upsample
                assert upsample is not None
                hidden_states = upsample(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        return self.conv_out(hidden_states)


class JanusVQVAE(NativeModel):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        self.quantize = JanusVQVectorQuantizer(self.config)
        self.post_quant_conv = nn.Conv2d(
            self.config["embed_dim"], self.config["latent_channels"], 1
        )
        self.decoder = JanusVQDecoder(self.config)

    def decode(self, image_tokens: Tensor) -> Tensor:
        expected_tokens = self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]
        if image_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected `image_tokens` to have shape `(batch_size, {expected_tokens})`, "
                f"but got shape `{image_tokens.shape}`."
            )
        codebook_entry = self.quantize.get_codebook_entry(image_tokens)
        hidden_states = self.post_quant_conv(codebook_entry)
        return self.decoder(hidden_states)


class JanusVQVAEAlignerMLP(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config["embed_dim"], config["projection_dim"])
        self.hidden_layers = nn.ModuleList(
            nn.Linear(config["projection_dim"], config["projection_dim"])
            for _ in range(1, config["num_hidden_layers"])
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        for layer in self.hidden_layers:
            hidden_states = F.gelu(hidden_states)
            hidden_states = layer(hidden_states)
        return hidden_states


class JanusVQVAEHead(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.proj_out = nn.Linear(config["image_token_embed_dim"], config["projection_dim"])
        self.vision_head = nn.Linear(config["projection_dim"], config["num_embeddings"])

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj_out(hidden_states)
        hidden_states = F.gelu(hidden_states)
        return self.vision_head(hidden_states)


class JanusGenerationModel(NativeModel):
    """All Janus image-generation tensors below the published ``model.`` prefix."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        if self.config["hidden_act"] != "gelu":
            raise ValueError("Unsupported Janus VQ activation")
        self.vqmodel = JanusVQVAE(self.config)
        self.generation_embeddings = nn.Embedding(
            self.config["num_embeddings"], self.config["embed_dim"]
        )
        self.generation_aligner = JanusVQVAEAlignerMLP(self.config)
        self.generation_head = JanusVQVAEHead(self.config)

    def decode_image_tokens(self, image_tokens: Tensor) -> Tensor:
        decoded_image = self.vqmodel.decode(image_tokens)
        return decoded_image.permute(0, 2, 3, 1)


def postprocess_janus(decoded_image: Tensor) -> Tensor:
    """Return the uint8 HWC pixels produced by JanusImageProcessor.postprocess."""
    images = decoded_image.float().permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor([-1.0, -1.0, -1.0], device=images.device)
    std = torch.tensor([2.0, 2.0, 2.0], device=images.device) * (1.0 / 255.0)
    images = images.sub(mean[:, None, None]).div(std[:, None, None])
    images = images.clip(0, 255).to(torch.uint8)
    return images.permute(0, 2, 3, 1)
