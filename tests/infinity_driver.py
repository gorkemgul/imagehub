"""Import helpers for the pinned original Infinity checkout with unused packages stubbed."""

from __future__ import annotations

import importlib
import importlib.machinery
import math
import sys
import types
from pathlib import Path

import torch
from reference_sources import reference_source
from torch.nn import functional as F

REFERENCE = reference_source("Infinity")


def _drop_path(value, drop_prob=0.0, training=False, scale_by_keep=True):
    return value


class _DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, value):
        return value


def _flash_attention(query, key, value, dropout_p=0.0, softmax_scale=None, **_kwargs):
    result = F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout_p,
        scale=softmax_scale,
    )
    return result.transpose(1, 2)


def _flash_varlen(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
):
    del max_seqlen_q, max_seqlen_k
    query = q
    outputs = []
    for index in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = int(cu_seqlens_q[index]), int(cu_seqlens_q[index + 1])
        k_start, k_end = int(cu_seqlens_k[index]), int(cu_seqlens_k[index + 1])
        key, value = kv[k_start:k_end].unbind(1)
        output = F.scaled_dot_product_attention(
            query[q_start:q_end].transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs)


def _install_stubs() -> None:
    pytz = types.ModuleType("pytz")
    pytz.timezone = lambda _name: None
    sys.modules["pytz"] = pytz
    imageio = types.ModuleType("imageio")
    sys.modules["imageio"] = imageio
    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_loss = types.ModuleType("timm.loss")
    timm_models.register_model = lambda function: function
    timm_layers.DropPath = _DropPath
    timm_layers.drop_path = _drop_path
    timm_loss.SoftTargetCrossEntropy = torch.nn.CrossEntropyLoss
    timm.models = timm_models
    sys.modules.update(
        {
            "timm": timm,
            "timm.models": timm_models,
            "timm.models.layers": timm_layers,
            "timm.loss": timm_loss,
        }
    )
    flash = types.ModuleType("flash_attn")
    flash.flash_attn_func = _flash_attention
    flash.flash_attn_varlen_kvpacked_func = _flash_varlen
    sys.modules["flash_attn"] = flash
    einops = types.ModuleType("einops")
    einops.rearrange = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("unused rearrange path")
    )
    sys.modules["einops"] = einops
    for module in (
        pytz,
        imageio,
        timm,
        timm_models,
        timm_layers,
        timm_loss,
        flash,
        einops,
    ):
        module.__spec__ = importlib.machinery.ModuleSpec(module.__name__, loader=None)


def reference_modules():
    if not REFERENCE.is_dir():
        raise FileNotFoundError(REFERENCE)
    _install_stubs()
    sys.path.insert(0, str(REFERENCE))
    basic = importlib.import_module("infinity.models.basic")
    multiscale = types.ModuleType("infinity.models.bsq_vae.multiscale_bsq")
    multiscale.MultiScaleBSQ = torch.nn.Identity
    sys.modules["infinity.models.bsq_vae.multiscale_bsq"] = multiscale
    vae = importlib.import_module("infinity.models.bsq_vae.flux_vqgan")
    return basic, vae


def _fused_norm(*, C, eps, x, scale, shift):
    return F.layer_norm(x, (C,), eps=eps) * (scale + 1) + shift


class _LFQ:
    def __init__(self):
        self.mask = 2 ** torch.arange(31, -1, -1)

    def indices_to_codes(self, indices, label_type="bit_label"):
        assert label_type == "bit_label"
        scale = 1.0 / math.sqrt(32)
        codes = indices * (2.0 * scale) - scale
        codes = F.normalize(codes, dim=-1)
        return codes.movedim(-1, 1)


class _Quantizer:
    def __init__(self):
        self.lfq = _LFQ()
        self.z_interplote_up = "trilinear"


class _ReferenceVAE:
    embed_dim = 32
    vocab_size = 2**32

    def __init__(self, decoder):
        self.decoder = decoder
        self.quantizer = _Quantizer()

    def decode(self, latent):
        return self.decoder(latent).clamp(-1, 1)


def reference_image(
    snapshot: str,
    prompt: str,
    seed: int,
    out_path: str,
    condition_path: str | None = None,
) -> None:
    """Run the original ``Infinity.autoregressive_infer_cfg`` on its smallest schedule."""
    root = Path(snapshot)
    if condition_path is None:
        from transformers import AutoTokenizer, T5EncoderModel

        text_root = root / "text_encoder"
        tokenizer = AutoTokenizer.from_pretrained(text_root, legacy=True)
        encoder = T5EncoderModel.from_pretrained(text_root, torch_dtype=torch.float32).eval()
        tokens = tokenizer(
            text=[prompt],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        features = encoder(
            input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
        ).last_hidden_state.float()
        lengths = tokens.attention_mask.sum(-1).tolist()
        compact = torch.cat([row[:length] for row, length in zip(features, lengths, strict=True)])
        cumulative = F.pad(tokens.attention_mask.sum(-1, dtype=torch.int32).cumsum(0), (1, 0))
        condition = compact, lengths, cumulative, max(lengths)
    else:
        condition = torch.load(condition_path, map_location="cpu", weights_only=True)

    basic, vae_module = reference_modules()
    infinity_module = importlib.import_module("infinity.models.infinity")
    cnn = {
        "cnn_type": "2d",
        "conv_in_out_2d": "no",
        "res_conv_2d": "no",
        "cnn_attention": "no",
        "cnn_norm_axis": "spatial",
        "conv_inner_2d": "no",
    }
    decoder = vae_module.Decoder(
        ch=128,
        ch_mult=[1, 2, 4, 4, 4],
        num_res_blocks=2,
        z_channels=32,
        patch_size=16,
        cnn_param=cnn,
    )
    vae_state = torch.load(
        root / "infinity_vae_d32reg.pth", map_location="cpu", weights_only=True, mmap=True
    )["vae"]
    decoder.load_state_dict(
        {
            name.removeprefix("decoder."): value
            for name, value in vae_state.items()
            if name.startswith("decoder.")
        },
        assign=True,
    )
    vae = _ReferenceVAE(decoder.eval())
    model = infinity_module.Infinity(
        vae_local=vae,
        text_channels=2048,
        text_maxlen=512,
        shared_aln=True,
        raw_scale_schedule=None,
        checkpointing="full-block",
        customized_flash_attn=False,
        fused_norm=True,
        pad_to_multiplier=128,
        use_flex_attn=False,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        pn="1M",
        apply_spatial_patchify=0,
        inference_mode=True,
        train_h_div_w_list=[1.0],
        depth=32,
        embed_dim=2048,
        num_heads=16,
        mlp_ratio=4,
        drop_path_rate=0.1,
        block_chunks=8,
    )
    state = torch.load(
        root / "infinity_2b_reg.pth", map_location="cpu", weights_only=True, mmap=True
    )
    model.load_state_dict(state, strict=True, assign=True)
    model.eval()
    for block in model.unregistered_blocks:
        block.fused_norm_func = _fused_norm
    model.head_nm.fused_norm_func = _fused_norm

    schedule = tuple((1, size, size) for size in (1, 2, 4, 6, 8, 12, 16))
    with torch.inference_mode():
        _, _, images = model.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=schedule,
            label_B_or_BLT=condition,
            g_seed=seed,
            B=1,
            cfg_sc=3,
            cfg_list=[3.0] * len(schedule),
            tau_list=[0.5] * len(schedule),
            top_k=900,
            top_p=0.97,
            returns_vemb=1,
            cfg_insertion_layer=[0],
            vae_type=32,
            ret_img=True,
            gt_leak=0,
            inference_mode=True,
            sampling_per_bits=1,
        )
    # The original returns BGR for OpenCV; imagehub stores RGB.
    torch.save(images.flip(-1).cpu(), out_path)
