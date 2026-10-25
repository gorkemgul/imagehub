"""Infinity-2B parity against the pinned original ``Infinity`` implementation."""

import gc
import importlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import reference_in_subprocess, snapshot_dir
from infinity_driver import _fused_norm, _Quantizer, reference_modules

from imagehub import pipeline
from imagehub.backends.infinity import InfinityComponents
from imagehub.models.infinity import (
    InfinityCrossAttention,
    InfinitySelfAttention,
    InfinityTransformer,
)
from imagehub.models.infinity_vae import InfinityDecoder, InfinityVAE
from imagehub.models.t5 import T5EncoderModel

pytestmark = pytest.mark.reference
MODEL_ID = "infinity-2b"
PROMPT = "A tiny blue bird perched on a flowering branch."
FIXTURES = Path(__file__).parents[1] / "fixtures" / "infinity_configs"
CPU = torch.device("cpu")


def test_random_self_attention_is_exact(record):
    basic, _ = reference_modules()
    torch.manual_seed(0)
    reference = basic.SelfAttention(
        embed_dim=64,
        num_heads=4,
        customized_flash_attn=False,
        cos_attn=True,
    ).eval()
    ours = InfinitySelfAttention(64, 4).eval()
    ours.load_state_dict(reference.state_dict())
    value = torch.randn(2, 5, 64)
    with torch.no_grad():
        expected = reference(value, None)
        actual = ours(value)
    assert torch.equal(actual, expected)
    record("infinity_self_attention_random_weights", max_abs=0.0, equal=True)


def test_random_cross_attention_is_exact(record):
    basic, _ = reference_modules()
    torch.manual_seed(1)
    reference = basic.CrossAttention(embed_dim=64, kv_dim=128, num_heads=4).eval()
    ours = InfinityCrossAttention(64, 128, 4).eval()
    ours.load_state_dict(reference.state_dict())
    query = torch.randn(2, 3, 64)
    compact = torch.randn(9, 128)
    cumulative = torch.tensor([0, 4, 9], dtype=torch.int32)
    with torch.no_grad():
        expected = reference(query, (compact, cumulative, 5))
        actual = ours(query, (compact, [4, 5], cumulative, 5))
    assert torch.equal(actual, expected)
    record("infinity_cross_attention_random_weights", max_abs=0.0, equal=True)


def test_random_transformer_first_scale_is_exact(record):
    reference_modules()
    reference_class = importlib.import_module("infinity.models.infinity").Infinity
    reference = reference_class(
        vae_local=SimpleNamespace(embed_dim=32, vocab_size=2**32, quantizer=_Quantizer()),
        text_channels=128,
        text_maxlen=8,
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_ratio=4,
        shared_aln=True,
        block_chunks=1,
        customized_flash_attn=False,
        fused_norm=False,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        pn="0.06M",
        train_h_div_w_list=[1.0],
        inference_mode=True,
    ).eval()
    for block in reference.unregistered_blocks:
        block.fused_norm_func = _fused_norm
    reference.head_nm.fused_norm_func = _fused_norm
    config = {
        "text_channels": 128,
        "text_max_length": 8,
        "embed_dim": 128,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4,
        "bit_channels": 32,
        "block_chunks": 1,
        "level_count": 15,
    }
    ours = InfinityTransformer(config).eval()
    ours.load_state_dict(reference.state_dict())
    torch.manual_seed(3)
    compact = torch.randn(5, 128)
    cumulative = torch.tensor([0, 5], dtype=torch.int32)
    condition = compact, [5], cumulative, 5
    schedule = tuple((1, size, size) for size in (1, 2, 4, 6, 8, 12, 16))

    with torch.no_grad():
        native_stage, adaptive, pooled, native_condition = ours.begin_generation(
            condition, schedule, True
        )
        _, actual = ours.stage_logits(native_stage, adaptive, pooled, native_condition, schedule, 0)
        ours.end_generation()

        unconditioned = reference.cfg_uncond[:5]
        reference_compact = torch.cat((compact, unconditioned))
        reference_cumulative = torch.tensor([0, 5, 10], dtype=torch.int32)
        reference_compact = reference.text_norm(reference_compact)
        ca_input = (reference_compact, reference_cumulative, 5)
        reference_pooled = reference.text_proj_for_sos(ca_input)
        reference_condition = (
            reference.text_proj_for_ca(reference_compact),
            reference_cumulative,
            5,
        )
        reference_adaptive = reference.shared_ada_lin(reference_pooled.float())
        stage = reference_pooled[:, None] + reference.pos_start
        for block in reference.unregistered_blocks:
            block.sa.kv_caching(True)
        stage = reference.add_lvl_embeding(stage, 0, schedule)
        for block in reference.blocks:
            stage = block(
                stage,
                reference_adaptive,
                reference_condition,
                None,
                None,
                schedule,
                reference.rope2d_freqs_grid,
                0,
            )
        expected = reference.get_logits(stage, reference_pooled)
        for block in reference.unregistered_blocks:
            block.sa.kv_caching(False)
    assert torch.equal(actual, expected)
    record("infinity_transformer_random_weights", first_scale_equal=True, max_abs=0.0)


def test_random_vae_decoder_is_exact(record):
    _, vae_module = reference_modules()
    cnn = {
        "cnn_type": "2d",
        "conv_in_out_2d": "no",
        "res_conv_2d": "no",
        "cnn_attention": "no",
        "cnn_norm_axis": "spatial",
        "conv_inner_2d": "no",
    }
    config = {
        "base_channels": 32,
        "channel_multipliers": (1, 1),
        "num_res_blocks": 1,
        "latent_channels": 4,
        "patch_size": 2,
    }
    torch.manual_seed(2)
    reference = vae_module.Decoder(
        ch=32,
        ch_mult=[1, 1],
        num_res_blocks=1,
        z_channels=4,
        patch_size=2,
        cnn_param=cnn,
    ).eval()
    ours = InfinityDecoder(config).eval()
    ours.load_state_dict(reference.state_dict())
    latent = torch.randn(1, 4, 3, 4)
    with torch.no_grad():
        expected = reference(latent)
        actual = ours(latent)
    assert torch.equal(actual, expected)
    record("infinity_vae_decoder_random_weights", max_abs=0.0, equal=True)


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    transformer_state = torch.load(
        root / "infinity_2b_reg.pth", map_location="cpu", weights_only=True, mmap=True
    )
    vae_state = torch.load(
        root / "infinity_vae_d32reg.pth", map_location="cpu", weights_only=True, mmap=True
    )["vae"]
    with torch.device("meta"):
        models = {
            "transformer": InfinityTransformer(),
            "vae": InfinityVAE(),
            "text_encoder": T5EncoderModel(
                json.loads((root / "text_encoder" / "config.json").read_text())
            ),
        }
    published = {
        "transformer": transformer_state,
        "vae": vae_state,
    }
    text_shapes = {}
    index = json.loads((root / "text_encoder" / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        with (root / "text_encoder" / shard).open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        text_shapes.update(
            {
                name: torch.empty(entry["shape"], device="meta")
                for name, entry in header.items()
                if name != "__metadata__"
            }
        )
    published["text_encoder"] = {
        name: value
        for name, value in text_shapes.items()
        if not name.startswith(T5EncoderModel.skipped_prefixes)
    }
    results = {}
    for name in ("transformer", "vae", "text_encoder"):
        prepared = models[name].prepare_state_dict(published[name])
        expected = models[name].state_dict()
        results[name] = {
            "missing": sorted(expected.keys() - prepared.keys()),
            "extra": sorted(prepared.keys() - expected.keys()),
            "mismatch": sorted(
                key
                for key in expected.keys() & prepared.keys()
                if expected[key].shape != prepared[key].shape
            ),
            "tensors": len(published[name]),
        }
    record(f"header_{MODEL_ID}", **results)
    assert all(
        not result[key] for result in results.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer_text_encoder_and_first_scale_load(record):
    root = snapshot_dir(MODEL_ID)
    transformers = pytest.importorskip("transformers")
    reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "text_encoder", legacy=True
    )
    expected = reference_tokenizer(
        [PROMPT], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
    )
    reference_encoder = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder", torch_dtype=torch.float32
    ).eval()
    with torch.no_grad():
        expected_features = reference_encoder(
            expected.input_ids, attention_mask=expected.attention_mask
        ).last_hidden_state
    del reference_encoder
    gc.collect()
    components = InfinityComponents.load(root, device=CPU, dtype=torch.float32)
    actual_ids, actual_mask = components.tokenizer([PROMPT], 512)
    assert torch.equal(actual_ids, expected.input_ids)
    assert torch.equal(actual_mask, expected.attention_mask)
    with torch.no_grad():
        actual_features = components.text_encoder(actual_ids, actual_mask)[0]
    assert torch.equal(actual_features, expected_features)
    # Loading all three strict native components is the real-checkpoint component smoke test;
    # end-to-end comparisons below exercise the first and all following autoregressive scales.
    record(
        f"components_{MODEL_ID}",
        tokenizer_equal=True,
        text_encoder_equal=True,
        first_scale_loaded=True,
        vae_decoder_loaded=True,
    )
    del components
    gc.collect()


def _e2e(root, tmp_path, record, seed, key):
    expected = reference_in_subprocess(
        "infinity_driver.reference_image",
        {"snapshot": str(root), "prompt": PROMPT, "seed": seed},
        tmp_path / f"reference-{seed}.pt",
    )
    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=seed, width=256, height=256)
        .images[0]
        .tensor.unsqueeze(0)
    )
    passed = torch.equal(actual, expected)
    record(key, passed=passed, seed=seed, width=256, height=256, scales=7)
    assert passed


def test_e2e_infinity_2b(tmp_path, record):
    _e2e(snapshot_dir(MODEL_ID), tmp_path, record, 0, f"e2e_{MODEL_ID}")


def test_seed_e2e_infinity_2b(tmp_path, record):
    _e2e(snapshot_dir(MODEL_ID), tmp_path, record, 42, f"seed_e2e_{MODEL_ID}")
