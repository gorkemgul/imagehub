"""Exact BAGEL parity against the pinned author implementation."""

import contextlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from helpers import reference_in_subprocess, snapshot_dir
from reference_sources import reference_source

from imagehub import pipeline
from imagehub.backends.bagel import (
    BagelComponents,
    _flow_prediction,
    _prompt_cache,
)
from imagehub.models.bagel import BagelModel, BagelTransformer
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

REFERENCE_ROOT = reference_source("BAGEL")
PROMPT = "A small red boat on a calm blue lake."


def _install_reference_stubs():
    flash = types.ModuleType("flash_attn")
    flash.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    flash.__version__ = "2.7.0"

    def flash_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False, **_kwargs):
        outputs = []
        for index in range(len(cu_seqlens_q) - 1):
            query = q[cu_seqlens_q[index] : cu_seqlens_q[index + 1]].transpose(0, 1)[None]
            key = k[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)[None]
            value = v[cu_seqlens_k[index] : cu_seqlens_k[index + 1]].transpose(0, 1)[None]
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=causal, enable_gqa=True
            )
            outputs.append(output[0].transpose(0, 1))
        return torch.cat(outputs)

    flash.flash_attn_func = lambda *args, **kwargs: None
    flash.flash_attn_varlen_func = flash_varlen
    sys.modules["flash_attn"] = flash
    einops = types.ModuleType("einops")
    einops.__spec__ = importlib.machinery.ModuleSpec("einops", loader=None)

    def rearrange(value, pattern, **sizes):
        if pattern == "b c h w -> b 1 (h w) c":
            return value.permute(0, 2, 3, 1).flatten(1, 2).unsqueeze(1)
        if pattern == "b 1 (h w) c -> b c h w":
            return (
                value.squeeze(1)
                .reshape(sizes["b"], sizes["h"], sizes["w"], sizes["c"])
                .permute(0, 3, 1, 2)
            )
        raise RuntimeError(f"unsupported reference rearrange pattern: {pattern}")

    einops.rearrange = rearrange
    sys.modules["einops"] = einops


def _patch_reference_rope():
    import modeling.qwen2.modeling_qwen2 as base

    def rope(config, device=None, **_kwargs):
        dim = config.hidden_size // config.num_attention_heads
        frequency = 1 / (
            config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim)
        )
        return frequency, 1.0

    base.ROPE_INIT_FUNCTIONS["default"] = rope


def _reference_qwen_module():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("BAGEL reference checkout not found")
    _install_reference_stubs()
    sys.path.insert(0, str(REFERENCE_ROOT))
    _patch_reference_rope()
    import modeling.bagel.qwen2_navit as module

    module.sdpa_kernel = lambda *args, **kwargs: contextlib.nullcontext()
    return module


def _tiny_llm_config():
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "qk_norm": True,
        "layer_module": "Qwen2MoTDecoderLayer",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "freeze_und": False,
        "pad_token_id": 0,
        "max_position_embeddings": 128,
    }


def test_random_mot_decoder_is_exact(record):
    reference_module = _reference_qwen_module()
    config = _tiny_llm_config()
    torch.manual_seed(0)
    reference = reference_module.Qwen2Model(reference_module.Qwen2Config(**config)).bfloat16()
    reference.train()
    ours = BagelTransformer(config).bfloat16().train()
    ours.load_state_dict(reference.state_dict(), strict=False)
    value = torch.randn(7, 32).bfloat16()
    positions = torch.arange(7)
    generated = torch.tensor([False, False, False, True, True, True, True])
    with torch.no_grad():
        expected = reference(
            packed_sequence=value.clone(),
            sample_lens=[7],
            attention_mask=[torch.zeros(7, 7)],
            packed_position_ids=positions,
            packed_und_token_indexes=(~generated).nonzero().flatten(),
            packed_gen_token_indexes=generated.nonzero().flatten(),
        )
        actual, _ = ours(value.clone(), positions, generated, causal=False)
    passed = torch.equal(actual, expected)
    record("bagel_mot_random", passed=passed, tolerance="exact")
    assert passed


def test_published_tensor_headers_match(record):
    root = snapshot_dir("bagel-7b-mot")
    config = json.loads((root / "config.json").read_text())
    config["max_latent_size"] = 64
    with torch.device("meta"):
        model = BagelModel(config)
    with (root / "model.safetensors.index.json").open() as source:
        weight_map = json.load(source)["weight_map"]
    expected = json.loads(
        (Path(__file__).parents[1] / "fixtures/bagel_7b_mot_configs/ema_header.json").read_text()
    )
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    result = {
        "missing": sorted(actual.keys() - expected.keys()),
        "extra": sorted(expected.keys() - actual.keys()),
        "mismatch": sorted(
            name for name in actual.keys() & expected.keys() if actual[name] != expected[name]
        ),
        "tensors": len(weight_map),
    }
    record("header_bagel-7b-mot", **result)
    assert not result["missing"] and not result["extra"] and not result["mismatch"]


def test_tokenizer_is_exact(record):
    root = snapshot_dir("bagel-7b-mot")
    reference = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    ours = Qwen2Tokenizer(root)
    prompts = [PROMPT, "naïve café — 東京", "<|vision_start|>cat<|vision_end|>"]
    expected = [reference.encode(text, add_special_tokens=False) for text in prompts]
    actual = [ours.tokenize(text) for text in prompts]
    passed = actual == expected
    record("bagel_tokenizer", passed=passed, prompts=len(prompts), tolerance="exact")
    assert passed


def _reference_components(snapshot, out_path):
    _install_reference_stubs()
    sys.path.insert(0, str(REFERENCE_ROOT))
    _patch_reference_rope()
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel.bagel import Bagel, BagelConfig
    from modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel.siglip_navit import SiglipVisionConfig, SiglipVisionModel
    from safetensors.torch import load_file

    llm_config = Qwen2Config.from_json_file(str(Path(snapshot) / "llm_config.json"))
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = None
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(str(Path(snapshot) / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae, vae_config = load_ae(str(Path(snapshot) / "ae.safetensors"))
    model = Bagel(
        Qwen2ForCausalLM(llm_config),
        SiglipVisionModel(vit_config),
        BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=2,
            max_latent_size=64,
            vit_max_num_patch_per_side=70,
        ),
    )
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    model.load_state_dict(load_file(str(Path(snapshot) / "ema.safetensors")), assign=True)
    model = model.bfloat16().eval()
    vae = vae.bfloat16().eval()
    model.language_model.model.enable_taylorseer = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    token_ids = {
        "bos_token_id": 151644,
        "eos_token_id": 151645,
        "start_of_image": 151652,
        "end_of_image": 151653,
    }
    inferencer = InterleaveInferencer(model, vae, tokenizer, None, None, token_ids)
    context = inferencer.update_context_text(PROMPT, inferencer.init_gen_context())
    prompt_key = context["past_key_values"].key_cache[0].clone()
    torch.manual_seed(123)
    generation = model.prepare_vae_latent(
        context["kv_lens"], context["ropes"], [(64, 64)], token_ids
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        velocity = model._forward_flow(
            x_t=generation["packed_init_noises"],
            timestep=torch.full((16,), 0.5),
            packed_vae_token_indexes=generation["packed_vae_token_indexes"],
            packed_vae_position_ids=generation["packed_vae_position_ids"],
            packed_text_ids=generation["packed_text_ids"],
            packed_text_indexes=generation["packed_text_indexes"],
            packed_indexes=generation["packed_indexes"],
            packed_position_ids=generation["packed_position_ids"],
            packed_seqlens=generation["packed_seqlens"],
            key_values_lens=generation["key_values_lens"],
            past_key_values=context["past_key_values"],
            packed_key_value_indexes=generation["packed_key_value_indexes"],
        )
    torch.manual_seed(321)
    decoded = vae.decode(torch.randn(1, 16, 8, 8, dtype=torch.bfloat16))
    torch.save({"prompt_key": prompt_key, "velocity": velocity, "decoded": decoded}, out_path)


def test_real_components(record, tmp_path):
    root = snapshot_dir("bagel-7b-mot")
    expected = reference_in_subprocess(
        "reference.test_ref_bagel_7b_mot._reference_components",
        {"snapshot": str(root)},
        tmp_path / "reference-components.pt",
    )
    components = BagelComponents.load(root, device=torch.device("cpu"), dtype=torch.bfloat16)
    cache = _prompt_cache(components.model, components.tokenizer, PROMPT)
    torch.manual_seed(123)
    latent = torch.randn(16, 64)
    grid = (torch.arange(4)[:, None] * 64 + torch.arange(4)[None]).flatten()
    positions = torch.full((18,), cache[0][0].shape[0], dtype=torch.long)
    velocity = _flow_prediction(components.model, latent, torch.tensor(0.5), positions, grid, cache)
    torch.manual_seed(321)
    decoded = components.vae.decode(torch.randn(1, 16, 8, 8, dtype=torch.bfloat16))
    comparisons = {
        "bagel_text_encoder": torch.equal(cache[0][0], expected["prompt_key"]),
        "bagel_denoiser": torch.equal(velocity, expected["velocity"]),
        "bagel_vae_decode": torch.equal(decoded, expected["decoded"]),
    }
    for name, passed in comparisons.items():
        record(name, passed=passed, tolerance="exact")
    assert all(comparisons.values())


def _reference_e2e(snapshot, seed, out_path):
    _install_reference_stubs()
    sys.path.insert(0, str(REFERENCE_ROOT))
    _patch_reference_rope()
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel.bagel import Bagel, BagelConfig
    from modeling.bagel.qwen2_navit import Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel.siglip_navit import SiglipVisionConfig, SiglipVisionModel
    from safetensors.torch import load_file

    llm_config = Qwen2Config.from_json_file(str(Path(snapshot) / "llm_config.json"))
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = None
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(str(Path(snapshot) / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae, vae_config = load_ae(str(Path(snapshot) / "ae.safetensors"))
    model = Bagel(
        Qwen2ForCausalLM(llm_config),
        SiglipVisionModel(vit_config),
        BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            latent_patch_size=2,
            max_latent_size=64,
            vit_max_num_patch_per_side=70,
        ),
    )
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    model.load_state_dict(load_file(str(Path(snapshot) / "ema.safetensors")), assign=True)
    model = model.bfloat16().eval()
    vae = vae.bfloat16().eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    inferencer = InterleaveInferencer(
        model,
        vae,
        tokenizer,
        None,
        None,
        {
            "bos_token_id": 151644,
            "eos_token_id": 151645,
            "start_of_image": 151652,
            "end_of_image": 151653,
        },
    )
    torch.manual_seed(seed)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        image = inferencer.interleave_inference(
            [PROMPT],
            image_shapes=(64, 64),
            num_timesteps=2,
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            timestep_shift=3.0,
        )[-1]
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(64, 64, 3)
    torch.save(pixels, out_path)


@pytest.mark.parametrize("seed", [0, 42])
def test_real_e2e(seed, tmp_path, record):
    root = snapshot_dir("bagel-7b-mot")
    expected = reference_in_subprocess(
        "reference.test_ref_bagel_7b_mot._reference_e2e",
        {"snapshot": str(root), "seed": seed},
        tmp_path / f"reference-{seed}.pt",
    )
    actual = (
        pipeline("t2i", backend="bagel-7b-mot", model=root, dtype=torch.bfloat16)(
            PROMPT, width=64, height=64, steps=2, seed=seed
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    record(
        "e2e_bagel-7b-mot" if seed == 0 else "seed_e2e_bagel-7b-mot",
        passed=passed,
        seed=seed,
        width=64,
        height=64,
        steps=2,
        tolerance="exact",
    )
    assert passed
