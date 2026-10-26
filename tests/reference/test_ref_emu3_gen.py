"""Exact Emu3 parity; real tests use ``IMAGEHUB_SNAPSHOT_EMU3_GEN``."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.backends.emu3 import Emu3Components, _sample, generate_tokens
from imagehub.models.emu3 import Emu3ForCausalLM
from imagehub.models.emu3_vq import Emu3VisionVQ
from imagehub.tokenization_emu3 import Emu3Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

REFERENCE_ROOT = Path("/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/Emu3")
FIXTURES = Path(__file__).parents[1] / "fixtures/emu3_gen_configs"


def _reference_modules():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"Emu3 reference checkout not found at {REFERENCE_ROOT}")
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
            self.get_seq_length(layer_idx)
        )
    sys.path.insert(0, str(REFERENCE_ROOT))
    from emu3.mllm.configuration_emu3 import Emu3Config
    from emu3.mllm.modeling_emu3 import Emu3ForCausalLM as ReferenceLM
    from emu3.tokenizer.configuration_emu3visionvq import Emu3VisionVQConfig
    from emu3.tokenizer.modeling_emu3visionvq import Emu3VisionVQModel

    return Emu3Config, ReferenceLM, Emu3VisionVQConfig, Emu3VisionVQModel


def _materialize_reference_rotary_buffers(model: torch.nn.Module) -> None:
    """Rebuild non-persistent RoPE buffers after meta-device construction."""
    for module in model.modules():
        if module.__class__.__name__ != "Emu3RotaryEmbedding":
            continue
        module.inv_freq = 1.0 / (
            module.base ** (torch.arange(0, module.dim, 2, dtype=torch.float32) / module.dim)
        )
        module._set_cos_sin_cache(
            seq_len=module.max_position_embeddings,
            device=module.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )


def test_published_headers_are_exact(record):
    generator_config = json.loads((FIXTURES / "generator/config.json").read_text())
    vision_config = json.loads((FIXTURES / "vision_tokenizer/config.json").read_text())
    with torch.device("meta"):
        generator = Emu3ForCausalLM(generator_config)
        vision = Emu3VisionVQ(vision_config)
    expected_generator = {}
    for path in sorted((FIXTURES / "generator").glob("*-header.json")):
        expected_generator.update(json.loads(path.read_text()))
    expected_vision = json.loads((FIXTURES / "vision_tokenizer/model-header.json").read_text())
    actual_generator = {name: list(value.shape) for name, value in generator.state_dict().items()}
    actual_vision = {name: list(value.shape) for name, value in vision.state_dict().items()}
    passed = actual_generator == expected_generator and actual_vision == expected_vision
    record(
        "emu3_gen_headers",
        passed=passed,
        generator_tensors=len(actual_generator),
        vision_tensors=len(actual_vision),
        missing=0 if passed else -1,
        extra=0 if passed else -1,
        mismatch=0 if passed else -1,
    )
    assert passed


def test_random_language_model_is_exact(record):
    Emu3Config, ReferenceLM, _, _ = _reference_modules()
    config = {
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "rope_scaling": None,
        "pretraining_tp": 1,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "img_token_id": 3,
        "boi_token_id": 4,
        "eoi_token_id": 5,
        "eol_token_id": 6,
        "eof_token_id": 7,
    }
    reference_config = Emu3Config(**config)
    reference_config.rope_scaling = None
    torch.manual_seed(0)
    reference = ReferenceLM(reference_config).eval()
    ours = Emu3ForCausalLM(config).eval()
    ours.load_state_dict(reference.state_dict())
    ids = torch.randint(8, 64, (2, 7))
    with torch.no_grad():
        expected = reference(ids, use_cache=False).logits
        actual, _ = ours(ids)
        from transformers.cache_utils import DynamicCache

        reference_cache = DynamicCache(config=reference_config)
        reference_prefill = reference(ids, past_key_values=reference_cache, use_cache=True)
        next_ids = torch.tensor([[9], [10]])
        expected_cached = reference(
            next_ids, past_key_values=reference_prefill.past_key_values, use_cache=True
        ).logits
        _, native_cache = ours(ids)
        actual_cached, _ = ours(next_ids, native_cache)
    passed = torch.equal(actual, expected) and torch.equal(actual_cached, expected_cached)
    record("emu3_gen_language_model_random", passed=passed, cache=True, tolerance="exact")
    assert passed


def test_random_vision_decode_is_exact(record):
    _, _, ReferenceConfig, ReferenceVQ = _reference_modules()
    config = {
        "ch": 32,
        "ch_mult": [1, 1],
        "num_res_blocks": 1,
        "attn_resolutions": [1],
        "dropout": 0.0,
        "in_channels": 3,
        "out_channels": 3,
        "z_channels": 4,
        "embed_dim": 4,
        "codebook_size": 16,
        "double_z": False,
        "temporal_downsample_factor": 1,
    }
    torch.manual_seed(1)
    reference = ReferenceVQ(ReferenceConfig(**config)).eval()
    ours = Emu3VisionVQ(config).eval()
    ours.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[[1, 2], [3, 4]]])
    with torch.no_grad():
        expected = reference.decode(tokens)
        actual = ours.decode(tokens)
    passed = torch.equal(actual, expected)
    record("emu3_gen_vision_decode_random", passed=passed, tolerance="exact")
    assert passed


def test_tiny_end_to_end_is_exact(record):
    Emu3Config, ReferenceLM, ReferenceVQConfig, ReferenceVQ = _reference_modules()
    lm_config = {
        "vocab_size": 64,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "rope_scaling": None,
        "pretraining_tp": 1,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "img_token_id": 3,
        "boi_token_id": 4,
        "eoi_token_id": 5,
        "eol_token_id": 6,
        "eof_token_id": 7,
    }
    vq_config = {
        "ch": 32,
        "ch_mult": [1, 1],
        "num_res_blocks": 1,
        "attn_resolutions": [1],
        "dropout": 0.0,
        "in_channels": 3,
        "out_channels": 3,
        "z_channels": 4,
        "embed_dim": 4,
        "codebook_size": 16,
        "double_z": False,
        "temporal_downsample_factor": 1,
    }
    reference_config = Emu3Config(**lm_config)
    reference_config.rope_scaling = None
    torch.manual_seed(4)
    reference_lm = ReferenceLM(reference_config).eval()
    native_lm = Emu3ForCausalLM(lm_config).eval()
    native_lm.load_state_dict(reference_lm.state_dict())
    reference_vq = ReferenceVQ(ReferenceVQConfig(**vq_config)).eval()
    native_vq = Emu3VisionVQ(vq_config).eval()
    native_vq.load_state_dict(reference_vq.state_dict())
    tokenizer = SimpleNamespace(visual_start=16, visual_end=31, eol_id=6, eof_id=7, eoi_id=5)
    positive = torch.tensor([[1, 8, 4, 3]])
    negative = torch.tensor([[1, 9, 4, 3]])
    native_tokens = generate_tokens(
        native_lm,
        tokenizer,
        positive,
        negative,
        height=1,
        width=3,
        guidance=3.0,
        temperature=1.0,
        top_k=1,
        generator=torch.Generator().manual_seed(0),
    )
    from transformers.cache_utils import DynamicCache

    positive_cache = DynamicCache(config=reference_config)
    negative_cache = DynamicCache(config=reference_config)
    with torch.no_grad():
        positive_logits = reference_lm(
            positive, past_key_values=positive_cache, use_cache=True
        ).logits[:, -1]
        negative_logits = reference_lm(
            negative, past_key_values=negative_cache, use_cache=True
        ).logits[:, -1]
        visual = []
        generator = torch.Generator().manual_seed(0)
        for offset in range(1, (3 + 1) * 1 + 3):
            if offset % 4 == 0:
                token = torch.tensor([[6]])
            elif offset == 5:
                token = torch.tensor([[7]])
            elif offset == 6:
                token = torch.tensor([[5]])
            else:
                logits = negative_logits + 3.0 * (positive_logits - negative_logits)
                sampled = _sample(logits[:, 16:32], 1, 1.0, generator)
                visual.append(sampled)
                token = sampled + 16
            positive_logits = reference_lm(
                token, past_key_values=positive_cache, use_cache=True
            ).logits[:, -1]
            negative_logits = reference_lm(
                token, past_key_values=negative_cache, use_cache=True
            ).logits[:, -1]
        reference_tokens = torch.cat(visual, 1).reshape(1, 1, 3)
        expected = reference_vq.decode(reference_tokens)
        actual = native_vq.decode(native_tokens)
    passed = torch.equal(native_tokens, reference_tokens) and torch.equal(actual, expected)
    record("emu3_gen_tiny_e2e", passed=passed, tolerance="exact")
    assert passed


def test_real_tokenizer_is_exact(record):
    tiktoken = pytest.importorskip("tiktoken")
    del tiktoken
    root = snapshot_dir("emu3-gen")
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        from emu3.mllm.tokenization_emu3 import Emu3Tokenizer as ReferenceTokenizer
    finally:
        sys.path.pop(0)
    reference = ReferenceTokenizer(
        vocab_file=str(root / "emu3.tiktoken"),
        special_tokens_file=str(root / "emu3_vision_tokens.txt"),
    )
    ours = Emu3Tokenizer(root)
    prompts = (
        "a portrait of young girl.",
        "Café 日本語\nbox ┌─┐",
        "<|extra_203|>an emu<|image start|>2*3<|image token|>",
    )
    expected = [reference.encode(prompt) for prompt in prompts]
    actual = [ours.encode(prompt) for prompt in prompts]
    passed = actual == expected
    record("emu3_gen_tokenizer", passed=passed, prompts=len(prompts), tolerance="exact")
    assert passed


def _reference_components(snapshot: str, out_path: str) -> None:
    from imagehub.checkpoints import load_weights

    Emu3Config, ReferenceLM, ReferenceVQConfig, ReferenceVQ = _reference_modules()
    root = Path(snapshot)
    config = Emu3Config(**json.loads((root / "config.json").read_text()))
    config.rope_scaling = None
    with torch.device("meta"):
        model = ReferenceLM(config)
    model.load_state_dict(load_weights(root), assign=True)
    _materialize_reference_rotary_buffers(model)
    model.eval()
    ids = torch.tensor([[151849, 64, 151852, 16, 9, 16, 151851]])
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits
    del model
    gc.collect()
    vision_config = ReferenceVQConfig(
        **json.loads((root / "vision_tokenizer/config.json").read_text())
    )
    with torch.device("meta"):
        vision = ReferenceVQ(vision_config)
    vision.load_state_dict(load_weights(root / "vision_tokenizer"), assign=True)
    vision.eval()
    with torch.no_grad():
        decoded = vision.decode(torch.zeros((1, 2, 2), dtype=torch.long))
    torch.save({"logits": logits, "decoded": decoded}, out_path)


def test_real_checkpoint_components(record, tmp_path):
    root = snapshot_dir("emu3-gen")
    expected = reference_in_subprocess(
        "test_ref_emu3_gen._reference_components",
        {"snapshot": str(root)},
        tmp_path / "reference-components.pt",
    )
    components = Emu3Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    ids = torch.tensor([[151849, 64, 151852, 16, 9, 16, 151851]])
    with torch.no_grad():
        logits, _ = components.model(ids)
        decoded = components.vision_tokenizer.decode(torch.zeros((1, 2, 2), dtype=torch.long))
    passed = torch.equal(logits, expected["logits"]) and torch.equal(decoded, expected["decoded"])
    record("emu3_gen_real_components", passed=passed, tolerance="exact")
    assert passed


def _reference_e2e(snapshot: str, seed: int, out_path: str) -> None:
    from imagehub.backends.emu3 import NEGATIVE_PROMPT, POSITIVE_SUFFIX, _sample, prompt_ids
    from imagehub.checkpoints import load_weights
    from imagehub.tokenization_emu3 import Emu3Tokenizer

    Emu3Config, ReferenceLM, ReferenceVQConfig, ReferenceVQ = _reference_modules()
    root = Path(snapshot)
    config_data = json.loads((root / "config.json").read_text())
    config = Emu3Config(**config_data)
    config.rope_scaling = None
    with torch.device("meta"):
        model = ReferenceLM(config)
    model.load_state_dict(load_weights(root), assign=True)
    _materialize_reference_rotary_buffers(model)
    model.eval()
    from transformers.cache_utils import DynamicCache

    tokenizer = Emu3Tokenizer(root)
    height, width = 2, 3
    conditional = prompt_ids(tokenizer, "a red cube" + POSITIVE_SUFFIX, height, width)
    unconditional = prompt_ids(tokenizer, NEGATIVE_PROMPT, height, width)
    generator = torch.Generator().manual_seed(seed)
    positive_cache = DynamicCache(config=config)
    negative_cache = DynamicCache(config=config)
    with torch.no_grad():
        positive_output = model(conditional, past_key_values=positive_cache, use_cache=True)
        negative_output = model(unconditional, past_key_values=negative_cache, use_cache=True)
    positive_logits = positive_output.logits[:, -1]
    negative_logits = negative_output.logits[:, -1]
    visual = []
    offset = 1
    while True:
        if offset % (width + 1) == 0:
            token = torch.tensor([[tokenizer.eol_id]])
        elif offset == (width + 1) * height + 1:
            token = torch.tensor([[tokenizer.eof_id]])
        elif offset == (width + 1) * height + 2:
            token = torch.tensor([[tokenizer.eoi_id]])
        elif offset == (width + 1) * height + 3:
            break
        else:
            logits = negative_logits + 3.0 * (positive_logits - negative_logits)
            sampled = _sample(
                logits[:, tokenizer.visual_start : tokenizer.visual_end + 1],
                1,
                1.0,
                generator,
            )
            visual.append(sampled)
            token = sampled + tokenizer.visual_start
        with torch.no_grad():
            positive_output = model(token, past_key_values=positive_cache, use_cache=True)
            negative_output = model(token, past_key_values=negative_cache, use_cache=True)
        positive_logits = positive_output.logits[:, -1]
        negative_logits = negative_output.logits[:, -1]
        offset += 1
    tokens = torch.cat(visual, 1).reshape(1, height, width)
    del model
    gc.collect()
    vision_data = json.loads((root / "vision_tokenizer/config.json").read_text())
    with torch.device("meta"):
        vision = ReferenceVQ(ReferenceVQConfig(**vision_data))
    vision.load_state_dict(load_weights(root / "vision_tokenizer"), assign=True)
    vision.eval()
    with torch.no_grad():
        decoded = vision.decode(tokens)
    pixels = decoded.float().add(1).div(2).clamp(0, 1).mul(255).to(torch.uint8)
    torch.save(pixels.permute(0, 2, 3, 1)[0], out_path)


@pytest.mark.parametrize(("seed", "key"), [(0, "e2e_emu3-gen"), (42, "seed_e2e_emu3-gen")])
def test_real_end_to_end(seed, key, record, tmp_path):
    root = snapshot_dir("emu3-gen")
    expected = reference_in_subprocess(
        "test_ref_emu3_gen._reference_e2e",
        {"snapshot": str(root), "seed": seed},
        tmp_path / f"reference-{seed}.pt",
    )
    ours = pipeline("t2i", backend="emu3", model=root, model_id="emu3-gen")
    actual = ours("a red cube", seed=seed, width=24, height=16, top_k=1).images[0].tensor
    passed = torch.equal(actual, expected)
    record(key, passed=passed, seed=seed, width=24, height=16, tolerance="exact")
    assert passed
