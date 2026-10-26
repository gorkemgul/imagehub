"""Show-o 512 real-checkpoint parity against the official ``Showo`` pipeline class."""

import gc
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import max_diff, snapshot_dir
from test_ref_show_o_random import _official_magvit, _official_phi, _official_showo

from imagehub import pipeline
from imagehub.backends.show_o import (
    IMAGE_TOKENS,
    MASK_TOKEN_ID,
    _load_component,
    create_attention_mask,
    prompt_ids,
)
from imagehub.checkpoints import load_weights
from imagehub.models.magvit_v2 import MAGVITv2
from imagehub.models.show_o import ShowOModel
from imagehub.tokenization_codegen import CodeGenTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
pytest.importorskip("numpy")

MODEL_ID = "show-o-512"
PROMPT = "a photograph of an astronaut riding a horse on the moon"
SEED_PROMPT = "a red cat reading a book in a sunlit library"
SPECIAL_TOKENS = (
    "<|soi|>",
    "<|eoi|>",
    "<|sov|>",
    "<|eov|>",
    "<|t2i|>",
    "<|mmu|>",
    "<|t2v|>",
    "<|v2v|>",
    "<|lvg|>",
)
REDUCED_STEPS = 2


@pytest.fixture(scope="module")
def root():
    return snapshot_dir(MODEL_ID)


def _shapes(path: Path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(value["shape"]) for name, value in header.items() if name != "__metadata__"}


def _entry(expected, actual):
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual),
    }


def _reference_phi(root: Path):
    official = _official_phi()
    config_data = json.loads((root / "tokenizer/config.json").read_text(encoding="utf-8"))
    config_data.update(vocab_size=58498, qk_layernorm=True, use_cache=False)
    config = transformers.PhiConfig(**config_data)
    config.rope_theta = 10000.0
    config.rope_scaling = None
    config._attn_implementation = "eager"
    model = official.PhiForCausalLM(config).eval()
    state = {
        name.removeprefix("showo."): tensor
        for name, tensor in load_weights(root / "pytorch_model.safetensors").items()
    }
    model.load_state_dict(state)
    return model


def _reference_vq(root: Path):
    model = _official_magvit().MAGVITv2().eval()
    model.load_state_dict(load_weights(root / "vq/pytorch_model.safetensors"))
    return model


def test_header(root, record):
    result = {}
    for name, model_cls, path in (
        ("showo", ShowOModel, root / "pytorch_model.safetensors"),
        ("vq", MAGVITv2, root / "vq/pytorch_model.safetensors"),
    ):
        with torch.device("meta"):
            model = model_cls({})
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        result[name] = _entry(expected, _shapes(path))
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not item[key] for item in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(root, record):
    native = CodeGenTokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True, padding_side="left"
    )
    reference.add_special_tokens({"pad_token": "[PAD]"})
    reference.add_tokens(list(SPECIAL_TOKENS))
    prompts = ["", PROMPT, "naïve café 日本語", "line1\nline2", "   indented\t\t"]
    for prompt in prompts:
        assert native.tokenize(prompt) == reference(prompt, add_special_tokens=False).input_ids
    assert native.pad_id == reference.pad_token_id
    assert native.special_ids == {
        token: reference.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS
    }
    record("show_o_tokenizer", prompts=len(prompts), identical=True)


def test_phi_logits(root, record, tmp_path):
    ids = torch.randint(0, 58498, (1, 7), generator=torch.Generator().manual_seed(3))
    mask = torch.zeros(1, 1, 7, 7)
    reference = _reference_phi(root)
    with torch.inference_mode():
        expected = reference(ids, attention_mask=mask, use_cache=False).logits
    output = tmp_path / "phi.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = _load_component(ShowOModel, root, torch.device("cpu"), torch.float32)
    with torch.inference_mode():
        actual = native(ids, mask)
    expected = torch.load(output, weights_only=True)
    measured = max_diff(actual, expected)
    record("show_o_phi_logits", **measured, tolerance=0.0)
    assert torch.equal(actual, expected)


def test_vq_decode(root, record, tmp_path):
    tokens = torch.tensor([[0, 1, 2, 8191]])
    reference = _reference_vq(root)
    with torch.inference_mode():
        expected = reference.decode_code(tokens)
    output = tmp_path / "vq.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = _load_component(MAGVITv2, root / "vq", torch.device("cpu"), torch.float32)
    with torch.inference_mode():
        actual = native.decode_code(tokens)
    expected = torch.load(output, weights_only=True)
    measured = max_diff(actual, expected)
    record("show_o_vq_decode", **measured, tolerance=0.0)
    assert torch.equal(actual, expected)


def _reference_pixels(root: Path, prompt: str, seed: int, output: Path):
    tokenizer = CodeGenTokenizer(root / "tokenizer")
    image = torch.full((IMAGE_TOKENS,), MASK_TOKEN_ID)
    conditional = prompt_ids(tokenizer, prompt, image)
    unconditional = prompt_ids(tokenizer, "", image)
    attention = create_attention_mask(
        torch.cat((conditional, unconditional)),
        pad_id=tokenizer.pad_id,
        soi_id=tokenizer.special_ids["<|soi|>"],
        eoi_id=tokenizer.special_ids["<|eoi|>"],
    )
    phi = _reference_phi(root)

    class Adapter:
        config = SimpleNamespace(mask_token_id=MASK_TOKEN_ID)

        def __init__(self, model):
            self.model = model

        def __call__(self, input_ids, attention_mask=None):
            return self.model(input_ids, attention_mask=attention_mask, use_cache=False).logits

    config = SimpleNamespace(
        model=SimpleNamespace(
            showo=SimpleNamespace(
                num_vq_tokens=IMAGE_TOKENS,
                num_new_special_tokens=10,
                llm_vocab_size=50295,
                codebook_size=8192,
            )
        ),
        dataset=SimpleNamespace(preprocessing=SimpleNamespace(max_seq_length=128)),
    )
    with torch.inference_mode():
        tokens = (
            _official_showo()
            .Showo.t2i_generate(
                Adapter(phi),
                input_ids=conditional,
                uncond_input_ids=unconditional,
                attention_mask=attention,
                temperature=1.0,
                timesteps=REDUCED_STEPS,
                guidance_scale=10.0,
                generator=torch.Generator("cpu").manual_seed(seed),
                config=config,
            )
            .clamp(0, 8191)
        )
    del phi
    gc.collect()
    vq = _reference_vq(root)
    with torch.inference_mode():
        decoded = vq.decode_code(tokens)
        pixels = ((decoded + 1) / 2).clamp(0, 1).mul(255).permute(0, 2, 3, 1).to(torch.uint8)
    torch.save(pixels, output)
    del vq, decoded, pixels
    gc.collect()


@pytest.mark.parametrize(
    ("name", "prompt", "seed"),
    [("e2e_show-o-512", PROMPT, 0), ("seed_e2e_show-o-512", SEED_PROMPT, 42)],
)
def test_end_to_end(root, record, tmp_path, name, prompt, seed):
    output = tmp_path / f"reference-{seed}.pt"
    _reference_pixels(root, prompt, seed, output)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(prompt, steps=REDUCED_STEPS, seed=seed)
    actual = torch.stack([image.tensor for image in result.images])
    expected = torch.load(output, weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        name,
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        steps=REDUCED_STEPS,
        seed=seed,
    )
    assert passed
