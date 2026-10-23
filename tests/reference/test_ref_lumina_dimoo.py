"""Exact parity with the official Lumina-DiMOO implementation and checkpoint."""

import gc
import importlib
import random
import sys
from pathlib import Path

import pytest
import torch
from helpers import max_diff, reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.backends.lumina_dimoo import (
    ANSWER_END_ID,
    ANSWER_START_ID,
    BEGIN_IMAGE_ID,
    END_IMAGE_ID,
    MASK_TOKEN_ID,
    decode_tokens,
    format_prompt,
    generate_tokens,
)
from imagehub.checkpoints import read_json
from imagehub.models.llada import LLaDAForMultiModalGeneration
from imagehub.models.vq_model import VQModel
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")

MODEL_ID = "lumina-dimoo"
REFERENCE_ROOT = Path("/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/Lumina-DiMOO")
PROMPT = "a red cube on a white table"
TINY_CONFIG = {
    "d_model": 32,
    "embedding_size": 134548,
    "vocab_size": 134548,
    "n_heads": 4,
    "n_kv_heads": 4,
    "n_layers": 2,
    "mlp_hidden_size": 48,
    "max_sequence_length": 128,
    "init_device": "cpu",
    "include_bias": False,
    "include_qkv_bias": False,
    "activation_type": "silu",
    "block_type": "llama",
    "layer_norm_type": "rms",
    "rope": True,
    "rope_full_precision": True,
    "rope_theta": 500000.0,
    "attention_dropout": 0.0,
    "embedding_dropout": 0.0,
    "residual_dropout": 0.0,
    "rms_norm_eps": 1e-5,
    "weight_tying": False,
    "input_emb_norm": False,
    "attention_layer_norm": False,
    "block_group_size": 1,
    "alibi": False,
    "scale_logits": False,
}


@pytest.fixture(scope="module")
def official():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"Lumina-DiMOO reference checkout not found at {REFERENCE_ROOT}")
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        configuration = importlib.import_module("model.configuration_llada")
        modeling = importlib.import_module("model.modeling_xllmx_dimoo")
        generation = importlib.import_module("generators.image_generation_generator")
        yield configuration, modeling, generation
    finally:
        sys.path.remove(str(REFERENCE_ROOT))
        for name in list(sys.modules):
            if name == "model" or name.startswith(("model.", "generators.", "utils.")):
                del sys.modules[name]


def test_published_headers(record):
    fixtures = Path(__file__).parents[1] / "fixtures/lumina_dimoo_configs"
    result = {}
    for name, cls in {
        "model": LLaDAForMultiModalGeneration,
        "vqvae": VQModel,
    }.items():
        config = read_json(fixtures / name / "config.json")
        shapes = read_json(fixtures / name / "header.json")
        with torch.device("meta"):
            model = cls(config)
        expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        actual = {key: tuple(value) for key, value in shapes.items()}
        result[name] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                key for key in expected.keys() & actual.keys() if expected[key] != actual[key]
            ),
            "tensors": len(actual),
        }
    record("header_lumina-dimoo", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def _randomized_pair(configuration, modeling, config=TINY_CONFIG):
    reference = modeling.LLaDAForMultiModalGeneration(configuration.LLaDAConfig(**config))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(generator=generator)
    ours = LLaDAForMultiModalGeneration(config)
    ours.load_state_dict(reference.state_dict())
    return reference.eval(), ours.eval()


def test_random_weight_llada_logits_are_exact(official, record):
    configuration, modeling, _ = official
    reference, ours = _randomized_pair(configuration, modeling)
    ids = torch.randint(0, 134548, (2, 13), generator=torch.Generator().manual_seed(2))
    with torch.inference_mode():
        expected = reference(ids, infer=True).logits
        actual = ours(ids)
    difference = max_diff(actual, expected)
    record("lumina_dimoo_llada_random_weights", **difference, tolerance=0.0)
    assert torch.equal(actual, expected)


def test_published_geometry_random_weight_logits_are_exact(official, record):
    configuration, modeling, _ = official
    config = TINY_CONFIG | {
        "d_model": 4096,
        "embedding_size": 128,
        "vocab_size": 128,
        "n_heads": 32,
        "n_kv_heads": 32,
        "n_layers": 1,
        "mlp_hidden_size": 12288,
    }
    reference, ours = _randomized_pair(configuration, modeling, config)
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        expected = reference(ids, infer=True).logits
        actual = ours(ids)
    difference = max_diff(actual, expected)
    record("lumina_dimoo_llada_published_geometry", **difference, tolerance=0.0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(("guidance", "seed"), [(0.0, 0), (2.0, 42)])
def test_tiny_end_to_end_tokens_are_exact(official, record, guidance, seed):
    configuration, modeling, generation = official
    reference, ours = _randomized_pair(configuration, modeling)
    conditional = torch.tensor(
        [
            [
                1,
                2,
                ANSWER_START_ID,
                BEGIN_IMAGE_ID,
                MASK_TOKEN_ID,
                MASK_TOKEN_ID,
                126084,
                MASK_TOKEN_ID,
                MASK_TOKEN_ID,
                126084,
                END_IMAGE_ID,
                ANSWER_END_ID,
            ]
        ]
    )
    unconditional = torch.tensor([[3, 4]])
    expected = (
        generation.generate_image(
            reference,
            conditional.clone(),
            seq_len=4,
            newline_every=2,
            timesteps=3,
            temperature=1.0,
            cfg_scale=guidance,
            uncon_ids=unconditional,
            code_start=4,
            generator=torch.Generator().manual_seed(seed),
        )
        - 126356
    )
    actual = generate_tokens(
        ours,
        conditional,
        unconditional,
        code_start=4,
        image_tokens=4,
        steps=3,
        guidance=guidance,
        temperature=1.0,
        generator=torch.Generator().manual_seed(seed),
    )
    record(f"lumina_dimoo_tiny_e2e_g{guidance:g}", tokens_equal=torch.equal(actual, expected))
    assert torch.equal(actual, expected)


def test_real_tokenizer(record):
    root = snapshot_dir(MODEL_ID)
    ours = Qwen2Tokenizer(root)
    reference = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    prompts = [PROMPT, *format_prompt(PROMPT), "café 東京\n<uncondition>"]
    for text in prompts:
        assert ours.tokenize(text) == reference.encode(text)
    record("lumina_dimoo_tokenizer", prompts=len(prompts), mismatches=0, tolerance="exact")


def test_real_tokenizer_seeded_fuzz(record):
    root = snapshot_dir(MODEL_ID)
    ours = Qwen2Tokenizer(root)
    reference = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    pieces = ["cat", " ", "\n", "café", "東京", "🦊", "<system>", "<uncondition>", "123", "—"]
    generator = random.Random(0)
    prompts = [
        "".join(generator.choice(pieces) for _ in range(generator.randrange(20)))
        for _ in range(200)
    ]
    for text in prompts:
        assert ours.tokenize(text) == reference.encode(text)
    record("lumina_dimoo_tokenizer_fuzz", prompts=len(prompts), mismatches=0, tolerance="exact")


def _load_official_model(modeling, root: Path):
    model_class = modeling.LLaDAForMultiModalGeneration
    # Transformers 5 expects bookkeeping absent from the pinned official wrapper.
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    if not getattr(model_class.tie_weights, "_imagehub_transformers_5_compatible", False):
        original_tie_weights = model_class.tie_weights

        def tie_weights(self, **_kwargs):
            return original_tie_weights(self)

        tie_weights._imagehub_transformers_5_compatible = True
        model_class.tie_weights = tie_weights
    return model_class.from_pretrained(
        root, torch_dtype=torch.float32, local_files_only=True
    ).eval()


def test_real_model_logits(record):
    root = snapshot_dir(MODEL_ID)
    ours = LLaDAForMultiModalGeneration.from_pretrained(root, dtype=torch.float32)
    ids = torch.tensor([[126080, 126354, 126349, MASK_TOKEN_ID, 126350, 126355]])
    actual = ours(ids)
    del ours
    gc.collect()
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        modeling = importlib.import_module("model.modeling_xllmx_dimoo")
        reference = _load_official_model(modeling, root)
        with torch.inference_mode():
            expected = reference(ids, infer=True).logits
    finally:
        sys.path.remove(str(REFERENCE_ROOT))
    difference = max_diff(actual, expected)
    record("lumina_dimoo_real_llada", **difference, tolerance=0.0)
    assert torch.equal(actual, expected)


def test_real_vq_decode(record):
    root = snapshot_dir(MODEL_ID)
    ids = torch.tensor([[0, 1], [2, 3]]).reshape(1, 4)
    ours = VQModel.from_pretrained(root / "vqvae", dtype=torch.float32)
    actual = decode_tokens(ours, ids, 32, 32)
    del ours
    gc.collect()
    reference = diffusers.VQModel.from_pretrained(
        root, subfolder="vqvae", local_files_only=True, torch_dtype=torch.float32
    )
    expected = reference.decode(
        ids, force_not_quantize=True, shape=(1, 2, 2, reference.config.latent_channels)
    ).sample.clip(0, 1)
    expected = expected.permute(0, 2, 3, 1).float().cpu().mul(255).round().to(torch.uint8)
    difference = max_diff(actual.float(), expected.float())
    record("lumina_dimoo_real_vq", **difference, tolerance=0.0)
    assert torch.equal(actual, expected)


def _official_real_image(snapshot: str, prompt: str, seed: int, out_path: str) -> None:
    root = Path(snapshot)
    sys.path.insert(0, str(REFERENCE_ROOT))
    modeling = importlib.import_module("model.modeling_xllmx_dimoo")
    generation = importlib.import_module("generators.image_generation_generator")
    tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = _load_official_model(modeling, root)
    conditional_text, unconditional_text = format_prompt(prompt)
    conditional = tokenizer.encode(conditional_text)
    unconditional = torch.tensor([tokenizer.encode(unconditional_text)])
    image = [MASK_TOKEN_ID, MASK_TOKEN_ID, 126084, MASK_TOKEN_ID, MASK_TOKEN_ID, 126084]
    suffix = [ANSWER_START_ID, BEGIN_IMAGE_ID, *image, END_IMAGE_ID, ANSWER_END_ID]
    inputs = torch.tensor([conditional + suffix])
    tokens = (
        generation.generate_image(
            model,
            inputs,
            seq_len=4,
            newline_every=2,
            timesteps=2,
            temperature=1.0,
            cfg_scale=4.0,
            uncon_ids=unconditional,
            code_start=len(conditional) + 2,
            generator=torch.Generator().manual_seed(seed),
        )
        - 126356
    )
    del model
    gc.collect()
    vqvae = diffusers.VQModel.from_pretrained(
        root, subfolder="vqvae", local_files_only=True, torch_dtype=torch.float32
    )
    image = vqvae.decode(
        tokens, force_not_quantize=True, shape=(1, 2, 2, vqvae.config.latent_channels)
    ).sample.clip(0, 1)
    pixels = image.permute(0, 2, 3, 1).float().cpu().mul(255).round().to(torch.uint8)
    torch.save(pixels, out_path)


@pytest.mark.parametrize(
    ("seed", "record_name"), [(0, "e2e_lumina-dimoo"), (42, "seed_e2e_lumina-dimoo")]
)
def test_real_short_e2e(record, tmp_path, seed, record_name):
    root = snapshot_dir(MODEL_ID)
    expected = reference_in_subprocess(
        "test_ref_lumina_dimoo._official_real_image",
        {"snapshot": str(root), "prompt": PROMPT, "seed": seed},
        tmp_path / f"reference-{seed}.pt",
    )
    result = pipeline("t2i", backend="lumina-dimoo", model=root)(
        PROMPT, width=32, height=32, steps=2, guidance=4.0, seed=seed
    )
    actual = result.images[0].tensor.unsqueeze(0)
    passed = torch.equal(actual, expected)
    record(record_name, passed=passed, width=32, height=32, steps=2, seed=seed)
    assert passed
