"""PixelDiT parity against the pinned official implementation and local checkpoint."""

# Derivative of NVlabs PixelDiT under NSCLv1; non-commercial. See licenses/pixeldit.

import gc
import json
import sys

import pytest
import torch
from fixture_files import fixture_path
from helpers import max_diff, snapshot_dir
from reference_sources import reference_source

from imagehub import pipeline
from imagehub.backends.pixeldit import sample_flow_dpm
from imagehub.checkpoints import load_weights
from imagehub.models.gemma2 import Gemma2Model
from imagehub.models.pixeldit import PIXELDIT_DEFAULTS, PixelDiT
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
np = pytest.importorskip("numpy")
transformers = pytest.importorskip("transformers")

REFERENCE_ROOT = reference_source("pixeldit")
FIXTURES = fixture_path("pixeldit_configs")
PROMPT = "a blue ceramic cat on a wooden table"
TINY_CONFIG = {
    "in_channels": 3,
    "patch_size": 2,
    "num_groups": 4,
    "hidden_size": 32,
    "pixel_hidden_size": 8,
    "pixel_attn_hidden_size": 32,
    "pixel_num_groups": 4,
    "patch_depth": 2,
    "pixel_depth": 1,
    "num_text_blocks": 1,
    "txt_embed_dim": 12,
    "txt_max_length": 5,
    "use_text_rope": True,
    "text_rope_theta": 10000.0,
    "repa_encoder_index": -1,
    "use_pixel_abs_pos": True,
    "pit_adaln_post_modulation": False,
}


@pytest.fixture(scope="module")
def official():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"PixelDiT reference checkout not found at {REFERENCE_ROOT}")
    sys.path.insert(0, str(REFERENCE_ROOT))
    sys.path.insert(0, str(REFERENCE_ROOT / "t2i"))
    try:
        from diffusion import DPMS
        from pixdit_core.pixeldit_t2i import PixDiT_T2I

        yield PixDiT_T2I, DPMS
    finally:
        sys.path.remove(str(REFERENCE_ROOT / "t2i"))
        sys.path.remove(str(REFERENCE_ROOT))
        for name in list(sys.modules):
            if name == "pixdit_core" or name.startswith(("pixdit_core.", "diffusion")):
                del sys.modules[name]


def test_fixture_headers(record):
    config = json.loads((FIXTURES / "model_config.json").read_text())
    headers = json.loads((FIXTURES / "model_headers" / "pixeldit_t2i_v1.json").read_text())
    with torch.device("meta"):
        model = PixelDiT(config)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    actual = {name: tuple(value["shape"]) for name, value in headers.items()}
    result = {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual),
    }
    record("pixeldit_fixture_header", **result)
    assert not result["missing"] and not result["extra"] and not result["mismatch"]


def test_random_transformer_exact(official, record):
    reference_class, _ = official
    torch.manual_seed(1)
    reference = reference_class(**TINY_CONFIG).eval()
    ours = PixelDiT(TINY_CONFIG).eval()
    ours.core.load_state_dict(reference.state_dict())
    generator = torch.Generator().manual_seed(2)
    image = torch.randn(1, 3, 4, 6, generator=generator)
    timestep = torch.tensor([0.4])
    text = torch.randn(1, 5, 12, generator=generator)
    expected = reference(image, timestep, text)
    actual = ours(image, timestep, text)
    passed = torch.equal(actual, expected)
    record("pixeldit_transformer_random", passed=passed, max_abs_diff=0.0, tolerance="exact")
    assert passed


class _ToyDenoiser:
    def __call__(self, x, timestep, text, mask=None):
        return (
            x * 0.1
            + timestep[:, None, None, None] * 0.0002
            + text.mean((1, 2))[:, None, None, None] * 0.03
        )


def test_flow_dpm_solver_exact(official, record):
    _, solver_factory = official
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(1, 3, 4, 4, generator=generator)
    condition = torch.randn(1, 5, 6, generator=generator)
    unconditional = torch.randn(1, 5, 6, generator=generator)
    reference = solver_factory(
        _ToyDenoiser(),
        condition=condition,
        uncondition=unconditional,
        cfg_scale=2.75,
        model_type="flow",
        schedule="FLOW",
        interval_guidance=[0, 1],
    ).sample(
        noise.clone(),
        steps=4,
        order=2,
        skip_type="time_uniform_flow",
        method="multistep",
        flow_shift=4.0,
    )
    actual = sample_flow_dpm(
        _ToyDenoiser(), noise, condition, unconditional, steps=4, guidance=2.75
    )
    passed = torch.equal(actual, reference)
    record("pixeldit_flow_dpm_random", passed=passed, max_abs_diff=0.0, tolerance="exact")
    assert passed


def test_tiny_end_to_end_exact(official, record):
    reference_class, solver_factory = official
    torch.manual_seed(4)
    reference = reference_class(**TINY_CONFIG).eval()
    ours = PixelDiT(TINY_CONFIG).eval()
    ours.core.load_state_dict(reference.state_dict())
    generator = torch.Generator().manual_seed(5)
    noise = torch.randn(1, 3, 4, 4, generator=generator)
    condition = torch.randn(1, 5, 12, generator=generator)
    unconditional = torch.randn(1, 5, 12, generator=generator)
    expected = solver_factory(
        reference,
        condition=condition,
        uncondition=unconditional,
        cfg_scale=2.75,
        model_type="flow",
        schedule="FLOW",
        interval_guidance=[0, 1],
    ).sample(
        noise.clone(),
        steps=4,
        order=2,
        skip_type="time_uniform_flow",
        method="multistep",
        flow_shift=4.0,
    )
    actual = sample_flow_dpm(ours, noise, condition, unconditional, steps=4, guidance=2.75)
    expected_pixels = ((expected + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
    actual_pixels = ((actual + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
    passed = torch.equal(actual_pixels, expected_pixels)
    record("pixeldit_tiny_e2e", passed=passed, max_abs_diff=0, tolerance="exact")
    assert passed


def test_real_tokenizer_and_text_encoder(record):
    root = snapshot_dir("pixeldit")
    text_root = root / "text_encoder"
    ours_tokenizer = GemmaTokenizer(text_root)
    reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
        text_root, local_files_only=True
    )
    reference_tokenizer.padding_side = "right"
    cases = [PROMPT, "Café — 東京", "  spaced   text  ", "🐱"]
    token_match = True
    for text in cases:
        ids, mask = ours_tokenizer(text, 300, padding_side="right")
        expected = reference_tokenizer(
            text, max_length=300, padding="max_length", truncation=True, return_tensors="pt"
        )
        token_match &= torch.equal(ids, expected.input_ids)
        token_match &= torch.equal(mask, expected.attention_mask.bool())
    encoded = reference_tokenizer(
        PROMPT, max_length=32, padding="max_length", truncation=True, return_tensors="pt"
    )
    reference_model = transformers.Gemma2Model.from_pretrained(
        text_root, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    expected_hidden = reference_model(**encoded).last_hidden_state
    del reference_model
    gc.collect()
    config = json.loads((text_root / "config.json").read_text())
    with torch.device("meta"):
        ours_model = Gemma2Model(config)
    state = load_weights(text_root, skip_prefixes=("lm_head.",))
    ours_model.load_checkpoint(
        {
            name.removeprefix("model."): value
            for name, value in state.items()
            if name.startswith("model.")
        }
    )
    ours_model.init_non_persistent_buffers()
    ours_model = ours_model.float().eval()
    actual_hidden = ours_model(encoded.input_ids, encoded.attention_mask)
    passed = token_match and torch.equal(actual_hidden, expected_hidden)
    record(
        "pixeldit_text_encoder",
        passed=passed,
        **max_diff(actual_hidden, expected_hidden),
        tolerance="exact",
    )
    assert passed


def test_real_denoiser_step_exact(official, record):
    root = snapshot_dir("pixeldit")
    reference_class, _ = official
    config = json.loads((root / "config.json").read_text())
    state = load_weights(root / "pixeldit_t2i_v1.pth", container_key="state_dict")
    reference_config = PIXELDIT_DEFAULTS | {
        key: value for key, value in config.items() if key in PIXELDIT_DEFAULTS
    }
    reference = reference_class(**reference_config).eval()
    reference.load_state_dict(
        {
            name.removeprefix("core."): value
            for name, value in state.items()
            if name.startswith("core.")
        }
    )
    generator = torch.Generator().manual_seed(6)
    image = torch.randn(1, 3, 16, 16, generator=generator)
    text = torch.randn(1, 8, 2304, generator=generator)
    expected = reference(image, torch.tensor([0.5]), text)
    del reference
    gc.collect()
    with torch.device("meta"):
        ours = PixelDiT(config)
    ours.load_checkpoint(state)
    ours = ours.float().eval()
    actual = ours(image, torch.tensor([0.5]), text)
    passed = torch.equal(actual, expected)
    record("pixeldit_denoiser_step", passed=passed, **max_diff(actual, expected), tolerance="exact")
    assert passed


@pytest.mark.parametrize(("seed", "record_name"), [(0, "e2e_pixeldit"), (42, "seed_e2e_pixeldit")])
def test_real_short_public_pipeline(seed, record_name, record, official):
    root = snapshot_dir("pixeldit")
    reference_class, solver_factory = official
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "text_encoder", local_files_only=True
    )
    tokenizer.padding_side = "right"
    text_encoder = (
        transformers.AutoModelForCausalLM.from_pretrained(
            root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
        )
        .get_decoder()
        .eval()
    )
    positive_tokens = tokenizer(
        PROMPT, max_length=300, padding="max_length", truncation=True, return_tensors="pt"
    )
    negative_tokens = tokenizer(
        "", max_length=300, padding="max_length", truncation=True, return_tensors="pt"
    )
    select = [0, *range(-299, 0)]
    condition = text_encoder(**positive_tokens).last_hidden_state[:, select]
    unconditional = text_encoder(**negative_tokens).last_hidden_state[:, select]
    del text_encoder, tokenizer
    gc.collect()
    config = json.loads((root / "config.json").read_text())
    reference_config = PIXELDIT_DEFAULTS | {
        key: value for key, value in config.items() if key in PIXELDIT_DEFAULTS
    }
    reference = reference_class(**reference_config).eval()
    state = load_weights(root / "pixeldit_t2i_v1.pth", container_key="state_dict")
    reference.load_state_dict(
        {
            name.removeprefix("core."): value
            for name, value in state.items()
            if name.startswith("core.")
        }
    )
    noise = torch.randn((1, 3, 16, 16), generator=torch.Generator().manual_seed(seed))
    expected = solver_factory(
        reference,
        condition=condition,
        uncondition=unconditional,
        cfg_scale=2.75,
        model_type="flow",
        schedule="FLOW",
        interval_guidance=[0, 1],
    ).sample(
        noise,
        steps=2,
        order=2,
        skip_type="time_uniform_flow",
        method="multistep",
        flow_shift=4.0,
    )
    expected = (
        ((expected + 1) / 2).clamp(0, 1).mul(255).round().permute(0, 2, 3, 1).to(torch.uint8)[0]
    )
    del reference, state, condition, unconditional
    gc.collect()
    with pipeline("t2i", backend="pixeldit", model=root) as pipe:
        actual = (
            pipe(
                PROMPT,
                seed=seed,
                steps=2,
                width=16,
                height=16,
                guidance=2.75,
                negative_prompt="",
                use_chi_prompt=False,
            )
            .images[0]
            .tensor
        )
    passed = torch.equal(actual, expected)
    record(record_name, passed=passed, max_abs_diff=0, tolerance="exact")
    assert passed
