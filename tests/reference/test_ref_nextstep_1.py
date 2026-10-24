"""Exact NextStep-1 parity against the pinned official implementation."""

import importlib.util
from pathlib import Path

import pytest
import torch
from helpers import reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.models.nextstep import FlowNetwork, NextStepModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

REFERENCE_ROOT = Path("/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/NextStep-1")
PROMPT = "A small red boat on a calm blue lake."


def _tiny_config():
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "latent_channels": 2,
        "latent_patch_size": 2,
        "fm_head_dim": 16,
        "fm_head_layers": 2,
        "bos_token_id": 1,
        "boi": 2,
        "image_placeholder_id": 3,
    }


def test_random_qwen_decoder_is_exact(record):
    config = _tiny_config()
    reference = transformers.Qwen2Model(transformers.Qwen2Config(**config)).eval()
    ours = NextStepModel(config).eval()
    ours.load_state_dict(reference.state_dict(), strict=False)
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(0))
    mask = torch.tensor([[1] * 7, [0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    with torch.no_grad():
        expected = reference(ids, attention_mask=mask).last_hidden_state
        actual, _ = ours(ours.embed(ids), mask)
    passed = torch.equal(actual, expected)
    record("nextstep_qwen_random", passed=passed, tolerance="exact")
    assert passed


def test_random_flow_head_is_exact(record):
    if not REFERENCE_ROOT.is_dir():
        pytest.skip("NextStep-1 reference checkout not found")
    spec = importlib.util.spec_from_file_location(
        "nextstep_reference", REFERENCE_ROOT / "inference" / "nextstep_model.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(1)
    reference = module.SimpleMLPAdaLN(8, 32, dim=16, layers=2).eval()
    ours = FlowNetwork(8, 32, 16, 2).eval()
    ours.load_state_dict(reference.state_dict())
    value, timestep, condition = torch.randn(3, 8), torch.rand(3), torch.randn(3, 32)
    with torch.no_grad():
        expected = reference(value, timestep, condition)
        actual = ours(value, timestep, condition)
    passed = torch.equal(actual, expected)
    record("nextstep_flow_random", passed=passed, tolerance="exact")
    assert passed


def _reference_e2e(snapshot, seed, out_path):
    import sys

    sys.path.insert(0, str(REFERENCE_ROOT))
    from inference.gen_pipeline import NextStepPipeline

    result = NextStepPipeline(snapshot, device="cpu", dtype=torch.float32, local_files_only=True)
    image = result.generate_image(
        PROMPT, hw=(64, 64), num_sampling_steps=1, seed=seed, progress=False
    )[0]
    torch.save(
        torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(64, 64, 3), out_path
    )


@pytest.mark.parametrize("seed", [0, 42])
def test_real_e2e(seed, tmp_path, record):
    root = snapshot_dir("nextstep-1")
    expected = reference_in_subprocess(
        "test_ref_nextstep_1._reference_e2e",
        {"snapshot": str(root), "seed": seed},
        tmp_path / f"reference-{seed}.pt",
    )
    actual = (
        pipeline("t2i", backend="nextstep-1-large", model=root)(
            PROMPT, width=64, height=64, steps=1, seed=seed
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    record(
        ("e2e_nextstep-1-large" if seed == 0 else "seed_e2e_nextstep-1-large"),
        passed=passed,
        seed=seed,
        width=64,
        height=64,
        steps=1,
        tolerance="exact",
    )
    assert passed
