"""Exact random-weight parity with the pinned official Meissonic modules."""

import importlib.util
import sys

import pytest
import torch
from helpers import TINY_MEISSONIC_TRANSFORMER
from reference_sources import reference_source

from imagehub.models.meissonic_transformer import MeissonicTransformer2DModel
from imagehub.schedulers.amused import AmusedScheduler

pytestmark = pytest.mark.reference
pytest.importorskip("diffusers")
pytest.importorskip("transformers")

REFERENCE = reference_source("meissonic") / "src"


def _load(name):
    path = REFERENCE / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"Meissonic reference checkout not found at {REFERENCE}")
    spec = importlib.util.spec_from_file_location(f"meissonic_reference_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transformer_tiny_exact(record):
    module = _load("transformer")
    config = {
        key: value for key, value in TINY_MEISSONIC_TRANSFORMER.items() if not key.startswith("_")
    }
    torch.manual_seed(0)
    reference = module.Transformer2DModel(**config).eval()
    # diffusers 0.40 moved the processor's stored head dimension onto Attention.
    for block in [*reference.transformer_blocks, *reference.single_transformer_blocks]:
        block.attn.head_dim = config["attention_head_dim"]
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)
    model = MeissonicTransformer2DModel(config).eval()
    model.load_checkpoint(reference.state_dict())
    ids = torch.randint(0, 24, (2, 4, 6))
    text = torch.randn(2, 5, 32)
    pooled = torch.randn(2, 16)
    micro = torch.tensor([[64.0, 96.0, 0.0, 0.0, 6.0]]).expand(2, -1)
    image_ids = torch.zeros(24, 3)
    image_ids[:, 1] = torch.arange(4).repeat_interleave(6)
    image_ids[:, 2] = torch.arange(6).repeat(4)
    text_ids = torch.zeros(5, 3)
    timestep = torch.tensor([1, 1])
    with torch.inference_mode():
        expected = reference(
            hidden_states=ids,
            encoder_hidden_states=text,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=image_ids,
            txt_ids=text_ids,
            micro_conds=micro,
        )
        actual = model(ids, text, pooled, timestep, image_ids, text_ids, micro)
    difference = float((actual - expected).abs().max())
    record("meissonic_transformer_tiny", max_abs_diff=difference)
    assert torch.equal(actual, expected)


def test_scheduler_trajectory_exact(record):
    module = _load("scheduler")
    reference = module.Scheduler(mask_token_id=20, masking_schedule="cosine")
    ours = AmusedScheduler({"mask_token_id": 20, "masking_schedule": "cosine"})
    reference.set_timesteps(5, (2, 0), "cpu")
    ours.set_timesteps(5, (2, 0), "cpu")
    expected = torch.full((2, 3, 4), 20)
    actual = expected.clone()
    ref_gen = torch.Generator("cpu").manual_seed(7)
    our_gen = torch.Generator("cpu").manual_seed(7)
    logits_gen = torch.Generator("cpu").manual_seed(8)
    for timestep in reference.timesteps:
        logits = torch.randn(2, 20, 3, 4, generator=logits_gen)
        expected = reference.step(logits, timestep, expected, generator=ref_gen).prev_sample
        actual = ours.step(logits, timestep, actual, generator=our_gen).prev_sample
        assert torch.equal(actual, expected)
    record("meissonic_scheduler", trajectory_equal=True)
