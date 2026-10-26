"""GLM-Image sigma schedule and ByT5 tokenizer parity with the reference libraries."""

import random
import shutil
from pathlib import Path

import pytest
import torch

from imagehub.backends.cogview4 import calculate_shift
from imagehub.backends.glm_image import glm_image_sigmas
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_byt5 import ByT5Tokenizer

pytestmark = pytest.mark.reference
FIXTURES = Path(__file__).parents[1] / "fixtures" / "glm_image_configs"


def test_timesteps_and_sigmas_match_diffusers_exactly():
    np = pytest.importorskip("numpy")
    diffusers = pytest.importorskip("diffusers")
    import json

    config = json.loads((FIXTURES / "scheduler_scheduler_config.json").read_text(encoding="utf-8"))
    for steps in (1, 2, 7, 28, 50, 1000):
        for height, width in ((1024, 1024), (1152, 768), (512, 2048)):
            reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
            timesteps = np.linspace(config["num_train_timesteps"], 1.0, steps + 1)[:-1]
            timesteps = timesteps.astype(np.int64).astype(np.float32)
            sigmas = timesteps / config["num_train_timesteps"]
            seq_len = (height // 8) * (width // 8) // 4
            mu = calculate_shift(seq_len, config)
            reference.set_timesteps(steps, sigmas=sigmas, timesteps=timesteps, mu=mu)
            ours = FlowMatchEulerDiscreteScheduler(config)
            ours.set_timesteps(
                steps, sigmas=glm_image_sigmas(steps, config["num_train_timesteps"]), mu=mu
            )
            assert torch.equal(ours.timesteps, reference.timesteps)
            assert torch.equal(ours.sigmas, reference.sigmas)


def test_byt5_tokenizer_matches_transformers(tmp_path):
    transformers = pytest.importorskip("transformers")
    for name in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
        shutil.copy(FIXTURES / f"tokenizer_{name}", tmp_path / name.removeprefix(""))
    reference = transformers.ByT5Tokenizer.from_pretrained(str(tmp_path))
    ours = ByT5Tokenizer(tmp_path)
    pieces = [
        "a", "Hello", " ", "  ", "\n", "é", "日本", "\U0001f600", "<pad>", "</s>",
        "<unk>", "<extra_id_0>", "<extra_id_12>", "<extra_id_124>", "<extra_id_9", "x",
    ]  # fmt: skip
    rng = random.Random(0)
    texts = ["", "a cat", "don't “panic”", "x" * 3000]
    texts += ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 8))) for _ in range(1500)]
    for text in texts:
        for max_length in (2048, 5):
            expected = reference([text], max_length=max_length, truncation=True).input_ids[0]
            assert ours.tokenize(text, max_length) == expected, (text, max_length)
