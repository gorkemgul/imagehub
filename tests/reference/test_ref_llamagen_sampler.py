"""The LlamaGen sampling functions against the official ones with random logits and seeds."""

import pytest
import torch

from imagehub.backends.llamagen import sample, top_k_top_p_filtering

pytestmark = pytest.mark.reference

SETTINGS = [
    {},
    {"temperature": 0.7},
    {"temperature": 1.8},
    {"temperature": 0.0},
    {"top_k": 1},
    {"top_k": 50},
    {"top_k": 1000},
    {"top_k": 100000},
    {"top_p": 0.3},
    {"top_p": 0.95},
    {"top_k": 40, "top_p": 0.9, "temperature": 0.8},
]


@pytest.mark.parametrize("settings", SETTINGS, ids=str)
@pytest.mark.parametrize("seed", [0, 1, 20240607])
def test_sample_matches_the_reference_draw_for_draw(llamagen_reference, settings, seed):
    _, generate, _ = llamagen_reference
    data = torch.Generator().manual_seed(seed + 100)
    logits = torch.randn(4, 3, 16384, generator=data) * 3.0
    ours = torch.Generator("cpu").manual_seed(seed)
    torch.manual_seed(seed)
    for _ in range(20):  # successive draws consume the generators in step
        expected_idx, expected_probs = generate.sample(logits.clone(), **settings)
        actual_idx, actual_probs = sample(logits.clone(), **settings, generator=ours)
        assert torch.equal(actual_idx, expected_idx)
        assert torch.equal(actual_probs, expected_probs)
        logits = logits.roll(1, dims=-1)


@pytest.mark.parametrize(
    "settings",
    [
        {"top_k": 3},
        {"top_p": 0.5},
        {"top_k": 10, "top_p": 0.8},
        {"top_k": 5, "min_tokens_to_keep": 3},
    ],
    ids=str,
)
def test_filtering_matches_the_reference(llamagen_reference, settings):
    _, generate, _ = llamagen_reference
    logits = torch.randn(3, 500, generator=torch.Generator().manual_seed(7)) * 2
    expected = generate.top_k_top_p_filtering(logits.clone(), **settings)
    actual = top_k_top_p_filtering(logits.clone(), **settings)
    assert torch.equal(actual, expected)
