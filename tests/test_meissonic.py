import pytest
import torch
from helpers import write_tiny_meissonic

from imagehub import pipeline
from imagehub.backends.meissonic import MeissonicBackend, MeissonicComponents, denoise
from imagehub.types import T2IRequest

SMALL = {"steps": 2, "guidance": 1.0, "width": 32, "height": 32}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_meissonic(tmp_path_factory.mktemp("meissonic") / "snapshot")


def test_generates_images_with_metadata(tiny):
    result = pipeline("t2i", backend="meissonic", model=tiny)("a cat", seed=3, **SMALL)
    assert result.images[0].tensor.shape == (32, 32, 3)
    assert result.metadata["seed"] == 3
    assert result.metadata["scheduler"] == "meissonic"


def test_seed_is_reproducible_and_requests_are_isolated(tiny):
    pipe = pipeline("t2i", backend="meissonic", model=tiny)
    first = pipe("a cat", seed=7, **SMALL).images[0].tensor
    pipe("a dog", seed=9, **SMALL)
    second = pipe("a cat", seed=7, **SMALL).images[0].tensor
    assert torch.equal(first, second)


def test_guidance_and_non_square(tiny):
    result = pipeline("t2i", backend="meissonic", model=tiny)(
        "a cat", seed=1, steps=1, guidance=4.0, width=64, height=32
    )
    assert result.images[0].tensor.shape == (32, 64, 3)


def test_options_are_validated_before_loading(tmp_path):
    backend = MeissonicBackend(tmp_path / "missing")
    with pytest.raises(TypeError, match="Unknown Meissonic options"):
        backend.infer(T2IRequest("x"), unknown=True)
    with pytest.raises(ValueError, match="multiple of 32"):
        backend.infer(T2IRequest("x"), width=33)


def test_callers_autograd_mode_is_restored(tiny):
    previous = torch.is_grad_enabled()
    with torch.enable_grad():
        pipeline("t2i", backend="meissonic", model=tiny)("x", seed=1, **SMALL)
        assert torch.is_grad_enabled()
    assert torch.is_grad_enabled() == previous


def test_denoise_returns_complete_tokens(tiny):
    components = MeissonicComponents.load(tiny, device=torch.device("cpu"), dtype=torch.float32)
    tokens = denoise(
        components,
        prompt="a cat",
        negative_prompt="",
        steps=2,
        guidance=1.0,
        height=32,
        width=32,
        generator=torch.Generator("cpu").manual_seed(2),
    )
    assert tokens.shape == (1, 2, 2)
    assert not (tokens == 23).any()
