import pytest
import torch
from helpers import write_tiny_sd, write_tiny_sdxl
from lora_helpers import TEXT_TARGETS, UNET_TARGETS, random_lora, write_kohya, write_peft

from imagehub import pipeline
from imagehub.lora import lora_layers

REQUEST = dict(seed=3, steps=2, width=32, height=32)


@pytest.fixture(scope="module")
def tiny_sd(tmp_path_factory):
    return write_tiny_sd(tmp_path_factory.mktemp("lora-sd") / "sd")


@pytest.fixture(scope="module")
def tiny_sdxl(tmp_path_factory):
    return write_tiny_sdxl(tmp_path_factory.mktemp("lora-sdxl") / "sdxl", sample_size=4)


def components_of(backend, root):
    with pipeline("t2i", backend=backend, model=root) as pipe:
        c = pipe.backend.components
        return {
            name: getattr(c, name)
            for name in ("unet", "text_encoder", "text_encoder_2")
            if getattr(c, name, None) is not None
        }


def image(backend, root, **options):
    with pipeline("t2i", backend=backend, model=root, **options) as pipe:
        return pipe("a cat", **REQUEST).images[0].tensor


def adapter(path, components, *, seed=0, zero=False, layout="peft"):
    weights = random_lora({"unet": components["unet"]}, UNET_TARGETS, seed=seed)
    for name in ("text_encoder", "text_encoder_2"):
        if name in components:
            weights |= random_lora({name: components[name]}, TEXT_TARGETS, seed=seed + 1)
    if zero:
        weights = {key: (down, torch.zeros_like(up)) for key, (down, up) in weights.items()}
    if layout == "kohya":
        return write_kohya(path, weights, alpha=1.0)
    return write_peft(path, weights, alpha=4.0)


@pytest.mark.parametrize(
    ("backend", "root"), [("stable-diffusion", "tiny_sd"), ("stable-diffusion-xl", "tiny_sdxl")]
)
def test_lora_changes_the_image_and_scale_zero_restores_it(tmp_path, request, backend, root):
    root = request.getfixturevalue(root)
    components = components_of(backend, root)
    plain = image(backend, root)
    zero = adapter(tmp_path / "zero.safetensors", components, zero=True)
    assert torch.equal(image(backend, root, lora=zero), plain)
    path = adapter(tmp_path / "a.safetensors", components, layout="kohya")
    assert not torch.equal(image(backend, root, lora=path), plain)
    assert torch.equal(image(backend, root, lora=path, lora_scale=0.0), plain)


def test_fused_adapters_match_unfused(tmp_path, tiny_sd):
    components = components_of("stable-diffusion", tiny_sd)
    first = adapter(tmp_path / "a.safetensors", components)
    second = adapter(tmp_path / "b.safetensors", components, seed=5, layout="kohya")
    stacked = [(first, 0.7), (second, 0.4)]
    with pipeline("t2i", backend="stable-diffusion", model=tiny_sd, lora=stacked) as pipe:
        c = pipe.backend.components
        ids = torch.tensor([[0, 5, 9, 2]])
        sample = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            context = c.text_encoder(ids)
            unfused = c.unet(sample, torch.tensor([10]), context)
    with pipeline(
        "t2i", backend="stable-diffusion", model=tiny_sd, lora=stacked, lora_fuse=True
    ) as pipe:
        c = pipe.backend.components
        layers = lora_layers(c.unet) | lora_layers(c.text_encoder)
        assert layers
        assert all(layer.merged_adapters == ["lora_0", "lora_1"] for layer in layers.values())
        with torch.no_grad():
            context = c.text_encoder(ids)
            fused = c.unet(sample, torch.tensor([10]), context)
    torch.testing.assert_close(fused, unfused, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (dict(lora=["missing.safetensors"]), FileNotFoundError),
        (dict(lora="adapter.bin"), ValueError),
        (dict(lora_fuse="yes"), TypeError),
    ],
)
def test_invalid_options_fail_at_construction(tmp_path, options, error):
    with pytest.raises(error):
        pipeline("t2i", backend="stable-diffusion", model=tmp_path / "not-loaded", **options)


def test_list_with_lora_scale_fails_at_construction(tmp_path, tiny_sd):
    path = adapter(tmp_path / "a.safetensors", components_of("stable-diffusion", tiny_sd))
    with pytest.raises(ValueError, match="pairs"):
        pipeline("t2i", backend="stable-diffusion", model=tiny_sd, lora=[path], lora_scale=0.5)


def test_reloading_applies_adapters_once(tmp_path, tiny_sd):
    path = adapter(tmp_path / "a.safetensors", components_of("stable-diffusion", tiny_sd))
    pipe = pipeline("t2i", backend="stable-diffusion", model=tiny_sd, lora=path)
    first = pipe("a cat", **REQUEST).images[0].tensor
    pipe.close()
    second = pipe("a cat", **REQUEST).images[0].tensor
    assert torch.equal(first, second)
    layers = lora_layers(pipe.backend.components.unet)
    assert all(layer.adapters == ("lora_0",) for layer in layers.values())
    pipe.close()
