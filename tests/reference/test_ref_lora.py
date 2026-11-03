"""LoRA loading parity against diffusers ``load_lora_weights`` (PEFT backend) on CPU float32."""

import pytest
import torch
from helpers import write_tiny_sd, write_tiny_sdxl
from lora_helpers import (
    TEXT_TARGETS,
    UNET_TARGETS,
    random_lora,
    write_kohya,
    write_legacy,
    write_peft,
)

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("peft")

PROMPT = "a cat"
STEPS = 3
SIZE = 32


def tiny(family, directory):
    if family == "sd":
        return write_tiny_sd(directory)
    return write_tiny_sdxl(directory, sample_size=4)


def reference_pipeline(family, root):
    if family == "sd":
        ref = diffusers.StableDiffusionPipeline.from_pretrained(
            root,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            local_files_only=True,
        )
    else:
        ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
            root, local_files_only=True, add_watermarker=False
        )
    ref.set_progress_bar_config(disable=True)
    return ref


def backend_name(family):
    return "stable-diffusion" if family == "sd" else "stable-diffusion-xl"


def native_components(family, root):
    with pipeline("t2i", backend=backend_name(family), model=root) as pipe:
        c = pipe.backend.components
        return {
            name: getattr(c, name)
            for name in ("unet", "text_encoder", "text_encoder_2")
            if getattr(c, name, None) is not None
        }


def random_adapter(components, *, seed=0, legacy=False, text=True, std=0.1):
    unet_targets = UNET_TARGETS
    if legacy:
        unet_targets = tuple(t for t in UNET_TARGETS if t not in ("proj_in", "proj_out"))
    weights = random_lora({"unet": components["unet"]}, unet_targets, seed=seed, std=std)
    for name in ("text_encoder", "text_encoder_2"):
        if text and name in components:
            weights |= random_lora({name: components[name]}, TEXT_TARGETS, seed=seed + 1, std=std)
    return weights


def without_text_model(weights):
    """diffusers 0.40 with transformers 5 names CLIP layers without the ``text_model`` level."""
    return {(c, m.removeprefix("text_model.")): v for (c, m), v in weights.items()}


def write(layout, path, weights, layers_per_block, family):
    prefixes = {"text_encoder": "lora_te_" if family == "sd" else "lora_te1_"}
    if layout == "peft":
        return write_peft(path, without_text_model(weights), alpha=3.0)
    if layout == "peft-no-alpha":
        return write_peft(path, without_text_model(weights))
    if layout == "kohya":
        return write_kohya(path, weights, alpha=1.0, text_prefixes=prefixes)
    if layout == "kohya-sgm":
        return write_kohya(
            path,
            weights,
            alpha=1.0,
            layers_per_block=layers_per_block,
            text_prefixes=prefixes,
        )
    return write_legacy(path, without_text_model(weights))


def reference_images(ref):
    images = ref(
        PROMPT,
        num_inference_steps=STEPS,
        height=SIZE,
        width=SIZE,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    return torch.from_numpy((images * 255).round().astype("uint8"))


def native_images(family, root, **options):
    with pipeline("t2i", backend=backend_name(family), model=root, **options) as pipe:
        result = pipe(PROMPT, steps=STEPS, height=SIZE, width=SIZE, seed=42)
    return torch.stack([image.tensor for image in result.images])


@pytest.mark.parametrize(
    ("family", "layout"),
    [
        ("sd", "peft"),
        ("sd", "peft-no-alpha"),
        ("sd", "kohya"),
        ("sd", "legacy"),
        ("sdxl", "peft"),
        ("sdxl", "kohya"),
        ("sdxl", "kohya-sgm"),
    ],
)
def test_lora_generation_matches_diffusers(tmp_path, family, layout):
    root = tiny(family, tmp_path / "tiny")
    components = native_components(family, root)
    # diffusers 0.40 converts kohya and legacy text-encoder keys to the transformers 4 names,
    # which transformers 5 models no longer have, so those layouts are compared on the U-Net.
    text = layout.startswith("peft")
    weights = random_adapter(
        components, legacy=layout == "legacy", text=text, std=0.1 if text else 0.5
    )
    layers = components["unet"].config.get("layers_per_block")
    path = write(layout, tmp_path / "adapter.safetensors", weights, layers, family)
    ref = reference_pipeline(family, root)
    plain = reference_images(ref)
    ref.load_lora_weights(str(path), adapter_name="a")
    expected = reference_images(ref)
    assert not torch.equal(expected, plain)
    assert torch.equal(native_images(family, root, lora=path), expected)


@pytest.mark.parametrize("family", ["sd", "sdxl"])
def test_stacked_and_fused_adapters_match_diffusers(tmp_path, family):
    root = tiny(family, tmp_path / "tiny")
    components = native_components(family, root)
    first = write_peft(
        tmp_path / "a.safetensors", without_text_model(random_adapter(components)), alpha=3.0
    )
    second = write_kohya(
        tmp_path / "b.safetensors",
        random_adapter(components, seed=7, text=False),
        alpha=1.0,
        text_prefixes={"text_encoder": "lora_te_" if family == "sd" else "lora_te1_"},
    )
    ref = reference_pipeline(family, root)
    ref.load_lora_weights(str(first), adapter_name="a")
    ref.load_lora_weights(str(second), adapter_name="b")
    ref.set_adapters(["a", "b"], [0.8, 0.5])
    expected = reference_images(ref)
    stacked = [(first, 0.8), (second, 0.5)]
    assert torch.equal(native_images(family, root, lora=stacked), expected)
    ref.fuse_lora()
    fused = reference_images(ref)
    assert torch.equal(native_images(family, root, lora=stacked, lora_fuse=True), fused)
