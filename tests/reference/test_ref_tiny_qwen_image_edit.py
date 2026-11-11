"""Tiny random-weight Qwen-Image-Edit pipeline parity."""

import pytest
import torch
from helpers import write_tiny_qwen_image_edit

from imagehub import pipeline
from imagehub.image import Image

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")


@pytest.mark.parametrize(
    ("seed", "negative_prompt", "true_cfg", "width", "height"),
    [(0, None, 1.0, 32, 32), (42, "unchanged", 2.0, 32, 48)],
)
def test_tiny_qwen_image_edit_pipeline_exact(
    tmp_path, seed, negative_prompt, true_cfg, width, height
):
    root = write_tiny_qwen_image_edit(tmp_path / "snapshot")
    source = Image(torch.arange(32 * 48 * 3, dtype=torch.int64).reshape(32, 48, 3).to(torch.uint8))
    reference = diffusers.QwenImageEditPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = reference(
            image=source.to_pil(),
            prompt="turn it blue",
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg,
            num_inference_steps=1,
            width=width,
            height=height,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype(np.uint8))
    actual = (
        pipeline("t2i", backend="qwen-image-edit", model=root)(
            "turn it blue",
            image=source,
            negative_prompt=negative_prompt,
            true_cfg=true_cfg,
            steps=1,
            width=width,
            height=height,
            seed=seed,
        )
        .images[0]
        .tensor
    )
    assert torch.equal(actual, expected[0])


def test_tiny_qwen_image_edit_plus_pipeline_exact(tmp_path):
    root = write_tiny_qwen_image_edit(tmp_path / "snapshot-plus", plus=True)
    source = Image(torch.arange(32 * 48 * 3, dtype=torch.int64).reshape(32, 48, 3).to(torch.uint8))
    images = [source, Image(source.tensor.flip(1))]
    reference = diffusers.QwenImageEditPlusPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = reference(
            image=[image.to_pil() for image in images],
            prompt="turn it blue",
            true_cfg_scale=1.0,
            num_inference_steps=1,
            width=32,
            height=32,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype(np.uint8))
    actual = (
        pipeline("t2i", backend="qwen-image-edit-2509", model=root)(
            "turn it blue", image=images, true_cfg=1.0, steps=1, width=32, height=32, seed=0
        )
        .images[0]
        .tensor
    )
    assert torch.equal(actual, expected[0])


@pytest.mark.parametrize("model_id", ["qwen-image", "qwen-image-edit", "qwen-image-edit-2509"])
def test_tiny_vae_decode_input_matches_pipeline(tmp_path, monkeypatch, model_id):
    """The latents handed to the VAE decoder must equal the pipeline's bit for bit.

    The pipeline rescales with ``latents / (1 / std) + mean``; ``latents * std + mean`` differs in
    the last bit and moved one pixel of the real-checkpoint seed-42 run across a rounding boundary.
    """
    from helpers import write_tiny_qwen_image

    from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage

    plus = model_id == "qwen-image-edit-2509"
    if model_id == "qwen-image":
        root = write_tiny_qwen_image(tmp_path / "snapshot", seed=7)
        pipeline_class = diffusers.QwenImagePipeline
        image_options: dict = {}
        reference_images = None
    else:
        root = write_tiny_qwen_image_edit(tmp_path / "snapshot", plus=plus)
        pipeline_class = (
            diffusers.QwenImageEditPlusPipeline if plus else diffusers.QwenImageEditPipeline
        )
        source = Image(
            torch.arange(32 * 48 * 3, dtype=torch.int64).reshape(32, 48, 3).to(torch.uint8)
        )
        images = [source, Image(source.tensor.flip(1))] if plus else [source]
        image_options = {"image": images if plus else source}
        reference_images = [image.to_pil() for image in images] if plus else source.to_pil()
    # The tiny snapshots use mean 0 and std 1, where both formulas agree; use the published values.
    import json

    from helpers import FIXTURES

    published = json.loads(
        (FIXTURES / "qwen_image_edit_configs" / "published" / "vae_config.json").read_text()
    )
    config_path = root / "vae" / "config.json"
    config = json.loads(config_path.read_text())
    config["latents_mean"] = published["latents_mean"]
    config["latents_std"] = published["latents_std"]
    config_path.write_text(json.dumps(config))
    captured: dict[str, torch.Tensor] = {}

    def recorder(name, decode):
        def wrapped(self, latents, *args, **kwargs):
            captured[name] = latents.clone()
            return decode(self, latents, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        diffusers.AutoencoderKLQwenImage,
        "decode",
        recorder("reference", diffusers.AutoencoderKLQwenImage.decode),
    )
    monkeypatch.setattr(
        AutoencoderKLQwenImage, "decode", recorder("native", AutoencoderKLQwenImage.decode)
    )
    reference = pipeline_class.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    call = {"image": reference_images} if reference_images is not None else {}
    with torch.inference_mode():
        reference(
            prompt="turn it blue",
            true_cfg_scale=1.0,
            num_inference_steps=1,
            width=32,
            height=32,
            generator=torch.Generator("cpu").manual_seed(3),
            output_type="np",
            **call,
        )
    pipeline("t2i", backend=model_id, model=root)(
        "turn it blue", true_cfg=1.0, steps=1, width=32, height=32, seed=3, **image_options
    )
    assert captured["reference"].dtype == captured["native"].dtype
    assert torch.equal(captured["native"], captured["reference"])


@pytest.mark.parametrize("plus", [False, True], ids=["qwen-image-edit", "qwen-image-edit-2509"])
def test_tiny_qwen_image_edit_lora_matches_diffusers(tmp_path, monkeypatch, plus):
    """Transformer adapters (single, weighted pair, fused) match diffusers' LoRA loader."""
    pytest.importorskip("peft")
    from lora_helpers import random_lora, write_peft

    from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage

    targets = ("to_k", "to_q", "to_v", "to_out.0")
    model_id = "qwen-image-edit-2509" if plus else "qwen-image-edit"
    root = write_tiny_qwen_image_edit(tmp_path / "snapshot", plus=plus)
    source = Image(torch.arange(32 * 48 * 3, dtype=torch.int64).reshape(32, 48, 3).to(torch.uint8))
    images = [source, Image(source.tensor.flip(1))] if plus else [source]
    with pipeline("t2i", backend=model_id, model=root) as pipe:
        transformer = {"transformer": pipe.backend.components.transformer}
        first = write_peft(
            tmp_path / "first.safetensors",
            random_lora(transformer, targets, seed=4, std=0.1),
            alpha=3.0,
        )
        second = write_peft(
            tmp_path / "second.safetensors",
            random_lora(transformer, targets, seed=7, std=0.1),
            alpha=2.0,
        )
    captured: dict[str, torch.Tensor] = {}

    def recorder(name, decode):
        def wrapped(self, latents, *args, **kwargs):
            captured[name] = latents.clone()
            return decode(self, latents, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        diffusers.AutoencoderKLQwenImage,
        "decode",
        recorder("reference", diffusers.AutoencoderKLQwenImage.decode),
    )
    monkeypatch.setattr(
        AutoencoderKLQwenImage, "decode", recorder("native", AutoencoderKLQwenImage.decode)
    )
    pipeline_class = (
        diffusers.QwenImageEditPlusPipeline if plus else diffusers.QwenImageEditPipeline
    )
    reference = pipeline_class.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)

    def run_reference():
        with torch.inference_mode():
            pixels = reference(
                image=[image.to_pil() for image in images] if plus else source.to_pil(),
                prompt="turn it blue",
                true_cfg_scale=1.0,
                num_inference_steps=2,
                width=32,
                height=32,
                generator=torch.Generator("cpu").manual_seed(5),
                output_type="np",
            ).images
        return captured["reference"], torch.from_numpy((pixels * 255).round().astype(np.uint8))[0]

    def run_native(**options):
        result = pipeline("t2i", backend=model_id, model=root, **options)(
            "turn it blue",
            image=images if plus else source,
            true_cfg=1.0,
            steps=2,
            width=32,
            height=32,
            seed=5,
        )
        return captured["native"], result.images[0].tensor

    def check(plain, **options):
        expected = run_reference()
        # The adapter must change the decoder input, or this comparison proves nothing.
        assert not torch.equal(expected[0], plain[0])
        actual = run_native(**options)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])

    plain = run_reference()
    reference.load_lora_weights(str(first), adapter_name="first")
    check(plain, lora=first)
    reference.load_lora_weights(str(second), adapter_name="second")
    reference.set_adapters(["first", "second"], [0.8, 0.5])
    check(plain, lora=[(first, 0.8), (second, 0.5)])
    reference.fuse_lora()
    check(plain, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True)
