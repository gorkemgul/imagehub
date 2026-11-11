"""Real-checkpoint parity for Nucleus-Image (requires the local snapshot)."""

import gc
from types import SimpleNamespace

import pytest
import torch
from helpers import reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.backends.krea2 import _load_text_encoder
from imagehub.backends.nucleus_image import encode_prompt
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.nucleus_image_transformer import NucleusImageTransformer2DModel
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"


def _root():
    return snapshot_dir("nucleus-image")


class _TextOnlyProcessor:
    """The text path of Qwen3-VL without constructing its unused vision processors."""

    def __init__(self, root):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )

    def __call__(
        self,
        *,
        text,
        padding,
        pad_to_multiple_of,
        max_length,
        truncation,
        return_attention_mask,
        return_tensors,
    ):
        return self.tokenizer(
            text,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )


def _reference(root, *, dtype=None):
    model_kwargs = {"local_files_only": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    reference = diffusers.NucleusMoEImagePipeline(
        transformer=diffusers.NucleusMoEImageTransformer2DModel.from_pretrained(
            root / "transformer", **model_kwargs
        ),
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            root / "scheduler", local_files_only=True
        ),
        vae=diffusers.AutoencoderKLQwenImage.from_pretrained(root / "vae", **model_kwargs),
        text_encoder=transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", **model_kwargs
        ),
        processor=_TextOnlyProcessor(root / "processor"),
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _reference_text(root):
    reference = diffusers.NucleusMoEImagePipeline(
        transformer=None,
        scheduler=None,
        vae=None,
        text_encoder=transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
        ),
        processor=_TextOnlyProcessor(root / "processor"),
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def test_header(record):
    # Strict loading raises on missing, extra or shape-mismatched tensors.
    root = _root()
    counts = {}
    with torch.device("meta"):
        for name, model_class in (
            ("transformer", NucleusImageTransformer2DModel),
            ("vae", AutoencoderKLQwenImage),
        ):
            model = model_class.from_pretrained(root / name)
            counts[name] = {
                "missing": [],
                "extra": [],
                "mismatch": [],
                "tensors": len(model.state_dict()),
            }
            del model
    record("header_nucleus-image", **counts)


def test_text_encoder(record, tmp_path):
    root = _root()
    reference = _reference_text(root)
    expected, mask = reference.encode_prompt(PROMPT, device=CPU)
    torch.save({"text": expected, "mask": mask}, tmp_path / "reference.pt")
    del reference, expected, mask
    gc.collect()
    ours = SimpleNamespace(
        tokenizer=Qwen2Tokenizer(root / "processor"),
        text_encoder=_load_text_encoder(root / "text_encoder", device=CPU, dtype=torch.float32),
    )
    actual, actual_mask = encode_prompt(ours, PROMPT, CPU, 1024)
    expected = torch.load(tmp_path / "reference.pt", weights_only=True)
    masks_match = torch.equal(actual_mask, expected["mask"])
    valid = expected["mask"].bool()
    passed = masks_match and torch.equal(actual[valid], expected["text"][valid])
    record("nucleus-image_text_encoder", passed=passed, tolerance="exact")
    assert passed


def test_vae_decode(record, tmp_path):
    root = _root()
    latents = torch.randn(1, 16, 1, 128, 128, generator=torch.Generator().manual_seed(0))
    reference = diffusers.AutoencoderKLQwenImage.from_pretrained(
        root / "vae", local_files_only=True
    )
    expected = reference.decode(latents).sample
    torch.save(expected, tmp_path / "vae.pt")
    del reference, expected
    gc.collect()
    actual = AutoencoderKLQwenImage.from_pretrained(
        root / "vae", device=CPU, dtype=torch.float32
    ).decode(latents)
    expected = torch.load(tmp_path / "vae.pt", weights_only=True)
    passed = torch.equal(actual, expected)
    record("nucleus-image_vae_decode", passed=passed, tolerance="exact")
    assert passed


def _reference_e2e(root: str, seed: int, out_path: str) -> None:
    """Run the reference pipeline alone in its own interpreter and save the uint8 image."""
    from pathlib import Path

    reference = _reference(Path(root), dtype=torch.float32)
    with torch.inference_mode():
        images = reference(
            PROMPT,
            num_inference_steps=2,
            generator=torch.Generator().manual_seed(seed),
            output_type="np",
        ).images
    torch.save(torch.from_numpy((images * 255).round().astype("uint8"))[0], out_path)


def test_end_to_end_and_seed(record, tmp_path):
    # The 1024x1024 reference and the native model do not fit in memory together, so the
    # reference runs in a separate process and its result is compared from disk.
    root = _root()
    expected, seed_expected = (
        reference_in_subprocess(
            "reference.test_ref_nucleus_image._reference_e2e",
            {"root": str(root), "seed": seed},
            tmp_path / f"reference-{seed}.pt",
        )
        for seed in (0, 42)
    )
    actual = (
        pipeline("t2i", backend="nucleus-image", model=root)(PROMPT, steps=2, seed=0)
        .images[0]
        .tensor
    )
    seeded = (
        pipeline("t2i", backend="nucleus-image", model=root)(PROMPT, steps=2, seed=42)
        .images[0]
        .tensor
    )

    def pixel_diff(a, b):
        return int((a.to(torch.int16) - b.to(torch.int16)).abs().max())

    record(
        "e2e_nucleus-image",
        passed=torch.equal(actual, expected),
        max_pixel_diff=pixel_diff(actual, expected),
        seed=0,
        steps=2,
    )
    record(
        "seed_e2e_nucleus-image",
        passed=torch.equal(seeded, seed_expected),
        max_pixel_diff=pixel_diff(seeded, seed_expected),
        seed=42,
        steps=2,
    )
    assert torch.equal(actual, expected) and torch.equal(seeded, seed_expected)
