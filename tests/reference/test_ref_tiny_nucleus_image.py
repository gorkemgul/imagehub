"""Exact tiny Nucleus-Image pipeline parity."""

import pytest
import torch
from helpers import write_tiny_nucleus_image

from imagehub import pipeline
from imagehub.backends.nucleus_image import NucleusImageComponents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
transformers = pytest.importorskip("transformers")


def _uint8(images):
    return torch.from_numpy((images * 255).round().astype("uint8"))


class _TextOnlyProcessor:
    def __init__(self, root):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return (
            "".join(
                f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
                for message in messages
            )
            + "<|im_start|>assistant\n"
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
            padding="max_length",
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
            return_tensors=return_tensors,
        )


def _reference_pipeline(root):
    reference = diffusers.NucleusMoEImagePipeline(
        transformer=diffusers.NucleusMoEImageTransformer2DModel.from_pretrained(
            root / "transformer", local_files_only=True, torch_dtype=torch.float32
        ),
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler"),
        vae=diffusers.AutoencoderKLQwenImage.from_pretrained(
            root / "vae", local_files_only=True, torch_dtype=torch.float32
        ),
        text_encoder=transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
        ),
        processor=_TextOnlyProcessor(root / "processor"),
    )
    reference.set_progress_bar_config(disable=True)
    return reference


@pytest.mark.parametrize(
    ("guidance", "negative_prompt", "height", "width"),
    [(1.0, None, 32, 32), (4.0, "blurry", 32, 48), (4.0, "blurry", 128, 128)],
    ids=["unguided", "guided-non-square", "guided-token-dropping"],
)
@pytest.mark.parametrize("seed", [0, 42])
def test_tiny_nucleus_image_matches_diffusers_exactly(
    tmp_path, guidance, negative_prompt, height, width, seed
):
    root = write_tiny_nucleus_image(tmp_path / "tiny")
    reference = _reference_pipeline(root)
    raw = torch.randn(
        (1, 4, height // 8, width // 8), generator=torch.Generator("cpu").manual_seed(seed)
    )
    noise = raw.reshape(1, 4, height // 16, 2, width // 16, 2).permute(0, 3, 5, 1, 2, 4)
    noise = noise.reshape(1, height * width // 256, 16)
    kwargs = {
        "negative_prompt": negative_prompt,
        "guidance_scale": guidance,
        "num_inference_steps": 2,
        "height": height,
        "width": width,
        "latents": noise.clone(),
        "max_sequence_length": 16,
    }
    expected_latents = reference("a cat", output_type="latent", **kwargs).images
    expected = _uint8(reference("a cat", output_type="np", **kwargs).images)

    components = NucleusImageComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components, "a cat", negative_prompt, 2, guidance, noise.clone(), height, width, 16
    )
    assert torch.equal(actual_latents, expected_latents)
    actual = pipeline("t2i", backend="nucleus-image", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        height=height,
        width=width,
        latents=noise,
        max_sequence_length=16,
    )
    assert torch.equal(torch.stack([image.tensor for image in actual.images]), expected)
