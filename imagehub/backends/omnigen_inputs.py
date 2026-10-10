"""OmniGen input handling: the multimodal prompt, image preprocessing and the attention layout."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from imagehub.image import as_pil
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.tokenization_omnigen import OmniGenTokenizer

IMAGE_PLACEHOLDER = "<img><|image_1|></img>"
IMAGE_TAG = re.compile(r"<\|image_\d+\|>")
NEGATIVE_PROMPT = (
    "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated "
    "hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad "
    "anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed "
    "limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers."
)
PAD_TOKEN_ID = 2
# Each image token covers a 16x16 pixel patch (8x VAE downscale, 2x2 patches).
TOKEN_PIXELS = 16


def validate_prompt(prompt: str, has_image: bool) -> None:
    """Check the placeholder convention before anything is loaded."""
    if has_image:
        if IMAGE_PLACEHOLDER not in prompt:
            raise ValueError(
                f"the prompt must reference the input image with the placeholder {IMAGE_PLACEHOLDER}"
            )
        numbers = sorted({int(tag[len("<|image_") : -2]) for tag in IMAGE_TAG.findall(prompt)})
        if numbers != [1]:
            raise ValueError(
                "only one input image is supported, so the only placeholder is "
                f"{IMAGE_PLACEHOLDER} (found image numbers {numbers})"
            )
    elif IMAGE_PLACEHOLDER in prompt or IMAGE_TAG.search(prompt):
        raise ValueError(
            f"the prompt contains an image placeholder ({IMAGE_PLACEHOLDER}) but no image was given"
        )


def crop_image(pil_image: Any, max_image_size: int) -> Any:
    """Shrink to at most ``max_image_size`` and centre-crop both sides to a multiple of 16."""
    from PIL import Image as PILImage

    while min(*pil_image.size) >= 2 * max_image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=PILImage.BOX)
    if max(*pil_image.size) > max_image_size:
        scale = max_image_size / max(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PILImage.BICUBIC
        )
    if min(*pil_image.size) < 16:
        scale = 16 / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PILImage.BICUBIC
        )
    width, height = pil_image.size
    crop_y1 = (height % 16) // 2
    crop_y2 = height % 16 - crop_y1
    crop_x1 = (width % 16) // 2
    crop_x2 = width % 16 - crop_x1
    return pil_image.crop((crop_x1, crop_y1, width - crop_x2, height - crop_y2))


def preprocess_input_image(image: Any, max_image_size: int = 1024) -> Tensor:
    """Return a ``(3, height, width)`` float32 tensor in [-1, 1] for one input image."""
    pil = crop_image(as_pil(image), max_image_size)
    data = torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8)
    pixels = data.reshape(pil.height, pil.width, 3).permute(2, 0, 1).contiguous()
    return pixels.to(torch.float32).div(255).sub_(0.5).div_(0.5)


def encode_input_images(
    vae: AutoencoderKL, pixel_values: Sequence[Tensor], generator: torch.Generator
) -> list[Tensor]:
    """Sample the posterior of each image (``(1, 3, h, w)`` pixels) and scale the latents."""
    latents = []
    for pixels in pixel_values:
        mean, logvar = vae.encode_moments(pixels.to(vae.encoder.conv_in.weight.dtype))
        std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype).to(mean.device)
        latents.append((mean + std * noise).mul_(vae.scaling_factor))
    return latents


def add_prefix_instruction(prompt: str) -> str:
    return (
        "<|user|>\nGenerate an image according to the following instructions\n"
        f"{prompt}<|end|>\n<|assistant|>\n<|diffusion|>"
    )


@dataclass
class _Sequence:
    input_ids: list[int]
    pixel_values: list[Tensor] | None
    image_sizes: list[list[int]] | None


def _multimodal_prompt(
    tokenizer: OmniGenTokenizer, text: str, images: Sequence[Tensor] | None
) -> _Sequence:
    text = add_prefix_instruction(text)
    if not images:
        return _Sequence(tokenizer.tokenize(text), None, None)
    chunks = [tokenizer.tokenize(chunk) for chunk in IMAGE_TAG.split(text)]
    for i in range(1, len(chunks)):
        if chunks[i][0] == 1:
            chunks[i] = chunks[i][1:]
    tags = IMAGE_TAG.findall(text)
    image_ids = [int(tag.split("|")[1].split("_")[-1]) for tag in tags]
    unique = sorted(set(image_ids))
    if unique != list(range(1, len(unique) + 1)) or len(unique) != len(images):
        raise ValueError(
            f"found {len(unique)} image tags in the prompt for {len(images)} input images"
        )
    ordered = [images[x - 1] for x in image_ids]
    ids: list[int] = []
    spans: list[list[int]] = []
    for i, chunk in enumerate(chunks):
        ids.extend(chunk)
        if i != len(chunks) - 1:
            start = len(ids)
            size = ordered[i].size(-2) * ordered[i].size(-1) // TOKEN_PIXELS // TOKEN_PIXELS
            spans.append([start, start + size])
            ids.extend([0] * size)
    return _Sequence(ids, ordered, spans)


@dataclass
class OmniGenInputs:
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    input_pixel_values: list[Tensor]
    input_image_sizes: dict[int, list[list[int]]]
    height: int
    width: int


def build_inputs(
    tokenizer: OmniGenTokenizer,
    prompt: str,
    images: Sequence[Tensor] | None,
    *,
    height: int,
    width: int,
    use_img_cfg: bool,
    use_input_image_size_as_output: bool = False,
) -> OmniGenInputs:
    """Port of the reference processor for one prompt and at most a few preprocessed images.

    The batch is ``[prompt, negative prompt]`` plus, when ``use_img_cfg``, the prompt that holds
    only the image placeholders.
    """
    images = list(images) if images else None
    conditional = _multimodal_prompt(tokenizer, prompt, images)
    unconditional = _multimodal_prompt(tokenizer, NEGATIVE_PROMPT, None)
    sequences = [conditional, unconditional]
    if use_img_cfg:
        if images:
            tags = " ".join(f"<img><|image_{i + 1}|></img>" for i in range(len(images)))
            sequences.append(_multimodal_prompt(tokenizer, tags, images))
        else:
            sequences.append(unconditional)
    if use_input_image_size_as_output:
        assert conditional.pixel_values is not None
        height, width = conditional.pixel_values[0].size(-2), conditional.pixel_values[0].size(-1)
    image_tokens = height * width // TOKEN_PIXELS // TOKEN_PIXELS

    pixel_values: list[Tensor] = []
    image_sizes: dict[int, list[list[int]]] = {}
    for index, sequence in enumerate(sequences):
        if sequence.pixel_values is not None and sequence.image_sizes is not None:
            pixel_values.extend(sequence.pixel_values)
            image_sizes[index] = [list(span) for span in sequence.image_sizes]

    longest = max(len(s.input_ids) for s in sequences)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for index, sequence in enumerate(sequences):
        pad = longest - len(sequence.input_ids)
        padded.append([PAD_TOKEN_ID] * pad + sequence.input_ids)
        masks.append([0] * pad + [1] * len(sequence.input_ids))
        if index in image_sizes:
            image_sizes[index] = [[x + pad for x in span] for span in image_sizes[index]]
    input_ids = torch.tensor(padded, dtype=torch.long)
    text_mask = torch.tensor(masks, dtype=torch.long)

    positions = []
    extended = []
    seq_len = longest + image_tokens + 1
    for row in text_mask:
        real = int(row.sum())
        pad = longest - real
        positions.append([0] * pad + list(range(real + image_tokens + 1)))
        mask = torch.tril(torch.ones(real + 1, real + 1))
        mask = torch.cat([mask, torch.zeros(real + 1, image_tokens)], dim=-1)
        mask = torch.cat([mask, torch.ones(image_tokens, real + image_tokens + 1)], dim=0)
        if pad > 0:
            mask = torch.cat([torch.zeros(real + 1 + image_tokens, pad), mask], dim=-1)
            mask = torch.cat([torch.ones(pad, seq_len), mask], dim=0)
        extended.append(mask.unsqueeze(0))
    attention_mask = torch.cat(extended, dim=0)
    for index, spans in image_sizes.items():
        for start, end in spans:
            attention_mask[index][start:end, start:end] = 1

    return OmniGenInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=torch.tensor(positions, dtype=torch.long),
        input_pixel_values=[x.unsqueeze(0) for x in pixel_values],
        input_image_sizes=image_sizes,
        height=height,
        width=width,
    )
