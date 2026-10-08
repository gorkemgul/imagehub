"""FLUX.1 Kontext input handling: size selection, preprocessing and VAE encoding."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from imagehub.image import as_pil
from imagehub.models.autoencoder_kl import AutoencoderKL

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def kontext_input_size(
    image: Any, *, auto_resize: bool = True, vae_scale_factor: int = 8
) -> tuple[int, int]:
    """Return the ``(height, width)`` the input image is resized to before encoding."""
    pil = as_pil(image)
    multiple_of = vae_scale_factor * 2
    # The image processor first rounds the size down to a multiple of ``vae_scale_factor * 2``.
    width = pil.width - pil.width % multiple_of
    height = pil.height - pil.height % multiple_of
    aspect_ratio = width / height
    if auto_resize:
        _, width, height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of
    return height, width


def preprocess_kontext_image(image: Any, height: int, width: int) -> Tensor:
    """Resize with Lanczos and return a ``(1, 3, height, width)`` float32 tensor in [-1, 1]."""
    from PIL import Image as PILImage

    pil = as_pil(image).resize((width, height), resample=PILImage.Resampling.LANCZOS)
    data = torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8)
    pixels = (
        data.reshape(1, pil.height, pil.width, 3).permute(0, 3, 1, 2).float() / 255.0
    ).contiguous()
    return 2.0 * pixels - 1.0


def encode_kontext_image(vae: AutoencoderKL, pixels: Tensor) -> Tensor:
    """Encode pixels in [-1, 1] to shifted and scaled image latents."""
    latents = vae.encode(pixels)
    return (latents - (vae.shift_factor or 0.0)) * vae.scaling_factor
