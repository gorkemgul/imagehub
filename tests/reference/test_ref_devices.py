"""MPS/CUDA smoke tests: they run and produce finite images. No parity claim."""

import math
import platform

import pytest
import torch
from helpers import checkpoint_dir, max_diff

from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)

pytestmark = pytest.mark.reference
PROMPT = "a photograph of an astronaut riding a horse"


def _available(device):
    return torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()


def _name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


@pytest.mark.parametrize(
    "device",
    [pytest.param("mps", marks=pytest.mark.mps), pytest.param("cuda", marks=pytest.mark.cuda)],
)
def test_device_smoke(device, record):
    if not _available(device):
        pytest.skip(f"{device} is not available")
    root = checkpoint_dir(5)
    noise = initial_latents(7, (1, 4, 64, 64))
    outputs = {}
    for name in ("cpu", device):
        components = StableDiffusionComponents.load(
            root, device=torch.device(name), dtype=torch.float32
        )
        final = denoise(
            components,
            prompt=PROMPT,
            negative_prompt=None,
            steps=20,
            guidance=7.5,
            latents=noise.to(name),
        )
        assert torch.isfinite(final).all()
        outputs[name] = (final.cpu(), decode_latents(components, final))
        del components
    (cpu_latents, cpu_pixels), (dev_latents, dev_pixels) = outputs["cpu"], outputs[device]
    assert dev_pixels.shape == (1, 512, 512, 3)
    mse = float(((cpu_pixels.double() - dev_pixels.double()) ** 2).mean())
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    record(
        f"smoke_{device}",
        device_name=_name(device),
        steps=20,
        latents=max_diff(dev_latents, cpu_latents),
        pixels_max_abs=int((dev_pixels.int() - cpu_pixels.int()).abs().max()),
        psnr_db=psnr if math.isfinite(psnr) else "inf",
    )
