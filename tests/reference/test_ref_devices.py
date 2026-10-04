"""MPS/CUDA smoke tests: they run and produce finite images. No parity claim."""

import math
import platform
import subprocess

import pytest
import torch
from helpers import checkpoint_dir, max_diff, snapshot_dir

from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)
from imagehub.backends.stable_diffusion_xl import (
    StableDiffusionXLComponents,
)
from imagehub.backends.stable_diffusion_xl import (
    denoise as denoise_sdxl,
)

pytestmark = pytest.mark.reference
PROMPT = "a photograph of an astronaut riding a horse"


def _available(device):
    return torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()


def _name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            chip = ""
        if chip:
            return chip
    return platform.machine()


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


@pytest.mark.parametrize(
    "device",
    [pytest.param("mps", marks=pytest.mark.mps), pytest.param("cuda", marks=pytest.mark.cuda)],
)
def test_sdxl_device_smoke(device, record):
    if not _available(device):
        pytest.skip(f"{device} is not available")
    root = snapshot_dir("sdxl-base-1.0")
    noise = initial_latents(7, (1, 4, 128, 128))
    outputs = {}
    for name in ("cpu", device):
        components = StableDiffusionXLComponents.load(
            root, device=torch.device(name), dtype=torch.float32
        )
        final = denoise_sdxl(
            components,
            prompt=PROMPT,
            negative_prompt=None,
            steps=20,
            guidance=5.0,
            latents=noise.to(name),
            original_size=(1024, 1024),
            crops_coords_top_left=(0, 0),
            target_size=(1024, 1024),
        )
        assert torch.isfinite(final).all()
        outputs[name] = (final.cpu(), decode_latents(components, final))
        del components
    (cpu_latents, cpu_pixels), (dev_latents, dev_pixels) = outputs["cpu"], outputs[device]
    assert dev_pixels.shape == (1, 1024, 1024, 3)
    mse = float(((cpu_pixels.double() - dev_pixels.double()) ** 2).mean())
    psnr = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    record(
        f"smoke_{device}_sdxl-base-1.0",
        device_name=_name(device),
        steps=20,
        size=[1024, 1024],
        latents=max_diff(dev_latents, cpu_latents),
        pixels_max_abs=int((dev_pixels.int() - cpu_pixels.int()).abs().max()),
        psnr_db=psnr if math.isfinite(psnr) else "inf",
    )
