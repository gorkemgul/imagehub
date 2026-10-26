"""SDXL refiner (image-to-image) pipeline: input handling, latent preparation and denoising."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from imagehub.backends.flux_kontext import preprocess_kontext_image
from imagehub.image import as_pil
from imagehub.schedulers.euler import EulerDiscreteScheduler

if TYPE_CHECKING:
    from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents

VAE_SCALE_FACTOR = 8
REFINER_DEFAULTS: dict[str, Any] = {
    "image": None,
    "strength": 0.3,
    "aesthetic_score": 6.0,
    "negative_aesthetic_score": 2.5,
    "negative_original_size": None,
    "negative_crops_coords_top_left": (0, 0),
    "negative_target_size": None,
}


def refiner_input_size(image: Any, vae_scale_factor: int = VAE_SCALE_FACTOR) -> tuple[int, int]:
    """Return the ``(height, width)`` of the preprocessed input: rounded down to the VAE scale."""
    pil = as_pil(image)
    return (
        pil.height - pil.height % vae_scale_factor,
        pil.width - pil.width % vae_scale_factor,
    )


def preprocess_refiner_image(image: Any) -> Tensor:
    """Resize to a multiple of 8 with Lanczos and normalize to ``[-1, 1]`` (``(1, 3, h, w)``)."""
    height, width = refiner_input_size(image)
    return preprocess_kontext_image(image, height, width)


def refiner_start(steps: int, strength: float) -> int:
    """Index of the first kept timestep, as the reference ``get_timesteps`` computes it."""
    return max(steps - min(int(steps * strength), steps), 0)


def make_scheduler(config: dict[str, Any]) -> EulerDiscreteScheduler:
    if config.get("_class_name", "EulerDiscreteScheduler") != "EulerDiscreteScheduler":
        raise ValueError("The SDXL refiner supports the Euler discrete scheduler only")
    return EulerDiscreteScheduler(config)


@torch.inference_mode()
def prepare_refiner_latents(
    components: StableDiffusionXLComponents,
    pixels: Tensor,
    scheduler: EulerDiscreteScheduler,
    latent_timestep: Tensor,
    *,
    num_images: int,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    """Encode (sampling the posterior), scale, repeat and noise the image latents."""
    vae = components.vae
    vae_dtype = vae.encoder.conv_in.weight.dtype
    posterior = vae.encode_sample(pixels.to(pixels.device, vae_dtype), generator)
    latents = posterior.to(dtype)
    latents = vae.scaling_factor * latents
    if num_images > latents.shape[0]:
        latents = torch.cat([latents] * (num_images // latents.shape[0]), dim=0)
    draw_device = generator.device if generator is not None else latents.device
    noise = torch.randn(latents.shape, generator=generator, dtype=dtype, device=draw_device).to(
        latents.device
    )
    return scheduler.add_noise(latents, noise, latent_timestep)


@torch.inference_mode()
def refine(
    components: StableDiffusionXLComponents,
    *,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    strength: float,
    guidance: float,
    pixels: Tensor,
    num_images: int,
    original_size: tuple[int, int] | None,
    crops_coords_top_left: tuple[int, int],
    target_size: tuple[int, int] | None,
    negative_original_size: tuple[int, int] | None,
    negative_crops_coords_top_left: tuple[int, int],
    negative_target_size: tuple[int, int] | None,
    aesthetic_score: float,
    negative_aesthetic_score: float,
    generator: torch.Generator | None,
) -> Tensor:
    from imagehub.backends.stable_diffusion_xl import encode_prompt

    dtype = components.unet.conv_in.weight.dtype
    device = pixels.device
    prompt_embeds, pooled = encode_prompt(components, prompt, device)
    prompt_embeds = prompt_embeds.to(dtype).repeat(num_images, 1, 1)
    pooled = pooled.to(dtype).repeat(num_images, 1)
    use_cfg = guidance > 1.0

    scheduler = make_scheduler(components.scheduler_config)
    scheduler.set_timesteps(steps)
    start = refiner_start(steps, strength)
    timesteps = scheduler.timesteps[start:]
    scheduler.set_begin_index(start)
    latent_timestep = timesteps[:1].repeat(num_images)
    latents = prepare_refiner_latents(
        components,
        pixels,
        scheduler,
        latent_timestep,
        num_images=num_images,
        dtype=prompt_embeds.dtype,
        generator=generator,
    )

    height = latents.shape[-2] * VAE_SCALE_FACTOR
    width = latents.shape[-1] * VAE_SCALE_FACTOR
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)
    negative_original_size = negative_original_size or original_size
    negative_target_size = negative_target_size or target_size
    time_ids = torch.tensor(
        [[*original_size, *crops_coords_top_left, aesthetic_score]], dtype=dtype, device=device
    ).repeat(num_images, 1)

    if use_cfg:
        if negative_prompt is None and components.force_zeros_for_empty_prompt:
            negative_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled = torch.zeros_like(pooled)
        else:
            negative_embeds, negative_pooled = encode_prompt(
                components, negative_prompt or "", device
            )
            negative_embeds = negative_embeds.to(dtype).repeat(num_images, 1, 1)
            negative_pooled = negative_pooled.to(dtype).repeat(num_images, 1)
        negative_time_ids = torch.tensor(
            [
                [
                    *negative_original_size,
                    *negative_crops_coords_top_left,
                    negative_aesthetic_score,
                ]
            ],
            dtype=dtype,
            device=device,
        ).repeat(num_images, 1)
        context = torch.cat([negative_embeds, prompt_embeds])
        text_embeds = torch.cat([negative_pooled, pooled])
        time_ids = torch.cat([negative_time_ids, time_ids])
    else:
        context = prompt_embeds
        text_embeds = pooled

    for timestep in timesteps:
        model_input = torch.cat([latents] * 2) if use_cfg else latents
        model_input = scheduler.scale_model_input(model_input, timestep)
        noise = components.unet(
            model_input, timestep, context, text_embeds=text_embeds, time_ids=time_ids
        )
        if use_cfg:
            noise_uncond, noise_text = noise.chunk(2)
            noise = noise_uncond + guidance * (noise_text - noise_uncond)
        latents = scheduler.step(noise, timestep, latents)
    return latents
