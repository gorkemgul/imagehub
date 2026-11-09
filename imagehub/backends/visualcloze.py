"""VisualCloze grid generation and FLUX Fill upsampling."""

from __future__ import annotations

import math
import os
import secrets
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.flux import (
    FluxComponents,
    _pack_latents,
    _prepare_latent_image_ids,
    _sigmas,
    _unpack_latents,
    calculate_shift,
    encode_prompt,
)
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_snapshot
from imagehub.image import Image, as_pil
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.types import ImageResult, T2IRequest

OPTION_DEFAULTS: dict[str, Any] = {
    "image": None,
    "content_prompt": None,
    "steps": 30,
    "guidance": 30.0,
    "max_sequence_length": 512,
    "seed": None,
    "upsampling_strength": 0.3,
    "upsampling_height": None,
    "upsampling_width": None,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _resize_and_crop(image: Any, width: int, height: int) -> Any:
    from PIL import Image as PILImage

    ratio = width / height
    source_ratio = image.width / image.height
    source_width = width if ratio > source_ratio else image.width * height // image.height
    source_height = height if ratio <= source_ratio else image.height * width // image.width
    resized = image.resize((source_width, source_height), resample=PILImage.Resampling.LANCZOS)
    output = PILImage.new("RGB", (width, height))
    output.paste(resized, box=(width // 2 - source_width // 2, height // 2 - source_height // 2))
    return output


def _pixels(image: Any) -> Tensor:
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return 2 * (data.reshape(1, image.height, image.width, 3).permute(0, 3, 1, 2).float() / 255) - 1


def _layout_prompt(rows: int, columns: int) -> str:
    return (
        f"A grid layout with {rows} rows and {columns} columns, "
        f"displaying {rows * columns} images arranged side by side."
    )


def _prompt(task: str, content: str | None, rows: int, columns: int) -> str:
    suffix = f"The last image of the last row depicts: {content}" if content is not None else ""
    return f"{_layout_prompt(rows, columns)} {task} {suffix}".strip()


def _image_sizes(
    grid: list[list[Any | None]], resolution: int, vae_scale: int
) -> list[list[list[int]]]:
    result: list[list[list[int]]] = []
    divisible = 2 * vae_scale
    for row in grid:
        first = next((image for image in row if image is not None), None)
        if first is None:
            result.append([[resolution, resolution] for _ in row])
            continue
        aspect = first.width / first.height
        height = int((resolution * resolution / aspect) ** 0.5)
        width = int(height * aspect)
        width = max(width // divisible, 1) * divisible
        height = max(height // divisible, 1) * divisible
        result.append([[height, width] for _ in row])
    return result


def preprocess_image(
    grid: list[list[Any | None]], *, resolution: int, vae_scale: int
) -> tuple[list[list[Tensor]], list[list[list[int]]], list[int]]:
    """Reference grid preprocessing, without its erroneous multi-target resize pass."""
    sizes = _image_sizes(grid, resolution, vae_scale)
    target_positions: list[int] = []
    processed: list[list[Tensor]] = []
    for row_index, row in enumerate(grid):
        height, width = sizes[row_index][0]
        converted: list[Tensor] = []
        for image in row:
            if image is None:
                from PIL import Image as PILImage

                current = PILImage.new("RGB", (width, height), (0, 0, 0))
                if row_index == len(grid) - 1:
                    target_positions.append(1)
            else:
                current = _resize_and_crop(image, width, height)
                if row_index == len(grid) - 1:
                    target_positions.append(0)
            converted.append(_pixels(current))
        processed.append(converted)
    return processed, sizes, target_positions


def _image_ids(
    images: list[Tensor], vae_scale: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    ids = []
    for index, image in enumerate(images, start=1):
        _, _, height, width = image.shape
        patch_height, patch_width = height // vae_scale // 2, width // vae_scale // 2
        current = torch.zeros(patch_height, patch_width, 3, device=device, dtype=dtype)
        current[..., 0] = index
        current[..., 1] = torch.arange(patch_height, device=device, dtype=dtype)[:, None]
        current[..., 2] = torch.arange(patch_width, device=device, dtype=dtype)[None, :]
        ids.append(current.reshape(-1, 3))
    return torch.cat(ids)


def _mask_tokens(mask: Tensor, vae_scale: int) -> Tensor:
    _, channels, height, width = mask.shape
    mask = mask.view(1, channels, height // vae_scale, vae_scale, width // vae_scale, vae_scale)
    mask = mask.permute(0, 1, 3, 5, 2, 4).reshape(
        1, channels * vae_scale**2, height // vae_scale, width // vae_scale
    )
    return _pack_latents(mask, 1, mask.shape[1], mask.shape[2], mask.shape[3])


def _generation_latents(
    components: FluxComponents,
    images: list[list[Tensor]],
    masks: list[list[Tensor]],
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    vae_scale: int,
) -> tuple[Tensor, Tensor, Tensor]:
    image_latents, mask_latents, mask_tokens, rows = [], [], [], []
    vae = components.vae
    for row, row_masks in zip(images, masks, strict=True):
        image = torch.cat(row, dim=3).to(device=device, dtype=dtype)
        mask = torch.cat(row_masks, dim=3).to(device=device, dtype=dtype)
        rows.append(image)
        encoded = vae.encode_sample(image, generator)
        encoded = (encoded - (vae.shift_factor or 0.0)) * vae.scaling_factor
        image_latents.append(
            _pack_latents(encoded, 1, encoded.shape[1], encoded.shape[2], encoded.shape[3])
        )
        mask_latents.append(
            _pack_latents(encoded.clone(), 1, encoded.shape[1], encoded.shape[2], encoded.shape[3])
        )
        mask_tokens.append(_mask_tokens(mask, vae_scale))
    image_latent = torch.cat(image_latents, dim=1)
    condition = torch.cat((torch.cat(mask_latents, dim=1), torch.cat(mask_tokens, dim=1)), dim=-1)
    noise = torch.randn(image_latent.shape, generator=generator, dtype=dtype).to(device)
    return noise, condition.to(dtype), _image_ids(rows, vae_scale, device, dtype)


def _run_denoising(
    components: FluxComponents,
    latents: Tensor,
    condition: Tensor,
    image_ids: Tensor,
    prompt_embeds: Tensor,
    pooled: Tensor,
    text_ids: Tensor,
    steps: int,
    guidance: float,
    scheduler: FlowMatchEulerDiscreteScheduler,
) -> Tensor:
    dtype = cast(torch.dtype, components.transformer.x_embedder.weight.dtype)
    scheduler.set_begin_index(0)
    guide = None
    if components.transformer.config["guidance_embeds"]:
        guide = torch.full(
            (latents.shape[0],), guidance, device=latents.device, dtype=torch.float32
        )
    for timestep in scheduler.timesteps:
        current = timestep.expand(latents.shape[0]).to(latents.dtype) / 1000
        predicted = components.transformer(
            torch.cat((latents, condition), dim=2).to(dtype),
            prompt_embeds.to(dtype),
            pooled.to(dtype),
            current,
            image_ids,
            text_ids,
            guide,
        )
        latents = scheduler.step(predicted, timestep, latents)
    return latents


def _decode_generation(
    components: FluxComponents,
    latents: Tensor,
    sizes: list[list[list[int]]],
    targets: list[int],
    vae_scale: int,
) -> list[Any]:
    from PIL import Image as PILImage

    start = sum(
        (row[0][0] // vae_scale) * (sum(size[1] for size in row) // vae_scale) // 4
        for row in sizes[:-1]
    )
    height = sizes[-1][0][0] // vae_scale
    width = sum(item[1] for item in sizes[-1]) // vae_scale
    last = latents[:, start : start + height * width // 4].view(
        1, height // 2, width // 2, latents.shape[-1] // 4, 2, 2
    )
    last = last.permute(0, 3, 1, 4, 2, 5).reshape(1, latents.shape[-1] // 4, height, width)
    decoded = components.vae.decode(
        (last / components.vae.scaling_factor + (components.vae.shift_factor or 0.0)).to(
            components.vae.decoder.conv_in.weight.dtype
        )
    )
    pixels = (
        ((decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu() * 255)
        .round()
        .to(torch.uint8)
    )
    output, left = [], 0
    for target, (_, width) in zip(targets, sizes[-1], strict=True):
        if target:
            current = pixels[0, : sizes[-1][0][0], left : left + width]
            output.append(
                PILImage.frombytes(
                    "RGB", (width, current.shape[0]), bytes(current.flatten().tolist())
                )
            )
        left += width
    return output


def _upsample(
    components: FluxComponents,
    images: list[Any],
    prompt: str,
    height: int,
    width: int,
    strength: float,
    steps: int,
    guidance: float,
    max_sequence_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> Tensor:
    """The FLUX Fill stage used by the reference combined pipeline."""
    from PIL import Image as PILImage

    vae = components.vae
    vae_scale = 2 ** (len(vae.config["block_out_channels"]) - 1)
    dtype = cast(torch.dtype, components.transformer.x_embedder.weight.dtype)
    init = torch.cat(
        [
            _pixels(image.resize((width, height), resample=PILImage.Resampling.LANCZOS))
            for image in images
        ]
    ).to(device=device, dtype=torch.float32)
    embeds, pooled, text_ids = encode_prompt(
        components, prompt, device, len(images), max_sequence_length
    )
    scheduler = FlowMatchEulerDiscreteScheduler(components.scheduler_config)
    sequence = (height // vae_scale // 2) * (width // vae_scale // 2)
    scheduler.set_timesteps(
        steps, sigmas=_sigmas(steps), mu=calculate_shift(sequence, components.scheduler_config)
    )
    begin = int(max(steps - min(steps * strength, steps), 0))
    timesteps = scheduler.timesteps[begin:]
    scheduler.set_begin_index(begin)
    latent_height, latent_width = 2 * (height // (vae_scale * 2)), 2 * (width // (vae_scale * 2))
    init_latents = vae.encode_sample(init.to(dtype), generator)
    init_latents = (init_latents - (vae.shift_factor or 0.0)) * vae.scaling_factor
    noise = torch.randn(
        (len(images), vae.config["latent_channels"], latent_height, latent_width),
        generator=generator,
        dtype=dtype,
    ).to(device)
    sigma = scheduler.sigmas[begin]
    latents = _pack_latents(
        (1 - sigma) * init_latents + sigma * noise,
        len(images),
        init_latents.shape[1],
        latent_height,
        latent_width,
    )
    mask = torch.ones((len(images), 1, height, width), device=device, dtype=dtype)
    masked = init * (1 - mask)
    masked_latents = vae.encode_sample(masked.to(dtype), generator)
    masked_latents = (masked_latents - (vae.shift_factor or 0.0)) * vae.scaling_factor
    masked_tokens = _pack_latents(
        masked_latents, len(images), masked_latents.shape[1], latent_height, latent_width
    )
    mask = mask[:, 0].view(len(images), latent_height, vae_scale, latent_width, vae_scale)
    mask = mask.permute(0, 2, 4, 1, 3).reshape(
        len(images), vae_scale**2, latent_height, latent_width
    )
    mask_tokens = _pack_latents(mask, len(images), mask.shape[1], latent_height, latent_width).to(
        dtype
    )
    condition = torch.cat((masked_tokens, mask_tokens), dim=-1)
    ids = _prepare_latent_image_ids(
        len(images), latent_height // 2, latent_width // 2, device, dtype
    )
    guide = torch.full((len(images),), guidance, device=device, dtype=torch.float32)
    for timestep in timesteps:
        prediction = components.transformer(
            torch.cat((latents, condition), dim=2).to(dtype),
            embeds.to(dtype),
            pooled.to(dtype),
            (timestep.expand(len(images)).to(dtype) / 1000),
            ids,
            text_ids,
            guide,
        )
        latents = scheduler.step(prediction, timestep, latents)
    unpacked = _unpack_latents(latents, height, width, vae_scale)
    decoded = vae.decode(
        (unpacked / vae.scaling_factor + (vae.shift_factor or 0.0)).to(
            vae.decoder.conv_in.weight.dtype
        )
    )
    return (
        ((decoded / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().cpu() * 255)
        .round()
        .to(torch.uint8)
    )


class VisualClozeBackend(TextToImageBackend):
    def __init__(
        self,
        model: str | os.PathLike[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        model_id: str | None = None,
    ) -> None:
        super().__init__()
        self.source = Path(model)
        self.model_id = model_id
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        if self.dtype not in _DTYPES:
            raise ValueError("dtype must be torch.float32, torch.float16 or torch.bfloat16")
        self.components: FluxComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown VisualCloze options: {', '.join(unknown)}")
        result = OPTION_DEFAULTS | merged
        if not isinstance(result["content_prompt"], str | type(None)):
            raise TypeError("content_prompt must be a string or None")
        grid = result["image"]
        if (
            not isinstance(grid, list)
            or not grid
            or not all(isinstance(row, list) and row for row in grid)
        ):
            raise ValueError("image must be a nonempty list of nonempty rows")
        if len({len(row) for row in grid}) != 1:
            raise ValueError("All image rows must have the same length")
        if not any(image is None for image in grid[-1]):
            raise ValueError("The last row must contain at least one None target")
        if any(image is None for row in grid[:-1] for image in row):
            raise ValueError("Earlier rows cannot contain None")
        converted: list[list[Any | None]] = []
        for row in grid:
            converted.append([None if image is None else as_pil(image) for image in row])
        result["image"] = converted
        for name in ("steps", "max_sequence_length"):
            if type(result[name]) is not int or result[name] <= 0:
                raise ValueError(f"{name} must be a positive int")
        if result["steps"] > 1000 or result["max_sequence_length"] > 512:
            raise ValueError(
                f"{('steps' if result['steps'] > 1000 else 'max_sequence_length')} is too large"
            )
        for name in ("guidance", "upsampling_strength"):
            value = result[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if result["guidance"] < 0 or not 0 <= result["upsampling_strength"] <= 1:
            raise ValueError("guidance must be >= 0 and upsampling_strength must be in [0, 1]")
        if result["seed"] is not None and (
            type(result["seed"]) is not int or not 0 <= result["seed"] < 2**64
        ):
            raise ValueError("seed must be an int in [0, 2**64)")
        dimensions = (result["upsampling_height"], result["upsampling_width"])
        if (dimensions[0] is None) != (dimensions[1] is None):
            raise ValueError("Give both upsampling_height and upsampling_width, or neither")
        if dimensions[0] is not None:
            for name in ("upsampling_height", "upsampling_width"):
                if type(result[name]) is not int or result[name] <= 0 or result[name] % 16:
                    raise ValueError(f"{name} must be a positive multiple of 16")
        return result

    def _load(self) -> None:
        self.components = FluxComponents.load(
            self.source, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        generator = torch.Generator("cpu").manual_seed(seed)
        vae_scale = 2 ** (len(self.components.vae.config["block_out_channels"]) - 1)
        resolution = (
            int(MODEL_CATALOG[self.model_id].config["resolution"]) if self.model_id else 384
        )
        grid, sizes, targets = preprocess_image(
            options["image"], resolution=resolution, vae_scale=vae_scale
        )
        masks = []
        for row_index, row in enumerate(grid):
            masks.append(
                [
                    torch.full((1, 1, row[0].shape[2], row[0].shape[3]), value)
                    for value in (targets if row_index == len(grid) - 1 else [0] * len(row))
                ]
            )
        dtype = cast(torch.dtype, self.components.transformer.x_embedder.weight.dtype)
        latents, condition, ids = _generation_latents(
            self.components, grid, masks, generator, self.device, dtype, vae_scale
        )
        prompt_embeds, pooled, text_ids = encode_prompt(
            self.components,
            _prompt(inputs.prompt, options["content_prompt"], len(grid), len(grid[0])),
            self.device,
            1,
            options["max_sequence_length"],
        )
        scheduler = FlowMatchEulerDiscreteScheduler(self.components.scheduler_config)
        scheduler.set_timesteps(
            options["steps"],
            sigmas=_sigmas(options["steps"]),
            mu=calculate_shift(latents.shape[1], self.components.scheduler_config),
        )
        final = _run_denoising(
            self.components,
            latents,
            condition,
            ids,
            prompt_embeds,
            pooled,
            text_ids,
            options["steps"],
            float(options["guidance"]),
            scheduler,
        )
        generated = _decode_generation(self.components, final, sizes, targets, vae_scale)
        if options["upsampling_strength"]:
            up_height, up_width = options["upsampling_height"], options["upsampling_width"]
            if up_height is None:
                first = next((image for image in options["image"][-1] if image is not None), None)
                if first is None:
                    up_height = up_width = 1024
                else:
                    up_width, up_height = first.width, first.height
                    if up_width * up_height > 1024**2:
                        aspect = up_width / up_height
                        up_height = int((1024**2 / aspect) ** 0.5)
                        up_width = int(up_height * aspect)
                    up_height, up_width = up_height // 16 * 16, up_width // 16 * 16
            pixels = _upsample(
                self.components,
                generated,
                options["content_prompt"] or "",
                up_height,
                up_width,
                float(options["upsampling_strength"]),
                options["steps"],
                float(options["guidance"]),
                options["max_sequence_length"],
                generator,
                self.device,
            )
        else:
            up_height = up_width = None
            pixels = torch.stack(
                [
                    torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
                        image.height, image.width, 3
                    )
                    for image in generated
                ]
            )
        snapshot = self.snapshot or {}
        return ImageResult(
            tuple(Image(pixel) for pixel in pixels),
            {
                "model_id": snapshot.get("model_id", self.model_id),
                "revision": snapshot.get("revision"),
                "seed": seed,
                "steps": options["steps"],
                "guidance": float(options["guidance"]),
                "max_sequence_length": options["max_sequence_length"],
                "content_prompt": options["content_prompt"],
                "image_input": True,
                "resolution": resolution,
                "upsampling_strength": float(options["upsampling_strength"]),
                "upsampling_height": up_height,
                "upsampling_width": up_width,
                "scheduler": "flow_match_euler",
                "device": str(self.device),
                "dtype": str(self.dtype).removeprefix("torch."),
            },
        )
