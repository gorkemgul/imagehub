"""JoyAI-Image-Edit inference backend."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import CheckpointError, read_json, read_snapshot
from imagehub.image import Image, as_pil
from imagehub.models.autoencoder_kl_wan import AutoencoderKLWan
from imagehub.models.joyimage_transformer import JoyImageEditTransformer3DModel
from imagehub.models.qwen3_vl import Qwen3VLForConditionalGeneration
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

_DEFAULTS: dict[str, Any] = {
    "steps": 40,
    "guidance": 4.0,
    "width": None,
    "height": None,
    "seed": None,
    "image": None,
}
_TEMPLATE = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
_MULTI_TEMPLATE = (
    "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "{}<|im_start|>assistant\n"
)


@dataclass
class _Components:
    tokenizer: Qwen2Tokenizer
    text: Qwen3VLForConditionalGeneration
    transformer: JoyImageEditTransformer3DModel
    vae: AutoencoderKLWan
    scheduler: dict[str, Any]

    @classmethod
    def load(cls, root: Path, device: torch.device, dtype: torch.dtype) -> _Components:
        needed = ("tokenizer", "text_encoder", "transformer", "vae", "scheduler")
        if any(not (root / part).is_dir() for part in needed):
            raise CheckpointError(f"{root} is not a JoyAI-Image snapshot")
        return cls(
            Qwen2Tokenizer(root / "tokenizer"),
            Qwen3VLForConditionalGeneration.from_pretrained(
                root / "text_encoder", device=device, dtype=dtype
            ),
            JoyImageEditTransformer3DModel.from_pretrained(
                root / "transformer", device=device, dtype=dtype
            ),
            AutoencoderKLWan.from_pretrained(root / "vae", device=device, dtype=dtype),
            read_json(root / "scheduler" / "scheduler_config.json"),
        )


def _bucket(height: int, width: int) -> tuple[int, int]:
    ratio = height / width
    candidates = (
        (512, 1792),
        (512, 1856),
        (512, 1920),
        (512, 1984),
        (512, 2048),
        (576, 1600),
        (576, 1664),
        (576, 1728),
        (576, 1792),
        (640, 1472),
        (640, 1536),
        (640, 1600),
        (704, 1344),
        (704, 1408),
        (704, 1472),
        (768, 1216),
        (768, 1280),
        (768, 1344),
        (832, 1152),
        (832, 1216),
        (896, 1088),
        (896, 1152),
        (960, 1024),
        (960, 1088),
        (1024, 960),
        (1024, 1024),
        (1088, 896),
        (1088, 960),
        (1152, 832),
        (1152, 896),
        (1216, 768),
        (1216, 832),
        (1280, 768),
        (1344, 704),
        (1344, 768),
        (1408, 704),
        (1472, 640),
        (1472, 704),
        (1536, 640),
        (1600, 576),
        (1600, 640),
        (1664, 576),
        (1728, 576),
        (1792, 512),
        (1792, 576),
        (1856, 512),
        (1920, 512),
        (1984, 512),
        (2048, 512),
    )
    return min(candidates, key=lambda item: abs(item[0] / item[1] - ratio))


def _resize_crop(image: Any, height: int, width: int) -> Any:
    from PIL import Image as PILImage

    pil = as_pil(image)
    scale = max(height / pil.height, width / pil.width)
    resized = pil.resize(
        (math.ceil(pil.width * scale), math.ceil(pil.height * scale)), PILImage.Resampling.BILINEAR
    )
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _pixels(image: Any, height: int, width: int) -> Tensor:
    cropped = _resize_crop(image, height, width)
    return (
        torch.frombuffer(bytearray(cropped.tobytes()), dtype=torch.uint8)
        .reshape(1, height, width, 3)
        .permute(0, 3, 1, 2)
        .float()
        .div(127.5)
        .sub(1)
    )


def _vision_pixels(image: Any, patch: int = 16, merge: int = 2) -> tuple[Tensor, Tensor]:
    """Apply the Qwen image processor's resize, normalization, and merge-block patch order."""
    from PIL import Image as PILImage

    pil = as_pil(image)
    factor = patch * merge
    h, w = round(pil.height / factor) * factor, round(pil.width / factor) * factor
    if h * w < 256 * 256:
        scale = math.sqrt((256 * 256) / (pil.height * pil.width))
        h = math.ceil(pil.height * scale / factor) * factor
        w = math.ceil(pil.width * scale / factor) * factor
    pil = pil.resize((w, h), PILImage.Resampling.BICUBIC)
    pixels = (
        torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8)
        .reshape(h, w, 3)
        .permute(2, 0, 1)
        .float()
        .sub(127.5)
        .div(127.5)
    )
    gh, gw = h // patch, w // patch
    patches = pixels.reshape(3, gh // merge, merge, patch, gw // merge, merge, patch)
    patches = patches.permute(1, 4, 2, 5, 0, 3, 6)
    patches = patches.unsqueeze(5).expand(-1, -1, -1, -1, -1, 2, -1, -1)
    return patches.reshape(gh * gw, 3 * 2 * patch * patch), torch.tensor([[1, gh, gw]])


def _conditioning(
    components: _Components,
    prompt: str,
    image: Any | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if image is None:
        text = _TEMPLATE.format(prompt)
        ids = torch.tensor([components.tokenizer.tokenize(text)], device=device)
        mask = torch.ones_like(ids)
        return components.text(ids, mask)[-1][:, 34:].to(dtype), mask[:, 34:]

    pixels, grid = _vision_pixels(image)
    count = int(grid.prod() // 4)
    user = (
        "<|im_start|>user\n<|vision_start|>"
        + "<|image_pad|>" * count
        + f"<|vision_end|>{prompt}<|im_end|>\n"
    )
    text = _MULTI_TEMPLATE.format(user)
    row = components.tokenizer.tokenize(text)
    ids = torch.tensor([row], device=device)
    mask = torch.ones_like(ids)
    image_id = components.text.config["image_token_id"]
    types = ids.eq(image_id).long()
    states = components.text(
        ids,
        mask,
        pixel_values=pixels.to(device, dtype),
        image_grid_thw=grid.to(device),
        mm_token_type_ids=types,
    )[-1]
    return states[:, 34:].to(dtype), mask[:, 34:]


class JoyAIImageBackend(TextToImageBackend):
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
        self.device = resolve_device(device)
        self.dtype = dtype or torch.float32
        self.model_id = model_id
        self.components: _Components | None = None
        self.snapshot: dict[str, Any] | None = None
        self.defaults = _DEFAULTS | (
            dict(MODEL_CATALOG[model_id].config.get("defaults", {})) if model_id else {}
        )

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        selected = self.defaults | dict(inputs.parameters) | options
        unknown = sorted(set(selected) - set(_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown JoyAI-Image options: {', '.join(unknown)}")
        if selected["image"] is None and self.model_id == "joyai-image-edit-plus":
            raise ValueError("image is required for joyai-image-edit-plus")
        if type(selected["steps"]) is not int or not 0 < selected["steps"] <= 1000:
            raise ValueError("steps must be an int in [1, 1000]")
        if (
            not isinstance(selected["guidance"], int | float)
            or isinstance(selected["guidance"], bool)
            or not math.isfinite(selected["guidance"])
        ):
            raise ValueError("guidance must be a finite number")
        if selected["seed"] is not None and (
            type(selected["seed"]) is not int or not 0 <= selected["seed"] < 2**64
        ):
            raise ValueError("seed must be an int in [0, 2**64)")
        if (selected["height"] is None) != (selected["width"] is None):
            raise ValueError("height and width must be given together")
        if selected["image"] is not None:
            as_pil(selected["image"])
        return selected

    def _load(self) -> None:
        self.components = _Components.load(self.source, self.device, self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    def _plus_latents(
        self, noise: Tensor, reference: Tensor
    ) -> tuple[Tensor, Tensor, list[list[tuple[int, int, int]]]]:
        assert self.components is not None
        pt, ph, pw = self.components.transformer.patch_size
        patches, shapes, masks = [], [], []
        for target, item in ((True, noise[0, 0]), (False, reference[0])):
            c, t, h, w = item.shape
            shape = (t // pt, h // ph, w // pw)
            block = item.reshape(c, shape[0], pt, shape[1], ph, shape[2], pw)
            block = block.permute(1, 3, 5, 0, 2, 4, 6).reshape(-1, c, pt, ph, pw)
            patches.append(block)
            shapes.append(shape)
            masks.append(torch.full((len(block),), target, device=item.device, dtype=torch.bool))
        return torch.cat(patches)[None], torch.cat(masks)[None], [shapes]

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        image = options["image"]
        plus = self.components.transformer.plus
        height, width = (
            (
                (options["height"], options["width"])
                if plus
                else _bucket(options["height"], options["width"])
            )
            if options["height"]
            else _bucket(
                *(
                    (as_pil(image).height, as_pil(image).width)
                    if image is not None
                    else (1024, 1024)
                )
            )
        )
        reference_size = (
            _bucket(as_pil(image).height, as_pil(image).width)
            if plus and image is not None
            else (height, width)
        )
        processed_image = _resize_crop(image, *reference_size) if image is not None else None
        states, states_mask = _conditioning(
            self.components, inputs.prompt or " ", processed_image, self.device, self.dtype
        )
        seed = secrets.randbits(63) if options["seed"] is None else options["seed"]
        noise = torch.randn(
            (1, 1, self.components.transformer.config["in_channels"], 1, height // 8, width // 8),
            generator=torch.Generator("cpu").manual_seed(seed),
            dtype=self.dtype,
        ).to(self.device)
        refs = None
        if image is not None:
            source_pixels = _pixels(image, *reference_size).to(self.device, self.dtype)
            encoded = (
                self.components.vae.encode(source_pixels)
                if plus
                else self.components.vae.encode_sample(source_pixels)
            )
            mean = torch.tensor(
                self.components.vae.latents_mean, device=self.device, dtype=self.dtype
            )[None, :, None, None]
            std = torch.tensor(
                self.components.vae.latents_std, device=self.device, dtype=self.dtype
            )[None, :, None, None]
            refs = ((encoded - mean) / std).unsqueeze(1).unsqueeze(3)
        target_mask = None
        shape_list = None
        if plus:
            assert refs is not None
            latents, target_mask, shape_list = self._plus_latents(noise, refs[:, 0])
        else:
            latents = noise if refs is None else torch.cat((refs, noise), 1)
        negative = None
        negative_mask = None
        if options["guidance"] > 1:
            negative, negative_mask = _conditioning(
                self.components, " " if plus else "", processed_image, self.device, self.dtype
            )
            if plus:
                length = max(states.shape[1], negative.shape[1])
                states = F.pad(states, (0, 0, 0, length - states.shape[1]))
                negative = F.pad(negative, (0, 0, 0, length - negative.shape[1]))
                states_mask = F.pad(states_mask, (0, length - states_mask.shape[1]))
                negative_mask = F.pad(negative_mask, (0, length - negative_mask.shape[1]))
                states = torch.cat((negative, states))
                states_mask = torch.cat((negative_mask, states_mask))
        scheduler = FlowMatchEulerDiscreteScheduler(self.components.scheduler)
        scheduler.set_timesteps(options["steps"])
        for timestep in scheduler.timesteps:
            if plus:
                assert target_mask is not None and shape_list is not None
                clean = latents.clone()
                model_input = torch.cat((latents, latents)) if negative is not None else latents
                prediction = self.components.transformer(
                    model_input,
                    timestep.expand(len(model_input)).to(self.dtype),
                    states,
                    states_mask,
                    shape_list * len(model_input),
                )
                if negative is not None:
                    uncond, cond = prediction.chunk(2)
                    combined = uncond + options["guidance"] * (cond - uncond)
                    cond_norm = torch.norm(cond, dim=2, keepdim=True)
                    prediction = combined * (
                        cond_norm / torch.norm(combined, dim=2, keepdim=True).clamp_min(1e-6)
                    )
            else:
                if refs is not None:
                    latents[:, :1] = refs
                prediction = self.components.transformer(
                    latents, timestep.expand(1).to(self.dtype), states
                )
                if negative is not None:
                    uncond = self.components.transformer(
                        latents, timestep.expand(1).to(self.dtype), negative
                    )
                    combined = uncond + options["guidance"] * (prediction - uncond)
                    cond_norm = torch.norm(prediction, dim=2, keepdim=True)
                    prediction = combined * (
                        cond_norm / torch.norm(combined, dim=2, keepdim=True).clamp_min(1e-6)
                    )
            latents = scheduler.step(prediction, timestep, latents)
            if plus:
                assert target_mask is not None
                latents[~target_mask] = clean[~target_mask]
        if plus:
            assert shape_list is not None
            lt, lh, lw = shape_list[0][0]
            count = lt * lh * lw
            pt, ph, pw = self.components.transformer.patch_size
            target = latents[0, :count]
            final = target.reshape(lt, lh, lw, target.shape[1], pt, ph, pw)
            final = final.permute(3, 0, 4, 1, 5, 2, 6).reshape(
                1, target.shape[1], lt * pt, lh * ph, lw * pw
            )[:, :, 0]
        else:
            final = latents.flatten(0, 1)[:, :, 0]
        mean = torch.tensor(self.components.vae.latents_mean, device=self.device, dtype=self.dtype)[
            None, :, None, None
        ]
        std = torch.tensor(self.components.vae.latents_std, device=self.device, dtype=self.dtype)[
            None, :, None, None
        ]
        decoded = self.components.vae.decode(final * std + mean)
        if not plus:
            decoded = decoded.unflatten(0, (latents.shape[0], latents.shape[1]))[:, -1]
        pixels = (
            decoded.add(1)
            .div(2)
            .clamp(0, 1)
            .permute(0, 2, 3, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .cpu()
        )
        meta = {
            "model_id": self.model_id,
            "revision": (self.snapshot or {}).get("revision"),
            "seed": seed,
            "steps": options["steps"],
            "guidance": options["guidance"],
            "width": width,
            "height": height,
            "image_input": image is not None,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(item) for item in pixels), meta)
