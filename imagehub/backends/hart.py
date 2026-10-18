"""HART text-to-image backend."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.base import TextToImageBackend
from imagehub.backends.stable_diffusion import resolve_device
from imagehub.checkpoints import load_weights, read_json, read_snapshot
from imagehub.image import Image
from imagehub.models.hart import HARTAutoencoder, HARTTransformer
from imagehub.models.qwen2 import Qwen2Model
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import ImageResult, T2IRequest

SYSTEM_PROMPT = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual '
    "descriptions suitable for image generation.\n\n"
    "Examples:\n"
    "- User Prompt: A cat sleeping -> A small, fluffy white cat curled up in a round shape, "
    "sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red "
    "flowers.\n"
    "- User Prompt: A busy city street -> A bustling city street scene at dusk, featuring "
    "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker "
    "bus passing by towering glass skyscrapers.\n\n"
    "Please generate only the enhanced description for the prompt below and DO NOT include any "
    'additional sentences. Start your response with "Enhanced Prompt:".\n\nUser Prompt:\n'
)

OPTION_DEFAULTS: dict[str, Any] = {
    "cfg_scale": 4.5,
    "seed": 1,
    "width": 1024,
    "height": 1024,
    "top_p": 0.0,
    "more_smooth": False,
    "use_system_prompt": True,
}
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _load_model(model: Any, path: Path, *, device: torch.device, dtype: torch.dtype) -> Any:
    model.load_checkpoint(load_weights(path, skip_prefixes=model.skipped_prefixes))
    model.init_non_persistent_buffers()
    return model.to(device=device, dtype=dtype).eval()


@dataclass
class HARTComponents:
    tokenizer: Qwen2Tokenizer
    text_encoder: Qwen2Model
    transformer: HARTTransformer
    autoencoder: HARTAutoencoder

    @classmethod
    def load(
        cls, directory: str | os.PathLike[str], *, device: torch.device, dtype: torch.dtype
    ) -> HARTComponents:
        root = Path(directory)
        text_config = read_json(root / "text_encoder" / "config.json")
        hart_config = read_json(root / "llm" / "config.json")
        vae_config = read_json(root / "tokenizer" / "config.json")
        with torch.device("meta"):
            text_encoder = Qwen2Model(text_config)
            transformer = HARTTransformer(
                hart_config
                | {
                    "latent_channels": vae_config["z_channels"],
                    "vocab_size": vae_config["vocab_size"],
                }
            )
            autoencoder = HARTAutoencoder(vae_config)
        text_encoder = _load_model(text_encoder, root / "text_encoder", device=device, dtype=dtype)
        transformer = _load_model(
            transformer, root / "llm" / "ema_model.bin", device=device, dtype=dtype
        )
        autoencoder = _load_model(
            autoencoder, root / "tokenizer" / "pytorch_model.bin", device=device, dtype=dtype
        )
        return cls(Qwen2Tokenizer(root / "text_encoder"), text_encoder, transformer, autoencoder)


def sample_tokens(
    logits: Tensor, *, top_k: int, top_p: float, generator: torch.Generator
) -> Tensor:
    if top_k:
        cutoff = logits.topk(top_k, dim=-1, sorted=False).values.amin(-1, keepdim=True)
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
    if top_p > 0:
        sorted_logits, indices = logits.sort(dim=-1)
        remove = sorted_logits.softmax(-1).cumsum(-1) <= 1 - top_p
        remove[..., -1] = False
        logits = logits.masked_fill(remove.scatter(-1, indices, remove), -torch.inf)
    batch, length, vocab = logits.shape
    return torch.multinomial(
        logits.softmax(-1).reshape(-1, vocab), 1, replacement=True, generator=generator
    ).view(batch, length)


def _diffusion_sample(
    model: HARTTransformer,
    condition: Tensor,
    guidance: float,
    generator: torch.Generator,
) -> Tensor:
    latent = model.config["latent_channels"]
    half = condition.shape[0] // 2
    noise = torch.randn((half, latent), generator=generator, dtype=condition.dtype)
    noise = torch.cat((noise, noise)).to(condition.device)

    # The published sampler builds an eight-step SpacedDiffusion schedule in
    # float64, then converts each coefficient to float32 before tensor math.
    # Reproduce those coefficients and its operation order exactly: applying
    # the general DDPM formulas to the original 1,000-step cumulative products
    # is mathematically equivalent, but differs after float32 rounding.
    alpha_cumprod = 1.0
    selected_cumprod: list[float] = []
    selected = [round(index * 999 / 7) for index in range(8)]
    selected_set = set(selected)
    for index in range(1000):
        start = math.cos((index / 1000 + 0.008) / 1.008 * math.pi / 2) ** 2
        end = math.cos(((index + 1) / 1000 + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_cumprod *= 1 - min(1 - end / start, 0.999)
        if index in selected_set:
            selected_cumprod.append(alpha_cumprod)

    betas: list[float] = []
    previous = 1.0
    for cumulative in selected_cumprod:
        betas.append(1 - cumulative / previous)
        previous = cumulative
    alphas = [1 - beta for beta in betas]
    cumulative = 1.0
    spaced_cumprod = []
    for alpha in alphas:
        cumulative *= alpha
        spaced_cumprod.append(cumulative)
    previous_cumprod = [1.0, *spaced_cumprod[:-1]]
    posterior_variance = [
        beta * (1 - previous_alpha) / (1 - alpha)
        for beta, previous_alpha, alpha in zip(betas, previous_cumprod, spaced_cumprod, strict=True)
    ]
    posterior_log_variance = [
        math.log(posterior_variance[1]),
        *(math.log(value) for value in posterior_variance[1:]),
    ]
    coefficient_1 = [
        beta * math.sqrt(previous_alpha) / (1 - alpha)
        for beta, previous_alpha, alpha in zip(betas, previous_cumprod, spaced_cumprod, strict=True)
    ]
    coefficient_2 = [
        (1 - previous_alpha) * math.sqrt(step_alpha) / (1 - alpha)
        for step_alpha, previous_alpha, alpha in zip(
            alphas, previous_cumprod, spaced_cumprod, strict=True
        )
    ]
    constants = torch.tensor(
        [
            [
                math.sqrt(1 / alpha),
                math.sqrt(1 / alpha - 1),
                coefficient_1[index],
                coefficient_2[index],
                posterior_log_variance[index],
                math.log(betas[index]),
            ]
            for index, alpha in enumerate(spaced_cumprod)
        ],
        dtype=torch.float32,
        device=condition.device,
    )

    for index in reversed(range(8)):
        timestep = selected[index]
        t = torch.full((len(noise),), timestep, device=condition.device, dtype=torch.long)
        combined = torch.cat((noise[:half], noise[:half]))
        output = model.diffloss.net(combined, t, condition)
        epsilon, variance = output.split(latent, dim=1)
        conditional, unconditional = epsilon.chunk(2)
        guided = conditional * (1 + guidance) - unconditional * guidance
        epsilon = torch.cat((guided, guided))
        reciprocal, reciprocal_minus_one, coef_1, coef_2, minimum, maximum = constants[index]
        predicted = (reciprocal * noise - reciprocal_minus_one * epsilon).clamp(-1, 1)
        mean = coef_1 * predicted + coef_2 * noise
        fraction = (variance + 1) / 2
        log_variance = fraction * maximum + (1 - fraction) * minimum
        variance_noise = torch.randn(
            noise.shape,
            generator=generator,
            device="cpu" if generator.device.type == "cpu" else noise.device,
            dtype=noise.dtype,
        ).to(noise.device)
        nonzero = float(index != 0)
        noise = mean + nonzero * torch.exp(0.5 * log_variance) * variance_noise
    return noise * 0.6


@torch.inference_mode()
def generate(
    components: HARTComponents,
    context: Tensor,
    context_mask: Tensor,
    positions: Tensor,
    *,
    cfg_scale: float,
    top_p: float,
    generator: torch.Generator,
    diffusion_generator: torch.Generator | None = None,
) -> Tensor:
    model, vae = components.transformer, components.autoencoder
    batch = len(context)
    condition = model.context_embed(
        model.context_norm(torch.cat((context, torch.zeros_like(context))))
    )
    level_positions = model.lvl_embed(model.lvl_1L)
    next_tokens = condition + level_positions[:, : model.config["context_token"]]
    positions = torch.cat((positions, torch.zeros_like(positions)))
    f_hat = context.new_zeros(
        batch,
        model.config["latent_channels"],
        model.config["patch_nums"][-1],
        model.config["patch_nums"][-1],
    )
    for block in model.blocks:
        block.attn.kv_caching(True)  # type: ignore[union-attr,operator]
    current_length = 0
    patch_nums = model.config["patch_nums"]
    try:
        for stage, size in enumerate(patch_nums[:-1]):
            current_length += model.config["context_token"] if stage == 0 else size * size
            hidden, logits = model.stage_forward(next_tokens, condition, stage, positions)
            ratio = stage / (len(patch_nums) - 1)
            logits = (1 + cfg_scale * ratio) * logits[:batch] - cfg_scale * ratio * logits[batch:]
            if stage == 0:
                logits = logits[:, -1:]
            ids = sample_tokens(
                logits,
                top_k=600 if stage < 7 else 300,
                top_p=top_p,
                generator=generator,
            )
            discrete = (
                vae.quantize.embedding(ids)
                .transpose(1, 2)
                .reshape(batch, model.config["latent_channels"], size, size)
            )
            f_hat, next_map = vae.quantize.next_input(stage, patch_nums, f_hat, discrete)
            next_tokens = model.word_embed(next_map.flatten(2).transpose(1, 2))
            length = patch_nums[stage + 1] ** 2
            next_tokens = (
                next_tokens + level_positions[:, current_length : current_length + length]
            ).repeat(2, 1, 1)

        stage = len(patch_nums) - 1
        hidden, logits = model.stage_forward(next_tokens, condition, stage, positions)
        final_guidance = cfg_scale * (stage - 1) / (len(patch_nums) - 1)
        logits = (1 + final_guidance) * logits[:batch] - final_guidance * logits[batch:]
        ids = sample_tokens(logits, top_k=300, top_p=top_p, generator=generator)
        discrete = vae.quantize.embedding(ids)
        discrete_condition = model.word_embed(discrete).repeat(2, 1, 1)
        diffusion_condition = model.decoder_norm(hidden + discrete_condition).flatten(0, 1)
        residual = _diffusion_sample(
            model,
            diffusion_condition,
            final_guidance,
            diffusion_generator or generator,
        )
        residual = residual.view(2 * batch, -1, model.config["latent_channels"])[:batch]
        tokens = discrete + residual
        f_hat = f_hat + tokens.transpose(1, 2).reshape(
            batch, model.config["latent_channels"], patch_nums[-1], patch_nums[-1]
        )
        return vae.decode(f_hat).add(1).mul(0.5)
    finally:
        for block in model.blocks:
            block.attn.kv_caching(False)  # type: ignore[union-attr,operator]


@torch.inference_mode()
def encode_prompt(
    components: HARTComponents, prompt: str, device: torch.device, use_system_prompt: bool
) -> tuple[Tensor, Tensor, Tensor]:
    ids, mask = components.tokenizer(prompt, 300)
    ids, mask = ids.to(device), mask.to(device)
    positions = mask.long().cumsum(1) - 1
    if use_system_prompt:
        system_ids = components.tokenizer.tokenize(SYSTEM_PROMPT)[:300]
        prefix = torch.tensor([system_ids], dtype=torch.long, device=device)
        prefix_mask = torch.ones_like(prefix, dtype=torch.bool)
        hidden = components.text_encoder(
            torch.cat((prefix, ids), dim=1), torch.cat((prefix_mask, mask), dim=1)
        )[:, len(system_ids) :]
    else:
        hidden = components.text_encoder(ids, mask)
    return hidden.float(), mask, positions


class HARTBackend(TextToImageBackend):
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
        self.components: HARTComponents | None = None
        self.snapshot: dict[str, Any] | None = None

    def validate_options(self, inputs: T2IRequest, options: dict[str, Any]) -> dict[str, Any]:
        merged = dict(inputs.parameters) | options
        unknown = sorted(set(merged) - set(OPTION_DEFAULTS))
        if unknown:
            raise TypeError(f"Unknown HART options: {', '.join(unknown)}")
        result = OPTION_DEFAULTS | merged
        if isinstance(result["cfg_scale"], bool) or not isinstance(
            result["cfg_scale"], int | float
        ):
            raise ValueError("cfg_scale must be a finite number >= 0")
        if not math.isfinite(result["cfg_scale"]) or result["cfg_scale"] < 0:
            raise ValueError("cfg_scale must be a finite number >= 0")
        if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**64:
            raise ValueError("seed must be an int in [0, 2**64)")
        if (result["width"], result["height"]) != (1024, 1024):
            raise ValueError("HART width and height must be 1024")
        if not isinstance(result["top_p"], int | float) or not 0 <= result["top_p"] <= 1:
            raise ValueError("top_p must be in [0, 1]")
        for name in ("more_smooth", "use_system_prompt"):
            if type(result[name]) is not bool:
                raise ValueError(f"{name} must be a bool")
        if result["more_smooth"]:
            raise ValueError("more_smooth is not supported")
        return result

    def _load(self) -> None:
        self.components = HARTComponents.load(self.source, device=self.device, dtype=self.dtype)
        self.snapshot = read_snapshot(self.source)

    def _unload(self) -> None:
        self.components = None
        self.snapshot = None

    @torch.inference_mode()
    def _infer(self, inputs: T2IRequest, **options: Any) -> ImageResult:
        assert self.components is not None
        seed = options["seed"] if options["seed"] is not None else secrets.randbits(63)
        context, mask, positions = encode_prompt(
            self.components, inputs.prompt, self.device, options["use_system_prompt"]
        )
        pixels = generate(
            self.components,
            context,
            mask,
            positions,
            cfg_scale=float(options["cfg_scale"]),
            top_p=float(options["top_p"]),
            generator=torch.Generator("cpu").manual_seed(seed),
            diffusion_generator=torch.Generator("cpu").manual_seed(seed),
        )
        images = pixels.mul(255).permute(0, 2, 3, 1).to("cpu", torch.uint8)
        snapshot = self.snapshot or {}
        metadata = {
            "model_id": snapshot.get("model_id"),
            "revision": snapshot.get("revision"),
            "seed": seed,
            "cfg_scale": float(options["cfg_scale"]),
            "width": 1024,
            "height": 1024,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        return ImageResult(tuple(Image(image) for image in images), metadata)
