"""A small LoRA training loop: seeded sampling, AdamW, schedules, evaluation, data-parallel
processes, checkpoints and exact resume."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.stable_diffusion import (
    StableDiffusionComponents,
    decode_latents,
    denoise,
    initial_latents,
)
from imagehub.checkpoints import save_safetensors
from imagehub.lora import LoRAConfig, LoRAInjection, inject_lora
from imagehub.lora_formats import read_lora_file
from imagehub.training.arguments import TrainingArguments
from imagehub.training.data import StableDiffusionCollator, _pil
from imagehub.training.distributed import (
    average_gradients,
    barrier,
    gather,
    reduce_sum,
    setup_distributed,
)
from imagehub.training.schedules import lr_lambda
from imagehub.training.sd1 import StableDiffusionLoRAObjective
from imagehub.training.specs import get_training_spec, list_training_specs

_DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ADAPTER_FILE = "adapter.safetensors"
CONFIG_FILE = "adapter_config.json"
STATE_FILE = "trainer_state.pt"
Collator = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Tensor]]


@dataclass
class TrainOutput:
    global_step: int
    training_loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    log_history: list[dict[str, float]] = field(default_factory=list)


def _check_dataset(dataset: Any, name: str) -> None:
    if not hasattr(dataset, "__len__") or len(dataset) == 0:
        raise ValueError(f"{name} must be a non-empty sized dataset")
    first = dataset[0]
    if not isinstance(first, Mapping) or not {"image", "text"} <= set(first):
        raise ValueError(f'{name} items must be mappings with "image" and "text"')


class Trainer:
    """Train a LoRA adapter for the U-Net of a Stable Diffusion 1.x / 2.x checkpoint.

    Datasets are sized and their items are ``{"image": PIL image or (3, H, W) tensor in [0, 1],
    "text": str}``, such as :class:`ImageFolderDataset`. ``data_collator`` replaces the default
    :class:`StableDiffusionCollator`; it maps a list of items to ``{"pixel_values": (B, 3, H, W)
    in [-1, 1], "input_ids": (B, 77)}``. Under ``torchrun`` every process trains on its own share
    of each batch and the LoRA gradients are averaged.
    """

    def __init__(
        self,
        model: str,
        checkpoint: str | os.PathLike[str],
        args: TrainingArguments,
        train_dataset: Any,
        eval_dataset: Any = None,
        data_collator: Collator | None = None,
    ) -> None:
        if not isinstance(args, TrainingArguments):
            raise TypeError("args must be a TrainingArguments")
        self.spec = get_training_spec(model)
        if not self.spec.support.is_trainable:
            trainable = list_training_specs(support="native")
            supported = ", ".join(s.model_id for s in trainable)
            raise ValueError(f"{model!r} cannot be trained yet. Trainable models: {supported}")
        _check_dataset(train_dataset, "train_dataset")
        if eval_dataset is not None:
            _check_dataset(eval_dataset, "eval_dataset")
        if data_collator is not None and not callable(data_collator):
            raise TypeError("data_collator must be callable")
        self.model_id = model
        self.checkpoint = Path(checkpoint)
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.global_step = 0
        self.log_history: list[dict[str, float]] = []
        self.components: StableDiffusionComponents | None = None
        self.injection: LoRAInjection | None = None
        self._run: Any = None

    # Setup -------------------------------------------------------------------------------

    def _setup(self) -> None:
        if self.components is not None:
            return
        args = self.args
        self.distributed, self.device = setup_distributed(args.device)
        self.dtype = _DTYPES[args.mixed_precision]
        components = StableDiffusionComponents.load(
            self.checkpoint, device=self.device, dtype=self.dtype, model_id=self.model_id
        )
        for module in (components.unet, components.text_encoder, components.vae):
            module.requires_grad_(False)
        components.text_encoder.eval()
        components.vae.eval()
        self.target_modules = args.lora_target_modules or self.spec.target_modules
        injection = inject_lora(
            components.unet,
            LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=self.target_modules,
                seed=args.seed,
                init="gaussian",
            ),
        )
        # Trainable parameters and optimizer state stay in float32 under mixed precision.
        for parameter in injection.parameters():
            parameter.data = parameter.data.float()
        components.unet.train()
        self.objective = StableDiffusionLoRAObjective(
            components,
            prediction_type=args.prediction_type,
            snr_gamma=args.snr_gamma,
            noise_offset=args.noise_offset,
        )
        self.parameters = list(injection.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda(args.lr_scheduler, args.lr_warmup_steps, args.max_steps)
        )
        self.scaler = (
            torch.amp.GradScaler(self.device.type) if args.mixed_precision == "fp16" else None
        )
        rank = self.distributed.rank
        # Every process shares the data order; augmentation and noise differ per process.
        self.order_generator = torch.Generator().manual_seed(args.seed)
        self.augment_generator = torch.Generator().manual_seed(args.seed + 1 + 2 * rank)
        self.noise_generator = torch.Generator().manual_seed(args.seed + 2 + 2 * rank)
        self.collator: Collator = self.data_collator or StableDiffusionCollator(
            components.tokenizer,
            resolution=args.resolution,
            center_crop=args.center_crop,
            random_flip=args.random_flip,
            generator=self.augment_generator,
        )
        self.epoch = 0
        self.position = 0
        self.order = torch.randperm(len(self.train_dataset), generator=self.order_generator)
        self.components = components
        self.injection = injection

    @property
    def is_main_process(self) -> bool:
        return self.components is None or self.distributed.is_main

    # Data --------------------------------------------------------------------------------

    def _next_indices(self, count: int) -> list[int]:
        indices = []
        for _ in range(count):
            if self.position >= len(self.order):
                self.epoch += 1
                self.position = 0
                self.order = torch.randperm(len(self.train_dataset), generator=self.order_generator)
            indices.append(int(self.order[self.position]))
            self.position += 1
        return indices

    def _to_device(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        if not {"pixel_values", "input_ids"} <= set(batch):
            raise ValueError('data_collator must return "pixel_values" and "input_ids"')
        pixels = batch["pixel_values"].to(device=self.device, dtype=self.dtype)
        return pixels, batch["input_ids"].to(device=self.device, dtype=torch.long)

    def _next_batch(self) -> tuple[Tensor, Tensor]:
        size = self.args.per_device_train_batch_size
        indices = self._next_indices(size * self.distributed.world_size)
        own = indices[self.distributed.rank * size : (self.distributed.rank + 1) * size]
        return self._to_device(self.collator([self.train_dataset[i] for i in own]))

    # Loop --------------------------------------------------------------------------------

    def _step(self) -> tuple[float, float]:
        args = self.args
        lr = float(self.optimizer.param_groups[0]["lr"])
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(args.gradient_accumulation_steps):
            pixels, ids = self._next_batch()
            loss = self.objective(pixels, ids, self.noise_generator)
            loss = loss / args.gradient_accumulation_steps
            (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
            total += float(loss.detach())
        average_gradients(self.parameters, self.distributed)
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters, args.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()
        self.global_step += 1
        total = reduce_sum([total], self.distributed, self.device)[0] / self.distributed.world_size
        return total, lr

    def _log(self, record: dict[str, float], run: Any) -> None:
        self.log_history.append(record)
        if not self.is_main_process:
            return
        print("  ".join(f"{k} {_format(v)}" for k, v in record.items()), flush=True)
        if run is not None:
            # Named like diffusers' training scripts, so dashboards line up.
            names = {"loss": "train_loss"}
            values = {names.get(k, k): v for k, v in record.items() if k != "step"}
            run.log(values, step=self.global_step)

    def train(self, resume_from_checkpoint: bool | str | os.PathLike[str] = False) -> TrainOutput:
        """Train until ``args.max_steps`` optimizer steps; optionally resume a checkpoint first."""
        self._setup()
        if resume_from_checkpoint:
            path = (
                self._latest_checkpoint()
                if resume_from_checkpoint is True
                else Path(resume_from_checkpoint)
            )
            self._load_checkpoint(path)
        run = self._run = self._start_reporting()
        interval = self.args.evaluation_interval
        losses = []
        started = time.perf_counter()
        last_saved = None
        last_sampled = None
        try:
            while self.global_step < self.args.max_steps:
                loss, lr = self._step()
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Loss is {loss} at step {self.global_step}")
                losses.append(loss)
                if self.global_step % self.args.logging_steps == 0:
                    self._log({"step": float(self.global_step), "loss": loss, "lr": lr}, run)
                if interval is not None and self.global_step % interval == 0:
                    if self.eval_dataset is not None:
                        metrics = self.evaluate()
                        self._log({"step": float(self.global_step)} | metrics, run)
                    if self.args.validation_prompts:
                        self.generate_samples()
                        last_sampled = self.global_step
                if self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint()
                    last_saved = self.global_step
            if losses and last_saved != self.global_step:
                self._save_checkpoint()
            if losses and self.args.validation_prompts and last_sampled != self.global_step:
                self.generate_samples()
        finally:
            self._run = None
            if run is not None:
                run.finish()
        runtime = time.perf_counter() - started
        mean = sum(losses) / len(losses) if losses else 0.0
        metrics = {
            "train_runtime": runtime,
            "train_steps_per_second": len(losses) / runtime if runtime > 0 else 0.0,
            "train_loss": mean,
        }
        return TrainOutput(self.global_step, mean, metrics, list(self.log_history))

    def _start_reporting(self) -> Any:
        if self.args.report_to != "wandb" or not self.is_main_process:
            return None
        try:
            import wandb
        except ImportError as error:
            raise ImportError('report_to="wandb" needs the wandb package') from error
        return wandb.init(project="imagehub", name=self.args.run_name, config=self.args.to_dict())

    # Evaluation --------------------------------------------------------------------------

    def evaluate(self, eval_dataset: Any = None) -> dict[str, float]:
        """Mean training loss over ``eval_dataset`` without gradients.

        Images are center-cropped and noise and timesteps are seeded by dataset position, so
        repeated evaluations of the same adapter agree, with any number of processes when the
        eval batch size is 1.
        """
        self._setup()
        assert self.components is not None
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("evaluate() needs an eval_dataset")
        if eval_dataset is not None:
            _check_dataset(eval_dataset, "eval_dataset")
        args, state = self.args, self.distributed
        collator = self.data_collator or StableDiffusionCollator(
            self.components.tokenizer,
            resolution=args.resolution,
            center_crop=True,
            random_flip=False,
            generator=torch.Generator().manual_seed(args.seed + 3),
        )
        start = len(dataset) * state.rank // state.world_size
        end = len(dataset) * (state.rank + 1) // state.world_size
        size = args.per_device_eval_batch_size
        total, count = 0.0, 0
        unet = self.components.unet
        unet.eval()
        try:
            with torch.no_grad():
                for first in range(start, end, size):
                    items = [dataset[i] for i in range(first, min(first + size, end))]
                    pixels, ids = self._to_device(collator(items))
                    # Seeded by position, so the result does not depend on the process count.
                    noise = torch.Generator().manual_seed(args.seed + 4 + first)
                    total += float(self.objective(pixels, ids, noise)) * len(items)
                    count += len(items)
        finally:
            unet.train()
        summed_loss, summed_count = reduce_sum([total, float(count)], state, self.device)
        return {"eval_loss": summed_loss / summed_count}

    def generate_samples(self) -> Path:
        """Write one image per validation prompt to ``output_dir/samples/step-N/``.

        During a ``report_to="wandb"`` run the images are also logged as ``validation``.
        """
        self._setup()
        assert self.components is not None
        args = self.args
        directory = Path(args.output_dir) / "samples" / f"step-{self.global_step}"
        if self.is_main_process:
            directory.mkdir(parents=True, exist_ok=True)
            components = self.components
            image_module = _pil()
            shape = (
                1,
                self.objective.latent_channels,
                args.resolution // self.objective.downsample,
                args.resolution // self.objective.downsample,
            )
            components.unet.eval()
            paths = []
            try:
                for index, prompt in enumerate(args.validation_prompts):
                    latents = denoise(
                        components,
                        prompt=prompt,
                        negative_prompt=None,
                        steps=args.validation_num_inference_steps,
                        guidance=args.validation_guidance_scale,
                        latents=initial_latents(args.seed, shape).to(self.device),
                    )
                    pixels = decode_latents(components, latents)[0]
                    image = image_module.frombytes(
                        "RGB", (pixels.shape[1], pixels.shape[0]), bytes(pixels.flatten().tolist())
                    )
                    image.save(directory / f"{index}.png")
                    paths.append(directory / f"{index}.png")
            finally:
                components.unet.train()
            if self._run is not None:
                import wandb

                images = [
                    wandb.Image(str(path), caption=prompt)
                    for path, prompt in zip(paths, args.validation_prompts, strict=True)
                ]
                self._run.log({"validation": images}, step=self.global_step)
        barrier(self.distributed)
        return directory

    # Saving ------------------------------------------------------------------------------

    def _adapter_tensors(self) -> dict[str, Tensor]:
        assert self.injection is not None
        return {f"unet.{k}": v.float() for k, v in self.injection.adapter_state_dict().items()}

    def _write_adapter(self, directory: Path) -> Path:
        assert self.injection is not None
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "unet.r": self.args.lora_rank,
            "unet.lora_alpha": self.args.lora_alpha,
            "unet.target_modules": list(self.injection.module_names),
        }
        path = directory / ADAPTER_FILE
        save_safetensors(
            self._adapter_tensors(),
            path,
            metadata={
                "format": "pt",
                "lora_adapter_metadata": json.dumps(metadata, indent=2, sort_keys=True),
            },
        )
        config = {
            "base_model": self.model_id,
            "format": "diffusers",
            "rank": self.args.lora_rank,
            "alpha": self.args.lora_alpha,
            "target_modules": list(self.target_modules),
            "global_step": self.global_step,
        }
        (directory / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return path

    def save_model(self, path: str | os.PathLike[str] | None = None) -> Path:
        """Write ``adapter.safetensors`` (diffusers layout) and ``adapter_config.json``.

        In a distributed run only the main process writes; every process gets the path.
        """
        if self.injection is None:
            raise RuntimeError("Call train() before save_model()")
        directory = Path(path) if path is not None else Path(self.args.output_dir)
        if self.is_main_process:
            self._write_adapter(directory)
        barrier(self.distributed)
        return directory / ADAPTER_FILE

    def _save_checkpoint(self) -> Path:
        directory = Path(self.args.output_dir) / f"checkpoint-{self.global_step}"
        streams = gather(
            {
                "augment": self.augment_generator.get_state(),
                "noise": self.noise_generator.get_state(),
            },
            self.distributed,
        )
        if self.is_main_process:
            self._write_adapter(directory)
            state = {
                "global_step": self.global_step,
                "world_size": self.distributed.world_size,
                "epoch": self.epoch,
                "position": self.position,
                "order": self.order,
                "order_rng": self.order_generator.get_state(),
                "process_rng": streams,
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "log_history": self.log_history,
                "args": self.args.to_dict(),
            }
            torch.save(state, directory / STATE_FILE)
        barrier(self.distributed)
        return directory

    def _latest_checkpoint(self) -> Path:
        found: Sequence[tuple[int, Path]] = sorted(
            (int(match.group(1)), path)
            for path in Path(self.args.output_dir).glob("checkpoint-*")
            if (match := re.fullmatch(r"checkpoint-(\d+)", path.name)) and path.is_dir()
        )
        if not found:
            raise FileNotFoundError(f"No checkpoint-* folder in {self.args.output_dir}")
        return found[-1][1]

    def _load_checkpoint(self, directory: Path) -> None:
        assert self.components is not None and self.injection is not None
        state_path = directory / STATE_FILE
        if not state_path.is_file():
            raise FileNotFoundError(f"{state_path} is missing")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["world_size"] != self.distributed.world_size:
            raise ValueError(
                f"{directory} was written by {state['world_size']} processes; resume it with "
                f"the same number (this run has {self.distributed.world_size})"
            )
        weights = read_lora_file(directory / ADAPTER_FILE, {"unet": self.components.unet})
        self.injection.load_adapter_state_dict(
            {
                f"{item.module}.lora_{'A' if kind == 'down' else 'B'}.weight": tensor
                for item in weights
                for kind, tensor in (("down", item.down), ("up", item.up))
            }
        )
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.order = state["order"]
        self.order_generator.set_state(state["order_rng"])
        streams = state["process_rng"][self.distributed.rank]
        self.augment_generator.set_state(streams["augment"])
        self.noise_generator.set_state(streams["noise"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if self.scaler is not None and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.log_history = list(state["log_history"])


def _format(value: float) -> str:
    if float(value).is_integer() and abs(value) < 1e9:
        return str(int(value))
    return f"{value:.4g}"
