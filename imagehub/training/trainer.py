"""A small LoRA training loop: seeded sampling, AdamW, schedules, checkpoints and exact resume."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.backends.stable_diffusion import StableDiffusionComponents, resolve_device
from imagehub.checkpoints import save_safetensors
from imagehub.lora import LoRAConfig, LoRAInjection, inject_lora
from imagehub.lora_formats import read_lora_file
from imagehub.training.arguments import TrainingArguments
from imagehub.training.data import preprocess_image
from imagehub.training.schedules import lr_lambda
from imagehub.training.sd1 import StableDiffusionLoRAObjective
from imagehub.training.specs import TrainingSupport, get_training_spec, list_training_specs

_DTYPES = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
ADAPTER_FILE = "adapter.safetensors"
CONFIG_FILE = "adapter_config.json"
STATE_FILE = "trainer_state.pt"


@dataclass
class TrainOutput:
    global_step: int
    training_loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    log_history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Train a LoRA adapter for the U-Net of a Stable Diffusion 1.x checkpoint.

    ``train_dataset`` is any sized dataset whose items are ``{"image": PIL image or (3, H, W)
    tensor in [0, 1], "text": str}``, such as :class:`ImageFolderDataset`.
    """

    def __init__(
        self,
        model: str,
        checkpoint: str | os.PathLike[str],
        args: TrainingArguments,
        train_dataset: Any,
    ) -> None:
        if not isinstance(args, TrainingArguments):
            raise TypeError("args must be a TrainingArguments")
        self.spec = get_training_spec(model)
        if self.spec.support is not TrainingSupport.LORA:
            supported = ", ".join(s.model_id for s in list_training_specs(support="lora"))
            raise ValueError(f"{model!r} cannot be trained yet. Trainable models: {supported}")
        if not hasattr(train_dataset, "__len__") or len(train_dataset) == 0:
            raise ValueError("train_dataset must be a non-empty sized dataset")
        first = train_dataset[0]
        if not isinstance(first, dict) or not {"image", "text"} <= set(first):
            raise ValueError('train_dataset items must be dicts with "image" and "text"')
        self.model_id = model
        self.checkpoint = Path(checkpoint)
        self.args = args
        self.train_dataset = train_dataset
        self.global_step = 0
        self.log_history: list[dict[str, float]] = []
        self.components: StableDiffusionComponents | None = None
        self.injection: LoRAInjection | None = None

    # Setup -------------------------------------------------------------------------------

    def _setup(self) -> None:
        if self.components is not None:
            return
        args = self.args
        self.device = resolve_device(args.device)
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
        self.data_generator = torch.Generator().manual_seed(args.seed)
        self.noise_generator = torch.Generator().manual_seed(args.seed + 1)
        self.epoch = 0
        self.position = 0
        self.order = torch.randperm(len(self.train_dataset), generator=self.data_generator)
        self.components = components
        self.injection = injection

    # Data --------------------------------------------------------------------------------

    def _next_batch(self) -> tuple[Tensor, Tensor]:
        assert self.components is not None
        args = self.args
        pixels, ids = [], []
        for _ in range(args.train_batch_size):
            if self.position >= len(self.order):
                self.epoch += 1
                self.position = 0
                self.order = torch.randperm(len(self.train_dataset), generator=self.data_generator)
            item = self.train_dataset[int(self.order[self.position])]
            self.position += 1
            pixels.append(
                preprocess_image(
                    item["image"],
                    resolution=args.resolution,
                    center_crop=args.center_crop,
                    random_flip=args.random_flip,
                    generator=self.data_generator,
                )
            )
            ids.append(self.components.tokenizer(item["text"]))
        batch = torch.stack(pixels).to(device=self.device, dtype=self.dtype)
        return batch, torch.tensor(ids, device=self.device)

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
        return total, lr

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
        run = self._start_reporting()
        losses = []
        started = time.perf_counter()
        last_saved = None
        try:
            while self.global_step < self.args.max_steps:
                loss, lr = self._step()
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Loss is {loss} at step {self.global_step}")
                losses.append(loss)
                if self.global_step % self.args.logging_steps == 0:
                    record = {"step": float(self.global_step), "loss": loss, "lr": lr}
                    self.log_history.append(record)
                    print(
                        f"step {self.global_step}/{self.args.max_steps}  "
                        f"loss {loss:.4f}  lr {lr:.3g}",
                        flush=True,
                    )
                    if run is not None:
                        run.log({"train_loss": loss, "lr": lr}, step=self.global_step)
                if self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint()
                    last_saved = self.global_step
            if losses and last_saved != self.global_step:
                self._save_checkpoint()
        finally:
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
        if self.args.report_to != "wandb":
            return None
        try:
            import wandb
        except ImportError as error:
            raise ImportError('report_to="wandb" needs the wandb package') from error
        return wandb.init(project="imagehub", name=self.args.run_name, config=self.args.to_dict())

    # Saving ------------------------------------------------------------------------------

    def _adapter_tensors(self) -> dict[str, Tensor]:
        assert self.injection is not None
        return {f"unet.{k}": v.float() for k, v in self.injection.adapter_state_dict().items()}

    def _write_adapter(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        modules = sorted({name for name in self.injection.module_names}) if self.injection else []
        metadata = {
            "unet.r": self.args.lora_rank,
            "unet.lora_alpha": self.args.lora_alpha,
            "unet.target_modules": modules,
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
        """Write ``adapter.safetensors`` (diffusers layout) and ``adapter_config.json``."""
        if self.injection is None:
            raise RuntimeError("Call train() before save_model()")
        return self._write_adapter(Path(path) if path is not None else Path(self.args.output_dir))

    def _save_checkpoint(self) -> Path:
        directory = Path(self.args.output_dir) / f"checkpoint-{self.global_step}"
        self._write_adapter(directory)
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "position": self.position,
            "order": self.order,
            "data_rng": self.data_generator.get_state(),
            "noise_rng": self.noise_generator.get_state(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "log_history": self.log_history,
            "args": self.args.to_dict(),
        }
        torch.save(state, directory / STATE_FILE)
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
        weights = read_lora_file(directory / ADAPTER_FILE, {"unet": self.components.unet})
        self.injection.load_adapter_state_dict(
            {
                f"{item.module}.lora_{'A' if kind == 'down' else 'B'}.weight": tensor
                for item in weights
                for kind, tensor in (("down", item.down), ("up", item.up))
            }
        )
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.position = int(state["position"])
        self.order = state["order"]
        self.data_generator.set_state(state["data_rng"])
        self.noise_generator.set_state(state["noise_rng"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if self.scaler is not None and state["scaler"] is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.log_history = list(state["log_history"])
