"""Native low-rank adapters for PyTorch linear and convolution layers.

The arithmetic follows PEFT's LoRA layers operation for operation, so adapters loaded here match
diffusers ``load_lora_weights`` bit for bit on the same weights.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from imagehub.models.fp8 import Fp8Linear

_INITS = ("uniform", "gaussian")


@dataclass(frozen=True)
class LoRAConfig:
    """Adapter settings for :func:`inject_lora`."""

    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    freeze_base: bool = True
    seed: int = 0
    init: str = "uniform"

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, int | float)
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("LoRA alpha must be finite and positive")
        object.__setattr__(self, "alpha", float(self.alpha))
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, int | float)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("LoRA dropout must be in [0, 1)")
        object.__setattr__(self, "dropout", float(self.dropout))
        patterns = tuple(self.target_modules)
        if not patterns or any(not isinstance(p, str) or not p.strip() for p in patterns):
            raise ValueError("LoRA target_modules must contain non-empty patterns")
        patterns = tuple(p.strip() for p in patterns)
        if len(patterns) != len(set(patterns)):
            raise ValueError("LoRA target_modules cannot contain duplicates")
        object.__setattr__(self, "target_modules", patterns)
        if not isinstance(self.freeze_base, bool):
            raise TypeError("LoRA freeze_base must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("LoRA seed must be an int")
        if self.init not in _INITS:
            raise ValueError(f"LoRA init must be one of {_INITS}")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


_Kernel = nn.Linear | nn.Conv2d
_Base = nn.Linear | nn.Conv2d | Fp8Linear


class LoRALayer(nn.Module):
    """A base layer plus named low-rank adapters, each with its own scale."""

    base: _Base

    def __init__(self, base: _Base) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.lora_dropout = nn.ModuleDict()
        self.scaling: dict[str, float] = {}
        self.merged_adapters: list[str] = []
        self._rank: dict[str, int] = {}
        self._alpha: dict[str, float] = {}
        self._output_rows: dict[str, slice] = {}
        self._base_requires_grad = {n: p.requires_grad for n, p in base.named_parameters()}

    @property
    def in_features(self) -> int:
        raise NotImplementedError

    @property
    def out_features(self) -> int:
        raise NotImplementedError

    def _make_pair(self, rank: int, out_features: int | None = None) -> tuple[_Kernel, _Kernel]:
        raise NotImplementedError

    def _fan_in(self) -> int:
        raise NotImplementedError

    def weights(self, name: str) -> tuple[nn.Parameter, nn.Parameter]:
        """The down (``A``) and up (``B``) weights of adapter ``name``."""
        return (
            cast(nn.Parameter, self.lora_A[name].weight),
            cast(nn.Parameter, self.lora_B[name].weight),
        )

    @property
    def adapters(self) -> tuple[str, ...]:
        return tuple(self.lora_A.keys())

    def add_adapter(
        self,
        name: str,
        down: Tensor | None,
        up: Tensor | None,
        *,
        rank: int,
        alpha: float,
        scale: float = 1.0,
        dropout: float = 0.0,
        generator: torch.Generator | None = None,
        init: str = "uniform",
        output_rows: slice | None = None,
    ) -> None:
        """Add adapter ``name``; with no tensors, ``A`` is seeded and ``B`` is zero.

        ``init="uniform"`` draws ``A`` like PEFT's default (Kaiming uniform, ``a=sqrt(5)``);
        ``init="gaussian"`` draws ``N(0, 1/rank)`` like PEFT's ``init_lora_weights="gaussian"``.
        """
        if init not in _INITS:
            raise ValueError(f"LoRA init must be one of {_INITS}")
        if not name or "." in name:
            raise ValueError(f"Invalid LoRA adapter name {name!r}")
        if name in self.lora_A:
            raise ValueError(f"LoRA adapter {name!r} is already loaded")
        if self.merged_adapters:
            raise RuntimeError("Unmerge LoRA adapters before adding another")
        if rank <= 0 or rank > min(self.in_features, self.out_features):
            raise ValueError(
                f"LoRA rank {rank} does not fit a layer with {self.in_features} inputs and "
                f"{self.out_features} outputs"
            )
        if (down is None) != (up is None):
            raise ValueError("LoRA down and up weights must be given together")
        if output_rows is not None and (
            output_rows.step is not None
            or output_rows.start is None
            or output_rows.stop is None
            or not 0 <= output_rows.start < output_rows.stop <= self.out_features
        ):
            raise ValueError("LoRA output rows must be a non-empty contiguous layer slice")
        lora_a, lora_b = self._make_pair(
            rank, None if output_rows is None else output_rows.stop - output_rows.start
        )
        weight = self.base.weight
        adapter_dtype = (
            self.base.compute_dtype if isinstance(self.base, Fp8Linear) else weight.dtype
        )
        lora_a.to(device=weight.device, dtype=adapter_dtype)
        lora_b.to(device=weight.device, dtype=adapter_dtype)
        with torch.no_grad():
            if down is None or up is None:
                if init == "gaussian":
                    values = torch.randn(lora_a.weight.shape, generator=generator) * (1 / rank)
                else:
                    bound = 1.0 / math.sqrt(self._fan_in())
                    values = torch.empty(lora_a.weight.shape).uniform_(
                        -bound, bound, generator=generator
                    )
                lora_a.weight.copy_(values)
                lora_b.weight.zero_()
            else:
                lora_a.weight.copy_(_fit(down, lora_a.weight, name, "down"))
                lora_b.weight.copy_(_fit(up, lora_b.weight, name, "up"))
        self.lora_A[name] = lora_a
        self.lora_B[name] = lora_b
        self.lora_dropout[name] = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._rank[name] = rank
        self._alpha[name] = float(alpha)
        self.scaling[name] = scale * float(alpha) / rank
        if output_rows is not None:
            self._output_rows[name] = output_rows

    def remove_adapter(self, name: str) -> None:
        if name in self.merged_adapters:
            raise RuntimeError("Unmerge LoRA adapters before removing one")
        for store in (self.lora_A, self.lora_B, self.lora_dropout):
            del store[name]
        del self.scaling[name], self._rank[name], self._alpha[name]
        self._output_rows.pop(name, None)

    def set_scale(self, name: str, scale: float) -> None:
        if name not in self.lora_A:
            raise KeyError(f"No LoRA adapter named {name!r}")
        if name in self.merged_adapters:
            raise RuntimeError("Unmerge before changing a LoRA scale")
        self.scaling[name] = scale * self._alpha[name] / self._rank[name]

    def delta_weight(self, name: str) -> Tensor:
        raise NotImplementedError

    def merge(self) -> None:
        """Add every unmerged adapter to the base weight, in insertion order."""
        with torch.no_grad():
            for name in self.lora_A:
                if name not in self.merged_adapters:
                    self.base.weight.data += self.delta_weight(name).to(self.base.weight.dtype)
                    self.merged_adapters.append(name)

    def unmerge(self) -> None:
        with torch.no_grad():
            while self.merged_adapters:
                name = self.merged_adapters.pop()
                self.base.weight.data -= self.delta_weight(name).to(self.base.weight.dtype)

    def restore_base_trainability(self) -> None:
        for name, parameter in self.base.named_parameters():
            parameter.requires_grad_(self._base_requires_grad[name])

    def forward(self, x: Tensor) -> Tensor:
        result = self.base(x)
        result_dtype = result.dtype
        for name in self.lora_A:
            if name in self.merged_adapters:
                continue
            lora_a, lora_b = self.lora_A[name], self.lora_B[name]
            h = x.to(self.weights(name)[0].dtype)
            update = lora_b(lora_a(self.lora_dropout[name](h))) * self.scaling[name]
            rows = self._output_rows.get(name)
            if rows is None:
                result = result + update
            else:
                result = result.clone()
                result[..., rows] = result[..., rows] + update
        return result.to(result_dtype)


class LoRALinear(LoRALayer):
    base: nn.Linear

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def _fan_in(self) -> int:
        return self.base.in_features

    def _make_pair(self, rank: int, out_features: int | None = None) -> tuple[_Kernel, _Kernel]:
        return (
            nn.Linear(self.base.in_features, rank, bias=False),
            nn.Linear(rank, out_features or self.base.out_features, bias=False),
        )

    def delta_weight(self, name: str) -> Tensor:
        weight_a, weight_b = self.weights(name)
        delta = (weight_b @ weight_a) * self.scaling[name]
        rows = self._output_rows.get(name)
        if rows is None:
            return delta
        result = delta.new_zeros(self.base.weight.shape)
        result[rows] = delta
        return result


class LoRAFp8Linear(LoRALayer):
    """A LoRA layer over an fp8 weight-only linear."""

    base: Fp8Linear | nn.Linear

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def _fan_in(self) -> int:
        return self.base.in_features

    def _make_pair(self, rank: int, out_features: int | None = None) -> tuple[_Kernel, _Kernel]:
        return (
            nn.Linear(self.base.in_features, rank, bias=False),
            nn.Linear(rank, out_features or self.base.out_features, bias=False),
        )

    def delta_weight(self, name: str) -> Tensor:
        weight_a, weight_b = self.weights(name)
        delta = (weight_b @ weight_a) * self.scaling[name]
        rows = self._output_rows.get(name)
        if rows is None:
            return delta
        result = delta.new_zeros(self.base.weight.shape)
        result[rows] = delta
        return result

    def merge(self) -> None:
        """Dequantize to the compute dtype, then merge as the reference loader does."""
        if not isinstance(self.base, Fp8Linear):
            super().merge()
            return
        base = self.base
        linear = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.compute_dtype,
        )
        with torch.no_grad():
            linear.weight.copy_(
                base.weight.to(base.compute_dtype)
                * base.weight_scale.to(base.compute_dtype).unsqueeze(1)
            )
            if base.bias is not None and linear.bias is not None:
                linear.bias.copy_(base.bias.to(base.compute_dtype))
        self.base = linear
        super().merge()


class LoRAConv2d(LoRALayer):
    base: nn.Conv2d

    @property
    def in_features(self) -> int:
        return self.base.in_channels

    @property
    def out_features(self) -> int:
        return self.base.out_channels

    def _fan_in(self) -> int:
        kh, kw = self.base.kernel_size
        return self.base.in_channels * kh * kw

    def _make_pair(self, rank: int, out_features: int | None = None) -> tuple[_Kernel, _Kernel]:
        if out_features is not None:
            raise ValueError("LoRA output rows are supported only for linear layers")
        base = self.base
        return (
            nn.Conv2d(
                base.in_channels,
                rank,
                base.kernel_size,  # type: ignore[arg-type]
                base.stride,  # type: ignore[arg-type]
                base.padding,  # type: ignore[arg-type]
                bias=False,
            ),
            nn.Conv2d(rank, base.out_channels, 1, 1, bias=False),
        )

    def delta_weight(self, name: str) -> Tensor:
        weight_a, weight_b = self.weights(name)
        if self.base.weight.shape[2:4] == (1, 1):
            product = weight_b.squeeze(3).squeeze(2) @ weight_a.squeeze(3).squeeze(2)
            return product.unsqueeze(2).unsqueeze(3) * self.scaling[name]
        return F.conv2d(weight_a.transpose(0, 1), weight_b).transpose(0, 1) * self.scaling[name]


def _fit(value: Tensor, target: Tensor, name: str, kind: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"LoRA {kind} weight for {name!r} must be a tensor")
    if value.shape != target.shape:
        # Published files store 1x1 convolutions as matrices or matrices as 1x1 kernels.
        if value.numel() == target.numel() and [d for d in value.shape if d != 1] == [
            d for d in target.shape if d != 1
        ]:
            value = value.reshape(target.shape)
        else:
            raise ValueError(
                f"LoRA {kind} weight for {name!r} has shape {tuple(value.shape)}; "
                f"expected {tuple(target.shape)}"
            )
    return value.to(device=target.device, dtype=target.dtype)


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, attribute = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), attribute


def wrap_module(model: nn.Module, name: str) -> LoRALayer:
    """Return the LoRA wrapper at ``name``, replacing a plain layer on first use."""
    module = model.get_submodule(name)
    if isinstance(module, LoRALayer):
        return module
    wrapper: LoRALayer
    if type(module) is nn.Linear:
        wrapper = LoRALinear(module)
    elif isinstance(module, Fp8Linear):
        wrapper = LoRAFp8Linear(module)
    elif type(module) is nn.Conv2d:
        wrapper = LoRAConv2d(module)
    else:
        raise TypeError(f"LoRA cannot wrap {name!r} ({type(module).__name__})")
    parent, attribute = _parent(model, name)
    setattr(parent, attribute, wrapper)
    return wrapper


def lora_layers(model: nn.Module) -> dict[str, LoRALayer]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALayer)}


def set_lora_scale(model: nn.Module, adapter: str, scale: float) -> None:
    for layer in lora_layers(model).values():
        for name in layer.adapters:
            # A diffusers-layout Ideogram 4 file can address q, k and v separately,
            # while the native model stores them in one qkv projection.  The loader
            # keeps those independent updates under a common adapter-name prefix.
            if name == adapter or name.startswith(f"{adapter}:"):
                layer.set_scale(name, scale)


def merge_lora(model: nn.Module) -> None:
    for layer in lora_layers(model).values():
        layer.merge()


def unmerge_lora(model: nn.Module) -> None:
    for layer in lora_layers(model).values():
        layer.unmerge()


def _unwrap(model: nn.Module, name: str, layer: LoRALayer) -> None:
    layer.unmerge()
    layer.restore_base_trainability()
    parent, attribute = _parent(model, name)
    setattr(parent, attribute, layer.base)


def remove_lora(model: nn.Module) -> nn.Module:
    """Put every wrapped base layer back, with its original ``requires_grad`` flags."""
    layers = lora_layers(model)
    for name in sorted(layers, key=lambda value: value.count("."), reverse=True):
        _unwrap(model, name, layers[name])
    return model


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if any(c in pattern for c in "*?["):
            if fnmatchcase(name, pattern):
                return True
        elif name == pattern or name.endswith("." + pattern):
            return True
    return False


def _targets(model: nn.Module, patterns: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Names of matching layers and of every candidate layer, skipping wrapper internals."""
    wrapped = tuple(f"{name}." for name in lora_layers(model))
    selected, available = [], []
    for name, module in model.named_modules():
        if not name or name.startswith(wrapped):
            continue
        if isinstance(module, nn.Linear | nn.Conv2d | Fp8Linear | LoRALayer):
            available.append(name)
            if _matches(name, patterns):
                selected.append(name)
    return selected, available


class LoRAInjection:
    """Handle for one adapter added by :func:`inject_lora`."""

    def __init__(
        self,
        model: nn.Module,
        layers: Mapping[str, LoRALayer],
        created: set[str],
        config: LoRAConfig,
        adapter: str,
    ) -> None:
        self.model = model
        self.layers = dict(layers)
        self.config = config
        self.adapter = adapter
        self._created = set(created)
        self._restored = False

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.layers))

    def parameters(self) -> Iterator[nn.Parameter]:
        for name in self.module_names:
            yield from self.layers[name].weights(self.adapter)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        self._ensure_active()
        return {
            key: tensor.detach().clone()
            for name in self.module_names
            for key, tensor in zip(
                (f"{name}.lora_A.weight", f"{name}.lora_B.weight"),
                self.layers[name].weights(self.adapter),
                strict=True,
            )
        }

    def load_adapter_state_dict(
        self, state_dict: Mapping[str, Tensor], *, strict: bool = True
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        self._ensure_active()
        if any(self.adapter in layer.merged_adapters for layer in self.layers.values()):
            raise RuntimeError("Unmerge LoRA adapters before loading adapter state")
        destinations = {
            key: parameter
            for name in self.module_names
            for key, parameter in zip(
                (f"{name}.lora_A.weight", f"{name}.lora_B.weight"),
                self.layers[name].weights(self.adapter),
                strict=True,
            )
        }
        missing = tuple(sorted(set(destinations) - set(state_dict)))
        unexpected = tuple(sorted(set(state_dict) - set(destinations)))
        if strict and (missing or unexpected):
            raise ValueError(f"LoRA state mismatch: missing={missing!r}, unexpected={unexpected!r}")
        shared = sorted(set(destinations) & set(state_dict))
        for key in shared:
            source = state_dict[key]
            if not isinstance(source, Tensor):
                raise TypeError(f"LoRA state {key!r} must be a tensor")
            if source.shape != destinations[key].shape:
                raise ValueError(
                    f"LoRA state {key!r} has shape {tuple(source.shape)}; "
                    f"expected {tuple(destinations[key].shape)}"
                )
        with torch.no_grad():
            for key in shared:
                destinations[key].copy_(state_dict[key])
        return missing, unexpected

    def merge(self) -> None:
        self._ensure_active()
        for layer in self.layers.values():
            layer.merge()

    def unmerge(self) -> None:
        self._ensure_active()
        for layer in self.layers.values():
            layer.unmerge()

    def restore(self) -> nn.Module:
        """Remove this adapter, and the wrappers it created, from the model."""
        self._ensure_active()
        for name in sorted(self.layers, key=lambda value: value.count("."), reverse=True):
            layer = self.layers[name]
            layer.unmerge()
            layer.remove_adapter(self.adapter)
            if name in self._created:
                _unwrap(self.model, name, layer)
        self._restored = True
        return self.model

    def _ensure_active(self) -> None:
        if self._restored:
            raise RuntimeError("This LoRA injection has already been restored")


def inject_lora(model: nn.Module, config: LoRAConfig, *, adapter: str = "default") -> LoRAInjection:
    """Add a new trainable adapter to every layer matching ``config.target_modules``."""
    if not isinstance(model, nn.Module):
        raise TypeError("LoRA injection requires a torch.nn.Module")
    if not isinstance(config, LoRAConfig):
        raise TypeError("config must be a LoRAConfig")
    selected, available = _targets(model, config.target_modules)
    if not selected:
        raise ValueError(
            f"LoRA target patterns {config.target_modules!r} matched no layers. "
            f"Available: {tuple(available)!r}"
        )
    for name in selected:
        module = model.get_submodule(name)
        if isinstance(module, LoRALayer):
            layer_in, layer_out = module.in_features, module.out_features
        elif isinstance(module, nn.Linear):
            layer_in, layer_out = module.in_features, module.out_features
        else:
            layer_in, layer_out = (
                cast(nn.Conv2d, module).in_channels,
                cast(nn.Conv2d, module).out_channels,
            )
        if config.rank > min(layer_in, layer_out):
            raise ValueError(
                f"LoRA rank {config.rank} does not fit {name!r} ({layer_in} -> {layer_out})"
            )
    layers: dict[str, LoRALayer] = {}
    created: set[str] = set()
    try:
        for index, name in enumerate(selected):
            existing = isinstance(model.get_submodule(name), LoRALayer)
            layer = wrap_module(model, name)
            if not existing:
                created.add(name)
            generator = torch.Generator().manual_seed(config.seed + index)
            layer.add_adapter(
                adapter,
                None,
                None,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                generator=generator,
                init=config.init,
            )
            layers[name] = layer
            if config.freeze_base:
                for parameter in layer.base.parameters():
                    parameter.requires_grad_(False)
    except BaseException:
        for name in reversed(selected):
            module = model.get_submodule(name)
            if isinstance(module, LoRALayer):
                if adapter in module.lora_A and name in layers:
                    module.remove_adapter(adapter)
                if name in created:
                    _unwrap(model, name, module)
        raise
    return LoRAInjection(model, layers, created, config, adapter)


__all__ = [
    "LoRAConfig",
    "LoRAConv2d",
    "LoRAInjection",
    "LoRALayer",
    "LoRALinear",
    "inject_lora",
    "lora_layers",
    "merge_lora",
    "remove_lora",
    "set_lora_scale",
    "unmerge_lora",
    "wrap_module",
]
