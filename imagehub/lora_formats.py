"""Read published LoRA files (diffusers, legacy diffusers and kohya layouts) into native models.

Native modules use the diffusers and transformers state-dict names, so a diffusers key needs only
its component prefix removed. kohya keys spell module paths with underscores; they are resolved
against the component's own module names, after mapping SDXL's original (SGM) U-Net block names
to the diffusers ones the way diffusers does.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor, nn

from imagehub.checkpoints import load_safetensors, read_safetensors_metadata
from imagehub.lora import LoRALayer, merge_lora, wrap_module

LoRASource = str | os.PathLike[str]
LoRAOption = LoRASource | Sequence[LoRASource | tuple[LoRASource, float]] | None

_KOHYA_PREFIXES = (
    ("lora_unet_", "unet"),
    ("lora_te1_", "text_encoder"),
    ("lora_te2_", "text_encoder_2"),
    ("lora_te_", "text_encoder"),
)
_UNSUPPORTED = ("dora_scale", "lora_magnitude_vector", "hada_", "lokr_", ".diff", ".oft_")
_LEGACY_MARKERS = (".processor.", "_lora.down.", "_lora.up.", ".lora.down.", ".lora.up.")
_LEGACY_UNET = (
    (".to_out_lora.up", ".to_out.0.lora_B"),
    (".to_out_lora.down", ".to_out.0.lora_A"),
    (".to_q_lora.down", ".to_q.lora_A"),
    (".to_q_lora.up", ".to_q.lora_B"),
    (".to_k_lora.down", ".to_k.lora_A"),
    (".to_k_lora.up", ".to_k.lora_B"),
    (".to_v_lora.down", ".to_v.lora_A"),
    (".to_v_lora.up", ".to_v.lora_B"),
    (".lora.up", ".lora_B"),
    (".lora.down", ".lora_A"),
)
_LEGACY_TEXT = (
    (".to_q_lora.up", ".q_proj.lora_B"),
    (".to_q_lora.down", ".q_proj.lora_A"),
    (".to_k_lora.up", ".k_proj.lora_B"),
    (".to_k_lora.down", ".k_proj.lora_A"),
    (".to_v_lora.up", ".v_proj.lora_B"),
    (".to_v_lora.down", ".v_proj.lora_A"),
    (".to_out_lora.up", ".out_proj.lora_B"),
    (".to_out_lora.down", ".out_proj.lora_A"),
    (".lora_linear_layer.up", ".lora_B"),
    (".lora_linear_layer.down", ".lora_A"),
    (".lora.up", ".lora_B"),
    (".lora.down", ".lora_A"),
)
_COMPONENTS = ("unet", "text_encoder", "text_encoder_2")
_SGM_SINGLE = {
    "input_blocks_0_0": "conv_in",
    "out_2": "conv_out",
    "time_embed_0": "time_embedding_linear_1",
    "time_embed_2": "time_embedding_linear_2",
    "label_emb_0_0": "add_embedding_linear_1",
    "label_emb_0_2": "add_embedding_linear_2",
}
_SGM_RESNET = (
    ("in_layers_2", "conv1"),
    ("out_layers_3", "conv2"),
    ("emb_layers_1", "time_emb_proj"),
    ("skip_connection", "conv_shortcut"),
)


@dataclass(frozen=True)
class LoRASpec:
    """One adapter file and the scale it is applied with."""

    path: Path
    scale: float = 1.0


@dataclass(frozen=True)
class LoRAModuleWeights:
    """The adapter weights for one native module."""

    component: str
    module: str
    down: Tensor
    up: Tensor
    alpha: float


def _scale(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("LoRA scales must be finite numbers")
    return float(value)


def _source(value: object) -> Path:
    if not isinstance(value, str | os.PathLike):
        raise TypeError("lora must be a path, or a list of paths or (path, scale) pairs")
    path = Path(value)
    if path.suffix != ".safetensors":
        raise ValueError(f"LoRA files must be .safetensors files: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"LoRA file not found: {path}")
    return path


def parse_lora_option(lora: LoRAOption, lora_scale: float = 1.0) -> tuple[LoRASpec, ...]:
    """Normalize the ``lora`` / ``lora_scale`` backend options; files must exist."""
    scale = _scale(lora_scale)
    if lora is None:
        return ()
    if isinstance(lora, str | os.PathLike):
        return (LoRASpec(_source(lora), scale),)
    if isinstance(lora, Sequence):
        if scale != 1.0:
            raise ValueError(
                "lora_scale applies to a single lora file; give (path, scale) pairs instead"
            )
        specs = []
        for item in lora:
            if isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError("LoRA pairs must be (path, scale)")
                specs.append(LoRASpec(_source(item[0]), _scale(item[1])))
            else:
                specs.append(LoRASpec(_source(item)))
        if not specs:
            raise ValueError("lora must not be an empty list")
        return tuple(specs)
    raise TypeError("lora must be a path, or a list of paths or (path, scale) pairs")


def _first(keys: Sequence[str]) -> str:
    return ", ".join(sorted(keys)[:5])


def _module_names(component: nn.Module) -> set[str]:
    wrapped = tuple(f"{name}." for name, m in component.named_modules() if isinstance(m, LoRALayer))
    return {
        name
        for name, module in component.named_modules()
        if name
        and not name.startswith(wrapped)
        and isinstance(module, nn.Linear | nn.Conv2d | LoRALayer)
    }


def _underscored(component: nn.Module) -> dict[str, str]:
    table: dict[str, str] = {}
    for name in _module_names(component):
        key = name.replace(".", "_")
        if key in table:
            raise ValueError(f"Module names {table[key]!r} and {name!r} collide in kohya spelling")
        table[key] = name
    return table


def _sgm_to_diffusers(name: str, layers_per_block: int | None) -> str:
    if name in _SGM_SINGLE:
        return _SGM_SINGLE[name]
    match = re.fullmatch(r"(input_blocks|middle_block|output_blocks)_(\d+)_(?:(\d+)_)?(.*)", name)
    if match is None:
        return name
    if not isinstance(layers_per_block, int):
        raise ValueError("kohya SDXL names need the U-Net's integer layers_per_block")
    kind, index, inner, rest = match.group(1), int(match.group(2)), match.group(3), match.group(4)
    for old, new in _SGM_RESNET:
        rest = re.sub(rf"(^|_){old}($|_)", rf"\g<1>{new}\g<2>", rest)
    width = layers_per_block + 1
    if kind == "middle_block":
        # middle_block_<i>_<rest>: the regex may have read a digit of rest as ``inner``.
        rest = f"{inner}_{rest}" if inner is not None else rest
        parts = {0: "resnets_0", 1: "attentions_0", 2: "resnets_1"}
        if index not in parts:
            raise ValueError(f"Invalid SGM middle block {index}")
        return f"mid_block_{parts[index]}_{rest}"
    if inner is None:
        if kind == "input_blocks" and rest.startswith("0_op"):
            inner, rest = "0", rest[2:]
        else:
            raise ValueError(f"Unsupported SGM U-Net name {name!r}")
    if kind == "input_blocks":
        block, layer = (index - 1) // width, (index - 1) % width
        if rest == "op":
            return f"down_blocks_{block}_downsamplers_0_conv"
        group = {"0": "resnets", "1": "attentions"}.get(inner)
        if group is None:
            raise ValueError(f"Unsupported SGM U-Net name {name!r}")
        return f"down_blocks_{block}_{group}_{layer}_{rest}"
    block, layer = index // width, index % width
    if inner == "2":
        return f"up_blocks_{block}_upsamplers_0_{rest}"
    group = {"0": "resnets", "1": "attentions"}.get(inner)
    if group is None:
        raise ValueError(f"Unsupported SGM U-Net name {name!r}")
    return f"up_blocks_{block}_{group}_{layer}_{rest}"


def _pairs(
    entries: Mapping[tuple[str, str], dict[str, Tensor]], source: str
) -> dict[tuple[str, str], tuple[Tensor, Tensor]]:
    result = {}
    unpaired = [f"{c}.{m}" for (c, m), t in entries.items() if set(t) != {"down", "up"}]
    if unpaired:
        raise ValueError(
            f"{source}: LoRA modules without both down and up weights: {_first(unpaired)}"
        )
    for key, tensors in entries.items():
        result[key] = (tensors["down"], tensors["up"])
    return result


def _diffusers_entries(
    tensors: Mapping[str, Tensor], legacy: bool, source: str
) -> dict[tuple[str, str], dict[str, Tensor]]:
    entries: dict[tuple[str, str], dict[str, Tensor]] = {}
    unknown = []
    for key, value in tensors.items():
        name = key
        component = name.split(".", 1)[0]
        if component not in _COMPONENTS:
            if not legacy:
                unknown.append(key)
                continue
            # Early attention-processor files saved the U-Net without a prefix.
            name, component = f"unet.{name}", "unet"
        if legacy:
            name = name.replace(".processor.", ".")
            for old, new in _LEGACY_UNET if component == "unet" else _LEGACY_TEXT:
                name = name.replace(old, new)
        match = re.fullmatch(r"([^.]+)\.(.+)\.lora_([AB])\.weight", name)
        if match is None:
            unknown.append(key)
            continue
        part = "down" if match.group(3) == "A" else "up"
        entries.setdefault((match.group(1), match.group(2)), {})[part] = value
    if unknown:
        raise ValueError(f"{source}: unrecognized LoRA keys: {_first(unknown)}")
    return entries


def _metadata_alpha(path: Path) -> dict[str, object]:
    raw = read_safetensors_metadata(path).get("lora_adapter_metadata")
    if raw is None:
        return {}
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid lora_adapter_metadata") from error
    if not isinstance(values, dict):
        raise ValueError(f"{path}: invalid lora_adapter_metadata")
    return values


def _alpha_from_metadata(
    values: Mapping[str, object], component: str, module: str, rank: int
) -> float:
    patterns = values.get(f"{component}.alpha_pattern")
    if isinstance(patterns, dict):
        for pattern, alpha in patterns.items():
            if module == pattern or module.endswith("." + pattern):
                return float(alpha)
    alpha = values.get(f"{component}.lora_alpha")
    return float(alpha) if isinstance(alpha, int | float) else float(rank)


def _native_name(module: str, names: set[str]) -> str | None:
    """The native spelling of a file's module name, or ``None`` if the model lacks it."""
    if module in names:
        return module
    # transformers 5 drops CLIP's ``text_model`` level, so newer files omit it.
    if f"text_model.{module}" in names:
        return f"text_model.{module}"
    return None


def read_lora_file(
    path: LoRASource,
    components: Mapping[str, nn.Module | None],
    *,
    unet_layers_per_block: int | None = None,
) -> list[LoRAModuleWeights]:
    """Read and resolve every adapter in ``path`` against the native ``components``."""
    source = Path(path)
    tensors = load_safetensors(source)
    keys = list(tensors)
    if not keys:
        raise ValueError(f"{source}: the LoRA file is empty")
    unsupported = [k for k in keys if any(marker in k for marker in _UNSUPPORTED)]
    if unsupported:
        raise ValueError(
            f"{source}: unsupported LoRA variant (DoRA/LyCORIS): {_first(unsupported)}"
        )

    alphas: dict[tuple[str, str], float] = {}
    if any(k.startswith(tuple(p for p, _ in _KOHYA_PREFIXES)) for k in keys):
        entries: dict[tuple[str, str], dict[str, Tensor]] = {}
        tables: dict[str, dict[str, str]] = {}
        unresolved = []
        for key, value in tensors.items():
            prefix_component = next(((p, c) for p, c in _KOHYA_PREFIXES if key.startswith(p)), None)
            match = re.fullmatch(r"(.+)\.(lora_down\.weight|lora_up\.weight|alpha)", key)
            if prefix_component is None or match is None:
                unresolved.append(key)
                continue
            prefix, component = prefix_component
            target = components.get(component)
            if target is None:
                unresolved.append(key)
                continue
            if component not in tables:
                tables[component] = _underscored(target)
            name = match.group(1)[len(prefix) :]
            if component == "unet":
                name = _sgm_to_diffusers(name, unet_layers_per_block)
            module = tables[component].get(name)
            if module is None and component != "unet":
                module = tables[component].get(f"text_model_{name}")
            if module is None:
                unresolved.append(key)
                continue
            if match.group(2) == "alpha":
                alphas[(component, module)] = float(value.item())
            else:
                part = "down" if match.group(2).startswith("lora_down") else "up"
                entries.setdefault((component, module), {})[part] = value
        if unresolved:
            raise ValueError(f"{source}: LoRA keys that match no module: {_first(unresolved)}")
        pairs = _pairs(entries, str(source))
        return [
            LoRAModuleWeights(c, m, down, up, alphas.get((c, m), float(down.shape[0])))
            for (c, m), (down, up) in sorted(pairs.items())
        ]

    legacy = any(marker in k for k in keys for marker in _LEGACY_MARKERS) or any(
        ".lora_linear_layer." in k for k in keys
    )
    pairs = _pairs(_diffusers_entries(tensors, legacy, str(source)), str(source))
    metadata = {} if legacy else _metadata_alpha(source)
    missing = []
    result = []
    names: dict[str, set[str]] = {}
    for (component, module), (down, up) in sorted(pairs.items()):
        target = components.get(component)
        if target is not None and component not in names:
            names[component] = _module_names(target)
        native = _native_name(module, names.get(component, set()))
        if native is None:
            missing.append(f"{component}.{module}")
            continue
        alpha = _alpha_from_metadata(metadata, component, module, int(down.shape[0]))
        result.append(LoRAModuleWeights(component, native, down, up, alpha))
    if missing:
        raise ValueError(f"{source}: LoRA keys that match no module: {_first(missing)}")
    return result


def load_lora_file(
    components: Mapping[str, nn.Module | None],
    spec: LoRASpec,
    *,
    adapter: str,
    unet_layers_per_block: int | None = None,
) -> list[str]:
    """Add the file as adapter ``adapter``; returns the qualified names of patched modules."""
    weights = read_lora_file(spec.path, components, unet_layers_per_block=unet_layers_per_block)
    added: list[tuple[LoRALayer, str]] = []
    try:
        for item in weights:
            component = components[item.component]
            assert component is not None
            layer = wrap_module(component, item.module)
            layer.add_adapter(
                adapter,
                item.down,
                item.up,
                rank=int(item.down.shape[0]),
                alpha=item.alpha,
                scale=spec.scale,
            )
            added.append((layer, f"{item.component}.{item.module}"))
    except BaseException:
        for layer, _ in added:
            layer.remove_adapter(adapter)
        raise
    return [name for _, name in added]


def apply_lora(
    components: Mapping[str, nn.Module | None],
    specs: Sequence[LoRASpec],
    *,
    fuse: bool,
    unet_layers_per_block: int | None = None,
) -> None:
    """Load ``specs`` in order as adapters ``lora_0``, ``lora_1``, ...; optionally merge them."""
    for index, spec in enumerate(specs):
        load_lora_file(
            components, spec, adapter=f"lora_{index}", unet_layers_per_block=unet_layers_per_block
        )
    if fuse:
        for component in components.values():
            if component is not None:
                merge_lora(component)


__all__ = [
    "LoRAModuleWeights",
    "LoRASpec",
    "apply_lora",
    "load_lora_file",
    "parse_lora_option",
    "read_lora_file",
]
