# LoRA loading implementation plan

**Goal:** Load diffusers-, legacy-diffusers- and kohya-format LoRA files into the native Stable
Diffusion 1.x and SDXL models, with outputs equal to diffusers `load_lora_weights` on CPU
float32 (#193).

**Architecture:** `imagehub/lora.py` holds torch-only LoRA wrappers (`LoRALinear`,
`inject_lora`), extended with a convolution wrapper and several named adapters per
layer. `imagehub/lora_formats.py` reads a safetensors file, detects its layout, converts keys to
native module names and adds one adapter per file. The SD 1.x and SDXL backends take `lora=`,
`lora_scale=` and `lora_fuse=` construction options and apply them at the end of `_load`.

**Tech stack:** torch only in the package; diffusers 0.40.0, transformers 5.18.0 and a pinned
`peft` in the `reference` extra for parity tests.

Design: `docs/design/2026-11-04-lora-and-training-design.md`.

## Global constraints

- Core dependency stays `torch>=2.2`; no PEFT, safetensors package or Accelerate import in
  `imagehub/`.
- Parity is checked on CPU float32 against diffusers 0.40.0 / transformers 5.18.0; tolerance is
  0 unless a measured difference forces the next power of ten (record it in the test).
- Repository text is English. Commit messages are English, without tool or assistant names.
- Pre-push checks are chained with `&&`:
  `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy imagehub && .venv/bin/pytest -q`.
- Branch `feat/lora-loading`, worktree `/Users/gorkem.gul/Personal/imagehub-wt/lora`; one PR to
  `main` with `Closes #193`.

## Numerical contract with PEFT

Bit-equal outputs require the same operation order as PEFT 0.x `LoraLayer`:

- unmerged linear: `result = base(x)`, then for each active adapter in insertion order
  `result = result + lora_B(lora_A(dropout(x))) * scaling`, where `lora_A` / `lora_B` are
  bias-free `nn.Linear` (`nn.Conv2d` for convolutions: `lora_A` copies the base kernel, stride,
  padding and dilation; `lora_B` is 1x1);
- `scaling = user_scale * lora_alpha / r` (this exact left-to-right order);
- merge delta, linear: `(B.weight @ A.weight) * scaling`; 1x1 conv:
  `(B.squeeze(3).squeeze(2) @ A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3) * scaling`;
  kxk conv: `F.conv2d(A.permute(1, 0, 2, 3), B).permute(1, 0, 2, 3) * scaling`;
  merge adds the delta to `base.weight` in place.

## File map

| File | Responsibility |
| --- | --- |
| `imagehub/checkpoints.py` | `save_safetensors(..., metadata=)`, `read_safetensors_metadata(path)` |
| `imagehub/lora.py` | `LoRAConfig`, `LoRALayer`, `LoRALinear`, `LoRAConv2d`, module-tree helpers, `inject_lora`, `LoRAInjection` |
| `imagehub/lora_formats.py` | layout detection, key conversion, `LoRASpec`, `parse_lora_option`, `load_lora_file` |
| `imagehub/backends/stable_diffusion.py` | `lora`, `lora_scale`, `lora_fuse` options |
| `imagehub/backends/stable_diffusion_xl.py` | same options |
| `pyproject.toml` | pinned `peft` in `reference` |
| `tests/test_checkpoints.py` | metadata round trip |
| `tests/test_lora.py` | core wrappers |
| `tests/test_lora_formats.py` | parsing and conversion without diffusers |
| `tests/test_lora_backends.py` | backend options on tiny snapshots |
| `tests/lora_helpers.py` | write random adapters in the three layouts |
| `tests/reference/test_ref_lora.py` | parity against diffusers + peft |
| `docs/models.md`, `README.md`, `docs/validation.md` | usage and validation |

---

### Task 1: safetensors metadata

**Files:** modify `imagehub/checkpoints.py`; test `tests/test_checkpoints.py`.

**Produces:** `save_safetensors(tensors, path, *, metadata: Mapping[str, str] | None = None)`;
`read_safetensors_metadata(path) -> dict[str, str]` (empty dict when absent).

- [ ] Write failing tests:

```python
def test_safetensors_metadata_round_trip(tmp_path):
    path = tmp_path / "a.safetensors"
    save_safetensors({"x": torch.ones(2)}, path, metadata={"format": "pt", "k": "{\"a\": 1}"})
    assert read_safetensors_metadata(path) == {"format": "pt", "k": "{\"a\": 1}"}
    assert torch.equal(load_safetensors(path)["x"], torch.ones(2))


def test_safetensors_metadata_absent(tmp_path):
    path = tmp_path / "a.safetensors"
    save_safetensors({"x": torch.ones(2)}, path)
    assert read_safetensors_metadata(path) == {}


def test_safetensors_metadata_must_be_strings(tmp_path):
    with pytest.raises(TypeError, match="metadata"):
        save_safetensors({"x": torch.ones(1)}, tmp_path / "a.safetensors", metadata={"k": 1})
```

- [ ] Run `.venv/bin/pytest tests/test_checkpoints.py -q -k metadata`; expect failures.
- [ ] Implement: in `save_safetensors`, validate that every key and value is `str` (else
  `TypeError("safetensors metadata keys and values must be strings")`) and put the mapping
  under `header["__metadata__"]` before `json.dumps`. Factor the header-reading lines of
  `load_safetensors` (length checks and JSON parse) into `_read_header(source, size) -> dict`
  used by both functions; `read_safetensors_metadata` returns `header.get("__metadata__", {})`
  after checking it is a `dict[str, str]` (else `CheckpointError("Invalid safetensors
  metadata")`).
- [ ] Run the tests; expect pass. Commit `Add safetensors metadata reading and writing`.

### Task 2: LoRA core (`imagehub/lora.py`)

**Files:** create `imagehub/lora.py`; test `tests/test_lora.py`.

**Produces:**

```python
@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    freeze_base: bool = True
    seed: int = 0
    @property
    def scaling(self) -> float: ...           # alpha / rank

class LoRALayer(nn.Module):                    # base of both wrappers
    base: nn.Linear | nn.Conv2d
    lora_A: nn.ModuleDict                      # adapter name -> nn.Linear / nn.Conv2d
    lora_B: nn.ModuleDict
    lora_dropout: nn.ModuleDict                # adapter name -> nn.Dropout / nn.Identity
    scaling: dict[str, float]
    merged_adapters: list[str]
    def add_adapter(self, name: str, down: Tensor | None, up: Tensor | None, *, rank: int,
                    alpha: float, scale: float = 1.0, dropout: float = 0.0,
                    generator: torch.Generator | None = None) -> None: ...
    def set_scale(self, name: str, scale: float) -> None: ...
    def delta_weight(self, name: str) -> Tensor: ...
    def merge(self) -> None: ...
    def unmerge(self) -> None: ...

class LoRALinear(LoRALayer): ...
class LoRAConv2d(LoRALayer): ...

def wrap_module(model: nn.Module, name: str) -> LoRALayer: ...   # idempotent
def lora_layers(model: nn.Module) -> dict[str, LoRALayer]: ...
def set_lora_scale(model: nn.Module, adapter: str, scale: float) -> None: ...
def merge_lora(model: nn.Module) -> None: ...
def unmerge_lora(model: nn.Module) -> None: ...
def remove_lora(model: nn.Module) -> nn.Module: ...             # restores base modules
def inject_lora(model: nn.Module, config: LoRAConfig, *, adapter: str = "default") -> LoRAInjection: ...

class LoRAInjection:
    model: nn.Module; config: LoRAConfig; adapter: str
    module_names: tuple[str, ...]
    def parameters(self) -> Iterator[nn.Parameter]: ...
    def adapter_state_dict(self) -> dict[str, Tensor]: ...   # "<module>.lora_A.weight" / "<module>.lora_B.weight"
    def load_adapter_state_dict(self, state: Mapping[str, Tensor], *, strict: bool = True) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
    def merge(self) -> None: ...
    def unmerge(self) -> None: ...
    def restore(self) -> nn.Module: ...
```

Behavior:

- `LoRAConfig.__post_init__` validates its fields (positive int rank, finite positive alpha,
  dropout in `[0, 1)`, non-empty unique stripped patterns, bool `freeze_base`, int `seed`) and
  stores `alpha` / `dropout` as floats.
- `add_adapter` with `down`/`up` tensors copies them (shape-checked against the base and
  `rank`) into new `lora_A[name]` / `lora_B[name]` modules on the base weight's device and
  dtype. With `down=None, up=None` it initializes `A` uniform in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`
  from `generator` (fan_in = `in_features`, or `in_channels * kh * kw`) and `B` to zeros.
  Duplicate names, merged layers, `rank > min(in, out)` and shape mismatches raise
  `ValueError`. `scaling[name] = scale * alpha / rank`. When the first adapter is added and
  `freeze_base` semantics apply (always for file loading), base parameters keep their own
  `requires_grad`; `inject_lora` freezes them when `config.freeze_base`.
- `forward(x)`: `result = self.base(x)`; if any adapter is merged, unmerged adapters still add;
  for each name in `lora_A` order not in `merged_adapters`:
  `result = result + self.lora_B[name](self.lora_A[name](self.lora_dropout[name](x))) * self.scaling[name]`.
- `merge()` adds `delta_weight(name)` (formulas in the numerical contract) to `base.weight`
  for each unmerged adapter, in order; `unmerge()` subtracts in reverse order.
- `set_scale` on a merged adapter raises `RuntimeError("Unmerge before changing a LoRA scale")`;
  it recomputes `scale * alpha / rank` from stored `alpha` and `rank` per adapter.
- `wrap_module` replaces `nn.Linear` with `LoRALinear` and `nn.Conv2d` with `LoRAConv2d` via the
  parent's `setattr` (support `nn.ModuleList` parents such as `to_out.0`); an existing
  `LoRALayer` is returned unchanged; other types raise `TypeError` naming the module.
- `remove_lora` unmerges and puts every `base` back, deepest names first.
- `inject_lora` selects `nn.Linear`/`nn.Conv2d` modules whose full name matches a glob or whose
  dotted suffix equals a non-glob pattern (so `to_out.0` matches `...attn1.to_out.0`), raises
  with the available names when nothing matches, and adds one adapter per module with a
  generator seeded `config.seed + index`. It is atomic: on error, wrapped modules are restored.
- `LoRAInjection.adapter_state_dict` keys are `"<module>.lora_A.weight"` and
  `"<module>.lora_B.weight"` (detached clones); `load_adapter_state_dict` checks keys
  and shapes strictly.

- [ ] Write failing tests in `tests/test_lora.py`:

```python
import pytest
import torch
from torch import nn

from imagehub.lora import (
    LoRAConfig, LoRAConv2d, LoRALinear, inject_lora, lora_layers, merge_lora, remove_lora,
    set_lora_scale, unmerge_lora, wrap_module,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_out = nn.ModuleList([nn.Linear(8, 8), nn.Dropout(0.0)])
        self.proj_in = nn.Conv2d(8, 8, 1)
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x, image):
        return self.to_out[0](self.to_q(x)), self.conv(self.proj_in(image))


def inputs():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, 5, 8, generator=g), torch.randn(2, 8, 6, 6, generator=g)


@pytest.mark.parametrize("bad", [dict(rank=0), dict(alpha=float("nan")), dict(dropout=1.0),
                                 dict(target_modules=()), dict(target_modules=("a", "a"))])
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        LoRAConfig(**bad)


def test_injection_is_a_no_op_until_trained():
    torch.manual_seed(0)
    model = Block()
    x, image = inputs()
    before = model(x, image)
    injection = inject_lora(model, LoRAConfig(rank=2, target_modules=("to_q", "to_out.0", "proj_in", "conv")))
    assert isinstance(model.to_q, LoRALinear) and isinstance(model.conv, LoRAConv2d)
    assert set(injection.module_names) == {"to_q", "to_out.0", "proj_in", "conv"}
    for a, b in zip(model(x, image), before):
        assert torch.equal(a, b)
    assert all(not p.requires_grad for p in model.to_q.base.parameters())
    assert all(p.requires_grad for p in injection.parameters())


def test_merge_unmerge_and_restore_round_trip():
    torch.manual_seed(0)
    model = Block()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    x, image = inputs()
    injection = inject_lora(model, LoRAConfig(rank=2, alpha=3.0, target_modules=("to_q", "proj_in", "conv")))
    with torch.no_grad():
        for p in injection.parameters():
            p.normal_()
    unmerged = model(x, image)
    injection.merge()
    for a, b in zip(model(x, image), unmerged):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    injection.unmerge()
    restored = injection.restore()
    assert isinstance(restored.to_q, nn.Linear) and not isinstance(restored.to_q, LoRALinear)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=1e-6, rtol=0)


def test_named_adapters_stack_with_independent_scales():
    torch.manual_seed(0)
    layer = nn.Linear(8, 4)
    model = nn.Sequential(layer)
    wrapped = wrap_module(model, "0")
    a1, b1 = torch.randn(2, 8), torch.randn(4, 2)
    a2, b2 = torch.randn(3, 8), torch.randn(4, 3)
    wrapped.add_adapter("style", a1, b1, rank=2, alpha=4.0, scale=0.5)
    wrapped.add_adapter("char", a2, b2, rank=3, alpha=3.0)
    x = torch.randn(5, 8)
    expected = layer(x) + (x @ a1.T @ b1.T) * (0.5 * 4.0 / 2) + (x @ a2.T @ b2.T) * (1.0 * 3.0 / 3)
    torch.testing.assert_close(model(x), expected)
    set_lora_scale(model, "style", 0.0)
    torch.testing.assert_close(model(x), layer(x) + (x @ a2.T @ b2.T))
    assert list(lora_layers(model)) == ["0"]


def test_merge_matches_unmerged_for_convolutions():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.Conv2d(6, 6, 1))
    for name in ("0", "1"):
        layer = wrap_module(model, name)
        kernel = layer.base.kernel_size
        down = torch.randn(2, layer.base.in_channels, *kernel)
        up = torch.randn(layer.base.out_channels, 2, 1, 1)
        layer.add_adapter("a", down, up, rank=2, alpha=2.0)
    image = torch.randn(1, 4, 5, 5)
    unmerged = model(image)
    merge_lora(model)
    torch.testing.assert_close(model(image), unmerged, atol=1e-4, rtol=1e-4)
    unmerge_lora(model)
    remove_lora(model)
    assert type(model[0]) is nn.Conv2d


def test_adapter_shape_and_name_errors():
    model = nn.Sequential(nn.Linear(8, 4))
    layer = wrap_module(model, "0")
    with pytest.raises(ValueError, match="shape"):
        layer.add_adapter("a", torch.zeros(2, 7), torch.zeros(4, 2), rank=2, alpha=2.0)
    layer.add_adapter("a", torch.zeros(2, 8), torch.zeros(4, 2), rank=2, alpha=2.0)
    with pytest.raises(ValueError, match="already"):
        layer.add_adapter("a", torch.zeros(2, 8), torch.zeros(4, 2), rank=2, alpha=2.0)
    with pytest.raises(TypeError):
        wrap_module(nn.Sequential(nn.LayerNorm(4)), "0")


def test_inject_reports_available_modules_when_nothing_matches():
    with pytest.raises(ValueError, match="to_q"):
        inject_lora(Block(), LoRAConfig(target_modules=("missing",)))


def test_adapter_state_dict_round_trip():
    torch.manual_seed(0)
    model = Block()
    injection = inject_lora(model, LoRAConfig(rank=2, target_modules=("to_q",)))
    state = injection.adapter_state_dict()
    assert set(state) == {"to_q.lora_A.weight", "to_q.lora_B.weight"}
    state["to_q.lora_B.weight"] = torch.ones_like(state["to_q.lora_B.weight"])
    injection.load_adapter_state_dict(state)
    assert torch.equal(model.to_q.lora_B["default"].weight, torch.ones(8, 2))
    with pytest.raises(ValueError, match="missing"):
        injection.load_adapter_state_dict({})
```

- [ ] Run `.venv/bin/pytest tests/test_lora.py -q`; expect import failure.
- [ ] Implement `imagehub/lora.py` to the contract above. Module docstring:
  `"""Native low-rank adapters for PyTorch linear and convolution layers."""`.
- [ ] Run tests, `ruff`, `mypy imagehub`; expect pass. Commit `Add native LoRA layers`.

### Task 3: LoRA file formats (`imagehub/lora_formats.py`)

**Files:** create `imagehub/lora_formats.py`; create `tests/lora_helpers.py`; test
`tests/test_lora_formats.py`.

**Consumes:** `wrap_module`, `LoRALayer.add_adapter`, `read_safetensors_metadata`,
`load_safetensors`.

**Produces:**

```python
LoRASource = str | os.PathLike[str]

@dataclass(frozen=True)
class LoRASpec:
    path: Path
    scale: float = 1.0

def parse_lora_option(lora: LoRASource | Sequence[LoRASource | tuple[LoRASource, float]] | None,
                      lora_scale: float = 1.0) -> tuple[LoRASpec, ...]: ...

@dataclass(frozen=True)
class LoRAModuleWeights:
    component: str          # "unet", "text_encoder", "text_encoder_2"
    module: str             # native module name inside the component
    down: Tensor
    up: Tensor
    alpha: float            # rank when the file has no alpha

def read_lora_file(path: LoRASource, components: Mapping[str, nn.Module | None],
                   *, unet_layers_per_block: int | None = None) -> list[LoRAModuleWeights]: ...

def load_lora_file(components: Mapping[str, nn.Module | None], spec: LoRASpec,
                   *, adapter: str, unet_layers_per_block: int | None = None) -> list[str]: ...

def apply_lora(components: Mapping[str, nn.Module | None], specs: Sequence[LoRASpec],
               *, fuse: bool, unet_layers_per_block: int | None = None) -> None: ...
```

Behavior:

- `parse_lora_option`: `None` → `()`. A single path uses `lora_scale`. A list may mix paths
  (scale 1.0) and `(path, scale)` pairs; passing `lora_scale != 1.0` together with a list raises
  `ValueError("lora_scale applies to a single lora file; give (path, scale) pairs instead")`.
  Scales must be finite real numbers (bool rejected). Files must exist and end with
  `.safetensors` (`FileNotFoundError` / `ValueError`).
- Layout detection on the key set:
  - **kohya**: any key starting with `lora_unet_`, `lora_te_`, `lora_te1_`, `lora_te2_`;
    tensors `<name>.lora_down.weight`, `<name>.lora_up.weight`, optional `<name>.alpha`.
  - **legacy diffusers**: any key containing `.processor.`, `_lora.down.`, `_lora.up.`,
    `.lora.down.`, `.lora.up.` or `.lora_linear_layer.`.
  - **diffusers / PEFT**: keys `<component>.<module>.lora_A.weight` / `.lora_B.weight`.
  - Keys containing `dora_scale`, `lora_magnitude_vector`, `hada_`, `lokr_`, `.diff` or
    `.diff_b` raise `ValueError("Unsupported LoRA variant (DoRA/LyCORIS): <first keys>")`.
  - Anything else raises `ValueError("Unrecognized LoRA layout: <first keys>")`.
- **PEFT keys:** component is the first dotted part (`unet`, `text_encoder`, `text_encoder_2`);
  module is the rest up to `.lora_A`/`.lora_B`. Alpha comes from the `lora_adapter_metadata`
  header JSON (`"<component>.lora_alpha"`, with `"<component>.alpha_pattern"` entries matched
  when the module equals the pattern or ends with `"." + pattern`); without metadata alpha
  equals the module's rank (`down.shape[0]`).
- **Legacy keys:** strip `.processor`; map the UNet suffixes `.to_q_lora.down`→`.to_q.lora_A`,
  `.to_q_lora.up`→`.to_q.lora_B` (same for `to_k`, `to_v`), `.to_out_lora.*`→`.to_out.0.lora_*`,
  `.lora.down`→`.lora_A`, `.lora.up`→`.lora_B`; for text encoders map
  `.to_q_lora.*`→`.q_proj.lora_*` (`to_k`→`k_proj`, `to_v`→`v_proj`, `to_out`→`out_proj`) and
  `.lora_linear_layer.down/up`→`.lora_A/.lora_B`; then treat as PEFT keys. Alpha equals rank.
- **kohya keys:** `lora_unet_` → `unet`, `lora_te_` and `lora_te1_` → `text_encoder`,
  `lora_te2_` → `text_encoder_2`. For the U-Net, first rewrite SGM names
  (`input_blocks_<i>_<j>_`, `middle_block_<i>_`, `output_blocks_<i>_<j>_`) to diffusers block
  names with `unet_layers_per_block` exactly as `diffusers.loaders.lora_conversion_utils.
  _maybe_map_sgm_blocks_to_diffusers` does (input block `i`: block `(i-1)//(L+1)`, layer
  `(i-1)%(L+1)`, inner `0`→`resnets`, `1`→`attentions`, `op`→`downsamplers_0`; middle `0/1/2` →
  `resnets_0`/`attentions_0`/`resnets_1`; output block `i`: block `i//(L+1)`, layer `i%(L+1)`,
  inner `0`/`1`/`2` → `resnets`/`attentions`/`upsamplers_0`), plus `input_blocks_0_0`→`conv_in`,
  `out_2`→`conv_out`, and the resnet sub-names `in_layers_2`→`conv1`, `out_layers_3`→`conv2`,
  `emb_layers_1`→`time_emb_proj`, `skip_connection`→`conv_shortcut`. Then resolve the
  underscored name against the component's own modules: build
  `{name.replace(".", "_"): name for name, m in component.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d, LoRALayer))}`
  (raise on a collision); text-encoder names are looked up under `text_model_` as in the
  file. Alpha is `float(<name>.alpha)` when present, else rank.
- Every resolved module must exist in its component and be `nn.Linear`/`nn.Conv2d`/`LoRALayer`;
  a component that is `None` or absent from `components`, or an unresolved name, raises
  `ValueError` listing up to five offending keys. Up/down pairs must both exist.
- `load_lora_file` reads, then for each module calls `wrap_module(component, module)` and
  `add_adapter(adapter, down, up, rank=down.shape[0], alpha=alpha, scale=spec.scale)`; returns the
  qualified module names. It reads and validates everything before wrapping anything.
- `apply_lora` loads specs as adapters `lora_0`, `lora_1`, ... in order and, with `fuse`, calls
  `merge_lora` on every component afterwards.

- [ ] Write `tests/lora_helpers.py` (no diffusers import):

```python
"""Write random LoRA files for native modules in the three published layouts."""

import json

import torch
from torch import nn

from imagehub.checkpoints import save_safetensors

KOHYA_PREFIX = {"unet": "lora_unet_", "text_encoder": "lora_te_", "text_encoder_2": "lora_te2_"}


def random_lora(components, targets, *, rank=2, seed=0):
    """``{(component, module): (down, up)}`` for every Linear/Conv2d whose name ends with a target."""
    g = torch.Generator().manual_seed(seed)
    result = {}
    for component, model in components.items():
        for name, module in model.named_modules():
            if not any(name == t or name.endswith("." + t) for t in targets):
                continue
            if isinstance(module, nn.Linear):
                down = torch.randn(rank, module.in_features, generator=g) * 0.1
                up = torch.randn(module.out_features, rank, generator=g) * 0.1
            elif isinstance(module, nn.Conv2d):
                down = torch.randn(rank, module.in_channels, *module.kernel_size, generator=g) * 0.1
                up = torch.randn(module.out_channels, rank, 1, 1, generator=g) * 0.1
            else:
                continue
            result[(component, name)] = (down, up)
    return result


def write_peft(path, weights, *, alpha=None):
    tensors = {}
    for (component, module), (down, up) in weights.items():
        tensors[f"{component}.{module}.lora_A.weight"] = down
        tensors[f"{component}.{module}.lora_B.weight"] = up
    metadata = None
    if alpha is not None:
        rank = next(iter(weights.values()))[0].shape[0]
        packed = {}
        for component in sorted({c for c, _ in weights}):
            packed[f"{component}.r"] = rank
            packed[f"{component}.lora_alpha"] = alpha
            packed[f"{component}.target_modules"] = sorted({m for c, m in weights if c == component})
        metadata = {"format": "pt", "lora_adapter_metadata": json.dumps(packed, indent=2, sort_keys=True)}
    save_safetensors(tensors, path, metadata=metadata)
    return path


def write_kohya(path, weights, *, alpha=None, sgm_names=None):
    """``sgm_names`` maps a diffusers U-Net module name to its SGM spelling (SDXL kohya files)."""
    tensors = {}
    for (component, module), (down, up) in weights.items():
        name = (sgm_names or {}).get(module, module) if component == "unet" else module
        key = KOHYA_PREFIX[component] + name.replace(".", "_")
        tensors[f"{key}.lora_down.weight"] = down
        tensors[f"{key}.lora_up.weight"] = up
        if alpha is not None:
            tensors[f"{key}.alpha"] = torch.tensor(float(alpha))
    save_safetensors(tensors, path)
    return path


def write_legacy(path, weights):
    tensors = {}
    for (component, module), (down, up) in weights.items():
        parent, _, leaf = module.rpartition(".")
        if component == "unet" and leaf in ("to_q", "to_k", "to_v"):
            stem = f"unet.{parent}.processor.{leaf}_lora"
        elif component == "unet" and module.endswith("to_out.0"):
            stem = f"unet.{module[: -len('.to_out.0')]}.processor.to_out_lora"
        elif component == "unet":
            stem = f"unet.{module}.lora"
        else:
            stem = f"{component}.{module}.lora_linear_layer"
        tensors[f"{stem}.down.weight"] = down
        tensors[f"{stem}.up.weight"] = up
    save_safetensors(tensors, path)
    return path
```

  For SDXL kohya tests, `sgm_names` is computed in the test from the tiny U-Net config by the
  inverse of the SGM mapping above for the attention modules used (`down_blocks.<b>.attentions.<j>.…`
  → `input_blocks.<1 + b*(L+1) + j>.1.…`, `mid_block.attentions.0.…` → `middle_block.1.…`,
  `up_blocks.<b>.attentions.<j>.…` → `output_blocks.<b*(L+1) + j>.1.…`).

- [ ] Write failing tests in `tests/test_lora_formats.py` using small native modules (a
  `CLIPTextModel` and `UNet2DConditionModel` built from the tiny configs in `tests/helpers.py`):
  - each layout loads and yields the same `LoRALayer` weights and scaling (PEFT with and
    without metadata alpha, kohya with and without `.alpha`, legacy);
  - kohya SGM names resolve on a tiny SDXL-shaped U-Net (`layers_per_block=2`);
  - kohya `proj_in` 1x1 convolution adapters load into `LoRAConv2d`;
  - `parse_lora_option` cases: single path + scale, list with pairs, list + `lora_scale`
    error, missing file, wrong suffix, non-finite scale;
  - DoRA / LyCORIS keys, unknown keys, a missing component (`text_encoder_2` on SD 1.x) and an
    unpaired down tensor raise `ValueError` with the key in the message and leave the model
    unwrapped;
  - `apply_lora(..., fuse=True)` leaves the same outputs as unfused within `1e-5`.
- [ ] Run `.venv/bin/pytest tests/test_lora_formats.py -q`; expect failures.
- [ ] Implement `imagehub/lora_formats.py` to the contract above.
- [ ] Run tests, `ruff`, `mypy imagehub`; expect pass. Commit `Read diffusers, legacy and kohya LoRA files`.

### Task 4: backend options

**Files:** modify `imagehub/backends/stable_diffusion.py`, `imagehub/backends/stable_diffusion_xl.py`;
test `tests/test_lora_backends.py`.

**Consumes:** `parse_lora_option`, `apply_lora`.

- Both `__init__` signatures gain keyword-only
  `lora: LoRASource | Sequence[LoRASource | tuple[LoRASource, float]] | None = None`,
  `lora_scale: float = 1.0`, `lora_fuse: bool = False`. `__init__` stores
  `self.lora = parse_lora_option(lora, lora_scale)` (so bad options fail before loading) and
  validates `lora_fuse` is a bool.
- At the end of `_load`, when `self.lora` is non-empty:
  - SD 1.x: `apply_lora({"unet": c.unet, "text_encoder": c.text_encoder}, self.lora, fuse=self.lora_fuse, unet_layers_per_block=c.unet.config["layers_per_block"])`;
  - SDXL: same with `"text_encoder_2": c.text_encoder_2` (and `text_encoder` possibly `None`
    for the refiner).
  - Read `layers_per_block` from the U-Net config the native model keeps; if it is stored
    under another attribute, use that.
- [ ] Write failing tests (tiny snapshots from `write_tiny_sd` / `write_tiny_sdxl`, adapters from
  `tests/lora_helpers.py`):
  - a zero-`up` adapter leaves generated images byte-equal to the plain backend;
  - a random adapter changes the output; `lora_scale=0.0` restores it;
  - two files stacked equal one file holding the summed effect when their modules are disjoint
    (compare against loading them in either order within `1e-5` on latents via
    `components.unet`);
  - `lora_fuse=True` matches unfused within `1e-5` and leaves no `LoRALayer` unmerged;
  - invalid options (`lora_scale` with a list, missing file, `lora_fuse="yes"`) raise at
    `pipeline(...)` construction without loading weights;
  - unloading and reloading the backend reapplies the adapters once (output unchanged).
- [ ] Run the tests; expect failures. Implement. Run tests, `ruff`, `mypy`; expect pass.
  Commit `Add lora options to the Stable Diffusion backends`.

### Task 5: diffusers parity

**Files:** modify `pyproject.toml` (`reference` extra gains `"peft==<latest release compatible
with diffusers 0.40.0>"`, pinned exactly; install it into `.venv`); create
`tests/reference/test_ref_lora.py`.

- [ ] Tests (marked `reference`, `pytest.importorskip("peft")`), each building the tiny snapshot,
  loading `diffusers.StableDiffusionPipeline` / `StableDiffusionXLPipeline` from it, writing a
  random adapter with `tests/lora_helpers.py` for the U-Net attention modules (`to_q`, `to_k`,
  `to_v`, `to_out.0`), `proj_in`, `proj_out`, `ff.net.0.proj`, `ff.net.2` and the text-encoder
  `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`:
  - parametrize layout × family: `peft` (alpha in metadata), `peft-no-alpha`, `kohya`
    (alpha 1.0 ≠ rank), `legacy` for SD 1.x; `peft`, `kohya-sgm` for SDXL;
  - `ref.load_lora_weights(path, adapter_name="a")`, 3 steps at 32x32, seed 42; imagehub
    `pipeline(..., lora=path)`; compare uint8 images with `torch.equal` (as the existing tiny
    pipeline tests do) and the U-Net / text-encoder outputs with `max_diff` and the tolerance
    rule;
  - two adapters: `ref.load_lora_weights(p1, adapter_name="a")`,
    `ref.load_lora_weights(p2, adapter_name="b")`, `ref.set_adapters(["a", "b"], [0.8, 0.5])`
    vs `lora=[(p1, 0.8), (p2, 0.5)]`;
  - fuse: `ref.fuse_lora()` vs `lora_fuse=True`.
- [ ] One real-weights test, skipped unless `IMAGEHUB_SD_V1_5` points at a snapshot and the
  adapter file exists: `latent-consistency/lcm-lora-sdv1-5` `pytorch_lora_weights.safetensors`
  (kohya layout with `.alpha` and `proj_in` convolutions), revision pinned in the test and
  fetched with `huggingface_hub` in the test environment; 2 steps at 256x256, guidance 1.0,
  `LCMScheduler` not required (compare with the snapshot's default scheduler on both sides);
  record measurements.
- [ ] Run `IMAGEHUB_REFERENCE_TESTS=1 .venv/bin/pytest tests/reference/test_ref_lora.py -q`;
  fix any mismatch at the source (operation order, scaling, key mapping) rather than loosening
  tolerances. Commit `Check LoRA loading against diffusers`.

### Task 6: docs and PR

- [ ] `docs/models.md`: a "LoRA adapters" section with the three examples from the design note,
  supported layouts, families (SD 1.x and SDXL backends) and rejected variants; remove "LoRA"
  from the SD 1.x / SDXL out-of-scope lines only.
- [ ] `README.md`: one short example under usage.
- [ ] `docs/validation.md`: the parity rows with measured differences.
- [ ] Full check chain (Global constraints) passes; push `feat/lora-loading`; open the PR to
  `main` with the usual body (Summary with `Closes #193.` and bold-labeled items, Validation
  table "CPU, float32, vs diffusers 0.40.0 / transformers 5.18.0 / peft <pin>", How to test).
