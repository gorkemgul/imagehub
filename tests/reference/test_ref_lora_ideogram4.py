"""Ideogram 4 diffusers-layout LoRA parity on dequantized tiny fp8 weights."""

import pytest
import torch
from lora_helpers import write_peft

from imagehub.lora_formats import LoRASpec, apply_lora
from imagehub.models.fp8 import load_fp8_state_dict, quantize_weight_to_fp8, swap_linears_to_fp8
from imagehub.models.ideogram4_transformer import Ideogram4Transformer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

CONFIG = {
    "num_layers": 1,
    "num_heads": 2,
    "attention_head_dim": 6,
    "intermediate_size": 16,
    "adanln_dim": 4,
    "in_channels": 8,
    "llm_features_dim": 8,
    "rope_theta": 10_000,
    "mrope_section": [1, 1, 1],
    "norm_eps": 1e-5,
}
TARGETS = ("to_q", "to_k", "to_v", "to_out.0", "w1", "w2", "w3")


def _fp8_state(model):
    state = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = f"{name}.weight"
            state[weight], state[f"{weight}_scale"] = quantize_weight_to_fp8(state[weight])
    return state


def _reference_state(state):
    """The dequantize / split / rename bridge from the fp8 snapshot to diffusers."""
    result = {}
    for name, value in state.items():
        if name.endswith(".weight_scale"):
            continue
        if name.endswith(".weight") and f"{name}_scale" in state:
            value = value.float() * state[f"{name}_scale"].float().unsqueeze(1)
        if ".attention.qkv." in name:
            stem, suffix = name.split(".attention.qkv.", 1)
            for projection, weight in zip(("to_q", "to_k", "to_v"), value.chunk(3, 0), strict=True):
                result[f"{stem}.attention.{projection}.{suffix}"] = weight
        else:
            result[name.replace(".attention.o.", ".attention.to_out.0.")] = value
    return result


def _models():
    torch.manual_seed(0)
    source = Ideogram4Transformer(CONFIG).eval()
    for parameter in source.parameters():
        parameter.data.normal_(0, 0.05)
    state = _fp8_state(source)
    native = Ideogram4Transformer(CONFIG).eval()
    swap_linears_to_fp8(native, state, torch.float32)
    load_fp8_state_dict(native, state, torch.device("cpu"), torch.float32, assign=True)
    reference_config = {
        key: value for key, value in CONFIG.items() if key not in ("num_heads", "adanln_dim")
    } | {"num_attention_heads": CONFIG["num_heads"], "adaln_dim": CONFIG["adanln_dim"]}
    reference = diffusers.Ideogram4Transformer2DModel(**reference_config).eval()
    reference.load_state_dict(_reference_state(state), strict=True)
    return native, reference


def _inputs():
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(1, 5, 8, generator=generator)
    features = torch.randn(1, 5, 8, generator=generator)
    positions = torch.tensor([[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]]])
    segment = torch.zeros(1, 5, dtype=torch.long)
    indicator = torch.tensor([[3, 3, 2, 2, 2]])
    timestep = torch.tensor([0.5])
    return hidden, timestep, features, positions, segment, indicator


def _output(model):
    hidden, timestep, features, positions, segment, indicator = _inputs()
    if isinstance(model, Ideogram4Transformer):
        return model(
            llm_features=features,
            x=hidden,
            t=timestep,
            position_ids=positions,
            segment_ids=segment,
            indicator=indicator,
        )
    return model(
        hidden,
        timestep,
        features,
        positions,
        segment,
        indicator,
        return_dict=False,
    )[0]


def _weights(reference, targets, seed):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name, module in reference.named_modules():
        if isinstance(module, torch.nn.Linear) and any(name.endswith(target) for target in targets):
            result[("transformer", name)] = (
                torch.randn(2, module.in_features, generator=generator) * 0.1,
                torch.randn(module.out_features, 2, generator=generator) * 0.1,
            )
    return result


def _registered(reference, name, targets):
    registered = set(reference.peft_config[name].target_modules)
    assert registered
    assert {
        target for target in targets if any(path.endswith(target) for path in registered)
    } == set(targets)


def _assert_loaded(native, reference, plain, path):
    expected = _output(reference)
    assert not torch.equal(expected, plain)
    apply_lora({"transformer": native}, (LoRASpec(path),), fuse=False)
    assert torch.equal(_output(native), expected)


def test_ideogram4_loader_accepts_only_the_conditional_transformer():
    assert diffusers.Ideogram4Pipeline._lora_loadable_modules == ["transformer"]


def test_ideogram4_diffusers_layout_adapters_match_dequantized_reference(tmp_path):
    native, reference = _models()
    plain = _output(reference)
    assert torch.equal(_output(native), plain)
    first = write_peft(tmp_path / "first.safetensors", _weights(reference, TARGETS, 4))
    second = write_peft(tmp_path / "second.safetensors", _weights(reference, TARGETS, 7))

    reference.load_lora_adapter(str(first), adapter_name="first")
    _registered(reference, "first", TARGETS)
    _assert_loaded(native, reference, plain, first)

    native, reference = _models()
    plain = _output(reference)
    reference.load_lora_adapter(str(first), adapter_name="first")
    reference.load_lora_adapter(str(second), adapter_name="second")
    _registered(reference, "first", TARGETS)
    _registered(reference, "second", TARGETS)
    reference.set_adapters(["first", "second"], [0.8, 0.5])
    expected = _output(reference)
    assert not torch.equal(expected, plain)
    apply_lora({"transformer": native}, (LoRASpec(first, 0.8), LoRASpec(second, 0.5)), fuse=False)
    assert torch.equal(_output(native), expected)

    native, reference = _models()
    plain = _output(reference)
    reference.load_lora_adapter(str(first), adapter_name="first")
    reference.load_lora_adapter(str(second), adapter_name="second")
    reference.set_adapters(["first", "second"], [0.8, 0.5])
    reference.fuse_lora()
    expected = _output(reference)
    assert not torch.equal(expected, plain)
    apply_lora({"transformer": native}, (LoRASpec(first, 0.8), LoRASpec(second, 0.5)), fuse=True)
    assert torch.equal(_output(native), expected)


def test_ideogram4_to_q_only_adapter_maps_to_the_first_fused_row_block(tmp_path):
    native, reference = _models()
    plain = _output(reference)
    path = write_peft(tmp_path / "q.safetensors", _weights(reference, ("to_q",), 4))
    reference.load_lora_adapter(str(path), adapter_name="q")
    _registered(reference, "q", ("to_q",))
    _assert_loaded(native, reference, plain, path)
