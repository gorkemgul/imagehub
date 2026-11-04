import pytest
import torch
from torch import nn

from imagehub.lora import (
    LoRAConfig,
    LoRAConv2d,
    LoRALinear,
    inject_lora,
    lora_layers,
    merge_lora,
    remove_lora,
    set_lora_scale,
    unmerge_lora,
    wrap_module,
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


@pytest.mark.parametrize(
    "bad",
    [
        dict(rank=0),
        dict(alpha=float("nan")),
        dict(dropout=1.0),
        dict(target_modules=()),
        dict(target_modules=("a", "a")),
        dict(init="normal"),
    ],
)
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        LoRAConfig(**bad)


def test_config_scaling():
    assert LoRAConfig(rank=4, alpha=8).scaling == 2.0


def test_injection_is_a_no_op_until_trained():
    torch.manual_seed(0)
    model = Block()
    x, image = inputs()
    before = model(x, image)
    injection = inject_lora(
        model, LoRAConfig(rank=2, target_modules=("to_q", "to_out.0", "proj_in", "conv"))
    )
    assert isinstance(model.to_q, LoRALinear)
    assert isinstance(model.conv, LoRAConv2d)
    assert set(injection.module_names) == {"to_q", "to_out.0", "proj_in", "conv"}
    for a, b in zip(model(x, image), before, strict=True):
        assert torch.equal(a, b)
    assert all(not p.requires_grad for p in model.to_q.base.parameters())
    assert all(p.requires_grad for p in injection.parameters())


def test_merge_unmerge_and_restore_round_trip():
    torch.manual_seed(0)
    model = Block()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    x, image = inputs()
    injection = inject_lora(
        model, LoRAConfig(rank=2, alpha=3.0, target_modules=("to_q", "proj_in", "conv"))
    )
    with torch.no_grad():
        for p in injection.parameters():
            p.normal_()
    unmerged = model(x, image)
    injection.merge()
    for a, b in zip(model(x, image), unmerged, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
    injection.unmerge()
    restored = injection.restore()
    assert type(restored.to_q) is nn.Linear
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, original[key], atol=1e-6, rtol=0)
    assert all(p.requires_grad for p in restored.parameters())


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
    expected = layer(x) + (x @ a1.T @ b1.T) * (0.5 * 4.0 / 2) + (x @ a2.T @ b2.T)
    torch.testing.assert_close(model(x), expected)
    set_lora_scale(model, "style", 0.0)
    torch.testing.assert_close(model(x), layer(x) + (x @ a2.T @ b2.T))
    assert list(lora_layers(model)) == ["0"]
    assert wrap_module(model, "0") is wrapped


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
    with pytest.raises(RuntimeError, match="Unmerge"):
        set_lora_scale(model, "a", 0.5)
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
    with pytest.raises(ValueError, match="rank"):
        layer.add_adapter("b", None, None, rank=5, alpha=2.0)
    with pytest.raises(TypeError):
        wrap_module(nn.Sequential(nn.LayerNorm(4)), "0")


def test_inject_reports_available_modules_when_nothing_matches():
    with pytest.raises(ValueError, match="to_q"):
        inject_lora(Block(), LoRAConfig(target_modules=("missing",)))


def test_inject_is_atomic():
    model = Block()
    model.nested = nn.Linear(8, 2)
    with pytest.raises(ValueError, match="rank"):
        inject_lora(model, LoRAConfig(rank=4, target_modules=("to_q", "nested")))
    assert type(model.to_q) is nn.Linear
    assert all(p.requires_grad for p in model.parameters())


def test_glob_patterns_match_full_names():
    model = Block()
    injection = inject_lora(model, LoRAConfig(rank=1, target_modules=("to_?",)))
    assert injection.module_names == ("to_q",)


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
    with pytest.raises(ValueError, match="shape"):
        injection.load_adapter_state_dict(state | {"to_q.lora_A.weight": torch.zeros(3, 8)})


def test_injection_is_seeded():
    first = inject_lora(Block(), LoRAConfig(rank=2, seed=3, target_modules=("to_q",)))
    second = inject_lora(Block(), LoRAConfig(rank=2, seed=3, target_modules=("to_q",)))
    a, b = first.adapter_state_dict(), second.adapter_state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert a["to_q.lora_A.weight"].abs().sum() > 0


def test_restored_injection_is_inactive():
    injection = inject_lora(Block(), LoRAConfig(rank=2, target_modules=("to_q",)))
    injection.restore()
    with pytest.raises(RuntimeError, match="restored"):
        injection.merge()


def test_gaussian_init_is_seeded_with_std_one_over_rank():
    def draw():
        model = nn.Sequential(nn.Linear(512, 512))
        injection = inject_lora(
            model, LoRAConfig(rank=4, init="gaussian", seed=1, target_modules=("0",))
        )
        return injection.adapter_state_dict()

    first, second = draw(), draw()
    down = first["0.lora_A.weight"]
    assert torch.equal(down, second["0.lora_A.weight"])
    assert abs(down.std().item() - 0.25) < 0.01
    assert not first["0.lora_B.weight"].any()
