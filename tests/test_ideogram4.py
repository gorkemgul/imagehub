import torch
from torch import nn

from imagehub.models.fp8 import Fp8Linear, load_fp8_state_dict, swap_linears_to_fp8
from imagehub.models.ideogram4_transformer import Ideogram4Transformer
from imagehub.schedulers.ideogram4 import PRESETS, get_schedule_for_resolution, ideogram4_step


class _Linears(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantized = nn.Linear(3, 2)
        self.regular = nn.Linear(2, 1)


def test_fp8_linear_forward_matches_dequantized_linear() -> None:
    layer = Fp8Linear(3, 2, bias=True, compute_dtype=torch.float32)
    layer.weight.copy_(torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.0, 4.0]]))
    layer.weight_scale.copy_(torch.tensor([0.5, 0.25]))
    assert layer.bias is not None
    layer.bias.copy_(torch.tensor([0.25, -0.5]))
    inputs = torch.tensor([[[2.0, -1.0, 0.5]]])

    weight = layer.weight.to(inputs.dtype) * layer.weight_scale.to(inputs.dtype).unsqueeze(1)
    expected = torch.nn.functional.linear(inputs, weight, layer.bias.to(inputs.dtype))

    torch.testing.assert_close(layer(inputs), expected, atol=0, rtol=0)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float32


def test_fp8_swap_and_load_preserve_checkpoint_dtypes() -> None:
    model = _Linears()
    state = {
        "quantized.weight": torch.ones(2, 3, dtype=torch.float8_e4m3fn),
        "quantized.weight_scale": torch.tensor([0.5, 0.25]),
        "quantized.bias": torch.tensor([1.0, 2.0]),
        "regular.weight": torch.tensor([[3.0, 4.0]]),
        "regular.bias": torch.tensor([5.0]),
    }

    swap_linears_to_fp8(model, state, torch.bfloat16)
    load_fp8_state_dict(model, state, torch.device("cpu"), torch.bfloat16, assign=True)

    assert isinstance(model.quantized, Fp8Linear)
    assert isinstance(model.regular, nn.Linear)
    assert model.quantized.weight.dtype == torch.float8_e4m3fn
    assert model.quantized.weight_scale.dtype == torch.float32
    assert model.quantized.bias is not None
    assert model.quantized.bias.dtype == torch.bfloat16
    assert model.regular.weight.dtype == torch.bfloat16
    assert model.regular.bias.dtype == torch.bfloat16


def test_ideogram4_published_config_and_layout() -> None:
    with torch.device("meta"):
        model = Ideogram4Transformer({})
        state = model.state_dict()

    assert model.config["adanln_dim"] == 512
    assert model.config["attention_head_dim"] == 256
    assert model.config["in_channels"] == 128
    assert model.config["intermediate_size"] == 12288
    assert model.config["llm_features_dim"] == 53248
    assert model.config["mrope_section"] == [24, 20, 20]
    assert model.config["norm_eps"] == 1e-5
    assert model.config["num_heads"] == 18
    assert model.config["num_layers"] == 34
    assert model.config["rope_theta"] == 5_000_000
    assert "input_proj.weight" in state
    assert "llm_cond_proj.weight" in state
    assert "layers.0.attention.qkv.weight" in state
    assert "layers.33.feed_forward.w3.weight" in state
    assert "final_layer.linear.weight" in state
    assert "rotary_emb.inv_freq" not in state


def test_ideogram4_schedule_presets_and_euler_step() -> None:
    assert set(PRESETS) == {"V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"}
    assert PRESETS["V4_QUALITY_48"].guidance_schedule == (3.0,) * 3 + (7.0,) * 45
    assert PRESETS["V4_DEFAULT_20"].guidance_schedule == (3.0,) * 2 + (7.0,) * 18
    assert PRESETS["V4_TURBO_12"].guidance_schedule == (3.0,) + (7.0,) * 11

    schedule = get_schedule_for_resolution((1024, 512), known_mean=0.5, std=1.75)
    sample = torch.tensor([[1.0, -2.0]])
    positive = torch.tensor([[0.5, 1.5]])
    negative = torch.tensor([[-0.25, 0.75]])
    guidance = 7.0
    t = float(schedule(torch.tensor([1.0])).item())
    s = float(schedule(torch.tensor([0.5])).item())
    expected = sample + (guidance * positive + (1.0 - guidance) * negative) * (s - t)

    torch.testing.assert_close(
        ideogram4_step(sample, positive, negative, guidance, t, s), expected, atol=0, rtol=0
    )
