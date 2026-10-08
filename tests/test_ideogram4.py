import pytest
import torch
from torch import nn

from imagehub.backends.ideogram4 import Ideogram4Backend
from imagehub.caption_verifier import CaptionVerifier
from imagehub.models.fp8 import Fp8Linear, load_fp8_state_dict, swap_linears_to_fp8
from imagehub.models.ideogram4_text import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    build_text_inputs,
    tokenize_prompt,
)
from imagehub.models.ideogram4_transformer import Ideogram4Transformer
from imagehub.schedulers.ideogram4 import PRESETS, get_schedule_for_resolution, ideogram4_step
from imagehub.types import T2IRequest


class _Linears(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quantized = nn.Linear(3, 2)
        self.regular = nn.Linear(2, 1)


class _Tokenizer:
    def chat_prompt(self, prompt: str) -> str:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def tokenize(self, text: str) -> list[int]:
        return list(range(1, len(text.split()) + 1))


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


def test_ideogram4_prompt_inputs_use_left_padding_and_image_positions() -> None:
    tokenizer = _Tokenizer()
    tokens, count = tokenize_prompt(tokenizer, "red fox")
    assert count == len(tokens)

    inputs = build_text_inputs(tokenizer, ["red fox", "fox"], 32, 48)
    assert inputs["num_image_tokens"] == 6
    assert inputs["grid_h"] == 2
    assert inputs["grid_w"] == 3
    assert inputs["max_text_tokens"] == count

    token_ids = inputs["token_ids"]
    indicator = inputs["indicator"]
    segment_ids = inputs["segment_ids"]
    position_ids = inputs["position_ids"]
    assert isinstance(token_ids, torch.Tensor)
    assert isinstance(indicator, torch.Tensor)
    assert isinstance(segment_ids, torch.Tensor)
    assert isinstance(position_ids, torch.Tensor)
    assert token_ids[1, 0].item() == 0
    assert indicator[1, 0].item() == 0
    assert segment_ids[1, 0].item() == SEQUENCE_PADDING_INDICATOR
    assert torch.all(indicator[:, -6:] == OUTPUT_IMAGE_INDICATOR)
    assert torch.all(indicator[0, :count] == LLM_TOKEN_INDICATOR)
    assert position_ids[0, count:].tolist() == [
        [IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET],
        [IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET + 1],
        [IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET + 2],
        [IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET + 1, IMAGE_POSITION_OFFSET],
        [IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET + 1, IMAGE_POSITION_OFFSET + 1],
        [IMAGE_POSITION_OFFSET, IMAGE_POSITION_OFFSET + 1, IMAGE_POSITION_OFFSET + 2],
    ]


def test_ideogram4_caption_verifier_accepts_valid_caption_and_reports_issues() -> None:
    valid = (
        '{"high_level_description":"A sign",'
        '"style_description":{"aesthetics":"clean","lighting":"daylight",'
        '"photo":"product","medium":"digital","color_palette":["#AABBCC"]},'
        '"compositional_deconstruction":{"background":"white","elements":['
        '{"type":"text","bbox":[100,200,300,800],"text":"HELLO",'
        '"desc":"black letters","color_palette":["#000000"]}]}}'
    )
    verifier = CaptionVerifier()
    assert verifier.verify_raw(valid) == []
    assert verifier.verify_raw("plain text")[0].startswith("invalid JSON:")
    assert verifier.verify_raw('{"compositional_deconstruction": {}}') == [
        "compositional_deconstruction: 'background' must exist"
    ]


def test_ideogram4_backend_validates_options_before_loading(tmp_path) -> None:
    backend = Ideogram4Backend(tmp_path, model_id="ideogram4-fp8")
    request = T2IRequest("a valid caption")

    options = backend.validate_options(request, {})
    assert options["preset"] == "V4_QUALITY_48"
    assert options["steps"] == 48
    assert options["guidance_schedule"] == PRESETS["V4_QUALITY_48"].guidance_schedule
    assert options["guidance"] is None
    assert options["width"] == options["height"] == 1024

    with pytest.raises(TypeError, match="Unknown Ideogram 4 options: typo"):
        backend.validate_options(request, {"typo": 1})
    with pytest.raises(ValueError, match="preset"):
        backend.validate_options(request, {"preset": "missing"})
    with pytest.raises(ValueError, match="multiple of 16"):
        backend.validate_options(request, {"width": 1000})
    with pytest.raises(ValueError, match="guidance_schedule must have length 2"):
        backend.validate_options(request, {"steps": 2, "guidance_schedule": [7.0]})
    with pytest.raises(ValueError, match="either seed or latents"):
        backend.validate_options(
            request,
            {
                "seed": 0,
                "latents": torch.zeros(1, 4096, 128),
            },
        )


def test_ideogram4_backend_latent_shape_and_overrides(tmp_path) -> None:
    backend = Ideogram4Backend(tmp_path)
    request = T2IRequest("a valid caption")
    latents = torch.zeros(1, 6, 128)
    options = backend.validate_options(
        request,
        {
            "preset": "V4_TURBO_12",
            "steps": 3,
            "guidance": 5.5,
            "guidance_schedule": (2.0, 3.0, 4.0),
            "width": 48,
            "height": 32,
            "latents": latents,
        },
    )
    assert options["steps"] == 3
    assert options["guidance"] == 5.5
    assert options["guidance_schedule"] == (2.0, 3.0, 4.0)
    assert options["latents"] is latents
