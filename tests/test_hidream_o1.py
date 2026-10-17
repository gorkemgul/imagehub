import json
from pathlib import Path

import torch

from imagehub.backends.hidream_o1 import (
    HiDreamO1Backend,
    HiDreamO1Components,
    build_text_sample,
    decode_patches,
    denoise,
    get_rope_index_fix_point,
)
from imagehub.models.hidream_o1 import (
    HiDreamO1Model,
    HiDreamO1PatchEmbed,
    HiDreamO1TimestepEmbedder,
)
from imagehub.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from imagehub.types import T2IRequest

FIXTURES = Path(__file__).parent / "fixtures" / "hidream_o1_configs"

TINY_CONFIG = {
    "image_token_id": 50,
    "video_token_id": 51,
    "vision_start_token_id": 52,
    "vision_end_token_id": 53,
    "tms_token_id": 7,
    "text_config": {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
    },
    "vision_config": {
        "depth": 1,
        "hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 16,
        "num_heads": 2,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 16,
        "out_hidden_size": 16,
        "deepstack_visual_indexes": [],
    },
}


def test_hidream_o1_published_header_layout() -> None:
    config = json.loads((FIXTURES / "config.json").read_text())
    published = {}
    for path in FIXTURES.glob("model-*-header.json"):
        published.update(json.loads(path.read_text()))
    with torch.device("meta"):
        model = HiDreamO1Model(config)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == published


def test_hidream_o1_generation_shapes() -> None:
    model = HiDreamO1Model(TINY_CONFIG).eval()
    input_ids = torch.tensor([[2, 7, 3]])
    patches = torch.randn(1, 2, 3 * 32 * 32)
    position_ids = torch.arange(5).view(1, 1, 5).expand(3, 1, 5)
    token_types = torch.tensor([[0, 1, 0, 1, 1]])
    with torch.inference_mode():
        output = model(input_ids, position_ids, patches, torch.tensor([0.25]), token_types)
    assert output.shape == (1, 5, 3 * 32 * 32)
    assert torch.isfinite(output).all()


def test_hidream_o1_new_component_shapes() -> None:
    assert HiDreamO1PatchEmbed(4, 16)(torch.randn(2, 3, 48)).shape == (2, 3, 16)
    assert HiDreamO1TimestepEmbedder(16)(torch.tensor([0.0, 1.0])).shape == (2, 16)


class _TinyTokenizer:
    def chat_prompt(self, prompt: str, *, enable_thinking: bool = True) -> str:
        del enable_thinking
        return f"<user>{prompt}<assistant>"

    def tokenize(self, text: str) -> list[int]:
        special = {"<|boi_token|>": 5, "<|tms_token|>": 7}
        for token, token_id in special.items():
            text = text.replace(token, chr(token_id))
        return [ord(character) % 64 for character in text]


def test_hidream_o1_text_sample_and_fixed_point_positions() -> None:
    tokenizer = _TinyTokenizer()
    config = {"image_token_id": 50, "vision_start_token_id": 52}
    sample = build_text_sample("x", 32, 64, tokenizer, config)  # type: ignore[arg-type]
    assert sample["input_ids"][0, -1].item() == 7
    assert sample["position_ids"].shape[-1] == sample["input_ids"].shape[-1] + 2
    assert int(sample["vinput_mask"].sum()) == 2
    assert int(sample["token_types"].sum()) == 3
    assert torch.equal(sample["position_ids"][:, 0, -2], torch.tensor([4096, 4096, 4096]))
    assert torch.equal(sample["position_ids"][:, 0, -1], torch.tensor([4096, 4096, 4097]))
    direct = get_rope_index_fix_point(
        torch.tensor([[1, 52, 50]]),
        torch.tensor([[1, 1, 2]]),
        image_token_id=50,
        vision_start_token_id=52,
    )
    assert torch.equal(
        direct[:, 0], torch.tensor([[0, 4096, 4096], [0, 4096, 4096], [0, 4096, 4097]])
    )


def test_flow_unipc_is_repeatable_and_uses_requested_timesteps() -> None:
    scheduler = FlowUniPCMultistepScheduler(shift=3.0)
    scheduler.set_timesteps(3, timesteps=[999, 500, 8])
    sample = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10
    for timestep in scheduler.timesteps:
        sample = scheduler.step(torch.full_like(sample, 0.25), timestep.float(), sample)
    assert torch.isfinite(sample).all()
    assert scheduler.step_index == 3


def test_hidream_o1_decode_patches_rounds_uint8() -> None:
    patches = torch.zeros(1, 2, 3 * 32 * 32)
    pixels = decode_patches(patches, 32, 64)
    assert pixels.shape == (1, 32, 64, 3)
    assert pixels.dtype == torch.uint8
    assert torch.all(pixels == 128)


def test_hidream_o1_sampling_is_repeatable_and_restores_autograd() -> None:
    components = HiDreamO1Components(_TinyTokenizer(), HiDreamO1Model(TINY_CONFIG).eval())  # type: ignore[arg-type]
    noise = torch.randn(1, 3, 32, 64, generator=torch.Generator().manual_seed(4))
    arguments = {
        "prompt": "x",
        "height": 32,
        "width": 64,
        "steps": 2,
        "guidance": 5.0,
        "shift": 3.0,
        "noise": noise,
        "timesteps": [999, 8],
    }
    torch.set_grad_enabled(True)
    first = denoise(components, **arguments)
    assert torch.is_grad_enabled()
    second = denoise(components, **arguments)
    assert torch.is_grad_enabled()
    assert torch.equal(first, second)


def test_hidream_o1_options_are_validated_before_loading(tmp_path) -> None:
    backend = HiDreamO1Backend(tmp_path)
    request = T2IRequest("a painting")
    try:
        backend.infer(request, unknown=True)
    except TypeError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown option was accepted")
    assert not backend.is_loaded


def test_hidream_o1_option_validation() -> None:
    backend = HiDreamO1Backend("unused")
    request = T2IRequest("a painting")
    selected = backend.validate_options(
        request, {"width": 64, "height": 32, "steps": 2, "timesteps": [999, 8]}
    )
    assert selected["steps"] == 2
    for options in ({"width": 63}, {"guidance": float("nan")}, {"seed": -1}):
        try:
            backend.validate_options(request, options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid options were accepted: {options}")
