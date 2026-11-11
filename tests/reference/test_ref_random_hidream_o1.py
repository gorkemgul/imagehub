"""Random-weight parity with the pinned official HiDream-O1 implementation."""

import importlib.util
import sys
import types
from contextlib import nullcontext

import pytest
import torch
from reference_sources import reference_source

from imagehub.backends.hidream_o1 import HiDreamO1Components, decode_patches, denoise
from imagehub.models.hidream_o1 import HiDreamO1Model
from imagehub.schedulers.flow_unipc import FlowUniPCMultistepScheduler
from tests.test_hidream_o1 import TINY_CONFIG

pytestmark = pytest.mark.reference
pytest.importorskip("transformers")

REFERENCE = reference_source("HiDream-O1-Image") / "models" / "qwen3_vl_transformers.py"


def _load_reference_module():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def default_rope(config, device=None, seq_len=None):
        del seq_len
        parameters = config.rope_parameters
        dimension = config.head_dim
        inverse = 1.0 / (
            parameters["rope_theta"]
            ** (torch.arange(0, dimension, 2, dtype=torch.float32, device=device) / dimension)
        )
        return inverse, 1.0

    # The pinned source targets transformers 4.57, while the reference environment carries
    # 5.18. These two compatibility shims restore the renamed default-RoPE entry and mask
    # argument without altering either numerical operation.
    ROPE_INIT_FUNCTIONS["default"] = default_rope
    spec = importlib.util.spec_from_file_location("hidream_o1_reference", REFERENCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.create_causal_mask = lambda **kwargs: kwargs["attention_mask"]
    return module


def _reference_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    text = TINY_CONFIG["text_config"]
    text_config = Qwen3VLTextConfig(**(text | {"rope_parameters": text["rope_scaling"]}))
    vision_config = Qwen3VLVisionConfig(**TINY_CONFIG["vision_config"])
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=50,
        video_token_id=51,
        vision_start_token_id=52,
        vision_end_token_id=53,
    )
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    return config


def test_hidream_o1_random_weights_exact(record) -> None:
    reference_module = _load_reference_module()
    torch.manual_seed(0)
    reference = reference_module.Qwen3VLForConditionalGeneration(_reference_config()).eval()
    reference.model.tms_token_id = 7
    reference_forward = reference.forward

    def without_flash(**kwargs):
        kwargs["use_flash_attn"] = False
        return reference_forward(**kwargs)

    reference.forward = without_flash
    model = HiDreamO1Model(TINY_CONFIG).eval()
    model.load_checkpoint(reference.state_dict())

    input_ids = torch.tensor([[2, 7, 3]])
    patches = torch.randn(1, 2, 3 * 32 * 32)
    position_ids = torch.arange(5).view(1, 1, 5).expand(3, 1, 5)
    token_types = torch.tensor([[0, 1, 0, 1, 1]])
    timestep = torch.tensor([0.25])
    with torch.inference_mode():
        expected = reference(
            input_ids=input_ids,
            position_ids=position_ids,
            vinputs=patches,
            timestep=timestep,
            token_types=token_types,
            use_flash_attn=False,
        ).x_pred
        actual = model(input_ids, position_ids, patches, timestep, token_types)

    difference = float((actual - expected).abs().max())
    record("hidream_o1_random_weights", max_abs_diff=difference)
    assert torch.equal(actual, expected)


def _load_reference_scheduler():
    path = REFERENCE.with_name("fm_solvers_unipc.py")
    spec = importlib.util.spec_from_file_location("hidream_o1_scheduler_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("timesteps", [None, [999, 700, 300, 8]])
def test_hidream_o1_flow_unipc_exact(record, timesteps) -> None:
    reference_module = _load_reference_scheduler()
    reference = reference_module.FlowUniPCMultistepScheduler(use_dynamic_shifting=False, shift=3.0)
    ours = FlowUniPCMultistepScheduler(shift=3.0)
    steps = 5 if timesteps is None else len(timesteps)
    reference.set_timesteps(steps, device="cpu")
    ours.set_timesteps(steps, device="cpu", timesteps=timesteps)
    if timesteps is not None:
        reference.timesteps = torch.tensor(timesteps, dtype=torch.long)
        reference.sigmas = torch.tensor([value / 1000 for value in timesteps] + [0.0])
    assert torch.equal(ours.timesteps, reference.timesteps)
    assert torch.equal(ours.sigmas, reference.sigmas)
    sample = torch.randn(1, 3, 4, generator=torch.Generator().manual_seed(2))
    generator = torch.Generator().manual_seed(3)
    difference = 0.0
    for timestep in reference.timesteps:
        model_output = torch.randn(sample.shape, generator=generator)
        expected = reference.step(model_output, timestep.float(), sample, return_dict=False)[0]
        actual = ours.step(model_output, timestep.float(), sample)
        difference = max(difference, float((actual - expected).abs().max()))
        assert torch.equal(actual, expected)
        sample = expected
    record(
        "hidream_o1_flow_unipc_" + ("default" if timesteps is None else "custom"),
        max_abs_diff=difference,
    )


class _TinyTokenizer:
    def chat_prompt(self, prompt, *, enable_thinking=True):
        del enable_thinking
        return f"<u>{prompt}<a>"

    def tokenize(self, text):
        text = text.replace("<|boi_token|>", chr(5)).replace("<|tms_token|>", chr(7))
        return [ord(character) % 64 for character in text]

    def encode(self, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and not add_special_tokens
        return torch.tensor([self.tokenize(text)])


class _TinyProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return self.tokenizer.chat_prompt(messages[0]["content"])


def _load_reference_pipeline():
    # The text-to-image path does not execute torchvision transforms. Supply its small import-time
    # surface and a two-pattern rearrange implementation so the pinned pipeline can be imported
    # without installing optional image-conditioning packages.
    einops = types.ModuleType("einops")

    def rearrange(tensor, pattern, **sizes):
        if pattern.startswith("B C (H p1)"):
            batch, channels, height, width = tensor.shape
            p1, p2 = sizes["p1"], sizes["p2"]
            return (
                tensor.reshape(batch, channels, height // p1, p1, width // p2, p2)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch, -1, channels * p1 * p2)
            )
        if pattern.startswith("B (H W) (C p1 p2)"):
            batch = tensor.shape[0]
            height, width, p1, p2 = sizes["H"], sizes["W"], sizes["p1"], sizes["p2"]
            return (
                tensor.reshape(batch, height, width, 3, p1, p2)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(batch, 3, height * p1, width * p2)
            )
        raise AssertionError(pattern)

    einops.rearrange = rearrange
    transforms = types.ModuleType("torchvision.transforms.v2")
    transforms.Compose = lambda values: values
    transforms.ToImage = transforms.ToDtype = transforms.Normalize = lambda *args, **kwargs: (
        object()
    )
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = types.SimpleNamespace(v2=transforms)
    sys.modules["einops"] = einops
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = torchvision.transforms
    sys.modules["torchvision.transforms.v2"] = transforms
    root = str(REFERENCE.parents[1])
    sys.path.insert(0, root)
    try:
        from models import pipeline as module
    finally:
        sys.path.remove(root)
    module.find_closest_resolution = lambda width, height: (width, height)
    module.tqdm.tqdm = lambda values, **kwargs: values
    return module


@pytest.mark.parametrize(
    ("seed", "guidance", "height", "width"),
    [(0, 0.0, 32, 64), (42, 5.0, 64, 32)],
)
def test_hidream_o1_tiny_end_to_end_exact(
    record, seed: int, guidance: float, height: int, width: int
) -> None:
    reference_module = _load_reference_module()
    pipeline_module = _load_reference_pipeline()
    torch.manual_seed(8)
    reference = reference_module.Qwen3VLForConditionalGeneration(_reference_config()).eval()
    reference.model.tms_token_id = 7
    reference_forward = reference.forward

    def without_flash(**kwargs):
        kwargs["use_flash_attn"] = False
        return reference_forward(**kwargs)

    reference.forward = without_flash
    ours = HiDreamO1Model(TINY_CONFIG).eval()
    ours.load_checkpoint(reference.state_dict())
    tokenizer = _TinyTokenizer()
    processor = _TinyProcessor(tokenizer)
    real_torch = pipeline_module.torch

    class _Float32Torch:
        bfloat16 = torch.float32

        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def autocast(*args, **kwargs):
            return nullcontext()

    pipeline_module.torch = _Float32Torch()
    try:
        expected_image = pipeline_module.generate_image(
            reference,
            processor,
            "x",
            height=height,
            width=width,
            num_inference_steps=1,
            guidance_scale=guidance,
            shift=3.0,
            timesteps_list=[999],
            seed=seed,
        )
    finally:
        pipeline_module.torch = real_torch
    noise = 8 * torch.randn(
        (1, 3, height, width), generator=torch.Generator("cpu").manual_seed(seed + 1)
    )
    actual_patches = denoise(
        HiDreamO1Components(tokenizer, ours),  # type: ignore[arg-type]
        prompt="x",
        height=height,
        width=width,
        steps=1,
        guidance=guidance,
        shift=3.0,
        noise=noise,
        timesteps=[999],
    )
    actual = decode_patches(actual_patches, height, width)[0]
    expected = torch.from_numpy(__import__("numpy").array(expected_image))
    difference = int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max())
    record(
        f"hidream_o1_tiny_end_to_end_seed_{seed}",
        max_pixel_diff=difference,
        passed=difference == 0,
        guidance=guidance,
        size=[height, width],
    )
    assert torch.equal(actual, expected)
