"""HiDream-O1-Image parity against the pinned official implementation on CPU float32."""

import gc
import importlib.util
import json
import struct
import sys
import types
from contextlib import nullcontext
from importlib.machinery import ModuleSpec

import pytest
import torch
from helpers import max_diff, reference_in_subprocess, snapshot_dir
from reference_sources import reference_source

from imagehub import pipeline
from imagehub.backends.hidream_o1 import build_text_sample
from imagehub.models.hidream_o1 import HiDreamO1Model
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
pytest.importorskip("diffusers")
numpy = pytest.importorskip("numpy")
transformers = pytest.importorskip("transformers")

MODEL_ID = "hidream-o1-image"
REFERENCE_ROOT = reference_source("HiDream-O1-Image")
REFERENCE_COMMIT = "2c2d29ff729e48f33e41f49edfdbd81d5ac103b4"
REVISION = "0b0901d99f200389e138c61946af1185f5f49a13"
PROMPT = "A small red sailboat on a calm lake at sunrise."
STEPS = 2
HEIGHT = 32
WIDTH = 64


def _load_reference_model_module():
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    def default_rope(config, device=None, seq_len=None):
        del seq_len
        dimension = config.head_dim
        inverse = 1.0 / (
            config.rope_parameters["rope_theta"]
            ** (torch.arange(0, dimension, 2, dtype=torch.float32, device=device) / dimension)
        )
        return inverse, 1.0

    ROPE_INIT_FUNCTIONS["default"] = default_rope
    path = REFERENCE_ROOT / "models" / "qwen3_vl_transformers.py"
    if not path.is_file():
        pytest.skip(f"Pinned HiDream-O1 reference checkout not found at {REFERENCE_ROOT}")
    spec = importlib.util.spec_from_file_location("hidream_o1_real_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.create_causal_mask = lambda **kwargs: kwargs["attention_mask"]
    return module


def _load_reference_pipeline_module():
    einops = types.ModuleType("einops")

    def rearrange(tensor, pattern, **axes):
        patch_height = axes["p1"]
        patch_width = axes["p2"]
        if pattern.startswith("B C"):
            batch, channels, height, width = tensor.shape
            return (
                tensor.reshape(
                    batch,
                    channels,
                    height // patch_height,
                    patch_height,
                    width // patch_width,
                    patch_width,
                )
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch, -1, channels * patch_height * patch_width)
            )
        if pattern.startswith("B (H W)"):
            batch, _, channels = tensor.shape
            height = axes["H"]
            width = axes["W"]
            return (
                tensor.reshape(batch, height, width, -1, patch_height, patch_width)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(
                    batch,
                    channels // (patch_height * patch_width),
                    height * patch_height,
                    width * patch_width,
                )
            )
        channels, height, width = tensor.shape
        return (
            tensor.reshape(
                channels,
                height // patch_height,
                patch_height,
                width // patch_width,
                patch_width,
            )
            .permute(1, 3, 0, 2, 4)
            .reshape(-1, channels * patch_height * patch_width)
        )

    einops.rearrange = rearrange
    sys.modules["einops"] = einops
    transforms = types.ModuleType("torchvision.transforms.v2")
    transforms.__spec__ = ModuleSpec(transforms.__name__, loader=None)
    transforms.Compose = lambda values: values
    transforms.ToImage = transforms.ToDtype = transforms.Normalize = lambda *args, **kwargs: (
        object()
    )
    torchvision = types.ModuleType("torchvision")
    torchvision.__spec__ = ModuleSpec(torchvision.__name__, loader=None)
    torchvision.transforms = types.SimpleNamespace(v2=transforms)
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = torchvision.transforms
    sys.modules["torchvision.transforms.v2"] = transforms
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        from models import pipeline as module
    finally:
        sys.path.remove(str(REFERENCE_ROOT))
    module.find_closest_resolution = lambda width, height: (width, height)
    module.tqdm.tqdm = lambda values, **kwargs: values
    return module


def _load_reference(root):
    module = _load_reference_model_module()
    model = module.Qwen3VLForConditionalGeneration.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    for child in model.modules():
        original = getattr(child, "original_inv_freq", None)
        if original is not None and original.device.type == "meta":
            child.original_inv_freq = child.inv_freq
    original = model.forward

    def without_flash(**kwargs):
        kwargs["use_flash_attn"] = False
        return original(**kwargs)

    model.forward = without_flash
    return model


def _headers(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    result = {}
    for name in sorted(set(index["weight_map"].values())):
        with (directory / name).open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result |= {
            key: tuple(value["shape"]) for key, value in header.items() if key != "__metadata__"
        }
    return result


def test_hidream_o1_headers(record):
    root = snapshot_dir(MODEL_ID)
    published = _headers(root)
    with torch.device("meta"):
        model = HiDreamO1Model(json.loads((root / "config.json").read_text()))
    expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    result = {
        "missing": sorted(expected.keys() - published.keys()),
        "extra": sorted(published.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & published.keys() if expected[name] != published[name]
        ),
        "tensors": len(published),
    }
    record(f"header_{MODEL_ID}", **result)
    assert not result["missing"] and not result["extra"] and not result["mismatch"]


def test_hidream_o1_tokenizer_chat_and_positions(record):
    root = snapshot_dir(MODEL_ID)
    reference = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    native = Qwen2Tokenizer(root)
    prompts = ["", PROMPT, "naïve café 日本語 🐱\nline two"]
    matches = []
    for prompt in prompts:
        rendered = reference.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        matches.append(
            rendered == native.chat_prompt(prompt)
            and reference.encode(rendered, add_special_tokens=False) == native.tokenize(rendered)
        )
    sample = build_text_sample(
        PROMPT, HEIGHT, WIDTH, native, json.loads((root / "config.json").read_text())
    )
    record(
        "hidream_o1_tokenizer_positions",
        prompts=len(prompts),
        exact=all(matches),
        sequence=int(sample["position_ids"].shape[-1]),
    )
    assert all(matches)
    assert sample["position_ids"].shape[0] == 3


def test_hidream_o1_denoiser_step(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    tokenizer = Qwen2Tokenizer(root)
    config = json.loads((root / "config.json").read_text())
    sample = build_text_sample(PROMPT, HEIGHT, WIDTH, tokenizer, config)
    patches = torch.randn(1, 2, 3 * 32 * 32, generator=torch.Generator().manual_seed(7))
    timestep = torch.tensor([0.4])

    reference = _load_reference(root)
    with torch.inference_mode():
        expected = reference(
            input_ids=sample["input_ids"],
            position_ids=sample["position_ids"],
            vinputs=patches,
            timestep=timestep,
            token_types=sample["token_types"],
            use_flash_attn=False,
        ).x_pred
    torch.save(expected, tmp_path / "step.pt")
    del expected, reference
    gc.collect()

    native = HiDreamO1Model.from_pretrained(root, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        actual = native(
            sample["input_ids"],
            sample["position_ids"],
            patches,
            timestep,
            sample["token_types"],
        )
    expected = torch.load(tmp_path / "step.pt", weights_only=True)
    difference = max_diff(actual, expected)
    record("hidream_o1_denoiser_step", **difference, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class _Float32Torch:
    bfloat16 = torch.float32

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def autocast(*args, **kwargs):
        return nullcontext()


def _reference_image(root, seed, output):
    reference = _load_reference(root)
    tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)
    module = _load_reference_pipeline_module()
    real_torch = module.torch
    module.torch = _Float32Torch()
    try:
        image = module.generate_image(
            reference,
            tokenizer,
            PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            guidance_scale=5.0,
            shift=3.0,
            seed=seed,
        )
        torch.save(torch.from_numpy(numpy.array(image)), output)
    finally:
        module.torch = real_torch
        del reference, tokenizer
        gc.collect()


def _end_to_end(record, tmp_path, seed, name):
    root = snapshot_dir(MODEL_ID)
    output = tmp_path / f"expected-{seed}.pt"
    expected = reference_in_subprocess(
        "reference.test_ref_hidream_o1._reference_image",
        {"root": root, "seed": seed},
        output,
    )
    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root, dtype=torch.float32)(
            PROMPT,
            seed=seed,
            steps=STEPS,
            guidance=5.0,
            shift=3.0,
            height=HEIGHT,
            width=WIDTH,
        )
        .images[0]
        .tensor
    )
    difference = int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max())
    passed = torch.equal(actual, expected)
    record(
        name,
        passed=passed,
        max_pixel_diff=difference,
        seed=seed,
        steps=STEPS,
        size=[HEIGHT, WIDTH],
        revision=REVISION,
        reference={"git_commit": REFERENCE_COMMIT},
    )
    assert passed


def test_hidream_o1_end_to_end(record, tmp_path):
    _end_to_end(record, tmp_path, 0, f"e2e_{MODEL_ID}")


def test_hidream_o1_seed_end_to_end(record, tmp_path):
    _end_to_end(record, tmp_path, 42, f"seed_e2e_{MODEL_ID}")
