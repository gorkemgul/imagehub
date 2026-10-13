"""Ovis-Image 7B checkpoint parity on CPU float32.

Reference results are saved and released before native components are loaded, so the
large text encoder and transformer are never resident twice.
"""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.ovis_image_transformer import OvisImageTransformer2DModel
from imagehub.models.qwen3 import Qwen3Model
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_ID = "ovis-image-7b"
PROMPT = "a photograph of an astronaut riding a horse on the moon"
SEED_PROMPT = "a red cat reading a book in a sunlit library"
REDUCED_STEPS = 2


@pytest.fixture(scope="module")
def root():
    return snapshot_dir(MODEL_ID)


def _shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    paths = (
        [
            directory / name
            for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))
        ]
        if index is not None
        else list(directory.glob("*.safetensors"))
    )
    result = {}
    for path in paths:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result.update(
            {
                name: tuple(value["shape"])
                for name, value in header.items()
                if name != "__metadata__"
            }
        )
    return result


def _entry(expected, actual):
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual),
    }


def _check(record, name, actual, expected):
    record(name, **max_diff(actual, expected), tolerance=0.0)
    assert torch.equal(actual, expected)


def test_header(root, record):
    result = {}
    for component, cls in {
        "text_encoder": Qwen3Model,
        "transformer": OvisImageTransformer2DModel,
        "vae": AutoencoderKL,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = _shapes(root / component)
        if component == "text_encoder":
            actual = {
                "model." + name: value for name, value in actual.items() if name != "lm_head.weight"
            }
        actual = {
            name: tuple(value.shape)
            for name, value in model.prepare_state_dict(
                {name: torch.empty(shape, device="meta") for name, shape in actual.items()}
            ).items()
        }
        result[component] = _entry(expected, actual)
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not item[key] for item in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(root, record):
    ours = Qwen2Tokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    prompts = ["", PROMPT, "naïve café 日本語", "line1\nline2 <|im_end|>"]
    for prompt in prompts:
        expected = reference(prompt, add_special_tokens=False).input_ids
        assert ours.tokenize(prompt) == expected
    record("ovis_image_tokenizer", prompts=len(prompts), identical=True)


def test_text_encoder(root, record, tmp_path):
    tokenizer = Qwen2Tokenizer(root / "tokenizer")
    ids, mask = tokenizer(PROMPT, max_length=64)
    reference = transformers.Qwen3Model.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(ids, attention_mask=mask).last_hidden_state
    output = tmp_path / "text.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    ours = Qwen3Model.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours(ids, mask)
    _check(record, "ovis_image_text_encoder", actual, torch.load(output, weights_only=True))


def test_transformer_step(root, record, tmp_path):
    generator = torch.Generator("cpu").manual_seed(3)
    hidden = torch.randn(1, 64, 64, generator=generator)
    context = torch.randn(1, 32, 2048, generator=generator)
    image_ids = torch.zeros(64, 3)
    image_ids[:, 1] = torch.arange(64) // 8
    image_ids[:, 2] = torch.arange(64) % 8
    text_ids = torch.zeros(32, 3)
    text_ids[:, 1:] = torch.arange(32)[:, None]
    timestep = torch.tensor([0.5])
    reference = diffusers.OvisImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(hidden, context, timestep, image_ids, text_ids, return_dict=False)[0]
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    ours = OvisImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(hidden, context, timestep, image_ids, text_ids)
    _check(record, "ovis_image_transformer_step", actual, torch.load(output, weights_only=True))


def test_vae_decode(root, record, tmp_path):
    latents = torch.randn(1, 16, 16, 16, generator=torch.Generator("cpu").manual_seed(4))
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference.decode(
            latents / reference.config.scaling_factor + reference.config.shift_factor,
            return_dict=False,
        )[0]
    output = tmp_path / "vae.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours.decode(latents / ours.scaling_factor + (ours.shift_factor or 0))
    _check(record, "ovis_image_vae_decode", actual, torch.load(output, weights_only=True))


def _reference_pixels(root, prompt, seed, tmp_path):
    reference = diffusers.OvisImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    images = reference(
        prompt,
        num_inference_steps=REDUCED_STEPS,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    ).images
    output = tmp_path / f"reference-{seed}.pt"
    torch.save(torch.from_numpy((images * 255).round().astype("uint8")), output)
    del reference, images
    gc.collect()
    return output


@pytest.mark.parametrize(
    ("name", "prompt", "seed"),
    [("e2e_ovis-image-7b", PROMPT, 0), ("seed_e2e_ovis-image-7b", SEED_PROMPT, 42)],
)
def test_end_to_end(root, record, tmp_path, name, prompt, seed):
    expected = _reference_pixels(root, prompt, seed, tmp_path)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(prompt, steps=REDUCED_STEPS, seed=seed)
    actual = torch.stack([image.tensor for image in result.images])
    expected_pixels = torch.load(expected, weights_only=True)
    passed = torch.equal(actual, expected_pixels)
    record(
        name,
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected_pixels.to(torch.int16)).abs().max()),
        steps=REDUCED_STEPS,
        seed=seed,
    )
    assert passed
