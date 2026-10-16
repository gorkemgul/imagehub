"""Original Qwen-Image real-checkpoint parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    OriginalQwenImageComponents,
    decode_original_latents,
    denoise_original,
    encode_original_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, qwen_image_mu
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE_PROMPT = "blurry, low quality"
REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
REFERENCE = {"diffusers_version": "0.40.0"}


def _shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    files = [directory / "diffusion_pytorch_model.safetensors"] if index is None else []
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    result = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result.update(
            {
                name: tuple(entry["shape"])
                for name, entry in header.items()
                if name != "__metadata__"
            }
        )
    return result


def _header(expected, actual, tensor_count=None):
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual) if tensor_count is None else tensor_count,
    }


def test_published_tensor_headers_match_models(record):
    root = snapshot_dir("qwen-image")
    with torch.device("meta"):
        text = Qwen25VLTextModel(read_json(root / "text_encoder/config.json"))
        transformer = QwenImageTransformer2DModel(read_json(root / "transformer/config.json"))
        vae = AutoencoderKLQwenImage(read_json(root / "vae/config.json"))
    text_shapes = _shapes(root / "text_encoder")
    mapped_text = {
        ("model.language_model." + name.removeprefix("model."))
        if name.startswith("model.")
        else name: shape
        for name, shape in text_shapes.items()
        if name.startswith("model.") or name == "lm_head.weight"
    }
    result = {
        "text_encoder": _header(
            {name: tuple(value.shape) for name, value in text.state_dict().items()},
            mapped_text,
            len(text_shapes),
        ),
        "transformer": _header(
            {name: tuple(value.shape) for name, value in transformer.state_dict().items()},
            _shapes(root / "transformer"),
        ),
        "vae": _header(
            {name: tuple(value.shape) for name, value in vae.state_dict().items()},
            _shapes(root / "vae"),
        ),
    }
    record("header_qwen-image", **result)
    for entry in result.values():
        assert not entry["missing"] and not entry["extra"] and not entry["mismatch"]


def test_tokenizer_and_text_encoder(record, tmp_path):
    root = snapshot_dir("qwen-image")
    template = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    ours_tokenizer = Qwen2Tokenizer(root / "tokenizer")
    reference_tokenizer = transformers.Qwen2Tokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    text = template.format(PROMPT)
    ids_match = ours_tokenizer.tokenize(text) == reference_tokenizer.encode(text)
    record("qwen_image_tokenizer", ids_match=ids_match, tolerance="exact")
    assert ids_match

    ids = torch.tensor([reference_tokenizer.encode(text)])
    mask = torch.ones_like(ids)
    reference = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            input_ids=ids, attention_mask=mask, output_hidden_states=True
        ).hidden_states[-1]
    expected_path = tmp_path / "text.pt"
    torch.save(expected, expected_path)
    del reference, expected
    gc.collect()
    ours = Qwen25VLTextModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours(ids, mask)[-1]
    expected = torch.load(expected_path, weights_only=True)
    measured = max_diff(actual, expected)
    record("qwen_image_text_encoder", **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_transformer_step(record, tmp_path):
    root = snapshot_dir("qwen-image")
    generator = torch.Generator("cpu").manual_seed(3)
    latents = torch.randn(1, 4, 64, generator=generator)
    context = torch.randn(1, 8, 3584, generator=generator)
    timestep = torch.tensor([0.75])
    kwargs = {
        "hidden_states": latents,
        "encoder_hidden_states": context,
        "timestep": timestep,
        "img_shapes": [[(1, 2, 2)]],
    }
    reference = diffusers.QwenImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(**kwargs, return_dict=False)[0]
    expected_path = tmp_path / "transformer.pt"
    torch.save(expected, expected_path)
    del reference, expected
    gc.collect()
    ours = QwenImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(**kwargs)
    expected = torch.load(expected_path, weights_only=True)
    measured = max_diff(actual, expected)
    record("qwen_image_transformer", **measured, calls=1, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_vae_and_scheduler(record, tmp_path):
    root = snapshot_dir("qwen-image")
    latents = torch.randn(1, 16, 1, 4, 4, generator=torch.Generator("cpu").manual_seed(4))
    reference = diffusers.AutoencoderKLQwenImage.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference.decode(latents, return_dict=False)[0]
    expected_path = tmp_path / "vae.pt"
    torch.save(expected, expected_path)
    del reference, expected
    gc.collect()
    ours = AutoencoderKLQwenImage.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours.decode(latents)
    expected = torch.load(expected_path, weights_only=True)
    measured = max_diff(actual, expected)
    record("qwen_image_vae", **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    config = read_json(root / "scheduler/scheduler_config.json")
    sigmas = torch.linspace(1.0, 1 / 4, 4, dtype=torch.float64).tolist()
    mu = qwen_image_mu(1024, 1024, config)
    native_scheduler = FlowMatchEulerDiscreteScheduler(config)
    reference_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    native_scheduler.set_timesteps(4, sigmas=sigmas, mu=mu)
    reference_scheduler.set_timesteps(4, sigmas=sigmas, mu=mu)
    assert torch.equal(native_scheduler.timesteps, reference_scheduler.timesteps)
    assert torch.equal(native_scheduler.sigmas, reference_scheduler.sigmas)


def _reference_output(root, tmp_path, *, seed):
    reference = diffusers.QwenImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    final = []

    def capture(_pipe, _index, _timestep, values):
        final[:] = [values["latents"].clone()]
        return values

    with torch.inference_mode():
        pixels = reference(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            true_cfg_scale=4.0,
            num_inference_steps=2,
            height=32,
            width=32,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
            callback_on_step_end=capture,
        ).images
    output = {
        "latents": final[0],
        "pixels": torch.from_numpy((pixels * 255).round().astype("uint8")),
    }
    path = tmp_path / f"reference-{seed}.pt"
    torch.save(output, path)
    del reference, output, pixels
    gc.collect()
    return path


def test_end_to_end(record, tmp_path):
    root = snapshot_dir("qwen-image")
    expected_path = _reference_output(root, tmp_path, seed=0)
    expected = torch.load(expected_path, weights_only=True)
    components = OriginalQwenImageComponents.load(root, device=CPU, dtype=torch.float32)
    embeds, _ = encode_original_prompt(components, PROMPT, CPU)
    del embeds
    raw = torch.randn(1, 1, 16, 4, 4, generator=torch.Generator("cpu").manual_seed(0))
    noise = raw.view(1, 16, 2, 2, 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 64)
    actual_latents = denoise_original(
        components,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        steps=2,
        true_cfg=4.0,
        width=32,
        height=32,
        latents=noise,
    )
    actual_pixels = decode_original_latents(components.vae, actual_latents, 32, 32)
    latent_diff = max_diff(actual_latents, expected["latents"])
    pixel_diff = int((actual_pixels.int() - expected["pixels"].int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        "e2e_qwen-image",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        steps=2,
        true_cfg=4.0,
        size=[32, 32],
        reference=REFERENCE,
        revision=REVISION,
    )
    assert passed


def test_seed_end_to_end(record, tmp_path):
    root = snapshot_dir("qwen-image")
    expected_path = _reference_output(root, tmp_path, seed=42)
    expected = torch.load(expected_path, weights_only=True)["pixels"][0]
    actual = (
        pipeline("t2i", backend="qwen-image", model=root)(
            PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            seed=42,
            steps=2,
            true_cfg=4.0,
            width=32,
            height=32,
        )
        .images[0]
        .tensor
    )
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        "seed_e2e_qwen-image",
        pixels_max_abs=pixel_diff,
        seed=42,
        steps=2,
        true_cfg=4.0,
        size=[32, 32],
        passed=pixel_diff == 0,
    )
    assert pixel_diff == 0
