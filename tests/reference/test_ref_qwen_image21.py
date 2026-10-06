"""Qwen-Image 2.1 real-checkpoint parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    QwenImageComponents,
    decode_latents,
    denoise,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_qwen_image21 import AutoencoderKLQwenImage21
from imagehub.models.qwen3_vl_text import Qwen3VLTextModel
from imagehub.models.qwen_image21_transformer import (
    QwenImage21KVCache,
    QwenImage21Transformer2DModel,
)
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, qwen_image_mu
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip(
        "needs the diffusers commit pinned for Qwen-Image 2.1 (.venv-ref-qwen-image-2.1)",
        allow_module_level=True,
    )
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE_PROMPT = "blurry, low quality"
REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"
DIFFUSERS_COMMIT = "7997e4eabb1f880fa5697e2b24de09a892fffad5"


def _sharded_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"))
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    result = {}
    for shard in sorted(set(weight_map.values())):
        with (directory / shard).open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    assert set(result) == set(weight_map)
    return result


def _single_shapes(directory):
    path = directory / "diffusion_pytorch_model.safetensors"
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


def _header_entry(expected, actual, tensor_count):
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": tensor_count,
    }


def test_published_tensor_headers_match_models(record):
    root = snapshot_dir("qwen-image-2.1")
    text_shapes = _sharded_shapes(root / "text_encoder")
    text_config = read_json(root / "text_encoder/config.json")["text_config"]
    with torch.device("meta"):
        text = Qwen3VLTextModel(text_config)
        transformer = QwenImage21Transformer2DModel(read_json(root / "transformer/config.json"))
        vae = AutoencoderKLQwenImage21(read_json(root / "vae/config.json"))
    expected_text = {name: tuple(value.shape) for name, value in text.state_dict().items()}
    actual_text = {
        name.removeprefix("model.language_model."): shape
        for name, shape in text_shapes.items()
        if name.startswith("model.language_model.")
    }
    transformer_shapes = _sharded_shapes(root / "transformer")
    vae_shapes = _single_shapes(root / "vae")
    result = {
        "text_encoder": _header_entry(expected_text, actual_text, len(text_shapes)),
        "transformer": _header_entry(
            {name: tuple(value.shape) for name, value in transformer.state_dict().items()},
            transformer_shapes,
            len(transformer_shapes),
        ),
        "vae": _header_entry(
            {name: tuple(value.shape) for name, value in vae.state_dict().items()},
            vae_shapes,
            len(vae_shapes),
        ),
    }
    record("header_qwen-image-2.1", **result)
    for entry in result.values():
        assert not entry["missing"] and not entry["extra"] and not entry["mismatch"]


def test_prompt_ids(record):
    root = snapshot_dir("qwen-image-2.1")
    ours = Qwen2Tokenizer(root / "processor")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "processor", local_files_only=True
    )
    template = (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    prompts = [template.format(PROMPT), template.format(NEGATIVE_PROMPT)]
    actual = [ours.tokenize(prompt) for prompt in prompts]
    expected = [reference.encode(prompt) for prompt in prompts]
    matches = actual == expected
    record("qwen_image_prompt_ids", prompts=len(prompts), ids_match=matches, tolerance="exact")
    assert matches


def test_text_encoder_hidden_states(record):
    root = snapshot_dir("qwen-image-2.1")
    components = QwenImageComponents.load_text_encoder
    tokenizer = Qwen2Tokenizer(root / "processor")
    ours_model = components(root / "text_encoder", device=CPU, dtype=torch.float32)
    reference = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    full_prompt = (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = torch.tensor([tokenizer.tokenize(full_prompt)])
    mask = torch.ones_like(ids)
    text_model = reference.model.language_model
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        with torch.inference_mode():
            expected = reference(
                input_ids=ids, attention_mask=mask, output_hidden_states=True
            ).hidden_states[-1]
            actual = ours_model(ids, mask)[-1]
    finally:
        handle.remove()
    measured = max_diff(actual, expected)
    record("qwen_image_text_encoder", **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del ours_model, reference
    gc.collect()


def test_transformer_cached_two_steps(record):
    root = snapshot_dir("qwen-image-2.1")
    ours = QwenImage21Transformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.QwenImage21Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(3)
    latents = torch.randn(1, 4096, 64, generator=generator)
    context = torch.randn(1, 32, 4096, generator=generator)
    img_mask = torch.cat(
        [torch.zeros(1, 32, dtype=torch.bool), torch.ones(1, 1024, dtype=torch.bool)], dim=1
    )
    shapes = [[(1, 64, 64)]]
    ref_cache = diffusers.models.transformers.transformer_qwenimage21.QwenImage21KVCache(32)
    our_cache = QwenImage21KVCache(32)
    pairs = []
    with torch.inference_mode():
        for index, value in enumerate((0.9, 0.5)):
            mode = "extract" if index == 0 else "cached"
            timestep = torch.tensor([value])
            expected = reference(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=context,
                img_shapes=shapes,
                img_mask=img_mask,
                kv_cache=ref_cache,
                kv_cache_mode=mode,
                return_dict=False,
            )[0]
            actual = ours(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=context,
                img_shapes=shapes,
                img_mask=img_mask,
                kv_cache=our_cache,
                kv_cache_mode=mode,
            )
            pairs.append((actual, expected))
            latents = latents - actual[:, -4096:] * 0.1
    worst = max(
        (max_diff(actual, expected) for actual, expected in pairs), key=lambda item: item["max_abs"]
    )
    record("qwen_image_transformer", **worst, calls=2, kv_cache=True, tolerance=0.0)
    for actual, expected in pairs:
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del ours, reference
    gc.collect()


def test_vae_decode_encode(record):
    root = snapshot_dir("qwen-image-2.1")
    ours = AutoencoderKLQwenImage21.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKLQwenImage21.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(4)
    latents = torch.randn(1, 64, 64, 64, generator=generator)
    image = torch.randn(1, 4, 1024, 1024, generator=generator)
    with torch.inference_mode():
        actual_decode = ours.decode(latents)
        expected_decode = reference.decode(latents.unsqueeze(2)).sample[:, :, 0]
        actual_encode = ours.encode(image)
        expected_encode = reference.encode(image.unsqueeze(2)).latent_dist.mode()[:, :, 0]
    decode_diff, encode_diff = (
        max_diff(actual_decode, expected_decode),
        max_diff(actual_encode, expected_encode),
    )
    record("qwen_image_vae_decode", **decode_diff, tolerance=0.0)
    record("qwen_image_vae_encode", **encode_diff, tolerance=0.0)
    torch.testing.assert_close(actual_decode, expected_decode, atol=0, rtol=0)
    torch.testing.assert_close(actual_encode, expected_encode, atol=0, rtol=0)
    del ours, reference
    gc.collect()


def test_scheduler(record):
    root = snapshot_dir("qwen-image-2.1")
    config = read_json(root / "scheduler/scheduler_config.json")
    mu = qwen_image_mu(1024, 1024, config)
    sigmas = torch.linspace(1.0, 1 / 8, 8, dtype=torch.float64).tolist()
    ours = FlowMatchEulerDiscreteScheduler(config)
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    ours.set_timesteps(8, sigmas=sigmas, mu=mu)
    reference.set_timesteps(8, sigmas=sigmas, mu=mu)
    measured = max_diff(ours.timesteps, reference.timesteps)
    record("qwen_image_scheduler", **measured, tolerance=0.0)
    torch.testing.assert_close(ours.timesteps, reference.timesteps, atol=0, rtol=0)
    torch.testing.assert_close(ours.sigmas, reference.sigmas, atol=0, rtol=0)


def _reference_output(root, *, steps, true_cfg, negative_prompt, noise):
    reference = diffusers.QwenImage21Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    final = []

    def capture(_pipe, index, timestep, values):
        del index, timestep
        final[:] = [values["latents"].clone()]
        return values

    with torch.inference_mode():
        pixels = reference(
            PROMPT,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg,
            num_inference_steps=steps,
            height=1024,
            width=1024,
            latents=noise.clone(),
            output_type="np",
            use_kv_cache=True,
            callback_on_step_end=capture,
        ).images
    pixels = torch.from_numpy((pixels * 255).round().astype("uint8"))
    del reference
    gc.collect()
    return final[0], pixels


def _end_to_end(record, *, steps, true_cfg, negative_prompt, record_name):
    root = snapshot_dir("qwen-image-2.1")
    raw = torch.randn(1, 1, 64, 64, 64, generator=torch.Generator("cpu").manual_seed(1234))
    noise = raw.view(1, 64, 4096).transpose(1, 2)
    expected_latents, expected_pixels = _reference_output(
        root, steps=steps, true_cfg=true_cfg, negative_prompt=negative_prompt, noise=noise
    )
    ours = QwenImageComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=negative_prompt,
        steps=steps,
        true_cfg=true_cfg,
        width=1024,
        height=1024,
        latents=noise.clone(),
    )
    actual_pixels = decode_latents(ours.vae, actual_latents, 1024, 1024)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        record_name,
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        steps=steps,
        true_cfg=true_cfg,
        size=[1024, 1024],
        reference={"diffusers_git_commit": DIFFUSERS_COMMIT},
        revision=REVISION,
    )
    assert passed


def test_end_to_end(record):
    _end_to_end(
        record, steps=8, true_cfg=1.0, negative_prompt=None, record_name="e2e_qwen-image-2.1"
    )


def test_guided_end_to_end(record):
    _end_to_end(
        record,
        steps=4,
        true_cfg=4.0,
        negative_prompt=NEGATIVE_PROMPT,
        record_name="e2e_guided_qwen-image-2.1",
    )


def test_seed_end_to_end(record):
    root = snapshot_dir("qwen-image-2.1")
    reference = diffusers.QwenImage21Pipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            true_cfg_scale=1.0,
            num_inference_steps=8,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
            use_kv_cache=True,
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    del reference
    gc.collect()
    actual = (
        pipeline("t2i", backend="qwen-image-2.1", model=root)(
            PROMPT, seed=42, steps=8, true_cfg=1.0, width=1024, height=1024
        )
        .images[0]
        .tensor
    )
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        "seed_e2e_qwen-image-2.1",
        pixels_max_abs=pixel_diff,
        seed=42,
        steps=8,
        true_cfg=1.0,
        size=[1024, 1024],
        passed=pixel_diff == 0,
    )
    assert pixel_diff == 0
