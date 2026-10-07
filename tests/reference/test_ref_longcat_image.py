"""LongCat-Image parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.longcat_image import (
    LongCatImageComponents,
    _pack_latents,
    calculate_shift,
    denoise,
    encode_prompt,
    rewrite_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.longcat_image_transformer import LongCatImageTransformer2DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 8


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = [next(directory.glob("*.safetensors"))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def test_header(record):
    root = snapshot_dir("longcat-image")
    result = {}
    for component, cls in {
        "text_encoder": Qwen25VLTextModel,
        "transformer": LongCatImageTransformer2DModel,
        "vae": AutoencoderKL,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    record("header_longcat_image", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_rewritten_prompt_and_token_ids(record, tmp_path):
    root = snapshot_dir("longcat-image")
    reference = diffusers.LongCatImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected_rewrite = reference.rewire_prompt(PROMPT, CPU)[0]
    expected_ids = reference.tokenizer(expected_rewrite, add_special_tokens=False).input_ids[:512]
    output = tmp_path / "rewrite.json"
    output.write_text(json.dumps({"prompt": expected_rewrite, "ids": expected_ids}))
    del reference
    gc.collect()

    components = LongCatImageComponents.load(root, device=CPU, dtype=torch.float32)
    actual_rewrite = rewrite_prompt(components, PROMPT, CPU)
    actual_ids = components.tokenizer.tokenize(actual_rewrite)[:512]
    expected = json.loads(output.read_text())
    exact = actual_rewrite == expected["prompt"] and actual_ids == expected["ids"]
    record("longcat_image_rewrite", exact=exact, tolerance="exact")
    assert exact


def test_text_encoder(record, tmp_path):
    root = snapshot_dir("longcat-image")
    reference = diffusers.LongCatImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected, expected_ids = reference.encode_prompt(PROMPT)
    output = tmp_path / "text.pt"
    torch.save({"embeds": expected, "ids": expected_ids}, output)
    del reference, expected, expected_ids
    gc.collect()

    components = LongCatImageComponents.load(root, device=CPU, dtype=torch.float32)
    actual, actual_ids = encode_prompt(components, PROMPT, CPU)
    expected = torch.load(output, weights_only=True)
    _check(record, "longcat_image_text_encoder", actual, expected["embeds"])
    assert torch.equal(actual_ids, expected["ids"])


def test_transformer(record, tmp_path):
    root = snapshot_dir("longcat-image")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 4096, 64, generator=generator)
    context = torch.randn(1, 512, 3584, generator=generator)
    img_ids = torch.zeros(4096, 3)
    img_ids[:, 0] = 1
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(64) + 512, torch.arange(64) + 512)
    txt_ids = torch.zeros(512, 3)
    txt_ids[:, 1] = torch.arange(512)
    txt_ids[:, 2] = torch.arange(512)
    timestep = torch.tensor([0.5])
    reference = diffusers.LongCatImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = LongCatImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(hidden, context, timestep, img_ids, txt_ids)
    _check(record, "longcat_image_transformer", actual, torch.load(output, weights_only=True))


def test_vae(record, tmp_path):
    root = snapshot_dir("longcat-image")
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 16, 128, 128, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = {
            "decode": reference.decode(latents).sample,
            "encode": reference.encode(image).latent_dist.mode(),
        }
    output = tmp_path / "vae.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_decode = native.decode(latents)
        actual_encode = native.encode(image)
    expected = torch.load(output, weights_only=True)
    _check(record, "longcat_image_vae_decode", actual_decode, expected["decode"])
    _check(record, "longcat_image_vae_encode", actual_encode, expected["encode"])


def test_scheduler(record):
    root = snapshot_dir("longcat-image")
    config = read_json(root / "scheduler/scheduler_config.json")
    sigmas = [1.0, 0.75, 0.5, 0.25]
    mu = calculate_shift(4096, config)
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=sigmas, mu=mu)
    native = FlowMatchEulerDiscreteScheduler(config)
    native.set_timesteps(4, sigmas=sigmas, mu=mu)
    exact = torch.equal(native.sigmas, reference.sigmas.cpu()) and torch.equal(
        native.timesteps, reference.timesteps.cpu()
    )
    record("longcat_image_scheduler", mu=mu, exact=exact, tolerance="exact")
    assert exact


def test_end_to_end(record, tmp_path):
    root = snapshot_dir("longcat-image")
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(42))
    reference = diffusers.LongCatImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": STEPS,
        "guidance_scale": 4.5,
        "height": 1024,
        "width": 1024,
        "latents": _pack_latents(noise.clone(), 1, 16, 128, 128),
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    output = tmp_path / "end-to-end.pt"
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, output)
    del reference, expected_latents, expected_pixels
    gc.collect()

    components = LongCatImageComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=STEPS,
        guidance=4.5,
        width=1024,
        height=1024,
        latents=noise.clone(),
        enable_cfg_renorm=True,
        cfg_renorm_min=0.0,
        enable_prompt_rewrite=True,
    )
    expected = torch.load(output, weights_only=True)
    _check(record, "longcat_image_e2e_latents", actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend="longcat-image", model=root)(
        PROMPT, steps=STEPS, latents=noise
    )
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    record(
        "e2e_longcat_image",
        passed=passed,
        max_pixel_diff=int(
            (pixels.to(torch.int16) - expected["pixels"].to(torch.int16)).abs().max()
        ),
        steps=STEPS,
    )
    assert passed


def test_seed_end_to_end(record, tmp_path):
    root = snapshot_dir("longcat-image")
    reference = diffusers.LongCatImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        PROMPT,
        num_inference_steps=STEPS,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    actual = (
        pipeline("t2i", backend="longcat-image", model=root)(PROMPT, seed=42, steps=STEPS)
        .images[0]
        .tensor
    )
    expected = torch.load(output, weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        "seed_e2e_longcat_image",
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        seed=42,
        steps=STEPS,
    )
    assert passed
