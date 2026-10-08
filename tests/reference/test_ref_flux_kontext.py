"""FLUX.1 Kontext parity on CPU float32 against the real checkpoint."""

import gc
import json
import struct

import numpy as np
import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends import flux as flux_backend
from imagehub.backends.flux import _pack_latents, _prepare_latent_image_ids
from imagehub.backends.flux_kontext import (
    encode_kontext_image,
    kontext_input_size,
    preprocess_kontext_image,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.flux_transformer import FluxTransformer2DModel
from imagehub.models.t5 import T5EncoderModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
PILImage = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

CPU = torch.device("cpu")
MODEL_ID = "flux1-kontext-dev"
PROMPT = "Add a red hat to the scene"
PLAIN_PROMPT = "a photograph of an astronaut riding a horse"
MAX_LENGTH = 512
GUIDANCE = 3.5
REDUCED_STEPS = 4


def make_image(width, height, variant):
    """A deterministic smooth gradient with a few filled shapes."""
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None].repeat(width, axis=1)
    channels = (x, y, (x + y) / 2) if variant == 0 else (y, (x + y) / 2, 255 - x)
    image = PILImage.fromarray(np.stack(channels, axis=-1).round().astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (width // 8, height // 8, width // 8 + width // 4, height // 8 + height // 3),
        fill=(220, 40, 40),
    )
    draw.rectangle(
        (width // 2, height // 2, width // 2 + width // 3, height // 2 + height // 4),
        fill=(30, 60, 200),
    )
    draw.polygon(
        [(width // 4, height - 10), (width // 2, height // 2 + 20), (3 * width // 4, height - 10)],
        fill=(240, 220, 40),
    )
    return image


def landscape():
    return make_image(1024, 768, 0)


def portrait():
    return make_image(640, 960, 1)


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _check_pixels(record, name, actual, expected, **extra):
    passed = torch.equal(actual, expected)
    record(
        name,
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        **extra,
    )
    assert passed


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


def _reference_input_pipeline(vae):
    """The reference pipeline's own input methods only need the VAE and the image processor."""
    pipe = object.__new__(diffusers.FluxKontextPipeline)
    pipe.vae_scale_factor = 8
    pipe.image_processor = diffusers.image_processor.VaeImageProcessor(vae_scale_factor=16)
    pipe.vae = vae
    return pipe


def _reference_input_size(pipe, image):
    # The arithmetic of FluxKontextPipeline.__call__ for the input image (auto resize).
    preferred = diffusers.pipelines.flux.pipeline_flux_kontext.PREFERRED_KONTEXT_RESOLUTIONS
    multiple_of = pipe.vae_scale_factor * 2
    height, width = pipe.image_processor.get_default_height_width(image)
    aspect_ratio = width / height
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in preferred)
    return height // multiple_of * multiple_of, width // multiple_of * multiple_of


def _reference_input(pipe, image):
    height, width = _reference_input_size(pipe, image)
    resized = pipe.image_processor.resize(image, height, width)
    pixels = pipe.image_processor.preprocess(resized, height, width)
    return (height, width), pixels


def _pixels(array):
    return torch.from_numpy((array * 255).round().astype("uint8"))


def _reference_run(root, **kwargs):
    """Run the reference pipeline once; return the final packed latents and uint8 pixels."""
    reference = diffusers.FluxKontextPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    captured = {}

    def capture(_pipe, _index, _timestep, callback_kwargs):
        captured["latents"] = callback_kwargs["latents"].clone()
        return callback_kwargs

    images = reference(
        prompt=kwargs.pop("prompt"),
        callback_on_step_end=capture,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pt",
        **kwargs,
    ).images
    pixels = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    del reference
    gc.collect()
    return {"latents": captured["latents"], "pixels": pixels}


def _our_run(root, monkeypatch, prompt, **options):
    """Run the library pipeline once; return the final packed latents and uint8 pixels."""
    captured = {}
    original = flux_backend.denoise

    def capture(*args, **kwargs):
        captured["latents"] = original(*args, **kwargs)
        return captured["latents"]

    monkeypatch.setattr(flux_backend, "denoise", capture)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(prompt, **options)
    monkeypatch.setattr(flux_backend, "denoise", original)
    gc.collect()
    return {
        "latents": captured["latents"],
        "pixels": torch.stack([i.tensor for i in result.images]),
    }


def test_header(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": CLIPTextModel,
        "text_encoder_2": T5EncoderModel,
        "transformer": FluxTransformer2DModel,
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
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("name", ["landscape", "portrait"])
def test_input_path(name, record):
    root = snapshot_dir(MODEL_ID)
    image = {"landscape": landscape, "portrait": portrait}[name]()
    reference_vae = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    pipe = _reference_input_pipeline(reference_vae)
    expected_size, expected_pixels = _reference_input(pipe, image)
    with torch.inference_mode():
        expected_latents = pipe._encode_vae_image(expected_pixels, generator=None)
    del pipe, reference_vae
    gc.collect()

    actual_size = kontext_input_size(image)
    assert actual_size == expected_size
    actual_pixels = preprocess_kontext_image(image, *actual_size)
    assert actual_pixels.stride() == expected_pixels.stride()
    _check(record, f"flux_kontext_preprocess_{name}", actual_pixels, expected_pixels)

    native = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_latents = encode_kontext_image(native, actual_pixels)
    _check(record, f"flux_kontext_encode_{name}", actual_latents, expected_latents)
    del native
    gc.collect()


def test_transformer(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(0)
    noise = torch.randn(1, 16, 128, 128, generator=generator)
    image = landscape()

    # Real conditioning from the reference text encoders, real image latents from the reference VAE.
    text_pipe = diffusers.FluxKontextPipeline.from_pretrained(
        root, transformer=None, vae=None, local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        context, pooled, txt_ids = text_pipe.encode_prompt(
            prompt=PROMPT, prompt_2=None, device=CPU, max_sequence_length=MAX_LENGTH
        )
    del text_pipe
    gc.collect()
    reference_vae = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    pipe = _reference_input_pipeline(reference_vae)
    _, pixels = _reference_input(pipe, image)
    with torch.inference_mode():
        image_latents = pipe._encode_vae_image(pixels, generator=None)
    del pipe, reference_vae
    gc.collect()

    packed_noise = diffusers.FluxKontextPipeline._pack_latents(noise.clone(), 1, 16, 128, 128)
    image_height, image_width = image_latents.shape[2:]
    packed_image = diffusers.FluxKontextPipeline._pack_latents(
        image_latents, 1, 16, image_height, image_width
    )
    noise_ids = diffusers.FluxKontextPipeline._prepare_latent_image_ids(
        1, 64, 64, CPU, torch.float32
    )
    image_ids = diffusers.FluxKontextPipeline._prepare_latent_image_ids(
        1, image_height // 2, image_width // 2, CPU, torch.float32
    )
    image_ids[..., 0] = 1
    img_ids = torch.cat([noise_ids, image_ids], dim=0)
    hidden = torch.cat([packed_noise, packed_image], dim=1)
    guidance = torch.full((1,), GUIDANCE, dtype=torch.float32)
    timesteps = [torch.tensor([1.0]), torch.tensor([0.5])]

    reference = diffusers.FluxTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = [
            reference(
                hidden_states=hidden,
                encoder_hidden_states=context,
                pooled_projections=pooled,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
            ).sample[:, : packed_noise.shape[1]]
            for timestep in timesteps
        ]
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()

    # The same inputs built with the library's own helpers.
    our_noise = _pack_latents(noise.clone(), 1, 16, 128, 128)
    our_image = _pack_latents(image_latents, 1, 16, image_height, image_width)
    our_image_ids = _prepare_latent_image_ids(
        1, image_height // 2, image_width // 2, CPU, torch.float32
    )
    our_image_ids[..., 0] = 1
    our_ids = torch.cat([_prepare_latent_image_ids(1, 64, 64, CPU, torch.float32), our_image_ids])
    assert torch.equal(our_ids, img_ids)
    assert torch.equal(torch.cat([our_noise, our_image], dim=1), hidden)

    native = FluxTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    expected = torch.load(output, weights_only=True)
    with torch.inference_mode():
        for index, (timestep, reference_output) in enumerate(zip(timesteps, expected, strict=True)):
            actual = native(hidden, context, pooled, timestep, our_ids, txt_ids, guidance)
            actual = actual[:, : our_noise.shape[1]]
            _check(record, f"flux_kontext_transformer_t{index}", actual, reference_output)
    del native
    gc.collect()


def test_end_to_end_with_image(record, tmp_path, monkeypatch):
    root = snapshot_dir(MODEL_ID)
    expected = _reference_run(
        root,
        prompt=PROMPT,
        image=landscape(),
        generator=torch.Generator("cpu").manual_seed(0),
    )
    output = tmp_path / "end-to-end.pt"
    torch.save(expected, output)
    del expected
    gc.collect()

    actual = _our_run(root, monkeypatch, PROMPT, image=landscape(), seed=0)
    expected = torch.load(output, weights_only=True)
    _check(record, "flux_kontext_e2e_latents", actual["latents"], expected["latents"])
    _check_pixels(
        record, "e2e_flux1_kontext_dev", actual["pixels"], expected["pixels"], steps=28, seed=0
    )


def test_end_to_end_without_image(record, tmp_path, monkeypatch):
    root = snapshot_dir(MODEL_ID)
    expected = _reference_run(
        root,
        prompt=PLAIN_PROMPT,
        num_inference_steps=REDUCED_STEPS,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    output = tmp_path / "no-image.pt"
    torch.save(expected, output)
    del expected
    gc.collect()

    actual = _our_run(root, monkeypatch, PLAIN_PROMPT, steps=REDUCED_STEPS, seed=0)
    expected = torch.load(output, weights_only=True)
    _check(record, "flux_kontext_e2e_no_image_latents", actual["latents"], expected["latents"])
    _check_pixels(
        record,
        "e2e_flux1_kontext_dev_no_image",
        actual["pixels"],
        expected["pixels"],
        steps=REDUCED_STEPS,
        seed=0,
    )


def test_seed_end_to_end(record, tmp_path, monkeypatch):
    root = snapshot_dir(MODEL_ID)
    expected = _reference_run(
        root,
        prompt=PROMPT,
        image=portrait(),
        num_inference_steps=REDUCED_STEPS,
        height=768,
        width=1360,
        generator=torch.Generator("cpu").manual_seed(1234),
    )["pixels"]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del expected
    gc.collect()

    actual = _our_run(
        root,
        monkeypatch,
        PROMPT,
        image=portrait(),
        steps=REDUCED_STEPS,
        height=768,
        width=1360,
        seed=1234,
    )["pixels"]
    expected = torch.load(output, weights_only=True)
    _check_pixels(
        record, "seed_e2e_flux1_kontext_dev", actual, expected, steps=REDUCED_STEPS, seed=1234
    )
