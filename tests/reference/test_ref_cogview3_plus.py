"""CogView3-Plus-3B parity on CPU float32 against a local snapshot.

The published transformer and VAE are bfloat16 and are upcast to float32 by both implementations.
The end-to-end and seed tests use 8 steps instead of the default 50 (the model is 7.6B parameters
with the text encoder); the reference result is saved and freed before the native model loads.
"""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.cogview3_plus import CogView3PlusComponents, denoise
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.cogview3_plus_transformer import CogView3PlusTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.ddim_cogvideox import CogVideoXDDIMScheduler
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "cogview3-plus-3b"
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 8
SEED = 0


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


def _noise():
    generator = torch.Generator("cpu").manual_seed(SEED)
    return torch.randn(1, 16, 128, 128, generator=generator)


def test_header(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {
        "text_encoder": T5EncoderModel,
        "transformer": CogView3PlusTransformer2DModel,
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
    record("header_cogview3-plus-3b", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokens_and_text_encoder(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_ids = reference_tokenizer(
        [PROMPT],
        padding="max_length",
        max_length=224,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids
    actual_ids, _ = T5Tokenizer(root / "tokenizer")([PROMPT], 224)
    tokens_match = torch.equal(actual_ids, expected_ids)
    record("cogview3_plus_tokens", exact=tokens_match, tolerance="exact")
    assert tokens_match

    reference = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(expected_ids)[0]
    output = tmp_path / "text-encoder.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = T5EncoderModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = native(actual_ids)[0]
    expected = torch.load(output, weights_only=True)
    _check(record, "cogview3_plus_text_encoder", actual, expected)
    del native
    gc.collect()


def test_transformer(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(2, 16, 128, 128, generator=generator)
    context = torch.randn(2, 224, config["text_embed_dim"], generator=generator)
    original_size = torch.tensor([[1024.0, 1024.0]]).repeat(2, 1)
    target_size = torch.tensor([[1024.0, 1024.0]]).repeat(2, 1)
    crop_coords = torch.zeros(2, 2)
    reference = diffusers.CogView3PlusTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    expected = []
    with torch.inference_mode():
        for timestep in (torch.tensor([999, 999]), torch.tensor([1, 501])):
            expected.append(
                reference(
                    hidden_states=hidden,
                    encoder_hidden_states=context,
                    timestep=timestep,
                    original_size=original_size,
                    target_size=target_size,
                    crop_coords=crop_coords,
                    return_dict=False,
                )[0]
            )
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    native = CogView3PlusTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    expected = torch.load(output, weights_only=True)
    with torch.inference_mode():
        for index, timestep in enumerate((torch.tensor([999, 999]), torch.tensor([1, 501]))):
            actual = native(hidden, context, timestep, original_size, target_size, crop_coords)
            _check(record, f"cogview3_plus_transformer_{index}", actual, expected[index])
    del native
    gc.collect()


def test_vae(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
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
    _check(record, "cogview3_plus_vae_decode", actual_decode, expected["decode"])
    _check(record, "cogview3_plus_vae_encode", actual_encode, expected["encode"])
    del native
    gc.collect()


@pytest.mark.parametrize("steps", [STEPS, 50])
def test_scheduler(record, steps):
    root = snapshot_dir(MODEL_ID)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    reference = diffusers.CogVideoXDDIMScheduler.from_config(config)
    reference.set_timesteps(steps)
    native = CogVideoXDDIMScheduler(config)
    native.set_timesteps(steps)
    exact = (
        torch.equal(native.timesteps, reference.timesteps)
        and torch.equal(native.alphas_cumprod, reference.alphas_cumprod)
        and torch.equal(native.final_alpha_cumprod, reference.final_alpha_cumprod)
    )
    record(f"cogview3_plus_scheduler_{steps}", exact=exact, tolerance="exact")
    assert exact


def test_end_to_end(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    noise = _noise()
    reference = diffusers.CogView3PlusPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": STEPS,
        "guidance_scale": 5.0,
        "height": 1024,
        "width": 1024,
        "max_sequence_length": 224,
        "latents": noise.clone(),
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    output = tmp_path / "end-to-end.pt"
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, output)
    del reference, expected_latents, expected_pixels
    gc.collect()
    components = CogView3PlusComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        steps=STEPS,
        guidance=5.0,
        width=1024,
        height=1024,
        latents=noise.clone(),
        max_sequence_length=224,
    )
    expected = torch.load(output, weights_only=True)
    _check(record, "cogview3_plus_e2e_latents", actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, steps=STEPS, guidance=5.0, latents=noise
    )
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    record(
        "e2e_cogview3-plus-3b",
        passed=passed,
        max_pixel_diff=int(
            (pixels.to(torch.int16) - expected["pixels"].to(torch.int16)).abs().max()
        ),
        steps=STEPS,
    )
    assert passed


def test_seed_end_to_end(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    reference = diffusers.CogView3PlusPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        prompt=PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=5.0,
        height=1024,
        width=1024,
        max_sequence_length=224,
        generator=torch.Generator("cpu").manual_seed(SEED),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    actual = (
        pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=SEED, steps=STEPS, guidance=5.0)
        .images[0]
        .tensor
    )
    expected = torch.load(output, weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        "seed_e2e_cogview3-plus-3b",
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        seed=SEED,
        steps=STEPS,
    )
    assert passed
