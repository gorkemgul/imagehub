"""ERNIE-Image parity on CPU float32 against the released diffusers/transformers code.

The transformer has 8B parameters, so the end-to-end runs use 4 steps instead of the default 50.
Every reference result is computed, saved and freed before the native model is loaded.
"""

import gc
import json
import random
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.ernie_image import (
    ErnieImageComponents,
    decode_latents,
    denoise,
    encode_prompt,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_flux2 import AutoencoderKLFlux2
from imagehub.models.ernie_image_transformer import ErnieImageTransformer2DModel
from imagehub.models.mistral3_text import Mistral3TextModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_mistral import MistralTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL = "ernie-image"
PROMPT = "a photograph of an astronaut riding a horse on mars, 'ERNIE' written on the helmet"
STEPS = 4


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


def _reference_pipeline(root):
    reference = diffusers.ErnieImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32, pe=None, pe_tokenizer=None
    )
    reference.transformer.to(torch.float32)
    reference.set_progress_bar_config(disable=True)
    return reference


def test_header(record):
    root = snapshot_dir(MODEL)
    result = {}
    for component, cls in {
        "transformer": ErnieImageTransformer2DModel,
        "vae": AutoencoderKLFlux2,
        "text_encoder": Mistral3TextModel,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        published = _checkpoint_shapes(root / component)
        if component == "text_encoder":
            published = {
                name: shape
                for name, shape in published.items()
                if not name.startswith(("vision_tower.", "multi_modal_projector."))
            }
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        result[component] = {
            "missing": sorted(expected.keys() - published.keys()),
            "extra": sorted(published.keys() - expected.keys()),
            "mismatch": sorted(
                name
                for name in expected.keys() & published.keys()
                if expected[name] != published[name]
            ),
            "tensors": len(published),
        }
    record("header_ernie_image", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer_matches_transformers_including_fuzz(record):
    root = snapshot_dir(MODEL)
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    native = MistralTokenizer(root / "tokenizer")
    prompts = [
        PROMPT,
        "",
        "  leading and trailing  ",
        "一只在火星上骑马的宇航员 🚀 café",
        "[INST]hello[/INST]<s>x</s>",
        "line one\n\nline two\r\n\ttabbed 12345 6.78",
        "word " * 3000,
    ]
    rng = random.Random(0)
    alphabet = list("abcXYZ \n\r\t.,!?-_/'\"()[]0123456789éß日本한ا🙂  ") + [
        "[INST]",
        "</s>",
        "<pad>",
    ]
    prompts += [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(500)
    ]
    mismatches = [
        text
        for text in prompts
        if reference(text, add_special_tokens=True, truncation=True, padding=False)["input_ids"]
        != native(text)
    ]
    record(
        "ernie_image_tokenizer", prompts=len(prompts), mismatches=len(mismatches), tolerance="exact"
    )
    assert not mismatches


def test_text_encoder_second_to_last_hidden_state(record, tmp_path):
    root = snapshot_dir(MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    reference = transformers.Mistral3Model.from_pretrained(
        root / "text_encoder", local_files_only=True, dtype=torch.float32
    ).eval()
    expected = {}
    with torch.inference_mode():
        for key, text in (("prompt", PROMPT), ("empty", "")):
            ids = tokenizer(text, add_special_tokens=True, truncation=True, padding=False)[
                "input_ids"
            ]
            outputs = reference(input_ids=torch.tensor([ids]), output_hidden_states=True)
            expected[key] = outputs.hidden_states[-2][0]
    torch.save(expected, tmp_path / "text.pt")
    del reference, outputs, expected
    gc.collect()
    components = ErnieImageComponents.__new__(ErnieImageComponents)
    components.tokenizer = MistralTokenizer(root / "tokenizer")
    components.text_encoder = Mistral3TextModel.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    expected = torch.load(tmp_path / "text.pt", weights_only=True)
    _check(
        record,
        "ernie_image_text_prompt",
        encode_prompt(components, PROMPT, CPU),
        expected["prompt"],
    )
    _check(record, "ernie_image_text_empty", encode_prompt(components, "", CPU), expected["empty"])
    del components
    gc.collect()


def test_transformer_single_step(record, tmp_path):
    root = snapshot_dir(MODEL)
    config = read_json(root / "transformer/config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(2, config["in_channels"], 32, 32, generator=generator)
    text = torch.randn(2, 96, config["text_in_dim"], generator=generator)
    lengths = torch.tensor([96, 60])
    timestep = torch.tensor([750.0, 250.0])
    reference = diffusers.ErnieImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    ).to(torch.float32)
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            timestep=timestep,
            text_bth=text,
            text_lens=lengths,
            return_dict=False,
        )[0]
    torch.save(expected, tmp_path / "transformer.pt")
    del reference, expected
    gc.collect()
    native = ErnieImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(hidden, timestep, text, lengths)
    _check(
        record,
        "ernie_image_transformer",
        actual,
        torch.load(tmp_path / "transformer.pt", weights_only=True),
    )
    del native
    gc.collect()


def test_vae_decode_with_latent_denormalisation(record, tmp_path):
    root = snapshot_dir(MODEL)
    latents = torch.randn(1, 128, 64, 64, generator=torch.Generator("cpu").manual_seed(1))
    reference = _reference_pipeline_vae(root)
    with torch.inference_mode():
        bn_mean = reference.bn.running_mean.view(1, -1, 1, 1)
        bn_std = torch.sqrt(reference.bn.running_var.view(1, -1, 1, 1) + 1e-5)
        unpatched = diffusers.ErnieImagePipeline._unpatchify_latents(latents * bn_std + bn_mean)
        image = reference.decode(unpatched, return_dict=False)[0]
        expected = torch.from_numpy(
            ((image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).float().numpy() * 255)
            .round()
            .astype("uint8")
        )
    torch.save(expected, tmp_path / "vae.pt")
    del reference, image, expected
    gc.collect()
    native = AutoencoderKLFlux2.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    actual = decode_latents(native, latents)
    expected = torch.load(tmp_path / "vae.pt", weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        "ernie_image_vae_decode",
        passed=passed,
        max_pixel_diff=int((actual.int() - expected.int()).abs().max()),
    )
    assert passed
    del native
    gc.collect()


def _reference_pipeline_vae(root):
    return diffusers.AutoencoderKLFlux2.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    ).eval()


def test_scheduler_timesteps(record):
    root = snapshot_dir(MODEL)
    config = read_json(root / "scheduler/scheduler_config.json")
    exact = True
    for steps in (1, 4, 50):
        sigmas = torch.linspace(1.0, 0.0, steps + 1)[:-1]
        reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
        reference.set_timesteps(sigmas=sigmas, device="cpu")
        native = FlowMatchEulerDiscreteScheduler(config)
        native.set_timesteps(steps, sigmas=sigmas.tolist())
        exact &= torch.equal(native.sigmas, reference.sigmas) and torch.equal(
            native.timesteps, reference.timesteps
        )
    record("ernie_image_scheduler", exact=bool(exact), tolerance="exact")
    assert exact


def test_end_to_end_defaults_with_fixed_noise(record, tmp_path):
    root = snapshot_dir(MODEL)
    noise = torch.randn(1, 128, 64, 64, generator=torch.Generator("cpu").manual_seed(0))
    reference = _reference_pipeline(root)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": STEPS,
        "height": 1024,
        "width": 1024,
        "latents": noise.clone(),
        "use_pe": False,
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, tmp_path / "end-to-end.pt")
    del reference, expected_latents, expected_pixels
    gc.collect()
    components = ErnieImageComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=STEPS,
        guidance=4.0,
        latents=noise.clone(),
    )
    expected = torch.load(tmp_path / "end-to-end.pt", weights_only=True)
    _check(record, "ernie_image_e2e_latents", actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend="ernie-image", model=root)(PROMPT, latents=noise, steps=STEPS)
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    record(
        "e2e_ernie_image",
        passed=passed,
        max_pixel_diff=int((pixels.int() - expected["pixels"].int()).abs().max()),
        steps=STEPS,
    )
    assert passed


def test_seed_end_to_end_0(record, tmp_path):
    root = snapshot_dir(MODEL)
    reference = _reference_pipeline(root)
    expected = reference(
        PROMPT,
        num_inference_steps=STEPS,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type="np",
        use_pe=False,
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    torch.save(expected, tmp_path / "seed.pt")
    del reference, expected
    gc.collect()
    actual = (
        pipeline("t2i", backend="ernie-image", model=root)(PROMPT, seed=0, steps=STEPS)
        .images[0]
        .tensor
    )
    expected = torch.load(tmp_path / "seed.pt", weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        "seed_e2e_ernie_image",
        passed=passed,
        max_pixel_diff=int((actual.int() - expected.int()).abs().max()),
        seed=0,
        steps=STEPS,
    )
    assert passed
