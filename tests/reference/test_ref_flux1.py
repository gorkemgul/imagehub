"""FLUX.1 parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.flux import FluxComponents, calculate_shift, denoise
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.flux_transformer import FluxTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_IDS = ("flux1-schnell", "flux1-dev")
PROMPT = "a photograph of an astronaut riding a horse"
MAX_LENGTH = {"flux1-schnell": 256, "flux1-dev": 512}
GUIDANCE = {"flux1-schnell": 0.0, "flux1-dev": 3.5}


def _record_name(key, model_id):
    return key if model_id == "flux1-schnell" else f"{key}_{model_id}"


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


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_header(model_id, record):
    root = snapshot_dir(model_id)
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
    record(f"header_{model_id}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_clip_tokenizer_and_pooled_encoder(model_id, record):
    root = snapshot_dir(model_id)
    reference_tokenizer = transformers.CLIPTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    expected_ids = reference_tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids
    native_tokenizer = CLIPTokenizer(root / "tokenizer")
    actual_ids = torch.tensor([native_tokenizer(PROMPT)])
    tokens_match = torch.equal(actual_ids, expected_ids)
    record(_record_name("flux1_clip_tokenizer", model_id), exact=tokens_match, tolerance="exact")
    assert tokens_match

    reference = transformers.CLIPTextModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(expected_ids).pooler_output
    del reference
    gc.collect()
    native = CLIPTextModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        hidden = native(actual_ids)
        positions = actual_ids.to(torch.int).argmax(dim=-1)
        actual = hidden[torch.arange(hidden.shape[0]), positions]
    _check(record, _record_name("flux1_clip", model_id), actual, expected)
    del native
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_t5_tokenizer_and_encoder(model_id, record):
    root = snapshot_dir(model_id)
    length = MAX_LENGTH[model_id]
    reference_tokenizer = transformers.T5TokenizerFast.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    expected_ids = reference_tokenizer(
        [PROMPT], padding="max_length", max_length=length, truncation=True, return_tensors="pt"
    ).input_ids
    native_tokenizer = T5Tokenizer(root / "tokenizer_2")
    actual_ids, _ = native_tokenizer([PROMPT], length)
    tokens_match = torch.equal(actual_ids, expected_ids)
    record(_record_name("flux1_t5_tokenizer", model_id), exact=tokens_match, tolerance="exact")
    assert tokens_match

    reference = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder_2", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(expected_ids).last_hidden_state
    del reference
    gc.collect()
    native = T5EncoderModel.from_pretrained(
        root / "text_encoder_2", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(actual_ids)[0]
    _check(record, _record_name("flux1_t5", model_id), actual, expected)
    del native
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_transformer(model_id, record, tmp_path):
    root = snapshot_dir(model_id)
    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    hidden = torch.randn(1, 4096, 64, generator=generator)
    context = torch.randn(
        1, MAX_LENGTH[model_id], config["joint_attention_dim"], generator=generator
    )
    pooled = torch.randn(1, config["pooled_projection_dim"], generator=generator)
    img_ids = torch.zeros(4096, 3)
    img_ids[:, 1:] = torch.cartesian_prod(torch.arange(64), torch.arange(64))
    txt_ids = torch.zeros(MAX_LENGTH[model_id], 3)
    timestep = torch.tensor([0.5])
    guidance = torch.tensor([GUIDANCE[model_id]]) if model_id == "flux1-dev" else None

    reference = diffusers.FluxTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(
            hidden_states=hidden,
            encoder_hidden_states=context,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        ).sample
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()

    native = FluxTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(hidden, context, pooled, timestep, img_ids, txt_ids, guidance)
    expected = torch.load(output, weights_only=True)
    _check(record, _record_name("flux1_transformer", model_id), actual, expected)
    del native
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_vae(model_id, record, tmp_path):
    root = snapshot_dir(model_id)
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
    _check(record, _record_name("flux1_vae_decode", model_id), actual_decode, expected["decode"])
    _check(record, _record_name("flux1_vae_encode", model_id), actual_encode, expected["encode"])
    del native
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_scheduler(model_id, record):
    root = snapshot_dir(model_id)
    config = read_json(root / "scheduler" / "scheduler_config.json")
    sigmas = [1.0, 0.75, 0.5, 0.25]
    mu = calculate_shift(4096, config)
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=sigmas, mu=mu)
    expected_sigmas = reference.sigmas.cpu()
    expected_timesteps = reference.timesteps.cpu()
    del reference
    gc.collect()
    native = FlowMatchEulerDiscreteScheduler(config)
    native.set_timesteps(4, sigmas=sigmas, mu=mu)
    exact = torch.equal(native.sigmas, expected_sigmas) and torch.equal(
        native.timesteps, expected_timesteps
    )
    record(_record_name("flux1_scheduler", model_id), mu=mu, exact=exact, tolerance="exact")
    assert exact


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(model_id, record, tmp_path):
    root = snapshot_dir(model_id)
    noise = torch.randn(1, 16, 128, 128, generator=torch.Generator("cpu").manual_seed(42))
    reference = diffusers.FluxPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "prompt": PROMPT,
        "num_inference_steps": 4,
        "guidance_scale": GUIDANCE[model_id],
        "height": 1024,
        "width": 1024,
        "max_sequence_length": MAX_LENGTH[model_id],
        "latents": noise.clone(),
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    output = tmp_path / "end-to-end.pt"
    torch.save({"latents": expected_latents, "pixels": expected_pixels}, output)
    del reference, expected_latents, expected_pixels
    gc.collect()

    components = FluxComponents.load(root, device=CPU, dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt=None,
        true_cfg_scale=1.0,
        steps=4,
        guidance=GUIDANCE[model_id],
        width=1024,
        height=1024,
        latents=noise.clone(),
        max_sequence_length=MAX_LENGTH[model_id],
    )
    expected = torch.load(output, weights_only=True)
    _check(record, _record_name("flux1_e2e_latents", model_id), actual_latents, expected["latents"])
    del components
    gc.collect()
    actual = pipeline("t2i", backend=model_id, model=root)(
        PROMPT,
        steps=4,
        guidance=GUIDANCE[model_id],
        width=1024,
        height=1024,
        max_sequence_length=MAX_LENGTH[model_id],
        latents=noise,
    )
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected["pixels"])
    record(
        f"e2e_{model_id}",
        passed=passed,
        max_pixel_diff=int(
            (pixels.to(torch.int16) - expected["pixels"].to(torch.int16)).abs().max()
        ),
        steps=4,
    )
    assert passed


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(model_id, record, tmp_path):
    root = snapshot_dir(model_id)
    reference = diffusers.FluxPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        prompt=PROMPT,
        num_inference_steps=4,
        guidance_scale=GUIDANCE[model_id],
        height=1024,
        width=1024,
        max_sequence_length=MAX_LENGTH[model_id],
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT,
            seed=42,
            steps=4,
            guidance=GUIDANCE[model_id],
            max_sequence_length=MAX_LENGTH[model_id],
        )
        .images[0]
        .tensor
    )
    expected = torch.load(output, weights_only=True)
    passed = torch.equal(actual, expected)
    record(
        f"seed_e2e_{model_id}",
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        seed=42,
        steps=4,
    )
    assert passed
