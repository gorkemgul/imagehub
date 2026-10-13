"""Parity of published DreamLite checkpoints on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub.backends.dreamlite import DreamLiteComponents, decode_latents, denoise, encode_prompt
from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.dreamlite import DreamLiteUNetModel
from imagehub.schedulers.flow_match_euler import (
    FlowMatchEulerDiscreteScheduler,
    qwen_image_mu,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

MODEL_IDS = ("dreamlite-base", "dreamlite-mobile")
PROMPT = "a photograph of an astronaut riding a horse"


def _checkpoint_shapes(path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: entry["shape"] for name, entry in header.items() if name != "__metadata__"}


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_published_headers(model_id, record):
    root = snapshot_dir(model_id)
    results = {}
    for name, model_class in (("unet", DreamLiteUNetModel), ("vae", AutoencoderTiny)):
        config = json.loads((root / name / "config.json").read_text())
        with torch.device("meta"):
            model = model_class(config)
        published = _checkpoint_shapes(root / name / "diffusion_pytorch_model.safetensors")
        actual = {key: list(value.shape) for key, value in model.state_dict().items()}
        results[name] = {
            "missing": sorted(set(published) - set(actual)),
            "extra": sorted(set(actual) - set(published)),
            "mismatch": sorted(
                key for key in actual.keys() & published.keys() if actual[key] != published[key]
            ),
            "tensors": len(published),
        }
    record(f"header_{model_id}", **results)
    assert all(
        not result[key] for result in results.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_components(model_id, record):
    root = snapshot_dir(model_id)
    ours = DreamLiteComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    reference_tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    reference_text = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder",
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    template = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    text = template.format(f"[Generate]: {PROMPT}")
    tokens = reference_tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        expected_text = reference_text(**tokens, output_hidden_states=True).hidden_states[-1][
            :, 34:
        ]
        actual_text, _ = encode_prompt(ours, [f"[Generate]: {PROMPT}"], torch.device("cpu"), 200)
    text_diff = max_diff(actual_text, expected_text)
    del reference_text, reference_tokenizer
    gc.collect()

    reference_unet = diffusers.DreamLiteUNetModel.from_pretrained(
        root / "unet", local_files_only=True, torch_dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(0)
    sample = torch.randn(1, 4, 8, 12, generator=generator)
    context = torch.randn(1, 5, 2048, generator=generator)
    time_ids = torch.tensor([[64.0, 96.0]])
    with torch.inference_mode():
        expected_unet = reference_unet(
            sample,
            torch.tensor(500.0),
            encoder_hidden_states=context,
            added_cond_kwargs={"time_ids": time_ids},
        ).sample
        actual_unet = ours.unet(sample, torch.tensor(500.0), context, time_ids=time_ids)
    unet_diff = max_diff(actual_unet, expected_unet)
    del reference_unet
    gc.collect()

    reference_vae = diffusers.AutoencoderTiny.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    latents = torch.randn(1, 4, 4, 6, generator=generator)
    with torch.inference_mode():
        expected_vae = reference_vae.decode(latents).sample
        actual_vae = ours.vae.decode(latents)
    vae_diff = max_diff(actual_vae, expected_vae)
    native_scheduler = FlowMatchEulerDiscreteScheduler(ours.scheduler_config)
    reference_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(
        ours.scheduler_config
    )
    sigmas = torch.linspace(1.0, 0.25, 4, dtype=torch.float64).tolist()
    mu = qwen_image_mu(1024, 768, ours.scheduler_config)
    native_scheduler.set_timesteps(4, sigmas=sigmas, mu=mu)
    reference_scheduler.set_timesteps(4, sigmas=sigmas, mu=mu)
    scheduler_diff = max_diff(native_scheduler.timesteps, reference_scheduler.timesteps)
    record(
        f"components_{model_id}",
        text_encoder=text_diff,
        unet=unet_diff,
        vae=vae_diff,
        scheduler=scheduler_diff,
        tolerance=0.0,
    )
    torch.testing.assert_close(actual_text, expected_text, atol=0, rtol=0)
    torch.testing.assert_close(actual_unet, expected_unet, atol=0, rtol=0)
    torch.testing.assert_close(actual_vae, expected_vae, atol=0, rtol=0)
    torch.testing.assert_close(
        native_scheduler.timesteps, reference_scheduler.timesteps, atol=0, rtol=0
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(model_id, record):
    root = snapshot_dir(model_id)
    mobile = model_id.endswith("mobile")
    steps = 4 if mobile else 30
    guidance = None if mobile else 3.5
    noise = torch.randn(1, 4, 128, 96, generator=torch.Generator("cpu").manual_seed(1234))
    pipeline_class = diffusers.DreamLiteMobilePipeline if mobile else diffusers.DreamLitePipeline
    reference = pipeline_class.from_pretrained(
        root,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    ).to("cpu")
    reference.set_progress_bar_config(disable=True)
    kwargs = {
        "num_inference_steps": steps,
        "height": 1024,
        "width": 768,
        "generator": torch.Generator("cpu").manual_seed(1234),
        "output_type": "np",
    }
    if not mobile:
        kwargs["guidance_scale"] = guidance
    with torch.inference_mode():
        expected = reference(PROMPT, **kwargs).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    del reference
    gc.collect()

    ours = DreamLiteComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=None,
        steps=steps,
        guidance=guidance,
        width=768,
        height=1024,
        latents=noise,
        mobile=mobile,
        max_sequence_length=200,
    )
    actual = decode_latents(ours.vae, actual_latents)
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        f"e2e_{model_id}",
        pixels_max_abs=pixel_diff,
        seed=1234,
        steps=steps,
        size=[1024, 768],
        tolerance=0,
    )
    record(f"seed_e2e_{model_id}", seed=1234, pixels_match=pixel_diff == 0)
    assert torch.equal(actual, expected)
