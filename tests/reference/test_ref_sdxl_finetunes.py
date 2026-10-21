"""Community SDXL fine-tune parity against diffusers 0.40.0 on CPU float32.

Set ``IMAGEHUB_SNAPSHOT_<ID>`` for each checkpoint. The end-to-end comparisons are deliberately
short; they are still run through the public pipeline API at two independent seeds.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel, CLIPTextModelWithProjection
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.schedulers.ddpm import DDPMScheduler
from imagehub.schedulers.deis_multistep import DEISMultistepScheduler
from imagehub.schedulers.euler import EulerDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parents[1] / "fixtures"
MODEL_IDS = ("juggernaut-xl-v9", "animagine-xl-4.0", "dreamshaper-xl-1.0")
FIXTURE_NAMES = {
    "juggernaut-xl-v9": "juggernaut_xl_v9_tensor_shapes.json",
    "animagine-xl-4.0": "animagine_xl_4_tensor_shapes.json",
    "dreamshaper-xl-1.0": "dreamshaper_xl_1_tensor_shapes.json",
}
COMPONENTS = {
    "text_encoder": CLIPTextModel,
    "text_encoder_2": CLIPTextModelWithProjection,
    "unet": UNet2DConditionModel,
    "vae": AutoencoderKL,
}
WEIGHT_FILES = (
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "model.fp16.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
)
SCHEDULERS = {
    "DDPMScheduler": DDPMScheduler,
    "DEISMultistepScheduler": DEISMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
}
PROMPT = "a photograph of an astronaut riding a horse"


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    path = next(directory / name for name in WEIGHT_FILES if (directory / name).is_file())
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


@pytest.fixture(scope="module", params=MODEL_IDS)
def real_pipelines(request):
    model_id = request.param
    root = snapshot_dir(model_id)
    ours = StableDiffusionXLComponents.load(root, device=CPU, dtype=torch.float32)
    reference = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False, torch_dtype=torch.float32
    )
    reference.to(CPU)
    reference.set_progress_bar_config(disable=True)
    return model_id, root, ours, reference


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_published_tensor_headers_match_models(model_id, record):
    root = snapshot_dir(model_id)
    published = json.loads((FIXTURES / FIXTURE_NAMES[model_id]).read_text(encoding="utf-8"))
    result = {}
    for component, model_class in COMPONENTS.items():
        with torch.device("meta"):
            model = model_class(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        assert shapes == {name: tuple(shape) for name, shape in published[component].items()}
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        actual = {name: tuple(tensor.shape) for name, tensor in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(actual),
        }
    record(f"header_{model_id}", **result)
    for entry in result.values():
        assert not entry["missing"]
        assert not entry["extra"]
        assert not entry["mismatch"]


def test_tokenizers_and_text_encoders(real_pipelines, record):
    model_id, _, ours, reference = real_pipelines
    prompts = [PROMPT, "1girl, masterpiece, high score", "Bir kedi İstanbul'da", ""]
    mismatches = []
    for index, (ours_tokenizer, reference_tokenizer) in enumerate(
        ((ours.tokenizer, reference.tokenizer), (ours.tokenizer_2, reference.tokenizer_2)), 1
    ):
        assert ours_tokenizer is not None
        for prompt in prompts:
            expected = reference_tokenizer(
                prompt, padding="max_length", max_length=77, truncation=True
            )["input_ids"]
            if ours_tokenizer(prompt) != expected:
                mismatches.append({"tokenizer": index, "prompt": prompt})
    record(
        f"sdxl_tokenizers_{model_id}",
        prompts=len(prompts),
        tokenizers=2,
        mismatches=mismatches,
        tolerance="exact",
    )
    assert not mismatches

    assert ours.tokenizer is not None and ours.text_encoder is not None
    ids_1 = torch.tensor([ours.tokenizer(prompt) for prompt in prompts])
    ids_2 = torch.tensor([ours.tokenizer_2(prompt) for prompt in prompts])
    with torch.inference_mode():
        _, ours_hidden_1 = ours.text_encoder(ids_1, output_hidden_states=True)
        expected_1 = reference.text_encoder(ids_1, output_hidden_states=True)
        ours_pooled_2, _, ours_hidden_2 = ours.text_encoder_2(ids_2, output_hidden_states=True)
        expected_2 = reference.text_encoder_2(ids_2, output_hidden_states=True)
    first = max_diff(ours_hidden_1[-2], expected_1.hidden_states[-2])
    second_hidden = max_diff(ours_hidden_2[-2], expected_2.hidden_states[-2])
    second_pooled = max_diff(ours_pooled_2, expected_2.text_embeds)
    record(f"sdxl_text_encoder_{model_id}", **first, tolerance=0.0)
    record(
        f"sdxl_text_encoder_2_{model_id}",
        **max((second_hidden, second_pooled), key=lambda value: value["max_abs"]),
        hidden_states=second_hidden,
        text_embeds=second_pooled,
        tolerance=0.0,
    )
    assert torch.equal(ours_hidden_1[-2], expected_1.hidden_states[-2])
    assert torch.equal(ours_hidden_2[-2], expected_2.hidden_states[-2])
    assert torch.equal(ours_pooled_2, expected_2.text_embeds)


def test_unet(real_pipelines, record):
    model_id, _, ours, reference = real_pipelines
    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(2, 4, 8, 8, generator=generator)
    context = torch.randn(2, 77, 2048, generator=generator)
    text_embeds = torch.randn(2, 1280, generator=generator)
    time_ids = torch.randn(2, 6, generator=generator)
    with torch.inference_mode():
        actual = ours.unet(
            sample, torch.tensor(500), context, text_embeds=text_embeds, time_ids=time_ids
        )
        expected = reference.unet(
            sample,
            torch.tensor(500),
            context,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
        ).sample
    record(f"sdxl_unet_{model_id}", **max_diff(actual, expected), tolerance=0.0, timestep=500)
    assert torch.equal(actual, expected)


def test_vae(real_pipelines, record):
    model_id, _, ours, reference = real_pipelines
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 4, 8, 8, generator=generator)
    image = torch.rand(1, 3, 64, 64, generator=generator) * 2 - 1
    with torch.inference_mode():
        pairs = {
            "decode": (ours.vae.decode(latents), reference.vae.decode(latents).sample),
            "encode": (ours.vae.encode(image), reference.vae.encode(image).latent_dist.mean),
        }
    for name, (actual, expected) in pairs.items():
        record(f"sdxl_vae_{name}_{model_id}", **max_diff(actual, expected), tolerance=0.0)
        assert torch.equal(actual, expected)


def test_scheduler(real_pipelines, record):
    model_id, _, ours, reference = real_pipelines
    scheduler_class = SCHEDULERS[ours.scheduler_config["_class_name"]]
    native = scheduler_class(ours.scheduler_config)
    expected_scheduler = type(reference.scheduler).from_config(reference.scheduler.config)
    native.set_timesteps(4)
    expected_scheduler.set_timesteps(4)
    assert torch.equal(native.timesteps, expected_scheduler.timesteps)
    generator = torch.Generator().manual_seed(2)
    actual = torch.randn(1, 4, 4, 6, generator=generator)
    expected = actual.clone()
    native_generator = torch.Generator().manual_seed(3)
    expected_generator = torch.Generator().manual_seed(3)
    worst = 0.0
    for timestep in native.timesteps:
        model_output = torch.randn(actual.shape, generator=generator)
        expected_kwargs = {}
        native_kwargs = {}
        if isinstance(native, DDPMScheduler):
            expected_kwargs["generator"] = expected_generator
            native_kwargs["generator"] = native_generator
        expected = expected_scheduler.step(
            model_output, timestep, expected, **expected_kwargs
        ).prev_sample
        actual = native.step(model_output, timestep, actual, **native_kwargs)
        worst = max(worst, max_diff(actual, expected)["max_abs"])
        assert torch.equal(actual, expected)
    record(f"sdxl_scheduler_{model_id}", max_abs=worst, steps=4, tolerance=0.0)


def _end_to_end(real_pipelines, record, *, seed: int, record_name: str):
    model_id, root, _, reference = real_pipelines
    with torch.inference_mode():
        expected = reference(
            PROMPT,
            num_inference_steps=4,
            guidance_scale=5.0,
            height=64,
            width=64,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    expected_pixels = torch.from_numpy((expected * 255).round().astype(np.uint8))[0]
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT, steps=4, guidance=5.0, width=64, height=64, seed=seed
        )
        .images[0]
        .tensor
    )
    pixels_max_abs = int((actual.int() - expected_pixels.int()).abs().max())
    passed = pixels_max_abs == 0
    record(
        record_name,
        pixels_max_abs=pixels_max_abs,
        passed=passed,
        seed=seed,
        steps=4,
        guidance=5.0,
        size=[64, 64],
        tolerance=0,
    )
    assert passed


def test_end_to_end_seed_zero(real_pipelines, record):
    model_id, _, _, _ = real_pipelines
    _end_to_end(real_pipelines, record, seed=0, record_name=f"e2e_{model_id}")


def test_seed_end_to_end(real_pipelines, record):
    model_id, _, _, _ = real_pipelines
    _end_to_end(real_pipelines, record, seed=42, record_name=f"seed_e2e_{model_id}")
