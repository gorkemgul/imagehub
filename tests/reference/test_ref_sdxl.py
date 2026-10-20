"""Parity of native SDXL components and pipeline against diffusers on CPU fp32.

Requires IMAGEHUB_REFERENCE_TESTS=1 and an IMAGEHUB_SNAPSHOT_<ID> variable pointing at a
snapshot made by ``python -m imagehub.pretrained <model-id> <dir>``.
"""

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_diffusion import decode_latents, initial_latents
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLComponents, denoise
from imagehub.catalog import MODEL_CATALOG
from imagehub.schedulers.edm_dpmsolver import EDMDPMSolverMultistepScheduler
from imagehub.schedulers.euler import EulerDiscreteScheduler
from imagehub.schedulers.euler_ancestral import EulerAncestralDiscreteScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

# Maximum absolute difference allowed against diffusers 0.40.0 / transformers 5.18.0, CPU fp32.
# Rule (spec §6.2): 0 if the measured difference is 0, otherwise the next power of ten above it.
# Random-weight runs at full SDXL size measured exactly 0 for every entry on 2026-10-04.
TOLERANCES = {
    "text_encoder": 0.0,
    "text_encoder_2": 0.0,
    "unet": 0.0,
    "vae_decode": 0.0,
    "vae_encode": 0.0,
    "e2e_latents": 0.0,
    "e2e_pixels": 0,
}
PROMPT = "a photograph of an astronaut riding a horse"
PROMPTS = [
    "a photo of an astronaut riding a horse on mars",
    "Bir kedi, yağlı boya; İSTANBUL'da ĞÜŞİÖÇ ığüşöç",
    "emoji 🐱🚀 and   multiple\tspaces\nnewline",
    "don't won't they're I've I'm we'll he'd 's '' '''s",
    "numbers 12345 3.14 1e-5 ½ ² Ⅷ ٣",
    "punct!!! ??? ... --- (a) [b] {c} <d> @#$%^&*",
    "x!<|endoftext|>y <|startoftext|> mid",
    "x <|ENDOFTEXT|> y",
    "<|StartOfText|>hi",
    "a b\u3000c d e",
    "Ünïcödé NFC é ﬁ ligature Ωmega ΣΟΦΟΣ",
    "日本語のテキストと中文和한국어",
    "word " * 100,
    "",
    "!",
    "a!b",
    "!!",
    "Hi CAT!",
]
CPU = torch.device("cpu")
SDXL_MODEL_IDS = tuple(
    spec.id
    for spec in MODEL_CATALOG.values()
    if spec.family == "stable-diffusion-xl"
    and spec.base is None
    and spec.id not in ("sdxl-refiner-1.0", "segmind-vega", "koala-700m", "koala-1b")
)
SCHEDULERS = {
    "EDMDPMSolverMultistepScheduler": EDMDPMSolverMultistepScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
    "EulerAncestralDiscreteScheduler": EulerAncestralDiscreteScheduler,
}


def _component_record(prefix, model_id):
    return prefix if model_id == "sdxl-base-1.0" else f"{prefix}_{model_id}"


@pytest.fixture(scope="module", params=SDXL_MODEL_IDS)
def sdxl(request):
    model_id = request.param
    root = snapshot_dir(model_id)
    ours = StableDiffusionXLComponents.load(root, device=CPU, dtype=torch.float32)
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False, torch_dtype=torch.float32
    )
    ref.to(CPU)
    ref.set_progress_bar_config(disable=True)
    return model_id, root, ours, ref


def _check(record, name, actual, expected, tolerance_key):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=TOLERANCES[tolerance_key])
    torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


def test_tokenizers(sdxl, record):
    model_id, _, ours, ref = sdxl
    mismatches = []
    for index, (ours_tokenizer, ref_tokenizer) in enumerate(
        ((ours.tokenizer, ref.tokenizer), (ours.tokenizer_2, ref.tokenizer_2)), start=1
    ):
        for text in PROMPTS:
            expected = ref_tokenizer(text, padding="max_length", max_length=77, truncation=True)[
                "input_ids"
            ]
            if ours_tokenizer(text) != expected:
                mismatches.append({"tokenizer": index, "prompt": text})
    record(
        _component_record("sdxl_tokenizers", model_id),
        prompts=len(PROMPTS),
        tokenizers=2,
        mismatches=mismatches,
        tolerance="exact",
    )
    assert not mismatches, mismatches[:5]


def test_text_encoders(sdxl, record):
    model_id, _, ours, ref = sdxl
    ids_1 = torch.tensor([ours.tokenizer(prompt) for prompt in PROMPTS])
    ids_2 = torch.tensor([ours.tokenizer_2(prompt) for prompt in PROMPTS])
    with torch.inference_mode():
        _, ours_hidden_1 = ours.text_encoder(ids_1, output_hidden_states=True)
        expected_1 = ref.text_encoder(ids_1, output_hidden_states=True)
        ours_pooled_2, _, ours_hidden_2 = ours.text_encoder_2(ids_2, output_hidden_states=True)
        expected_2 = ref.text_encoder_2(ids_2, output_hidden_states=True)
    _check(
        record,
        _component_record("sdxl_text_encoder", model_id),
        ours_hidden_1[-2],
        expected_1.hidden_states[-2],
        "text_encoder",
    )
    hidden_diff = max_diff(ours_hidden_2[-2], expected_2.hidden_states[-2])
    pooled_diff = max_diff(ours_pooled_2, expected_2.text_embeds)
    worst = max((hidden_diff, pooled_diff), key=lambda value: value["max_abs"])
    record(
        _component_record("sdxl_text_encoder_2", model_id),
        **worst,
        hidden_states=hidden_diff,
        text_embeds=pooled_diff,
        tolerance=TOLERANCES["text_encoder_2"],
    )
    torch.testing.assert_close(
        ours_hidden_2[-2],
        expected_2.hidden_states[-2],
        atol=TOLERANCES["text_encoder_2"],
        rtol=0,
    )
    torch.testing.assert_close(
        ours_pooled_2,
        expected_2.text_embeds,
        atol=TOLERANCES["text_encoder_2"],
        rtol=0,
    )


def test_unet_single_step(sdxl, record):
    model_id, _, ours, ref = sdxl
    generator = torch.Generator().manual_seed(0)
    latent_size = int(ours.unet.config["sample_size"])
    sample = torch.randn(2, 4, latent_size, latent_size, generator=generator)
    context = torch.randn(2, 77, 2048, generator=generator)
    text_embeds = torch.randn(2, 1280, generator=generator)
    time_ids = torch.randn(2, 6, generator=generator)
    pairs = []
    with torch.inference_mode():
        for value in (999, 500, 1):
            timestep = torch.tensor(value)
            actual = ours.unet(
                sample,
                timestep,
                context,
                text_embeds=text_embeds,
                time_ids=time_ids,
            )
            expected = ref.unet(
                sample,
                timestep,
                context,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample
            pairs.append((actual, expected, max_diff(actual, expected)))
    worst = max((measured for _, _, measured in pairs), key=lambda value: value["max_abs"])
    record(
        _component_record("sdxl_unet", model_id),
        **worst,
        tolerance=TOLERANCES["unet"],
        timesteps=[999, 500, 1],
    )
    for actual, expected, _ in pairs:
        torch.testing.assert_close(actual, expected, atol=TOLERANCES["unet"], rtol=0)


def test_vae(sdxl, record):
    model_id, _, ours, ref = sdxl
    generator = torch.Generator().manual_seed(1)
    latent_size = int(ours.unet.config["sample_size"])
    image_size = latent_size * 8
    latents = torch.randn(1, 4, latent_size, latent_size, generator=generator)
    image = torch.rand(1, 3, image_size, image_size, generator=generator) * 2 - 1
    with torch.inference_mode():
        decode = ours.vae.decode(latents), ref.vae.decode(latents).sample
        encode = ours.vae.encode(image), ref.vae.encode(image).latent_dist.mean
    for name, (actual, expected) in (
        ("sdxl_vae_decode", decode),
        ("sdxl_vae_encode", encode),
    ):
        tolerance_key = name.removeprefix("sdxl_")
        record(
            _component_record(name, model_id),
            **max_diff(actual, expected),
            tolerance=TOLERANCES[tolerance_key],
        )
    for name, (actual, expected) in (
        ("sdxl_vae_decode", decode),
        ("sdxl_vae_encode", encode),
    ):
        tolerance_key = name.removeprefix("sdxl_")
        torch.testing.assert_close(actual, expected, atol=TOLERANCES[tolerance_key], rtol=0)


def test_scheduler(sdxl, record):
    model_id, _, ours, ref = sdxl
    scheduler_cls = SCHEDULERS[ours.scheduler_config["_class_name"]]
    measurements = {}
    schedules = []
    for steps in (50, 30):
        actual = scheduler_cls(ours.scheduler_config)
        expected = type(ref.scheduler).from_config(ref.scheduler.config)
        actual.set_timesteps(steps)
        expected.set_timesteps(steps)
        measurements[str(steps)] = {
            "timesteps": max_diff(actual.timesteps, expected.timesteps),
            "sigmas": max_diff(actual.sigmas, expected.sigmas),
        }
        schedules.append((actual, expected))
    record(
        _component_record("sdxl_scheduler", model_id),
        steps=[50, 30],
        measurements=measurements,
        tolerance=0.0,
    )
    for actual, expected in schedules:
        torch.testing.assert_close(actual.timesteps, expected.timesteps, atol=0, rtol=0)
        torch.testing.assert_close(actual.sigmas, expected.sigmas, atol=0, rtol=0)


def _run_end_to_end(sdxl, record, *, steps, guidance, negative_prompt, record_name, runtime_note):
    _, _, ours, ref = sdxl
    latent_size = int(ours.unet.config["sample_size"])
    image_size = latent_size * 8
    latents = initial_latents(1234, (1, 4, latent_size, latent_size))
    with torch.inference_mode():
        ref_latents = ref(
            PROMPT,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=image_size,
            width=image_size,
            latents=latents.clone(),
            generator=torch.Generator("cpu").manual_seed(5678),
            output_type="latent",
        ).images
        if ref.vae.config.latents_mean is not None and ref.vae.config.latents_std is not None:
            latents_mean = torch.tensor(ref.vae.config.latents_mean).view(1, 4, 1, 1)
            latents_std = torch.tensor(ref.vae.config.latents_std).view(1, 4, 1, 1)
            decode_latents_input = (
                ref_latents * latents_std / ref.vae.config.scaling_factor + latents_mean
            )
        else:
            decode_latents_input = ref_latents / ref.vae.config.scaling_factor
        decoded = ref.vae.decode(decode_latents_input).sample
        ref_np = ref.image_processor.postprocess(decoded, output_type="np")
    ref_pixels = torch.from_numpy((ref_np * 255).round().astype("uint8"))
    our_latents = denoise(
        ours,
        prompt=PROMPT,
        negative_prompt=negative_prompt,
        steps=steps,
        guidance=guidance,
        latents=latents.clone(),
        original_size=(image_size, image_size),
        crops_coords_top_left=(0, 0),
        target_size=(image_size, image_size),
        generator=torch.Generator("cpu").manual_seed(5678),
    )
    our_pixels = decode_latents(ours, our_latents)
    latent_diff = max_diff(our_latents, ref_latents)
    pixel_diff = int((our_pixels.int() - ref_pixels.int()).abs().max())
    passed = (
        latent_diff["max_abs"] <= TOLERANCES["e2e_latents"]
        and pixel_diff <= TOLERANCES["e2e_pixels"]
    )
    record(
        record_name,
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=steps,
        guidance=guidance,
        size=[image_size, image_size],
        noise_seed=1234,
        tolerance={"latents": TOLERANCES["e2e_latents"], "pixels": TOLERANCES["e2e_pixels"]},
        runtime_note=runtime_note,
    )
    assert passed, (latent_diff, pixel_diff)


def test_end_to_end(sdxl, record):
    model_id, _, _, _ = sdxl
    defaults = MODEL_CATALOG[model_id].config.get("defaults", {})
    steps = int(defaults.get("steps", 30))
    guidance = float(defaults.get("guidance", 5.0))
    _run_end_to_end(
        sdxl,
        record,
        steps=steps,
        guidance=guidance,
        negative_prompt=None,
        record_name=f"e2e_{model_id}",
        runtime_note=(
            "30 steps are used instead of 50 to keep CPU runtime reasonable."
            if "steps" not in defaults
            else f"{steps} step(s): the catalog default for this model."
        ),
    )


def test_end_to_end_multistep(sdxl, record):
    model_id, _, _, _ = sdxl
    defaults = MODEL_CATALOG[model_id].config.get("defaults", {})
    if "steps" not in defaults:
        pytest.skip("model does not define a few-step default")
    _run_end_to_end(
        sdxl,
        record,
        steps=4,
        guidance=2.0,
        negative_prompt="blurry",
        record_name=f"e2e_multistep_{model_id}",
        runtime_note="4 steps exercise ancestral sampling and classifier-free guidance.",
    )


def test_seed_end_to_end(sdxl, record):
    model_id, root, ours, ref = sdxl
    defaults = MODEL_CATALOG[model_id].config.get("defaults", {})
    steps = int(defaults.get("steps", 30))
    guidance = float(defaults.get("guidance", 5.0))
    image_size = int(ours.unet.config["sample_size"]) * 8
    with torch.inference_mode():
        expected = ref(
            PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=image_size,
            width=image_size,
            generator=torch.Generator("cpu").manual_seed(42),
            output_type="np",
        ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT,
            seed=42,
            steps=steps,
            guidance=guidance,
            width=image_size,
            height=image_size,
        )
        .images[0]
        .tensor
    )
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        pixels_max_abs=pixel_diff,
        seed=42,
        steps=steps,
        guidance=guidance,
        size=[image_size, image_size],
        passed=pixel_diff <= TOLERANCES["e2e_pixels"],
        runtime_note="30 steps are used instead of 50 to keep CPU runtime reasonable.",
    )
    assert pixel_diff <= TOLERANCES["e2e_pixels"]
