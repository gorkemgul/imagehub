"""Lumina 2 LoRA-step and adapter parity against diffusers, CPU float32."""

import math

import pytest
import torch
from helpers import write_tiny_lumina2
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.backends.lumina2 import Lumina2Backend, decode_latents, denoise
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments
from imagehub.training.lumina2 import (
    LOGIT_MEAN,
    LOGIT_STD,
    MAX_SEQUENCE_LENGTH,
    MODE_SCALE,
    SYSTEM_PROMPT,
    WEIGHTING_SCHEME,
    Lumina2Family,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
training_utils = pytest.importorskip("diffusers.training_utils")
PIL = pytest.importorskip("PIL.Image")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")

CPU = torch.device("cpu")
TARGETS = ("to_k", "to_q", "to_v", "to_out.0")
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")


def image(seed=12, *, height=32, width=32):
    return PIL.fromarray(
        torch.randint(
            0,
            256,
            (height, width, 3),
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.uint8,
        ).numpy()
    )


def args(root, **kwargs):
    return TrainingArguments(
        output_dir=root, max_steps=1, resolution=32, max_sequence_length=8, **kwargs
    )


def test_lumina2_defaults_match_the_script(tmp_path):
    root = write_tiny_lumina2(tmp_path / "lumina2")
    training_args = TrainingArguments(output_dir=tmp_path / "run", max_steps=1, resolution=32)
    trainer = Trainer(
        "lumina-image-2.0", root, training_args, [{"image": image(), "text": "a cat"}]
    )
    trainer._setup()
    objective = trainer.objective
    assert (
        objective.weighting_scheme,
        objective.logit_mean,
        objective.logit_std,
        objective.mode_scale,
        objective.precondition_outputs,
    ) == ("none", 0.0, 1.0, 1.29, False)
    assert trainer.collator.max_sequence_length == MAX_SEQUENCE_LENGTH == 256
    assert (training_args.lora_rank, training_args.lora_alpha) == (None, None)
    assert (trainer.spec.lora_rank, trainer.spec.lora_alpha) == (4, 4.0)
    assert (WEIGHTING_SCHEME, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, MAX_SEQUENCE_LENGTH) == (
        "none",
        0.0,
        1.0,
        1.29,
        256,
    )
    assert trainer.collator.system_prompt == SYSTEM_PROMPT


def test_lumina2_transform_and_prompt_ids_match_the_script(tmp_path):
    source = image(3, height=97, width=64)
    expected = functional.resize(source, 32, interpolation=transforms.InterpolationMode.BILINEAR)
    torch.manual_seed(4)
    expected = functional.crop(expected, *transforms.RandomCrop.get_params(expected, (32, 32)))
    expected = functional.normalize(functional.to_tensor(expected), [0.5], [0.5])

    root = write_tiny_lumina2(tmp_path / "lumina2")
    family = Lumina2Family()
    components = family.load(root, device=CPU, dtype=torch.float32, model_id="lumina-image-2.0")
    data = family.collator(
        components,
        args(tmp_path / "run"),
        generator=torch.Generator().manual_seed(4),
        train=True,
    )([{"image": source, "text": "a cat"}])
    assert torch.equal(data["pixel_values"][0], expected)

    reference_pipeline = diffusers.Lumina2Pipeline.from_pretrained(root, local_files_only=True)
    assert reference_pipeline.system_prompt == SYSTEM_PROMPT
    tokens = reference_pipeline.tokenizer(
        [reference_pipeline.system_prompt + " <Prompt Start> a cat"],
        padding="max_length",
        max_length=8,
        truncation=True,
        return_tensors="pt",
    )
    assert torch.equal(data["input_ids"], tokens.input_ids)
    assert torch.equal(data["attention_mask"], tokens.attention_mask.bool())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lumina2_first_step_matches_diffusers(tmp_path, scheme):
    root = write_tiny_lumina2(tmp_path / "lumina2")
    family = Lumina2Family()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="lumina-image-2.0")
    for module in [ours.transformer, *family.frozen(ours)]:
        module.requires_grad_(False)
    injection = inject_lora(
        ours.transformer, LoRAConfig(rank=4, alpha=4, target_modules=TARGETS, init="gaussian")
    )
    generator = torch.Generator().manual_seed(5)
    batch = family.collator(ours, args(tmp_path / "run"), generator=generator, train=True)(
        [{"image": image(), "text": "a cat"}, {"image": image(13), "text": "a horse"}]
    )
    replay = torch.Generator()
    replay.set_state(generator.get_state())
    objective = family.objective(ours, args(tmp_path / "run", weighting_scheme=scheme))
    draws = objective.draw(batch, generator)
    loss = objective.loss_from(batch, draws)
    loss.backward()

    transformer = diffusers.Lumina2Transformer2DModel.from_pretrained(root, subfolder="transformer")
    vae = diffusers.AutoencoderKL.from_pretrained(root, subfolder="vae")
    text_encoder = transformers.Gemma2Model.from_pretrained(root, subfolder="text_encoder").to(
        torch.bfloat16
    )
    transformer.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS)
    )
    names = [
        f"{name}.lora_{letter}.default.weight"
        for name in injection.module_names
        for letter in ("A", "B")
    ]
    with torch.no_grad():
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        pipe = diffusers.Lumina2Pipeline.from_pretrained(
            root, transformer=None, vae=None, text_encoder=text_encoder, local_files_only=True
        )
        embeds, mask, _, _ = pipe.encode_prompt(
            ["a cat", "a horse"],
            device=CPU,
            do_classifier_free_guidance=False,
            max_sequence_length=8,
        )
        posterior_distribution = vae.encode(batch["pixel_values"]).latent_dist
        posterior_state = replay.get_state()
        sample = posterior_distribution.sample(generator=replay)
        posterior = torch.randn(
            posterior_distribution.mean.shape,
            generator=torch.Generator().set_state(posterior_state),
        )
        latents = (sample - vae.config.shift_factor) * vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=replay)
        u = training_utils.compute_density_for_timestep_sampling(
            weighting_scheme=scheme,
            batch_size=len(latents),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            generator=replay,
        )
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler"
        )
        indices = (u * scheduler.config.num_train_timesteps).long()
        timesteps = scheduler.timesteps[indices]
        sigmas = scheduler.sigmas[indices].view(len(latents), 1, 1, 1)
    prediction = transformer(
        (1.0 - sigmas) * noise + sigmas * latents,
        timesteps / scheduler.config.num_train_timesteps,
        embeds,
        mask,
        return_dict=False,
    )[0]
    expected = torch.mean(
        (
            training_utils.compute_loss_weighting_for_sd3(scheme, sigmas).float()
            * (prediction.float() - (latents - noise).float()) ** 2
        ).reshape(len(latents), -1),
        1,
    ).mean()
    expected.backward()
    assert torch.equal(draws["posterior"], posterior)
    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    assert torch.equal(draws["sigmas"], scheduler.sigmas[indices])
    assert torch.equal(loss, expected)
    assert all(
        torch.equal(mine.grad, transformer.get_parameter(name).grad)
        for mine, name in zip(injection.parameters(), names, strict=True)
    )


def _reference_output(reference_pipeline):
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(42))
    common = dict(
        prompt="a cat",
        negative_prompt="",
        num_inference_steps=3,
        guidance_scale=4.0,
        height=32,
        width=32,
        max_sequence_length=8,
        latents=noise,
    )
    latents = reference_pipeline(output_type="latent", **common).images
    pixels = reference_pipeline(output_type="np", **common).images
    return latents, torch.from_numpy((pixels * 255).round().astype("uint8"))


def _native_output(root, **options):
    backend = Lumina2Backend(root, **options)
    backend._load()
    assert backend.components is not None
    final = denoise(
        backend.components,
        prompt="a cat",
        negative_prompt="",
        system_prompt=SYSTEM_PROMPT,
        steps=3,
        guidance=4.0,
        cfg_trunc_ratio=1.0,
        cfg_normalization=True,
        max_sequence_length=8,
        latents=torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(42)),
    )
    return final, decode_latents(backend.components.vae, final)


def _assert_registered_targets(reference_pipeline, name):
    registered = set(reference_pipeline.transformer.peft_config[name].target_modules)
    assert registered
    assert {
        target for target in TARGETS if any(path.endswith(target) for path in registered)
    } == set(TARGETS)


def _assert_native_matches(root, reference_pipeline, plain, **options):
    expected = _reference_output(reference_pipeline)
    # Tiny-model updates can round to identical uint8 pixels, so final latents remain asserted.
    assert not (torch.equal(expected[0], plain[0]) and torch.equal(expected[1], plain[1]))
    actual = _native_output(root, **options)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_lumina2_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_lumina2(tmp_path / "lumina2")
    with pipeline("t2i", backend="lumina-image-2.0", model=root) as pipe:
        first = random_lora({"transformer": pipe.backend.components.transformer}, TARGETS, seed=4)
        second = random_lora({"transformer": pipe.backend.components.transformer}, TARGETS, seed=7)
    first = write_peft(tmp_path / "first.safetensors", first, alpha=3.0)
    second = write_peft(tmp_path / "second.safetensors", second, alpha=2.0)
    reference_pipeline = diffusers.Lumina2Pipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.set_progress_bar_config(disable=True)
    plain = _reference_output(reference_pipeline)
    reference_pipeline.load_lora_weights(str(first), adapter_name="first")
    _assert_registered_targets(reference_pipeline, "first")
    _assert_native_matches(root, reference_pipeline, plain, lora=first)
    reference_pipeline.load_lora_weights(str(second), adapter_name="second")
    _assert_registered_targets(reference_pipeline, "second")
    reference_pipeline.set_adapters(["first", "second"], [0.8, 0.5])
    _assert_native_matches(root, reference_pipeline, plain, lora=[(first, 0.8), (second, 0.5)])
    reference_pipeline.fuse_lora()
    _assert_native_matches(
        root, reference_pipeline, plain, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True
    )


def test_trained_lumina2_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_lumina2(tmp_path / "lumina2")
    trainer = Trainer(
        "lumina-image-2.0",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run",
            max_steps=3,
            resolution=32,
            max_sequence_length=8,
            learning_rate=1e-2,
        ),
        [{"image": image(9), "text": "a cat"}],
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    adapter = trainer.save_model(tmp_path / "adapter")
    reference_pipeline = diffusers.Lumina2Pipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.set_progress_bar_config(disable=True)
    plain = _reference_output(reference_pipeline)
    reference_pipeline.load_lora_weights(str(adapter), adapter_name="trained")
    _assert_registered_targets(reference_pipeline, "trained")
    _assert_native_matches(root, reference_pipeline, plain, lora=adapter)
