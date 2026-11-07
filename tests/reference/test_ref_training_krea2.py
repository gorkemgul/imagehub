"""Krea 2 training defaults and frozen-input parity against diffusers."""

import json
import math

import pytest
import torch
from helpers import write_tiny_krea2
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.backends.krea2 import Krea2Components
from imagehub.checkpoints import load_weights
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments, get_training_spec
from imagehub.training.krea2 import (
    LOGIT_MEAN,
    LOGIT_STD,
    MAX_SEQUENCE_LENGTH,
    MODE_SCALE,
    WEIGHTING_SCHEME,
    Krea2Family,
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
TARGETS = get_training_spec("krea2-turbo").target_modules
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")


def _args(root, **kwargs):
    return TrainingArguments(
        output_dir=root, max_steps=1, resolution=32, max_sequence_length=8, **kwargs
    )


def _image(seed=12, *, height=32, width=32):
    return torch.rand(3, height, width, generator=torch.Generator().manual_seed(seed))


def _reference_pipeline(root):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    text_encoder = transformers.Qwen3VLModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    transformer = diffusers.Krea2Transformer2DModel(
        **json.loads((root / "transformer/config.json").read_text())
    )
    transformer.load_state_dict(load_weights(root / "transformer"))
    vae = diffusers.AutoencoderKLQwenImage(**json.loads((root / "vae/config.json").read_text()))
    vae.load_state_dict(load_weights(root / "vae"))
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(
        json.loads((root / "scheduler/scheduler_config.json").read_text())
    )
    return diffusers.Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        text_encoder_select_layers=tuple(range(1, 13)),
        is_distilled=False,
        patch_size=2,
    )


def test_krea2_defaults_and_prompt_embeddings_match_diffusers(tmp_path):
    root = write_tiny_krea2(tmp_path / "raw", distilled=False)
    spec = get_training_spec("krea2-turbo")
    assert (spec.lora_rank, spec.lora_alpha) == (32, 32)
    assert (WEIGHTING_SCHEME, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, MAX_SEQUENCE_LENGTH) == (
        "none",
        0.0,
        1.0,
        1.29,
        512,
    )
    components = Krea2Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    family = Krea2Family()
    args = TrainingArguments(
        output_dir=tmp_path / "run", max_steps=1, resolution=32, max_sequence_length=8
    )
    collator = family.collator(
        components, args, generator=torch.Generator().manual_seed(0), train=True
    )
    batch = collator([{"image": torch.zeros(3, 32, 32), "text": "a tiny cat"}])
    actual = family.objective(components, args).encode_condition(batch)
    expected_embeds, expected_mask = _reference_pipeline(root).encode_prompt(
        "a tiny cat", max_sequence_length=8
    )
    assert torch.equal(actual["prompt_embeds"], expected_embeds)
    assert torch.equal(actual["prompt_mask"], expected_mask)

    source = PIL.fromarray(
        torch.randint(
            0, 256, (97, 64, 3), generator=torch.Generator().manual_seed(3), dtype=torch.uint8
        ).numpy()
    )
    replay = torch.Generator().manual_seed(4)
    resized = functional.resize(source, 32, interpolation=transforms.InterpolationMode.BILINEAR)
    top = int(torch.randint(0, resized.height - 32 + 1, (1,), generator=replay))
    left = int(torch.randint(0, resized.width - 32 + 1, (1,), generator=replay))
    expected_pixels = functional.crop(resized, top, left, 32, 32)
    if torch.rand(1, generator=replay) < 0.5:
        expected_pixels = functional.hflip(expected_pixels)
    expected_pixels = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(expected_pixels))
    transformed = family.collator(
        components,
        TrainingArguments(
            output_dir=tmp_path / "transform", max_steps=1, resolution=32, random_flip=True
        ),
        generator=torch.Generator().manual_seed(4),
        train=True,
    )([{"image": source, "text": "a tiny cat"}])
    assert torch.equal(transformed["pixel_values"][0], expected_pixels)
    tokens = family.collator(
        components,
        TrainingArguments(output_dir=tmp_path / "tokens", max_steps=1, resolution=32),
        generator=torch.Generator().manual_seed(0),
        train=True,
    )([{"image": torch.zeros(3, 32, 32), "text": "a tiny cat"}])
    assert torch.equal(transformed["input_ids"], tokens["input_ids"])
    assert torch.equal(transformed["attention_mask"], tokens["attention_mask"])

    native_trainer = Trainer(
        "krea2-turbo",
        root,
        TrainingArguments(
            output_dir=tmp_path / "defaults", max_steps=1, resolution=32, lora_rank=1, lora_alpha=1
        ),
        [{"image": _image(), "text": "a tiny cat"}],
    )
    native_trainer._setup()
    assert native_trainer.collator.max_sequence_length == MAX_SEQUENCE_LENGTH
    reference_transformer = _reference_pipeline(root).transformer
    reference_transformer.add_adapter(
        peft.LoraConfig(
            r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=TARGETS,
        )
    )
    registered = {
        name.removesuffix(".lora_A.default.weight")
        for name, _ in reference_transformer.named_parameters()
        if name.endswith(".lora_A.default.weight")
    }
    assert set(native_trainer.injection.module_names) == registered


@pytest.mark.parametrize("scheme", SCHEMES)
def test_krea2_first_step_matches_diffusers_for_every_weighting_scheme(tmp_path, scheme):
    root = write_tiny_krea2(tmp_path / "raw", distilled=False)
    family = Krea2Family()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="krea2-turbo")
    for module in [ours.transformer, *family.frozen(ours)]:
        module.requires_grad_(False)
    injection = inject_lora(
        ours.transformer, LoRAConfig(rank=1, alpha=1, target_modules=TARGETS, init="gaussian")
    )
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in injection.parameters():
            if not parameter.any():
                parameter.normal_(0, 0.02, generator=generator)
    ours.transformer.train()
    batch = family.collator(
        ours, _args(tmp_path / "batch"), generator=torch.Generator().manual_seed(12), train=True
    )(
        [
            {"image": _image(), "text": "a tiny cat"},
            {"image": _image(13), "text": "a tiny horse"},
        ]
    )
    replay = torch.Generator()
    replay.set_state(generator.get_state())
    objective = family.objective(ours, _args(tmp_path / "run", weighting_scheme=scheme))
    draws = objective.draw(batch, generator)
    loss = objective.loss_from(batch, draws)
    loss.backward()

    reference = _reference_pipeline(root)
    transformer = reference.transformer
    transformer.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(
            r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=TARGETS,
        )
    )
    names = [
        f"{name}.lora_{letter}.default.weight"
        for name in injection.module_names
        for letter in ("A", "B")
    ]
    with torch.no_grad():
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        posterior = reference.vae.encode(batch["pixel_values"].unsqueeze(2)).latent_dist
        posterior_state = replay.get_state()
        latents = posterior.sample(generator=replay)
        posterior_noise = torch.randn(
            posterior.mean.shape, generator=torch.Generator().set_state(posterior_state)
        )
        mean = torch.tensor(reference.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        inverse_std = (
            torch.tensor(reference.vae.config.latents_std).reciprocal().view(1, -1, 1, 1, 1)
        )
        latents = (latents - mean) * inverse_std
        noise = torch.randn(latents.shape, generator=replay)
        u = training_utils.compute_density_for_timestep_sampling(
            weighting_scheme=scheme,
            batch_size=len(latents),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            generator=replay,
        )
        scheduler = reference.scheduler
        indices = (u * scheduler.config.num_train_timesteps).long()
        timesteps = scheduler.timesteps[indices]
        sigmas = scheduler.sigmas[indices].view(len(latents), 1, 1, 1, 1)
        prompt, mask = reference.encode_prompt(
            ["a tiny cat", "a tiny horse"], max_sequence_length=8
        )
        noisy = (1 - sigmas) * latents + sigmas * noise
        batch_size, channels, frames, height, width = noisy.shape
        packed = noisy.permute(0, 2, 1, 3, 4).view(
            batch_size, frames, channels, height // 2, 2, width // 2, 2
        )
        packed = packed.permute(0, 3, 5, 2, 1, 4, 6).reshape(
            batch_size, (height // 2) * (width // 2), channels * frames * 4
        )
        position_ids = diffusers.Krea2Pipeline.prepare_position_ids(
            prompt.shape[1], height // 2, width // 2, CPU
        )
    prediction = transformer(
        packed,
        encoder_hidden_states=prompt,
        timestep=timesteps / 1000,
        position_ids=position_ids,
        encoder_attention_mask=mask,
        return_dict=False,
    )[0]
    prediction = prediction.view(
        batch_size, height // 2, width // 2, prediction.shape[-1] // 4, 2, 2
    )
    prediction = prediction.permute(0, 3, 1, 4, 2, 5).reshape(
        batch_size, channels, 1, height, width
    )
    expected = torch.mean(
        (
            training_utils.compute_loss_weighting_for_sd3(scheme, sigmas).float()
            * (prediction.float() - (noise - latents).float()) ** 2
        ).reshape(len(latents), -1),
        1,
    ).mean()
    expected.backward()
    assert torch.equal(draws["posterior"], posterior_noise)
    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    assert torch.equal(draws["sigmas"], scheduler.sigmas[indices])
    assert torch.equal(loss, expected)
    assert all(
        torch.equal(mine.grad, transformer.get_parameter(name).grad)
        for mine, name in zip(injection.parameters(), names, strict=True)
    )


def _reference_output(reference):
    raw = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(42))
    latents = reference._pack_latents(raw, 1, 4, 2, 2)
    common = dict(
        prompt="a cat",
        num_inference_steps=2,
        guidance_scale=0.0,
        height=16,
        width=16,
        latents=latents,
        max_sequence_length=8,
    )
    final = reference(output_type="latent", **common).images
    pixels = reference(output_type="np", **common).images
    return final, torch.from_numpy((pixels * 255).round().astype("uint8"))


def _native_output(root, **options):
    raw = torch.randn(1, 4, 2, 2, generator=torch.Generator().manual_seed(42))
    reference = _reference_pipeline(root)
    latents = reference._pack_latents(raw, 1, 4, 2, 2)
    with pipeline("t2i", backend="krea2", model=root, **options) as native:
        result = native(
            "a cat",
            steps=2,
            guidance=0.0,
            width=16,
            height=16,
            latents=latents,
            max_sequence_length=8,
        )
    from imagehub.backends.krea2 import denoise
    from imagehub.lora_formats import apply_lora, parse_lora_option

    components = Krea2Components.load(root, device=CPU, dtype=torch.float32)
    if options.get("lora"):
        apply_lora(
            {"transformer": components.transformer},
            parse_lora_option(options["lora"]),
            fuse=options.get("lora_fuse", False),
        )
    final = denoise(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=2,
        guidance=0.0,
        width=16,
        height=16,
        latents=latents,
        max_sequence_length=8,
    )
    return final, torch.stack([image.tensor for image in result.images])


def _assert_registered(reference, name):
    registered = set(reference.transformer.peft_config[name].target_modules)
    assert registered
    # PEFT records the terminal module spelling for nested Krea 2 targets.
    assert registered <= {
        "img_in",
        "linear",
        "linear_1",
        "linear_2",
        "projector",
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "to_gate",
        "up",
        "down",
        "time_mod_proj",
    }


def _assert_adapter(root, reference, plain, **options):
    expected = _reference_output(reference)
    assert not torch.equal(expected[0], plain[0])
    actual = _native_output(root, **options)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_krea2_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_krea2(tmp_path / "krea2", distilled=False)
    components = Krea2Components.load(root, device=CPU, dtype=torch.float32)
    first = write_peft(
        tmp_path / "first.safetensors",
        random_lora({"transformer": components.transformer}, TARGETS, rank=1, seed=4),
        alpha=1.0,
    )
    second = write_peft(
        tmp_path / "second.safetensors",
        random_lora({"transformer": components.transformer}, TARGETS, rank=1, seed=7),
        alpha=1.0,
    )
    reference = _reference_pipeline(root)
    plain = _reference_output(reference)
    reference.load_lora_weights(str(first), adapter_name="first")
    _assert_registered(reference, "first")
    _assert_adapter(root, reference, plain, lora=first)
    reference.load_lora_weights(str(second), adapter_name="second")
    _assert_registered(reference, "second")
    reference.set_adapters(["first", "second"], [0.8, 0.5])
    _assert_adapter(root, reference, plain, lora=[(first, 0.8), (second, 0.5)])
    reference.fuse_lora()
    _assert_adapter(root, reference, plain, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True)


def test_trained_krea2_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_krea2(tmp_path / "krea2", distilled=False)
    trainer = Trainer(
        "krea2-turbo",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run",
            max_steps=3,
            resolution=32,
            max_sequence_length=8,
            learning_rate=1e-2,
            lora_rank=1,
            lora_alpha=1,
        ),
        [{"image": _image(9), "text": "a cat"}],
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    adapter = trainer.save_model(tmp_path / "adapter")
    reference = _reference_pipeline(root)
    plain = _reference_output(reference)
    reference.load_lora_weights(str(adapter), adapter_name="trained")
    _assert_registered(reference, "trained")
    _assert_adapter(root, reference, plain, lora=adapter)
