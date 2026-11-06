"""Qwen-Image LoRA-step and adapter parity against diffusers, CPU float32."""

import math

import pytest
import torch
from helpers import write_tiny_qwen_image
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    OriginalQwenImageComponents,
    QwenImageBackend,
    decode_original_latents,
    denoise_original,
)
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments, get_training_spec
from imagehub.training.qwen_image import (
    LOGIT_MEAN,
    LOGIT_STD,
    MODE_SCALE,
    ORIGINAL_MAX_SEQUENCE_LENGTH,
    WEIGHTING_SCHEME,
    QwenImageFamily,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
training_utils = pytest.importorskip("diffusers.training_utils")
PIL = pytest.importorskip("PIL.Image")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")

CPU = torch.device("cpu")
TARGETS = ("to_k", "to_q", "to_v", "to_out.0")
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")
TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
DROP_INDEX = 34


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


def batch(family, components, generator):
    args = TrainingArguments(output_dir="unused", max_steps=1, resolution=32)
    return family.collator(components, args, generator=generator, train=True)(
        [{"image": image(), "text": "a cat"}, {"image": image(), "text": "a horse"}]
    )


def reference(root):
    transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
        root, subfolder="transformer", torch_dtype=torch.float32
    )
    vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        root, subfolder="vae", torch_dtype=torch.float32
    )
    for module in (transformer, vae):
        module.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS)
    )
    transformer.train()
    return transformer, vae


def test_qwen_image_default_arguments_give_the_script_defaults(tmp_path):
    # train_dreambooth_lora_qwen_image.py:343-346, 383-393, 563-578 and 1281-1292.
    root = write_tiny_qwen_image(tmp_path / "qwen")
    args = TrainingArguments(output_dir=tmp_path / "run", max_steps=1, resolution=32)
    trainer = Trainer("qwen-image", root, args, [{"image": image(), "text": "a cat"}])
    trainer._setup()

    assert trainer.objective.weighting_scheme == WEIGHTING_SCHEME == "none"
    assert (
        (trainer.objective.logit_mean, trainer.objective.logit_std, trainer.objective.mode_scale)
        == (
            LOGIT_MEAN,
            LOGIT_STD,
            MODE_SCALE,
        )
        == (0.0, 1.0, 1.29)
    )
    assert trainer.collator.original and ORIGINAL_MAX_SEQUENCE_LENGTH == 512
    assert (args.lora_rank, args.lora_alpha) == (None, None)
    assert (trainer.spec.lora_rank, trainer.spec.lora_alpha) == (4, 4)
    assert trainer.injection is not None
    assert (trainer.injection.config.rank, trainer.injection.config.alpha) == (4, 4.0)
    assert set(trainer.injection.module_names) and set(trainer.target_modules) == set(TARGETS)
    assert set(get_training_spec("qwen-image").target_modules) == set(TARGETS)


def test_qwen_image_transform_and_prompt_ids_match_the_script(tmp_path):
    source = image(3, height=97, width=64)
    expected = functional.resize(
        source, [48, 32], interpolation=transforms.InterpolationMode.BILINEAR
    )
    torch.manual_seed(4)
    expected = functional.crop(expected, *transforms.RandomCrop.get_params(expected, (32, 32)))
    expected = functional.normalize(functional.to_tensor(expected), [0.5], [0.5])

    root = write_tiny_qwen_image(tmp_path / "qwen")
    family = QwenImageFamily()
    components = family.load(root, device=CPU, dtype=torch.float32, model_id="qwen-image")
    data = family.collator(
        components,
        TrainingArguments(output_dir="unused", max_steps=1, resolution=32),
        generator=torch.Generator().manual_seed(4),
        train=True,
    )([{"image": source, "text": "a cat"}])
    assert torch.equal(data["pixel_values"][0], expected)

    reference_pipeline = diffusers.QwenImagePipeline.from_pretrained(root, local_files_only=True)
    tokens = reference_pipeline.tokenizer(
        [TEMPLATE.format("a cat")],
        max_length=1024 + DROP_INDEX,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    assert torch.equal(data["input_ids"][:, DROP_INDEX:], tokens.input_ids[:, DROP_INDEX:])
    assert torch.equal(
        data["attention_mask"][:, DROP_INDEX:], tokens.attention_mask.bool()[:, DROP_INDEX:]
    )
    expected_embeds, expected_mask = reference_pipeline.encode_prompt(
        "a cat", device=CPU, max_sequence_length=ORIGINAL_MAX_SEQUENCE_LENGTH
    )
    actual = family.objective(components, TrainingArguments(output_dir="unused", max_steps=1))
    condition = actual.encode_condition(data)
    assert torch.equal(condition["prompt_embeds"], expected_embeds)
    if expected_mask is None:
        assert condition["prompt_mask"] is None
    else:
        assert condition["prompt_mask"] is not None
        assert torch.equal(condition["prompt_mask"], expected_mask.bool())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_first_qwen_image_step_matches_diffusers(tmp_path, scheme):
    root = write_tiny_qwen_image(tmp_path / "qwen")
    family = QwenImageFamily()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="qwen-image")
    for module in [ours.transformer, *family.frozen(ours)]:
        module.requires_grad_(False)
    injection = inject_lora(
        ours.transformer, LoRAConfig(rank=4, alpha=4, target_modules=TARGETS, init="gaussian")
    )
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in injection.parameters():
            if not parameter.any():
                parameter.normal_(0, 0.02, generator=generator)
    ours.transformer.train()
    data = batch(family, ours, generator)
    replay = torch.Generator()
    replay.set_state(generator.get_state())
    objective = family.objective(
        ours,
        TrainingArguments(
            output_dir=tmp_path / "run", max_steps=1, resolution=32, weighting_scheme=scheme
        ),
    )
    draws = objective.draw(data, generator)
    loss = objective.loss_from(data, draws)
    loss.backward()

    transformer, vae = reference(root)
    names = [
        f"{name}.lora_{letter}.default.weight"
        for name in injection.module_names
        for letter in ("A", "B")
    ]
    with torch.no_grad():
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        reference_pipeline = diffusers.QwenImagePipeline.from_pretrained(
            root, transformer=None, vae=None, local_files_only=True
        )
        prompt_embeds, prompt_mask = reference_pipeline.encode_prompt(
            ["a cat", "a horse"], device=CPU, max_sequence_length=ORIGINAL_MAX_SEQUENCE_LENGTH
        )
        distribution = vae.encode(data["pixel_values"].unsqueeze(2)).latent_dist
        posterior_noise = torch.randn(distribution.mean.shape, generator=replay)
        posterior = distribution.mean + distribution.std * posterior_noise
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        latents = (posterior - mean) / std
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
        sigmas = scheduler.sigmas[indices].view(len(latents), 1, 1, 1, 1)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        packed = diffusers.QwenImagePipeline._pack_latents(
            noisy.permute(0, 2, 1, 3, 4),
            len(latents),
            latents.shape[1],
            latents.shape[3],
            latents.shape[4],
        )
    prediction = transformer(
        hidden_states=packed,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        timestep=timesteps / 1000,
        img_shapes=[(1, latents.shape[3] // 2, latents.shape[4] // 2)] * len(latents),
        return_dict=False,
    )[0]
    prediction = diffusers.QwenImagePipeline._unpack_latents(prediction, 32, 32, 8)
    weighting = training_utils.compute_loss_weighting_for_sd3(scheme, sigmas)
    expected = torch.mean(
        (weighting.float() * (prediction.float() - (noise - latents).float()) ** 2).reshape(
            len(latents), -1
        ),
        1,
    ).mean()
    expected.backward()

    assert torch.equal(draws["posterior"], posterior_noise)
    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    assert torch.equal(loss, expected)
    assert all(
        torch.equal(mine.grad, transformer.get_parameter(name).grad)
        for mine, name in zip(injection.parameters(), names, strict=True)
    )


def _noise(height=32, width=32):
    raw = torch.randn(
        1, 1, 16, height // 8, width // 8, generator=torch.Generator().manual_seed(42)
    )
    return diffusers.QwenImagePipeline._pack_latents(raw, 1, 16, height // 8, width // 8)


def reference_output(reference_pipeline):
    noise = _noise()
    common = dict(
        prompt="a cat",
        negative_prompt="",
        true_cfg_scale=1.0,
        num_inference_steps=3,
        height=32,
        width=32,
        latents=noise,
    )
    latents = reference_pipeline(output_type="latent", **common).images
    pixels = reference_pipeline(output_type="np", **common).images
    return latents, torch.from_numpy((pixels * 255).round().astype("uint8"))


def native_output(root, **options):
    backend = QwenImageBackend(root, model_id="qwen-image", **options)
    backend._load()
    assert isinstance(backend.components, OriginalQwenImageComponents)
    components = backend.components
    latents = denoise_original(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=3,
        true_cfg=1.0,
        width=32,
        height=32,
        latents=_noise(),
    )
    return latents, decode_original_latents(components.vae, latents, 32, 32)


def assert_native_matches(root, reference_pipeline, plain, **options):
    expected = reference_output(reference_pipeline)
    # Tiny-model adapter updates can round to the same uint8 pixels; latents retain that assertion.
    assert not (torch.equal(expected[0], plain[0]) and torch.equal(expected[1], plain[1]))
    actual = native_output(root, **options)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def assert_registered_targets(reference_pipeline, name):
    registered = set(reference_pipeline.transformer.peft_config[name].target_modules)
    assert registered
    assert {
        target for target in TARGETS if any(path.endswith(target) for path in registered)
    } == set(TARGETS)
    assert all(any(path.endswith(target) for target in TARGETS) for path in registered)


def qwen_adapters(root, tmp_path):
    with pipeline("t2i", backend="qwen-image", model=root) as pipe:
        first = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, seed=4, std=0.1
        )
        second = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, seed=7, std=0.1
        )
    return (
        write_peft(tmp_path / "first.safetensors", first, alpha=3.0),
        write_peft(tmp_path / "second.safetensors", second, alpha=2.0),
    )


def test_qwen_image_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_qwen_image(tmp_path / "qwen")
    first, second = qwen_adapters(root, tmp_path)
    reference_pipeline = diffusers.QwenImagePipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.set_progress_bar_config(disable=True)
    plain = reference_output(reference_pipeline)
    reference_pipeline.load_lora_weights(str(first), adapter_name="first")
    assert_registered_targets(reference_pipeline, "first")
    assert_native_matches(root, reference_pipeline, plain, lora=first)
    reference_pipeline.load_lora_weights(str(second), adapter_name="second")
    reference_pipeline.set_adapters(["first", "second"], [0.8, 0.5])
    assert_native_matches(root, reference_pipeline, plain, lora=[(first, 0.8), (second, 0.5)])
    reference_pipeline.fuse_lora()
    assert_native_matches(
        root, reference_pipeline, plain, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True
    )


def test_trained_qwen_image_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_qwen_image(tmp_path / "qwen")
    trainer = Trainer(
        "qwen-image",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
        ),
        [{"image": image(9), "text": "a cat"}],
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    adapter = trainer.save_model(tmp_path / "adapter")
    reference_pipeline = diffusers.QwenImagePipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.set_progress_bar_config(disable=True)
    plain = reference_output(reference_pipeline)
    reference_pipeline.load_lora_weights(str(adapter), adapter_name="trained")
    assert_registered_targets(reference_pipeline, "trained")
    assert_native_matches(root, reference_pipeline, plain, lora=adapter)
