"""Qwen-Image 2.1 LoRA-step and adapter parity against diffusers, CPU float32."""

import math

import pytest
import torch
from helpers import write_tiny_qwen_image21
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    QwenImageBackend,
    QwenImageComponents,
    decode_latents,
    denoise,
)
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments, get_training_spec
from imagehub.training.qwen_image import (
    LOGIT_MEAN,
    LOGIT_STD,
    MODE_SCALE,
    WEIGHTING_SCHEME,
    QwenImageFamily,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip(
        "needs the diffusers commit pinned for Qwen-Image 2.1 (.venv-ref-qwen-image-2.1)",
        allow_module_level=True,
    )
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
training_utils = pytest.importorskip("diffusers.training_utils")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")

CPU = torch.device("cpu")
TARGETS = ("to_k", "to_q", "to_v", "to_out.0")
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")
SYSTEM_TEMPLATE = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
TEMPLATE = SYSTEM_TEMPLATE + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class _TextOnlyProcessor:
    def __init__(self, root):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)

    def __call__(self, *, text, padding, padding_side, return_tensors):
        self.tokenizer.padding_side = padding_side
        return self.tokenizer(
            text, padding=padding, return_tensors=return_tensors, add_special_tokens=True
        )

    def apply_chat_template(self, messages, *, tokenize, return_dict):
        del tokenize, return_dict
        text = "".join(content["text"] for content in messages[0]["content"])
        rendered = f"<|im_start|>system\n{text}<|im_end|>\n"
        return [self.tokenizer.encode(rendered)]


def image(seed=12, *, height=32, width=32):
    pixels = torch.randint(
        0,
        256,
        (3, height, width),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.uint8,
    )
    return functional.to_pil_image(pixels)


def batch(family, components, generator):
    args = TrainingArguments(output_dir="unused", max_steps=1, resolution=32)
    return family.collator(components, args, generator=generator, train=True)(
        [{"image": image(), "text": "a cat"}, {"image": image(), "text": "a horse"}]
    )


def reference_pipeline(root, *, transformer=None, vae=None):
    text_encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    transformer = transformer or diffusers.QwenImage21Transformer2DModel.from_pretrained(
        root, subfolder="transformer", torch_dtype=torch.float32
    )
    vae = vae or diffusers.AutoencoderKLQwenImage21.from_pretrained(
        root, subfolder="vae", torch_dtype=torch.float32
    )
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        root, subfolder="scheduler"
    )
    result = diffusers.QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=_TextOnlyProcessor(root / "processor"),
        transformer=transformer,
    )
    result.set_progress_bar_config(disable=True)
    return result


def reference(root):
    transformer = diffusers.QwenImage21Transformer2DModel.from_pretrained(
        root, subfolder="transformer", torch_dtype=torch.float32
    )
    vae = diffusers.AutoencoderKLQwenImage21.from_pretrained(
        root, subfolder="vae", torch_dtype=torch.float32
    )
    for module in (transformer, vae):
        module.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian", target_modules=TARGETS)
    )
    transformer.train()
    return transformer, vae


def test_qwen_image21_default_arguments_give_the_script_defaults(tmp_path):
    # train_dreambooth_lora_qwenimage21.py:404-417, 581-600 and 1324-1334.
    root = write_tiny_qwen_image21(tmp_path / "qwen")
    args = TrainingArguments(output_dir=tmp_path / "run", max_steps=1, resolution=32)
    trainer = Trainer("qwen-image-2.1", root, args, [{"image": image(), "text": "a cat"}])
    trainer._setup()

    assert trainer.objective.weighting_scheme == WEIGHTING_SCHEME == "none"
    assert (
        (trainer.objective.logit_mean, trainer.objective.logit_std, trainer.objective.mode_scale)
        == (LOGIT_MEAN, LOGIT_STD, MODE_SCALE)
        == (0.0, 1.0, 1.29)
    )
    assert trainer.collator.original is False
    assert (args.lora_rank, args.lora_alpha, args.lora_dropout, args.lora_target_modules) == (
        None,
        None,
        0.0,
        None,
    )
    assert (trainer.spec.lora_rank, trainer.spec.lora_alpha) == (16, 16.0)
    assert trainer.injection is not None
    assert (trainer.injection.config.rank, trainer.injection.config.alpha) == (16, 16.0)
    assert set(trainer.injection.module_names) and set(trainer.target_modules) == set(TARGETS)
    assert set(get_training_spec("qwen-image-2.1").target_modules) == set(TARGETS)


def test_qwen_image21_transform_and_prompt_ids_match_the_script(tmp_path):
    source = image(3, height=97, width=64)
    expected = functional.resize(
        source, [48, 32], interpolation=transforms.InterpolationMode.BILINEAR
    )
    torch.manual_seed(4)
    expected = functional.crop(expected, *transforms.RandomCrop.get_params(expected, (32, 32)))
    expected = functional.normalize(functional.to_tensor(expected), [0.5], [0.5])

    root = write_tiny_qwen_image21(tmp_path / "qwen")
    family = QwenImageFamily()
    components = family.load(root, device=CPU, dtype=torch.float32, model_id="qwen-image-2.1")
    data = family.collator(
        components,
        TrainingArguments(output_dir="unused", max_steps=1, resolution=32),
        generator=torch.Generator().manual_seed(4),
        train=True,
    )([{"image": source, "text": "a cat"}])
    assert torch.equal(data["pixel_values"][0], expected)

    processor = _TextOnlyProcessor(root / "processor")
    tokens = processor(
        text=[TEMPLATE.format("a cat")], padding=True, padding_side="left", return_tensors="pt"
    )
    drop = len(processor.tokenizer(SYSTEM_TEMPLATE, add_special_tokens=True).input_ids)
    assert torch.equal(data["input_ids"][:, drop:], tokens.input_ids[:, drop:])
    assert torch.equal(data["attention_mask"][:, drop:], tokens.attention_mask.bool()[:, drop:])
    expected_embeds, expected_mask, expected_image_mask = reference_pipeline(root).encode_prompt(
        "a cat", device=CPU
    )
    actual = family.objective(components, TrainingArguments(output_dir="unused", max_steps=1))
    condition = actual.encode_condition(data)
    assert torch.equal(condition["prompt_embeds"], expected_embeds)
    if expected_mask is None:
        assert condition["prompt_mask"] is None
    else:
        assert condition["prompt_mask"] is not None
        assert torch.equal(condition["prompt_mask"], expected_mask.bool())
    assert not expected_image_mask.any()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_first_qwen_image21_step_matches_diffusers(tmp_path, scheme):
    root = write_tiny_qwen_image21(tmp_path / "qwen")
    family = QwenImageFamily()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="qwen-image-2.1")
    for module in [ours.transformer, *family.frozen(ours)]:
        module.requires_grad_(False)
    injection = inject_lora(
        ours.transformer, LoRAConfig(rank=16, alpha=16, target_modules=TARGETS, init="gaussian")
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
        reference_pipe = reference_pipeline(root, transformer=transformer, vae=vae)
        prompt_embeds, prompt_mask, image_pad_mask = reference_pipe.encode_prompt(
            ["a cat", "a horse"], device=CPU
        )
        pixels = torch.cat(
            [data["pixel_values"], torch.zeros_like(data["pixel_values"][:, :1])], dim=1
        ).unsqueeze(2)
        distribution = vae.encode(pixels).latent_dist
        posterior_noise = torch.randn(distribution.mean.shape, generator=replay)
        posterior = distribution.mean + distribution.std * posterior_noise
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        latents = (posterior - mean) * std.reciprocal()
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
        packed = diffusers.QwenImage21Pipeline._pack_latents(
            noisy,
            len(latents),
            latents.shape[1],
            latents.shape[3],
            latents.shape[4],
        )
        target_slots = (latents.shape[3] * latents.shape[4]) // 4
        image_mask = torch.cat(
            [
                image_pad_mask,
                torch.ones(len(latents), target_slots, dtype=torch.bool),
            ],
            dim=1,
        )
    assert torch.equal(objective.encode_images(data["pixel_values"], draws["posterior"]), latents)
    actual_condition = objective.encode_condition(data)
    assert torch.equal(actual_condition["prompt_embeds"], prompt_embeds)
    assert torch.equal(actual_condition["prompt_mask"], prompt_mask)
    assert torch.equal(
        packed,
        noisy[:, :, 0].view(len(latents), latents.shape[1], -1).transpose(1, 2),
    )
    assert torch.equal(image_mask[:, : prompt_embeds.shape[1]], image_pad_mask)
    assert image_mask[:, prompt_embeds.shape[1] :].all()
    prediction = transformer(
        hidden_states=packed,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        timestep=timesteps / 1000,
        img_shapes=[[(1, latents.shape[3], latents.shape[4])]] * len(latents),
        img_mask=image_mask,
        return_dict=False,
    )[0]
    prediction = prediction[:, -packed.shape[1] :]
    prediction = diffusers.QwenImage21Pipeline._unpack_latents(prediction, 32, 32, 16)
    actual_prediction = objective.predict(noisy, timesteps, actual_condition)
    assert torch.equal(actual_prediction, prediction)
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
    for mine, name in zip(injection.parameters(), names, strict=True):
        assert torch.equal(mine.grad, transformer.get_parameter(name).grad), name


def _noise(height=32, width=32):
    raw = torch.randn(1, 64, height // 16, width // 16, generator=torch.Generator().manual_seed(42))
    return diffusers.QwenImage21Pipeline._pack_latents(raw, 1, 64, height // 16, width // 16)


def reference_output(reference_pipe):
    noise = _noise()
    common = dict(
        prompt="a cat",
        negative_prompt=None,
        true_cfg_scale=1.0,
        num_inference_steps=3,
        height=32,
        width=32,
        latents=noise,
        use_kv_cache=True,
    )
    latents = reference_pipe(output_type="latent", **common).images
    pixels = reference_pipe(output_type="np", **common).images
    return latents, torch.from_numpy((pixels * 255).round().astype("uint8"))


def native_output(root, **options):
    backend = QwenImageBackend(root, model_id="qwen-image-2.1", **options)
    backend._load()
    assert isinstance(backend.components, QwenImageComponents)
    components = backend.components
    latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=3,
        true_cfg=1.0,
        width=32,
        height=32,
        latents=_noise(),
    )
    return latents, decode_latents(components.vae, latents, 32, 32)


def assert_native_matches(root, reference_pipe, plain, **options):
    expected = reference_output(reference_pipe)
    assert not (torch.equal(expected[0], plain[0]) and torch.equal(expected[1], plain[1]))
    actual = native_output(root, **options)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def assert_registered_targets(reference_pipe, name):
    registered = set(reference_pipe.transformer.peft_config[name].target_modules)
    assert registered
    assert {
        target for target in TARGETS if any(path.endswith(target) for path in registered)
    } == set(TARGETS)
    assert all(any(path.endswith(target) for target in TARGETS) for path in registered)


def qwen_adapters(root, tmp_path):
    with pipeline("t2i", backend="qwen-image-2.1", model=root) as pipe:
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


def test_qwen_image21_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_qwen_image21(tmp_path / "qwen")
    first, second = qwen_adapters(root, tmp_path)
    reference_pipe = reference_pipeline(root)
    plain = reference_output(reference_pipe)
    reference_pipe.load_lora_weights(str(first), adapter_name="first")
    assert_registered_targets(reference_pipe, "first")
    assert_native_matches(root, reference_pipe, plain, lora=first)
    reference_pipe.load_lora_weights(str(second), adapter_name="second")
    reference_pipe.set_adapters(["first", "second"], [0.8, 0.5])
    assert_native_matches(root, reference_pipe, plain, lora=[(first, 0.8), (second, 0.5)])
    reference_pipe.fuse_lora()
    assert_native_matches(
        root, reference_pipe, plain, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True
    )


def test_trained_qwen_image21_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_qwen_image21(tmp_path / "qwen")
    trainer = Trainer(
        "qwen-image-2.1",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
        ),
        [{"image": image(9), "text": "a cat"}],
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    adapter = trainer.save_model(tmp_path / "adapter")
    reference_pipe = reference_pipeline(root)
    plain = reference_output(reference_pipe)
    reference_pipe.load_lora_weights(str(adapter), adapter_name="trained")
    assert_registered_targets(reference_pipe, "trained")
    assert_native_matches(root, reference_pipe, plain, lora=adapter)
