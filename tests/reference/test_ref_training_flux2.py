"""FLUX.2 klein LoRA-step and adapter parity against diffusers, CPU float32."""

import inspect
import math

import pytest
import torch
from helpers import write_tiny_flux2
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments
from imagehub.training.flux2 import (
    GUIDANCE_SCALE,
    LOGIT_MEAN,
    LOGIT_STD,
    MAX_SEQUENCE_LENGTH,
    MODE_SCALE,
    TEXT_ENCODER_OUT_LAYERS,
    WEIGHTING_SCHEME,
    Flux2Family,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
training_utils = pytest.importorskip("diffusers.training_utils")
PIL = pytest.importorskip("PIL.Image")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")

CPU = torch.device("cpu")
TARGETS = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
    *(f"single_transformer_blocks.{index}.attn.to_out" for index in range(24)),
)
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")
# The script is run with --text_encoder_out_layers 9 18 27: the official FLUX.2 Qwen3 layers and
# the Flux2KleinPipeline default. The script's own default (10, 20, 30) is the FLUX.2 [dev] set.
SCRIPT_TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)


def image(seed=12, *, height=40, width=48):
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


def reference_pipeline(root):
    result = diffusers.Flux2KleinPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    result.set_progress_bar_config(disable=True)
    return result


def reference(root):
    transformer = diffusers.Flux2Transformer2DModel.from_pretrained(root, subfolder="transformer")
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(root, subfolder="vae")
    text = transformers.Qwen3ForCausalLM.from_pretrained(root / "text_encoder")
    for module in (transformer, vae, text):
        module.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=TARGETS,
        )
    )
    transformer.train()
    return transformer, vae, text


def test_flux2_defaults_targets_transform_and_tokens_match_the_script(tmp_path):
    root = write_tiny_flux2(tmp_path / "flux2")
    trainer = Trainer(
        "flux2-klein-4b",
        root,
        TrainingArguments(output_dir=tmp_path / "run", max_steps=1, resolution=32),
        [{"image": image(), "text": "a cat"}],
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
    assert trainer.collator.max_sequence_length == MAX_SEQUENCE_LENGTH == 512
    assert (trainer.spec.lora_rank, trainer.spec.lora_alpha) == (4, 4.0)
    assert (WEIGHTING_SCHEME, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, GUIDANCE_SCALE) == (
        "none",
        0.0,
        1.0,
        1.29,
        3.5,
    )
    assert TEXT_ENCODER_OUT_LAYERS == SCRIPT_TEXT_ENCODER_OUT_LAYERS
    encode_prompt = inspect.signature(diffusers.Flux2KleinPipeline.encode_prompt)
    assert encode_prompt.parameters["text_encoder_out_layers"].default == (9, 18, 27)

    source = image(3, height=97, width=64)
    expected = functional.resize(source, 32, interpolation=transforms.InterpolationMode.BILINEAR)
    torch.manual_seed(4)
    expected = functional.crop(expected, *transforms.RandomCrop(32).get_params(expected, (32, 32)))
    expected = functional.hflip(expected)
    expected = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(expected))
    from imagehub.training.data import preprocess_image

    actual = preprocess_image(
        source,
        resolution=32,
        center_crop=False,
        random_flip=True,
        generator=torch.Generator().manual_seed(4),
    )
    # torchvision consumes its random crop from the global stream; the native collator consumes
    # the identical values from the supplied generator.
    replay = torch.Generator().manual_seed(4)
    resized = functional.resize(source, 32, interpolation=transforms.InterpolationMode.BILINEAR)
    top = int(torch.randint(0, resized.height - 32 + 1, (1,), generator=replay))
    left = int(torch.randint(0, resized.width - 32 + 1, (1,), generator=replay))
    expected = functional.crop(resized, top, left, 32, 32)
    if torch.rand(1, generator=replay) < 0.5:
        expected = functional.hflip(expected)
    expected = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(expected))
    assert torch.equal(actual, expected)

    collator = trainer.family.collator(
        trainer.components,
        args(tmp_path / "out"),
        generator=torch.Generator().manual_seed(0),
        train=True,
    )
    batch = collator([{"image": image(), "text": "a cat"}])
    ref = reference_pipeline(root)
    text = ref.tokenizer.apply_chat_template(
        [{"role": "user", "content": "a cat"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    expected_tokens = ref.tokenizer(
        text, return_tensors="pt", padding="max_length", truncation=True, max_length=8
    )
    assert torch.equal(batch["input_ids"], expected_tokens.input_ids)
    assert torch.equal(batch["attention_mask"], expected_tokens.attention_mask.bool())

    transformer, _, _ = reference(root)
    registered = {
        name.removesuffix(".lora_A.default.weight")
        for name, _ in transformer.named_parameters()
        if name.endswith(".lora_A.default.weight")
    }
    assert set(trainer.injection.module_names) == registered


@pytest.mark.parametrize("scheme", SCHEMES)
def test_first_flux2_step_matches_diffusers(tmp_path, scheme):
    root = write_tiny_flux2(tmp_path / "flux2")
    family = Flux2Family()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="flux2-klein-4b")
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
    data = family.collator(
        ours, args(tmp_path / "batch"), generator=torch.Generator().manual_seed(12), train=True
    )([{"image": image(), "text": "a cat"}, {"image": image(13), "text": "a horse"}])
    replay = torch.Generator()
    replay.set_state(generator.get_state())
    objective = family.objective(ours, args(tmp_path / "run", weighting_scheme=scheme))
    draws = objective.draw(data, generator)
    loss = objective.loss_from(data, draws)
    loss.backward()

    transformer, vae, text = reference(root)
    names = [
        f"{name}.lora_{letter}.default.weight"
        for name in injection.module_names
        for letter in ("A", "B")
    ]
    with torch.no_grad():
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        latent = vae.encode(data["pixel_values"]).latent_dist.mode()
        latent = diffusers.Flux2KleinPipeline._patchify_latents(latent)
        mean, var = vae.bn.running_mean.view(1, -1, 1, 1), vae.bn.running_var.view(1, -1, 1, 1)
        latent = (latent - mean) / torch.sqrt(var + vae.config.batch_norm_eps)
        ids = diffusers.Flux2KleinPipeline._prepare_latent_ids(latent)
        noise = torch.randn(latent.shape, generator=replay)
        u = training_utils.compute_density_for_timestep_sampling(
            weighting_scheme=scheme,
            batch_size=len(latent),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            generator=replay,
        )
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler"
        )
        indices = (u * scheduler.config.num_train_timesteps).long()
        timesteps, sigmas = (
            scheduler.timesteps[indices],
            scheduler.sigmas[indices].view(len(latent), 1, 1, 1),
        )
        noisy = (1.0 - sigmas) * latent + sigmas * noise
        packed = diffusers.Flux2KleinPipeline._pack_latents(noisy)
        encoded = text(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states
        prompt = (
            torch.stack([encoded[index] for index in SCRIPT_TEXT_ENCODER_OUT_LAYERS], dim=1)
            .permute(0, 2, 1, 3)
            .flatten(2)
        )
        text_ids = diffusers.Flux2KleinPipeline._prepare_text_ids(prompt)
    prediction = transformer(
        packed,
        encoder_hidden_states=prompt,
        timestep=timesteps / 1000,
        guidance=None,
        txt_ids=text_ids,
        img_ids=ids,
        return_dict=False,
    )[0]
    prediction = diffusers.Flux2KleinPipeline._unpack_latents_with_ids(
        prediction[:, : packed.size(1) :], ids
    )
    expected = torch.mean(
        (
            training_utils.compute_loss_weighting_for_sd3(scheme, sigmas).float()
            * (prediction.float() - (noise - latent).float()) ** 2
        ).reshape(len(latent), -1),
        1,
    ).mean()
    expected.backward()
    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    assert torch.equal(draws["sigmas"], scheduler.sigmas[indices])
    assert torch.equal(loss, expected)
    assert all(
        torch.equal(mine.grad, transformer.get_parameter(name).grad)
        for mine, name in zip(injection.parameters(), names, strict=True)
    )


def _reference_output(reference):
    latents = torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(42))
    common = dict(
        prompt="a cat",
        num_inference_steps=2,
        guidance_scale=1.0,
        height=32,
        width=32,
        max_sequence_length=8,
        latents=latents,
    )
    final = reference(output_type="latent", **common).images
    pixels = reference(output_type="np", **common).images
    return final, torch.from_numpy((pixels * 255).round().astype("uint8"))


def _native_output(root, **options):
    with pipeline("t2i", backend="flux2", model=root, **options) as native:
        result = native(
            "a cat",
            steps=2,
            guidance=1.0,
            width=32,
            height=32,
            max_sequence_length=8,
            latents=torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(42)),
        )
        pixels = torch.stack([image.tensor for image in result.images])
    from imagehub.backends.flux2 import Flux2Components, denoise

    components = Flux2Components.load(root, device=CPU, dtype=torch.float32)
    if options.get("lora"):
        from imagehub.lora_formats import apply_lora, parse_lora_option

        apply_lora(
            {"transformer": components.transformer},
            parse_lora_option(options["lora"]),
            fuse=options.get("lora_fuse", False),
        )
    final = denoise(
        components,
        prompt="a cat",
        steps=2,
        guidance=1.0,
        width=32,
        height=32,
        max_sequence_length=8,
        latents=torch.randn(1, 16, 8, 8, generator=torch.Generator().manual_seed(42)),
    )
    return final, pixels


def _assert_registered(reference, name):
    registered = set(reference.transformer.peft_config[name].target_modules)
    assert registered
    assert all(any(path.endswith(target) for target in TARGETS) for path in registered)


def _assert_adapter(root, reference, plain, **options):
    expected = _reference_output(reference)
    assert not torch.equal(expected[0], plain[0])
    actual = _native_output(root, **options)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_flux2_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_flux2(tmp_path / "flux2")
    with pipeline("t2i", backend="flux2", model=root) as native:
        assert native.backend.components is not None
        first = write_peft(
            tmp_path / "first.safetensors",
            random_lora({"transformer": native.backend.components.transformer}, TARGETS, seed=4),
            alpha=3.0,
        )
        second = write_peft(
            tmp_path / "second.safetensors",
            random_lora({"transformer": native.backend.components.transformer}, TARGETS, seed=7),
            alpha=2.0,
        )
    reference = reference_pipeline(root)
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


def test_trained_flux2_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_flux2(tmp_path / "flux2")
    trainer = Trainer(
        "flux2-klein-4b",
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
    reference = reference_pipeline(root)
    plain = _reference_output(reference)
    reference.load_lora_weights(str(adapter), adapter_name="trained")
    _assert_registered(reference, "trained")
    _assert_adapter(root, reference, plain, lora=adapter)
