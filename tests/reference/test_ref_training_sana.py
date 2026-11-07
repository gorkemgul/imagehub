"""Sana LoRA-step parity against diffusers' training script, CPU float32."""

import pytest
import torch
from helpers import write_tiny_sana
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments
from imagehub.training.sana import (
    LOGIT_MEAN,
    LOGIT_STD,
    MAX_SEQUENCE_LENGTH,
    MODE_SCALE,
    WEIGHTING_SCHEME,
    SanaFamily,
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
TARGETS = ("to_k", "to_q", "to_v")
# The Sana training script's attention targets plus every other Linear suffix in the tiny
# transformer. SanaPipeline delegates these generic PEFT keys to ``load_lora_adapter``.
ADAPTER_TARGETS = TARGETS + ("to_out.0", "linear", "linear_1", "linear_2", "proj_out")
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none")
INSTRUCTION = ("Enhance: ",)


def batch(family, components, generator):
    image = PIL.fromarray(
        torch.randint(
            0, 256, (32, 32, 3), generator=torch.Generator().manual_seed(12), dtype=torch.uint8
        ).numpy()
    )
    args = TrainingArguments(output_dir="unused", max_steps=1, resolution=32, max_sequence_length=8)
    return family.collator(components, args, generator=generator, train=True)(
        [{"image": image, "text": "a cat"}, {"image": image, "text": "a horse"}]
    )


def reference(root):
    transformer = diffusers.SanaTransformer2DModel.from_pretrained(root, subfolder="transformer")
    vae = diffusers.AutoencoderDC.from_pretrained(root, subfolder="vae")
    text_encoder = transformers.Gemma2Model.from_pretrained(root, subfolder="text_encoder")
    for module in (transformer, vae, text_encoder):
        module.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS)
    )
    transformer.train()
    return transformer, vae, text_encoder


def test_sana_default_arguments_give_the_script_defaults(tmp_path):
    # train_dreambooth_lora_sana.py argparse: weighting_scheme "none", logit_mean 0.0,
    # logit_std 1.0, mode_scale 1.29, max_sequence_length 300, complex_human_instruction None,
    # rank 4, lora_alpha 4; target modules to_k, to_q, to_v.
    from imagehub.training import get_training_spec

    root = write_tiny_sana(tmp_path / "sana")
    args = TrainingArguments(output_dir=tmp_path / "run", max_steps=1, resolution=32)
    family = SanaFamily()
    components = family.load(
        root, device=torch.device("cpu"), dtype=torch.float32, model_id="sana-600m-1024"
    )
    objective = family.objective(components, args)
    collator = family.collator(components, args, generator=torch.Generator(), train=True)
    assert (
        objective.weighting_scheme,
        objective.logit_mean,
        objective.logit_std,
        objective.mode_scale,
        objective.precondition_outputs,
    ) == ("none", 0.0, 1.0, 1.29, False)
    assert objective.max_sequence_length == collator.max_sequence_length == 300
    assert args.complex_human_instruction is None
    assert (args.lora_rank, args.lora_alpha) == (4, 4)
    assert set(get_training_spec("sana-600m-1024").target_modules) == {"to_k", "to_q", "to_v"}
    assert (WEIGHTING_SCHEME, LOGIT_MEAN, LOGIT_STD, MODE_SCALE, MAX_SEQUENCE_LENGTH) == (
        "none",
        0.0,
        1.0,
        1.29,
        300,
    )


def test_sana_transform_and_prompt_ids_match_the_script(tmp_path):
    from imagehub.training.data import preprocess_image_xl

    image = PIL.fromarray(
        torch.randint(
            0, 256, (97, 64, 3), generator=torch.Generator().manual_seed(3), dtype=torch.uint8
        ).numpy()
    )
    expected = transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR)(image)
    torch.manual_seed(4)
    expected = functional.crop(expected, *transforms.RandomCrop(32).get_params(expected, (32, 32)))
    expected = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(
        expected
    )
    torch.manual_seed(4)
    actual, _, _ = preprocess_image_xl(image, resolution=32, center_crop=False, random_flip=False)
    # Resize/crop do not draw from the explicit generator when flipping is disabled.
    assert torch.equal(actual, expected)

    root = write_tiny_sana(tmp_path / "sana")
    family = SanaFamily()
    components = family.load(root, device=CPU, dtype=torch.float32, model_id="sana-600m-1024")
    data = batch(family, components, torch.Generator().manual_seed(0))
    reference_pipeline = diffusers.SanaPipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.tokenizer.padding_side = "right"
    expected_tokens = reference_pipeline.tokenizer(
        ["a cat", "a horse"],
        padding="max_length",
        max_length=8,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    assert torch.equal(data["input_ids"], expected_tokens.input_ids)
    assert torch.equal(data["attention_mask"], expected_tokens.attention_mask.bool())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_first_sana_step_matches_diffusers(tmp_path, scheme):
    root = write_tiny_sana(tmp_path / "sana")
    family = SanaFamily()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="sana-600m-1024")
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
            output_dir=tmp_path / "run",
            max_steps=1,
            resolution=32,
            max_sequence_length=8,
            weighting_scheme=scheme,
        ),
    )
    draws = objective.draw(data, generator)
    loss = objective.loss_from(data, draws)
    loss.backward()

    transformer, vae, text_encoder = reference(root)
    names = [
        f"{name}.lora_{letter}.default.weight"
        for name in injection.module_names
        for letter in ("A", "B")
    ]
    with torch.no_grad():
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        reference_pipeline = diffusers.SanaPipeline.from_pretrained(
            root, transformer=None, vae=None, text_encoder=text_encoder
        )
        prompt_embeds, prompt_mask, _, _ = reference_pipeline.encode_prompt(
            ["a cat", "a horse"], do_classifier_free_guidance=False, max_sequence_length=8
        )
        latents = vae.encode(data["pixel_values"]).latent * vae.config.scaling_factor
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
        noisy = (1.0 - sigmas) * latents + sigmas * noise
    prediction = transformer(
        hidden_states=noisy,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_mask,
        timestep=timesteps,
        return_dict=False,
    )[0]
    expected = torch.mean(
        (
            training_utils.compute_loss_weighting_for_sd3(scheme, sigmas).float()
            * (prediction.float() - (noise - latents).float()) ** 2
        ).reshape(len(latents), -1),
        1,
    ).mean()
    expected.backward()

    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    assert torch.equal(loss, expected)
    assert all(
        torch.equal(mine.grad, transformer.get_parameter(name).grad)
        for mine, name in zip(injection.parameters(), names, strict=True)
    )


def reference_image(reference_pipeline):
    images = reference_pipeline(
        "a cat",
        negative_prompt="",
        num_inference_steps=3,
        guidance_scale=4.0,
        height=32,
        width=32,
        use_resolution_binning=False,
        max_sequence_length=8,
        complex_human_instruction=list(INSTRUCTION),
        generator=torch.Generator("cpu").manual_seed(42),
        clean_caption=False,
        output_type="np",
    ).images
    return torch.from_numpy((images * 255).round().astype("uint8"))[0]


def native_image(root, **options):
    with pipeline("t2i", backend="sana", model=root, **options) as pipe:
        result = pipe(
            "a cat",
            negative_prompt="",
            steps=3,
            guidance=4.0,
            height=32,
            width=32,
            use_resolution_binning=False,
            max_sequence_length=8,
            complex_human_instruction=INSTRUCTION,
            seed=42,
        )
    return result.images[0].tensor


def sana_adapters(root, tmp_path):
    with pipeline("t2i", backend="sana", model=root) as pipe:
        weights = random_lora(
            {"transformer": pipe.backend.components.transformer}, ADAPTER_TARGETS, seed=4, std=0.1
        )
        second = random_lora(
            {"transformer": pipe.backend.components.transformer}, ADAPTER_TARGETS, seed=7
        )
    return (
        write_peft(tmp_path / "first.safetensors", weights, alpha=3.0),
        write_peft(tmp_path / "second.safetensors", second, alpha=2.0),
    )


def test_sana_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_sana(tmp_path / "sana")
    first, second = sana_adapters(root, tmp_path)
    reference_pipeline = diffusers.SanaPipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.set_progress_bar_config(disable=True)
    plain = reference_image(reference_pipeline)
    reference_pipeline.load_lora_weights(str(first), adapter_name="first")
    assert reference_pipeline.transformer.peft_config["first"].target_modules == set(
        ADAPTER_TARGETS
    )
    expected = reference_image(reference_pipeline)
    assert not torch.equal(expected, plain)
    assert torch.equal(native_image(root, lora=first), expected)
    reference_pipeline.load_lora_weights(str(second), adapter_name="second")
    reference_pipeline.set_adapters(["first", "second"], [0.8, 0.5])
    stacked = [(first, 0.8), (second, 0.5)]
    assert torch.equal(native_image(root, lora=stacked), reference_image(reference_pipeline))
    reference_pipeline.fuse_lora()
    assert torch.equal(
        native_image(root, lora=stacked, lora_fuse=True), reference_image(reference_pipeline)
    )


def test_trained_sana_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_sana(tmp_path / "sana")
    image = PIL.fromarray(
        torch.randint(
            0, 256, (32, 32, 3), generator=torch.Generator().manual_seed(9), dtype=torch.uint8
        ).numpy()
    )
    trainer = Trainer(
        "sana-600m-1024",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
        ),
        [{"image": image, "text": "a cat"}],
    )
    trainer.train()
    adapter = trainer.save_model(tmp_path / "adapter")
    reference_pipeline = diffusers.SanaPipeline.from_pretrained(root, local_files_only=True)
    reference_pipeline.set_progress_bar_config(disable=True)
    plain = reference_image(reference_pipeline)
    reference_pipeline.load_lora_weights(str(adapter), adapter_name="trained")
    expected = reference_image(reference_pipeline)
    assert not torch.equal(expected, plain)
    assert torch.equal(native_image(root, lora=adapter), expected)
