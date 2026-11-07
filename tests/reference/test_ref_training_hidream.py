"""HiDream-I1 LoRA-step and adapter parity against diffusers, CPU float32."""

import math

import pytest
import torch
from helpers import write_tiny_hidream
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.backends.hidream import HiDreamBackend, decode_latents, denoise
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel
from imagehub.training import Trainer, TrainingArguments
from imagehub.training.hidream import (
    LOGIT_MEAN,
    LOGIT_STD,
    MAX_SEQUENCE_LENGTH,
    MODE_SCALE,
    WEIGHTING_SCHEME,
    HiDreamFamily,
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
TARGETS = ("to_k", "to_q", "to_v", "to_out")
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
        output_dir=root, max_steps=1, resolution=16, max_sequence_length=8, **kwargs
    )


def reference_text_pipe(root):
    tokenizer_4 = transformers.AutoTokenizer.from_pretrained(root / "text_encoder_4")
    tokenizer_4.pad_token = tokenizer_4.eos_token
    return diffusers.HiDreamImagePipeline.from_pretrained(
        root,
        transformer=None,
        vae=None,
        tokenizer=transformers.CLIPTokenizer.from_pretrained(root / "tokenizer"),
        text_encoder=transformers.CLIPTextModelWithProjection.from_pretrained(
            root / "text_encoder"
        ),
        tokenizer_2=transformers.CLIPTokenizer.from_pretrained(root / "tokenizer_2"),
        text_encoder_2=transformers.CLIPTextModelWithProjection.from_pretrained(
            root / "text_encoder_2"
        ),
        tokenizer_3=transformers.T5Tokenizer.from_pretrained(root / "tokenizer_3"),
        text_encoder_3=transformers.T5EncoderModel.from_pretrained(root / "text_encoder_3"),
        tokenizer_4=tokenizer_4,
        text_encoder_4=transformers.LlamaForCausalLM.from_pretrained(
            root / "text_encoder_4", output_hidden_states=True, output_attentions=True
        ),
        local_files_only=True,
    )


def reference_pipeline(root):
    tokenizer_4 = transformers.AutoTokenizer.from_pretrained(root / "text_encoder_4")
    tokenizer_4.pad_token = tokenizer_4.eos_token
    pipeline = diffusers.HiDreamImagePipeline.from_pretrained(
        root,
        tokenizer_4=tokenizer_4,
        text_encoder_4=transformers.LlamaForCausalLM.from_pretrained(
            root / "text_encoder_4", output_hidden_states=True, output_attentions=True
        ),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def test_hidream_defaults_match_the_script(tmp_path):
    root = write_tiny_hidream(tmp_path / "hidream")
    training_args = TrainingArguments(output_dir=tmp_path / "run", max_steps=1, resolution=16)
    trainer = Trainer("hidream-i1-dev", root, training_args, [{"image": image(), "text": "a cat"}])
    trainer._setup()
    objective = trainer.objective
    assert (
        objective.weighting_scheme,
        objective.logit_mean,
        objective.logit_std,
        objective.mode_scale,
        objective.precondition_outputs,
    ) == ("none", 0.0, 1.0, 1.29, False)
    assert trainer.collator.max_sequence_length == MAX_SEQUENCE_LENGTH == 128
    assert (training_args.lora_rank, training_args.lora_alpha) == (None, None)
    assert (trainer.spec.lora_rank, trainer.spec.lora_alpha) == (4, 4.0)
    assert (WEIGHTING_SCHEME, LOGIT_MEAN, LOGIT_STD, MODE_SCALE) == ("none", 0.0, 1.0, 1.29)


def test_hidream_transform_and_prompt_ids_match_the_pipeline(tmp_path):
    source = image(3, height=97, width=64)
    expected = functional.resize(source, 16, interpolation=transforms.InterpolationMode.BILINEAR)
    torch.manual_seed(4)
    expected = functional.crop(expected, *transforms.RandomCrop.get_params(expected, (16, 16)))
    expected = functional.normalize(functional.to_tensor(expected), [0.5], [0.5])

    root = write_tiny_hidream(tmp_path / "hidream")
    family = HiDreamFamily()
    components = family.load(root, device=CPU, dtype=torch.float32, model_id="hidream-i1-dev")
    batch = family.collator(
        components, args(tmp_path / "run"), generator=torch.Generator().manual_seed(4), train=True
    )([{"image": source, "text": "a cat"}])
    assert torch.equal(batch["pixel_values"][0], expected)

    pipe = reference_text_pipe(root)
    for tokenizer, ids, mask, length in (
        (pipe.tokenizer, batch["input_ids"], None, 8),
        (pipe.tokenizer_2, batch["input_ids_2"], None, 8),
        (pipe.tokenizer_3, batch["input_ids_3"], batch["attention_mask_3"], 8),
        (pipe.tokenizer_4, batch["input_ids_4"], batch["attention_mask_4"], 8),
    ):
        inputs = tokenizer(
            ["a cat"],
            padding="max_length",
            max_length=length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        assert torch.equal(ids, inputs.input_ids)
        if mask is not None:
            assert torch.equal(mask, inputs.attention_mask.bool())


@pytest.mark.parametrize("scheme", SCHEMES)
def test_hidream_first_step_matches_diffusers(tmp_path, scheme):
    root = write_tiny_hidream(tmp_path / "hidream")
    family = HiDreamFamily()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="hidream-i1-dev")
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

    transformer = diffusers.HiDreamImageTransformer2DModel.from_pretrained(
        root, subfolder="transformer", force_inference_output=True
    )
    transformer.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS)
    )
    transformer.train()
    names = [
        f"{name}.lora_{letter}.default.weight"
        for name in injection.module_names
        for letter in ("A", "B")
    ]
    with torch.no_grad():
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        pipe = reference_text_pipe(root)
        t5, _, llama, _, pooled, _ = pipe.encode_prompt(
            prompt=["a cat", "a horse"],
            device=CPU,
            do_classifier_free_guidance=False,
            max_sequence_length=8,
        )
        vae = diffusers.AutoencoderKL.from_pretrained(root, subfolder="vae")
        posterior = vae.encode(batch["pixel_values"]).latent_dist
        posterior_state = replay.get_state()
        sample = posterior.sample(generator=replay)
        posterior_noise = torch.randn(
            posterior.mean.shape, generator=torch.Generator().set_state(posterior_state)
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
    prediction = -transformer(
        latents * (1.0 - sigmas) + noise * sigmas,
        timesteps=timesteps,
        encoder_hidden_states_t5=t5,
        encoder_hidden_states_llama3=llama,
        pooled_embeds=pooled,
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
    assert torch.equal(draws["posterior"], posterior_noise)
    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    assert torch.equal(draws["sigmas"], scheduler.sigmas[indices])
    assert torch.equal(loss, expected)
    assert all(
        torch.equal(mine.grad, transformer.get_parameter(name).grad)
        for mine, name in zip(injection.parameters(), names, strict=True)
    )
    registered = {
        name.removesuffix(".lora_A.default.weight")
        for name, _ in transformer.named_parameters()
        if name.endswith(".lora_A.default.weight")
    }
    assert set(injection.module_names) == registered
    assert not any(name.endswith(("to_q_t", "to_k_t", "to_v_t", "to_out_t")) for name in registered)


def _reference_output(reference_pipeline):
    latents = torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(42))
    common = dict(
        prompt="a cat",
        negative_prompt="",
        num_inference_steps=3,
        guidance_scale=4.0,
        height=256,
        width=256,
        max_sequence_length=8,
        latents=latents,
    )
    torch.manual_seed(42)
    final = reference_pipeline(output_type="latent", **common).images
    torch.manual_seed(42)
    pixels = reference_pipeline(output_type="np", **common).images
    return final, torch.from_numpy((pixels * 255).round().astype("uint8"))


def _native_output(root, **options):
    backend = HiDreamBackend(root, **options).load()
    try:
        assert backend.components is not None
        final = denoise(
            backend.components,
            prompt="a cat",
            negative_prompt="",
            steps=3,
            guidance=4.0,
            latents=torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(42)),
            max_sequence_length=8,
            noise_generator=torch.Generator().manual_seed(42),
            lora=backend.lora,
            lora_fuse=backend.lora_fuse,
        )
        return final, decode_latents(backend.components.vae, final)
    finally:
        backend.unload()


def _assert_registered_targets(reference_pipeline, name):
    registered = set(reference_pipeline.transformer.peft_config[name].target_modules)
    assert registered
    assert all(any(path.endswith(target) for target in TARGETS) for path in registered)
    assert {
        target for target in TARGETS if any(path.endswith(target) for path in registered)
    } == set(TARGETS)
    assert not any(path.endswith(f"{target}_t") for path in registered for target in TARGETS)


def _assert_native_matches(root, reference_pipeline, plain, **options):
    expected = _reference_output(reference_pipeline)
    assert not torch.equal(expected[0], plain[0])
    actual = _native_output(root, **options)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_hidream_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_hidream(tmp_path / "hidream")
    with pipeline("t2i", backend="hidream-i1-dev", model=root) as pipe:
        assert pipe.backend.components is not None
        transformer = pipe.backend.components._from_pretrained(
            HiDreamImageTransformer2DModel, "transformer"
        )
        first = random_lora({"transformer": transformer}, TARGETS, seed=4)
        second = random_lora({"transformer": transformer}, TARGETS, seed=7)
    first = write_peft(tmp_path / "first.safetensors", first, alpha=3.0)
    second = write_peft(tmp_path / "second.safetensors", second, alpha=2.0)
    reference = reference_pipeline(root)
    plain = _reference_output(reference)
    reference.load_lora_weights(str(first), adapter_name="first")
    _assert_registered_targets(reference, "first")
    _assert_native_matches(root, reference, plain, lora=first)
    reference.load_lora_weights(str(second), adapter_name="second")
    _assert_registered_targets(reference, "second")
    reference.set_adapters(["first", "second"], [0.8, 0.5])
    _assert_native_matches(root, reference, plain, lora=[(first, 0.8), (second, 0.5)])
    reference.fuse_lora()
    _assert_native_matches(
        root, reference, plain, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True
    )


def test_trained_hidream_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_hidream(tmp_path / "hidream")
    trainer = Trainer(
        "hidream-i1-dev",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run",
            max_steps=3,
            resolution=16,
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
    _assert_registered_targets(reference, "trained")
    _assert_native_matches(root, reference, plain, lora=adapter)
