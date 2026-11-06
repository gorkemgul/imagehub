"""FLUX.1 LoRA-step and adapter-loading parity against diffusers, CPU float32."""

import sys
from pathlib import Path

import pytest
import torch
from helpers import write_tiny_flux1
from lora_helpers import TEXT_TARGETS, random_lora, write_peft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_flux_kontext import make_image, write_tiny_flux_snapshot  # noqa: E402

from imagehub import pipeline  # noqa: E402
from imagehub.backends.flux import _clip_prompt_embeds  # noqa: E402
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import Trainer, TrainingArguments
from imagehub.training.flux import Flux1Family

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
training_utils = pytest.importorskip("diffusers.training_utils")
PIL = pytest.importorskip("PIL.Image")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")

CPU = torch.device("cpu")
TARGETS = (
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
    "proj_mlp",
)


def batch(family, components, generator):
    image_generator = torch.Generator().manual_seed(12)
    image = PIL.fromarray(
        torch.randint(0, 256, (32, 32, 3), generator=image_generator, dtype=torch.uint8).numpy()
    )
    args = TrainingArguments(output_dir="unused", max_steps=1, resolution=32)
    return family.collator(components, args, generator=generator, train=True)(
        [{"image": image, "text": "a cat"}, {"image": image, "text": "a horse"}]
    )


def reference(root, targets):
    transformer = diffusers.FluxTransformer2DModel.from_pretrained(root, subfolder="transformer")
    vae = diffusers.AutoencoderKL.from_pretrained(root, subfolder="vae")
    clip = transformers.CLIPTextModel.from_pretrained(root, subfolder="text_encoder")
    t5 = transformers.T5EncoderModel.from_pretrained(root, subfolder="text_encoder_2")
    for module in (transformer, vae, clip, t5):
        module.requires_grad_(False)
    transformer.add_adapter(
        peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=targets)
    )
    transformer.train()
    return transformer, vae, clip, t5


def test_flux_transform_matches_the_script():
    resize = transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR)
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    generator = torch.Generator().manual_seed(3)
    for size in ((97, 64), (64, 130), (81, 80), (300, 211)):
        image = PIL.fromarray(
            torch.randint(
                0, 256, (*reversed(size), 3), generator=generator, dtype=torch.uint8
            ).numpy()
        )
        for center in (True, False):
            for seed in range(4):
                torch.manual_seed(seed)
                flip = bool(torch.rand(1) < 0.5)
                expected = resize(image)
                if flip:
                    expected = functional.hflip(expected)
                expected = (
                    transforms.CenterCrop(64)(expected)
                    if center
                    else functional.crop(
                        expected, *transforms.RandomCrop(64).get_params(expected, (64, 64))
                    )
                )
                expected = to_tensor(expected)
                from imagehub.training.data import preprocess_image_xl

                torch.manual_seed(seed)
                actual, _, _ = preprocess_image_xl(
                    image, resolution=64, center_crop=center, random_flip=True
                )
                assert torch.equal(actual, expected)


def test_flux_token_ids_match_the_script(tmp_path):
    root = write_tiny_flux1(tmp_path / "flux")
    family = Flux1Family()
    components = family.load(root, device=CPU, dtype=torch.float32, model_id="flux1-dev")
    text = "a photograph of an astronaut riding a horse " * 20
    collator = family.collator(
        components,
        TrainingArguments(output_dir=tmp_path / "out", max_steps=1, resolution=32),
        generator=torch.Generator().manual_seed(0),
        train=True,
    )
    image = PIL.fromarray(torch.zeros(32, 32, 3, dtype=torch.uint8).numpy())
    data = collator([{"image": image, "text": "a cat"}, {"image": image, "text": text}])
    ref = diffusers.FluxPipeline.from_pretrained(root, local_files_only=True)
    assert torch.equal(
        data["input_ids"],
        ref.tokenizer(
            ["a cat", text],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids,
    )
    assert torch.equal(
        data["input_ids_2"],
        ref.tokenizer_2(
            ["a cat", text],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids,
    )


@pytest.mark.parametrize("scheme", ("logit_normal", "mode", "sigma_sqrt", "cosmap", "none"))
@pytest.mark.parametrize("guidance_embeds", (False, True))
def test_first_flux_step_matches_diffusers(tmp_path, scheme, guidance_embeds):
    root = write_tiny_flux1(tmp_path / "flux", guidance_embeds=guidance_embeds)
    family = Flux1Family()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="flux1-dev")
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
    args = TrainingArguments(
        output_dir=tmp_path / "run", max_steps=1, resolution=32, weighting_scheme=scheme
    )
    objective = family.objective(ours, args)
    draws = objective.draw(data, generator)
    loss = objective.loss_from(data, draws)
    loss.backward()

    transformer, vae, clip, t5 = reference(root, TARGETS)
    with torch.no_grad():
        names = [
            f"{name}.lora_{letter}.default.weight"
            for name in injection.module_names
            for letter in ("A", "B")
        ]
        for mine, name in zip(injection.parameters(), names, strict=True):
            transformer.get_parameter(name).copy_(mine)
        pooled = clip(data["input_ids"]).pooler_output
        prompt = t5(data["input_ids_2"])[0]
        posterior = torch.randn(
            tuple(vae.encode(data["pixel_values"]).latent_dist.mean.shape), generator=replay
        )
        dist = vae.encode(data["pixel_values"]).latent_dist
        latents = (
            dist.mean + dist.std * posterior - vae.config.shift_factor
        ) * vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=replay)
        u = training_utils.compute_density_for_timestep_sampling(
            weighting_scheme=scheme,
            batch_size=len(latents),
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            generator=replay,
        )
        indices = (u * 1000).long()
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler"
        )
        timesteps = scheduler.timesteps[indices]
        sigmas = scheduler.sigmas[indices].view(len(latents), 1, 1, 1)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        packed = diffusers.FluxPipeline._pack_latents(noisy, len(latents), 4, 4, 4)
        image_ids = diffusers.FluxPipeline._prepare_latent_image_ids(
            len(latents), 2, 2, CPU, torch.float32
        )
        text_ids = torch.zeros(prompt.shape[1], 3)
        guidance = torch.tensor([3.5]).expand(len(latents)) if guidance_embeds else None
    prediction = transformer(
        hidden_states=packed,
        timestep=timesteps / 1000,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=prompt,
        txt_ids=text_ids,
        img_ids=image_ids,
        return_dict=False,
    )[0]
    prediction = diffusers.FluxPipeline._unpack_latents(prediction, 32, 32, 8)
    expected = torch.mean(
        (
            training_utils.compute_loss_weighting_for_sd3(scheme, sigmas).float()
            * (prediction.float() - (noise - latents).float()) ** 2
        ).reshape(2, -1),
        1,
    ).mean()
    expected.backward()
    assert torch.equal(loss, expected)
    assert torch.equal(draws["posterior"], posterior)
    assert torch.equal(draws["noise"], noise)
    assert torch.equal(draws["timesteps"], timesteps)
    reference_gradients = [transformer.get_parameter(name).grad for name in names]
    assert all(
        torch.equal(a.grad, b)
        for a, b in zip(injection.parameters(), reference_gradients, strict=True)
    )


def render(root, *, backend="flux", **options):
    with pipeline("t2i", backend=backend, model=root, **options) as pipe:
        return (
            pipe(
                "a cat", seed=42, steps=3, guidance=3.5, width=32, height=32, max_sequence_length=8
            )
            .images[0]
            .tensor
        )


def diffusers_flux_weights(components):
    """Transformers 5 FLUX LoRAs omit the native CLIP model's ``text_model`` segment."""
    weights = random_lora({"transformer": components.transformer}, TARGETS, seed=4, std=0.1)
    text = random_lora({"text_encoder": components.text_encoder}, TEXT_TARGETS, seed=5, std=0.1)
    return weights | {
        (component, name.removeprefix("text_model.")): value
        for (component, name), value in text.items()
    }


def reference_image(reference_pipeline, *, kontext):
    options = dict(
        generator=torch.Generator("cpu").manual_seed(42),
        num_inference_steps=3,
        guidance_scale=3.5,
        height=32,
        width=32,
        max_sequence_length=8,
        output_type="np",
    )
    result = (
        reference_pipeline(prompt="a cat", **options)
        if kontext
        else reference_pipeline("a cat", **options)
    )
    return torch.from_numpy((result.images * 255).round().astype("uint8"))[0]


def flux_adapters(root, tmp_path, *, backend):
    with pipeline("t2i", backend=backend, model=root) as pipe:
        components = pipe.backend.components
        weights = diffusers_flux_weights(components)
        second = random_lora({"transformer": components.transformer}, TARGETS, seed=7)
    return (
        write_peft(tmp_path / "first.safetensors", weights, alpha=3.0),
        write_peft(tmp_path / "second.safetensors", second, alpha=2.0),
    )


def native_pooled(root, **options):
    with pipeline("t2i", backend="flux", model=root, **options) as pipe:
        return _clip_prompt_embeds(pipe.backend.components, "a cat", CPU, 1)


def test_flux_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_flux1(tmp_path / "flux")
    first, second = flux_adapters(root, tmp_path, backend="flux")
    ref = diffusers.FluxPipeline.from_pretrained(root, local_files_only=True)
    ref.set_progress_bar_config(disable=True)
    plain = reference_image(ref, kontext=False)
    with torch.no_grad():
        plain_pooled = ref.encode_prompt("a cat", None, max_sequence_length=8)[1]
    ref.load_lora_weights(str(first), adapter_name="first")
    expected = reference_image(ref, kontext=False)
    assert not torch.equal(expected, plain)
    assert torch.equal(render(root, lora=first), expected)
    # The CLIP adapter really loads on both sides (pixels of the tiny model barely move).
    with torch.no_grad():
        pooled = ref.encode_prompt("a cat", None, max_sequence_length=8)[1]
    assert not torch.equal(pooled, plain_pooled)
    assert torch.equal(native_pooled(root, lora=first), pooled)
    ref.load_lora_weights(str(second), adapter_name="second")
    ref.set_adapters(["first", "second"], [0.8, 0.5])
    expected = reference_image(ref, kontext=False)
    assert torch.equal(render(root, lora=[(first, 0.8), (second, 0.5)]), expected)
    ref.fuse_lora()
    assert torch.equal(render(root, lora=[(first, 0.8), (second, 0.5)], lora_fuse=True), expected)


def test_kontext_lora_loading_matches_diffusers(tmp_path):
    root = write_tiny_flux_snapshot(tmp_path / "kontext")
    first, second = flux_adapters(root, tmp_path, backend="flux1-kontext-dev")
    image = make_image(64, 64, seed=0)
    ref = diffusers.FluxKontextPipeline.from_pretrained(root, local_files_only=True)
    ref.set_progress_bar_config(disable=True)

    def expected_pixels():
        pixels = ref(
            image=image,
            prompt="a cat",
            num_inference_steps=2,
            guidance_scale=3.5,
            height=32,
            width=32,
            max_area=32 * 32,
            max_sequence_length=8,
            generator=torch.Generator("cpu").manual_seed(0),
            output_type="np",
        ).images
        return torch.from_numpy((pixels * 255).round().astype("uint8"))[0]

    def native_pixels(**options):
        with pipeline("t2i", backend="flux1-kontext-dev", model=root, **options) as pipe:
            result = pipe(
                "a cat",
                image=image,
                steps=2,
                width=32,
                height=32,
                max_area=32 * 32,
                max_sequence_length=8,
                seed=0,
            )
        return result.images[0].tensor

    plain = expected_pixels()
    ref.load_lora_weights(str(first), adapter_name="first")
    expected = expected_pixels()
    assert not torch.equal(expected, plain)
    assert torch.equal(native_pixels(lora=first), expected)
    ref.load_lora_weights(str(second), adapter_name="second")
    ref.set_adapters(["first", "second"], [0.8, 0.5])
    stacked = [(first, 0.8), (second, 0.5)]
    assert torch.equal(native_pixels(lora=stacked), expected_pixels())
    ref.fuse_lora()
    assert torch.equal(native_pixels(lora=stacked, lora_fuse=True), expected_pixels())


def test_trained_flux_adapter_loads_in_diffusers(tmp_path):
    root = write_tiny_flux1(tmp_path / "flux")
    image = PIL.fromarray(
        torch.randint(
            0, 256, (32, 32, 3), generator=torch.Generator().manual_seed(9), dtype=torch.uint8
        ).numpy()
    )
    trainer = Trainer(
        "flux1-dev",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
        ),
        [{"image": image, "text": "a cat"}],
    )
    trainer.train()
    adapter = trainer.save_model(tmp_path / "adapter")
    ref = diffusers.FluxPipeline.from_pretrained(root, local_files_only=True)
    ref.set_progress_bar_config(disable=True)
    ref.load_lora_weights(str(adapter), adapter_name="trained")
    assert torch.equal(render(root, lora=adapter), reference_image(ref, kontext=False))


def test_trained_kontext_adapter_loads_in_diffusers(tmp_path):
    # Without condition images, the Kontext script trains the plain FLUX.1 step, with the
    # single-stream ``proj_mlp`` layers added to the default targets.
    root = write_tiny_flux_snapshot(tmp_path / "kontext")
    image = make_image(32, 32, seed=9)
    trainer = Trainer(
        "flux1-kontext-dev",
        root,
        TrainingArguments(
            output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
        ),
        [{"image": image, "text": "a cat"}],
    )
    trainer.train()
    adapter = trainer.save_model(tmp_path / "adapter")
    from safetensors.torch import load_file

    assert any(".proj_mlp." in key for key in load_file(adapter))
    condition = make_image(64, 64, seed=0)
    ref = diffusers.FluxKontextPipeline.from_pretrained(root, local_files_only=True)
    ref.set_progress_bar_config(disable=True)
    ref.load_lora_weights(str(adapter), adapter_name="trained")
    expected = ref(
        image=condition,
        prompt="a cat",
        num_inference_steps=2,
        guidance_scale=3.5,
        height=32,
        width=32,
        max_area=32 * 32,
        max_sequence_length=8,
        generator=torch.Generator("cpu").manual_seed(0),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    with pipeline("t2i", backend="flux1-kontext-dev", model=root, lora=adapter) as pipe:
        result = pipe(
            "a cat",
            image=condition,
            steps=2,
            width=32,
            height=32,
            max_area=32 * 32,
            max_sequence_length=8,
            seed=0,
        )
    assert torch.equal(result.images[0].tensor, expected)
