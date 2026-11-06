"""SD 3 LoRA training parity against diffusers' ``train_dreambooth_lora_sd3.py`` step, CPU fp32."""

import json

import pytest
import torch
from helpers import write_tiny_sd3
from lora_helpers import TEXT_TARGETS, random_lora, write_peft

from imagehub import pipeline
from imagehub.backends.stable_diffusion_3 import encode_prompt
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments
from imagehub.training.data import preprocess_image_xl
from imagehub.training.flow_matching import loss_weighting, sample_density, training_sigmas
from imagehub.training.sd3 import StableDiffusion3Family

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")
PIL = pytest.importorskip("PIL.Image")
training_utils = pytest.importorskip("diffusers.training_utils")

CPU = torch.device("cpu")
# The script's default target modules.
TARGETS = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]
SCHEMES = ("logit_normal", "mode", "sigma_sqrt", "cosmap")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_sd3(tmp_path_factory.mktemp("ref-train-sd3") / "sd3")


def images(count, sizes):
    g = torch.Generator().manual_seed(0)
    result = []
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        pixels = torch.randint(0, 256, (height, width, 3), generator=g, dtype=torch.uint8)
        result.append(PIL.frombytes("RGB", (width, height), bytes(pixels.flatten().tolist())))
    return result


def write_folder(root):
    root.mkdir(parents=True)
    lines = []
    for index, image in enumerate(images(3, ((40, 48), (48, 40)))):
        image.save(root / f"{index}.png")
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_training_sigmas_match_the_scheduler(shift):
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    timesteps, sigmas = training_sigmas(dict(scheduler.config))
    assert torch.equal(timesteps, scheduler.timesteps)
    assert torch.equal(sigmas, scheduler.sigmas)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_density_and_weighting_match_diffusers(scheme):
    options = dict(logit_mean=0.3, logit_std=1.2, mode_scale=1.29)
    expected = training_utils.compute_density_for_timestep_sampling(
        scheme, 64, generator=torch.Generator().manual_seed(3), **options
    )
    actual = sample_density(scheme, 64, generator=torch.Generator().manual_seed(3), **options)
    assert torch.equal(actual, expected)
    sigmas = torch.linspace(0.01, 1.0, 50).view(-1, 1, 1, 1)
    assert torch.equal(
        loss_weighting(scheme, sigmas),
        training_utils.compute_loss_weighting_for_sd3(scheme, sigmas),
    )


def test_sd3_transform_matches_the_script():
    resize = transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR)
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    for image in images(4, ((97, 64), (64, 130), (81, 80), (300, 211))):
        for center in (True, False):
            for seed in range(4):
                # DreamBoothDataset: resize, flip, then crop. The script tosses the flip coin
                # with ``random`` and RandomHorizontalFlip(p=1.0) draws one more torch number;
                # here the coin is the torch draw imagehub makes, before the crop offsets.
                torch.manual_seed(seed)
                flip = bool(torch.rand(1) < 0.5)
                resized = resize(image)
                if flip:
                    resized = functional.hflip(resized)
                if center:
                    cropped = transforms.CenterCrop(64)(resized)
                else:
                    y1, x1, h, w = transforms.RandomCrop(64).get_params(resized, (64, 64))
                    cropped = functional.crop(resized, y1, x1, h, w)
                expected = to_tensor(cropped)
                torch.manual_seed(seed)
                actual, _, _ = preprocess_image_xl(
                    image, resolution=64, center_crop=center, random_flip=True
                )
                assert torch.equal(actual, expected)


def test_sd3_token_ids_match_the_script(tiny):
    family = StableDiffusion3Family()
    ours = family.load(tiny, device=CPU, dtype=torch.float32, model_id="sd3-medium")
    args = TrainingArguments(output_dir=tiny / "unused", max_steps=1, resolution=32)
    collator = family.collator(ours, args, generator=torch.Generator().manual_seed(0), train=True)
    texts = ["a cat", "a photograph of an astronaut riding a horse " * 8]
    image = images(1, ((32, 32),))[0]
    batch = collator([{"image": image, "text": text} for text in texts])
    ref = reference_pipeline(tiny)
    # tokenize_prompt for both CLIP tokenizers; T5 at the script's max_sequence_length of 77.
    for key, tokenizer in (
        ("input_ids", ref.tokenizer),
        ("input_ids_2", ref.tokenizer_2),
        ("input_ids_3", ref.tokenizer_3),
    ):
        expected = tokenizer(
            texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        ).input_ids
        assert torch.equal(batch[key], expected), key


class Reference:
    def __init__(self, root):
        self.transformer = diffusers.SD3Transformer2DModel.from_pretrained(
            root, subfolder="transformer"
        )
        self.vae = diffusers.AutoencoderKL.from_pretrained(root, subfolder="vae")
        self.text_encoders = [
            transformers.CLIPTextModelWithProjection.from_pretrained(root, subfolder=name)
            for name in ("text_encoder", "text_encoder_2")
        ]
        self.text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
            root, subfolder="text_encoder_3"
        )
        self.scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            root, subfolder="scheduler"
        )
        for module in (self.transformer, self.vae, self.text_encoder_3, *self.text_encoders):
            module.requires_grad_(False)
        self.transformer.add_adapter(
            peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS)
        )
        self.transformer.train()

    def lora_parameters(self, names):
        for name in names:
            layer = self.transformer.get_submodule(name)
            yield layer.lora_A["default"].weight
            yield layer.lora_B["default"].weight

    def encode_prompt(self, batch):
        # encode_prompt with text_input_ids_list, as in the script.
        clip_embeds, pooled = [], []
        for encoder, key in zip(self.text_encoders, ("input_ids", "input_ids_2"), strict=True):
            output = encoder(batch[key], output_hidden_states=True)
            pooled.append(output[0])
            clip_embeds.append(output.hidden_states[-2])
        clip = torch.cat(clip_embeds, dim=-1)
        t5 = self.text_encoder_3(batch["input_ids_3"])[0]
        clip = torch.nn.functional.pad(clip, (0, t5.shape[-1] - clip.shape[-1]))
        return torch.cat([clip, t5], dim=-2), torch.cat(pooled, dim=-1)

    def step(self, batch, generator, *, scheme, precondition):
        """The script's training step; the draws replay ``generator`` in our order."""
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(batch)
            dist = self.vae.encode(batch["pixel_values"]).latent_dist
            posterior = torch.randn(dist.mean.shape, generator=generator)
            model_input = dist.mean + dist.std * posterior
        config = self.vae.config
        model_input = (model_input - config.shift_factor) * config.scaling_factor
        noise = torch.randn(model_input.shape, generator=generator)
        u = training_utils.compute_density_for_timestep_sampling(
            weighting_scheme=scheme,
            batch_size=model_input.shape[0],
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            generator=generator,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices]
        # get_sigmas
        step_indices = [(self.scheduler.timesteps == t).nonzero().item() for t in timesteps]
        sigmas = self.scheduler.sigmas[step_indices].flatten()
        while len(sigmas.shape) < model_input.ndim:
            sigmas = sigmas.unsqueeze(-1)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        if precondition:
            model_pred = model_pred * (-sigmas) + noisy_model_input
        weighting = training_utils.compute_loss_weighting_for_sd3(
            weighting_scheme=scheme, sigmas=sigmas
        )
        target = model_input if precondition else noise - model_input
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        draws = {"posterior": posterior, "noise": noise, "timesteps": timesteps}
        return loss.mean(), draws


@pytest.mark.parametrize(
    ("scheme", "precondition"), [(scheme, True) for scheme in SCHEMES] + [("logit_normal", False)]
)
def test_first_sd3_step_matches(tmp_path, tiny, scheme, precondition):
    family = StableDiffusion3Family()
    ours = family.load(tiny, device=CPU, dtype=torch.float32, model_id="sd3-medium")
    for module in [ours.transformer, *family.frozen(ours)]:
        module.requires_grad_(False)
    injection = inject_lora(
        ours.transformer,
        LoRAConfig(rank=4, alpha=4, target_modules=tuple(TARGETS), init="gaussian"),
    )
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in injection.parameters():
            if not parameter.any():
                parameter.normal_(0, 0.02, generator=g)
    ours.transformer.train()
    reference = Reference(tiny)
    with torch.no_grad():
        for mine, theirs in zip(
            injection.parameters(), reference.lora_parameters(injection.module_names), strict=True
        ):
            theirs.copy_(mine)
    args = TrainingArguments(
        output_dir=tmp_path / "run",
        max_steps=1,
        resolution=32,
        weighting_scheme=scheme,
        precondition_outputs=precondition,
    )
    objective = family.objective(ours, args)
    collator = family.collator(ours, args, generator=g, train=True)
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    batch = collator([data[0], data[1]])
    replay = torch.Generator()
    replay.set_state(g.get_state())
    draws = objective.draw(batch, g)
    loss = objective.loss_from(batch, draws)
    loss.backward()
    expected, expected_draws = reference.step(
        batch, replay, scheme=scheme, precondition=precondition
    )
    expected.backward()
    for key, value in expected_draws.items():
        assert torch.equal(draws[key], value), key
    assert torch.equal(loss, expected)
    for mine, theirs in zip(
        injection.parameters(), reference.lora_parameters(injection.module_names), strict=True
    ):
        assert torch.equal(mine.grad, theirs.grad)


def reference_images(ref, prompt="a cat"):
    images = ref(
        prompt,
        negative_prompt="blurry",
        num_inference_steps=3,
        guidance_scale=4.0,
        height=32,
        width=32,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    return torch.from_numpy((images * 255).round().astype("uint8"))


def native_images(root, **options):
    with pipeline("t2i", backend="sd3-medium", model=root, **options) as pipe:
        result = pipe(
            "a cat", negative_prompt="blurry", steps=3, guidance=4.0, height=32, width=32, seed=42
        )
    return torch.stack([image.tensor for image in result.images])


def reference_pipeline(root):
    ref = diffusers.StableDiffusion3Pipeline.from_pretrained(root, local_files_only=True)
    ref.set_progress_bar_config(disable=True)
    return ref


def random_sd3_adapter(root, *, seed=0, std=0.1):
    with pipeline("t2i", backend="sd3-medium", model=root) as pipe:
        c = pipe.backend.components
        weights = random_lora(
            {"transformer": c.transformer}, (*TARGETS, "ff.net.0.proj", "ff.net.2"), seed=seed
        )
        for name in ("text_encoder", "text_encoder_2"):
            # The SD3 loader, unlike the SD / SDXL one, keeps the ``text_model`` level.
            weights |= random_lora({name: getattr(c, name)}, TEXT_TARGETS, seed=seed + 1, std=std)
    return weights


def native_prompt_embeds(root, **options):
    with pipeline("t2i", backend="sd3-medium", model=root, **options) as pipe:
        return encode_prompt(
            pipe.backend.components,
            "a cat",
            "blurry",
            CPU,
            1,
            max_sequence_length=8,
            do_classifier_free_guidance=True,
        )


def reference_prompt_embeds(ref):
    with torch.no_grad():
        return ref.encode_prompt(
            "a cat",
            None,
            None,
            negative_prompt="blurry",
            do_classifier_free_guidance=True,
            max_sequence_length=8,
        )


def test_sd3_lora_generation_matches_diffusers(tmp_path, tiny):
    path = write_peft(tmp_path / "a.safetensors", random_sd3_adapter(tiny), alpha=3.0)
    ref = reference_pipeline(tiny)
    plain = reference_images(ref)
    plain_embeds = reference_prompt_embeds(ref)
    ref.load_lora_weights(str(path), adapter_name="a")
    expected = reference_images(ref)
    expected_embeds = reference_prompt_embeds(ref)
    assert not torch.equal(expected, plain)
    assert not torch.equal(expected_embeds[2], plain_embeds[2])
    assert torch.equal(native_images(tiny, lora=path), expected)
    for actual, wanted in zip(native_prompt_embeds(tiny, lora=path), expected_embeds, strict=True):
        assert torch.equal(actual, wanted)


def test_stacked_and_fused_sd3_adapters_match_diffusers(tmp_path, tiny):
    first = write_peft(tmp_path / "a.safetensors", random_sd3_adapter(tiny), alpha=3.0)
    second = write_peft(tmp_path / "b.safetensors", random_sd3_adapter(tiny, seed=7))
    ref = reference_pipeline(tiny)
    ref.load_lora_weights(str(first), adapter_name="a")
    ref.load_lora_weights(str(second), adapter_name="b")
    ref.set_adapters(["a", "b"], [0.8, 0.5])
    stacked = [(first, 0.8), (second, 0.5)]
    assert torch.equal(native_images(tiny, lora=stacked), reference_images(ref))
    ref.fuse_lora()
    assert torch.equal(native_images(tiny, lora=stacked, lora_fuse=True), reference_images(ref))


def test_saved_sd3_adapter_matches_in_diffusers(tmp_path, tiny):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = TrainingArguments(
        output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
    )
    trainer = Trainer("sd3-medium", tiny, args, data)
    trainer.train()
    path = trainer.save_model(tmp_path / "final")
    ref = reference_pipeline(tiny)
    plain = reference_images(ref)
    ref.load_lora_weights(str(path), adapter_name="trained")
    expected = reference_images(ref)
    assert not torch.equal(expected, plain)
    assert torch.equal(native_images(tiny, lora=path), expected)
