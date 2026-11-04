"""LoRA training parity against diffusers' ``train_text_to_image_lora.py`` step (peft), CPU fp32."""

import json

import pytest
import torch
from helpers import checkpoint_dir, max_diff, write_tiny_sd

from imagehub import pipeline
from imagehub.backends.stable_diffusion import StableDiffusionComponents
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import (
    ImageFolderDataset,
    StableDiffusionLoRAObjective,
    Trainer,
    TrainingArguments,
    preprocess_image,
)
from imagehub.training.schedules import lr_lambda

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
transforms = pytest.importorskip("torchvision.transforms")
PIL = pytest.importorskip("PIL.Image")

CPU = torch.device("cpu")
TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def random_images(count, sizes):
    g = torch.Generator().manual_seed(0)
    images = []
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        pixels = torch.randint(0, 256, (height, width, 3), generator=g, dtype=torch.uint8)
        images.append(PIL.fromarray(pixels.numpy()))
    return images


def test_preprocessing_matches_torchvision():
    images = random_images(4, ((97, 64), (64, 130), (80, 80), (300, 211)))
    for center in (True, False):
        reference = transforms.Compose(
            [
                transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(64) if center else transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for image in images:
            torch.manual_seed(3)
            expected = reference(image)
            torch.manual_seed(3)
            actual = preprocess_image(image, resolution=64, center_crop=center, random_flip=True)
            assert torch.equal(actual, expected)


def test_lr_schedules_match_diffusers():
    from diffusers.optimization import get_scheduler

    for name in ("constant", "constant_with_warmup", "linear", "cosine"):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = get_scheduler(name, optimizer, num_warmup_steps=3, num_training_steps=20)
        multiplier = lr_lambda(name, 3, 20)
        for step in range(20):
            assert scheduler.get_last_lr()[0] == multiplier(step), (name, step)
            optimizer.step()
            scheduler.step()


class Reference:
    """The diffusers components of a snapshot with the script's LoRA adapter on the U-Net."""

    def __init__(self, root, rank):
        self.unet = diffusers.UNet2DConditionModel.from_pretrained(root, subfolder="unet")
        self.vae = diffusers.AutoencoderKL.from_pretrained(root, subfolder="vae")
        self.text_encoder = transformers.CLIPTextModel.from_pretrained(
            root, subfolder="text_encoder"
        )
        self.scheduler = diffusers.DDPMScheduler.from_pretrained(root, subfolder="scheduler")
        for module in (self.unet, self.vae, self.text_encoder):
            module.requires_grad_(False)
        self.unet.add_adapter(
            peft.LoraConfig(
                r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=TARGETS
            )
        )
        self.unet.train()

    def lora_parameters(self, names):
        for name in names:
            layer = self.unet.get_submodule(name)
            yield layer.lora_A["default"].weight
            yield layer.lora_B["default"].weight

    def copy_from(self, injection):
        with torch.no_grad():
            for ours, theirs in zip(
                injection.parameters(), self.lora_parameters(injection.module_names), strict=True
            ):
                theirs.copy_(ours)

    def loss(self, pixels, ids, posterior, noise, timesteps, *, prediction_type, snr_gamma):
        with torch.no_grad():
            dist = self.vae.encode(pixels).latent_dist
            latents = (dist.mean + dist.std * posterior) * self.vae.config.scaling_factor
            hidden = self.text_encoder(ids, return_dict=False)[0]
        scheduler = self.scheduler
        scheduler.register_to_config(prediction_type=prediction_type)
        noisy = scheduler.add_noise(latents, noise, timesteps)
        if prediction_type == "epsilon":
            target = noise
        else:
            target = scheduler.get_velocity(latents, noise, timesteps)
        prediction = self.unet(noisy, timesteps, hidden, return_dict=False)[0]
        if snr_gamma is None:
            return torch.nn.functional.mse_loss(prediction.float(), target.float())
        from diffusers.training_utils import compute_snr

        snr = compute_snr(scheduler, timesteps)
        weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        weights = weights / snr if prediction_type == "epsilon" else weights / (snr + 1)
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none")
        return (loss.mean(dim=list(range(1, len(loss.shape)))) * weights).mean()


def first_step(root, *, resolution, prediction_type, snr_gamma, record=None):
    ours = StableDiffusionComponents.load(root, device=CPU, dtype=torch.float32)
    for module in (ours.unet, ours.vae, ours.text_encoder):
        module.requires_grad_(False)
    injection = inject_lora(
        ours.unet, LoRAConfig(rank=4, alpha=4, target_modules=tuple(TARGETS), init="gaussian")
    )
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in injection.parameters():
            if not parameter.any():
                parameter.normal_(0, 0.02, generator=g)
    ours.unet.train()
    reference = Reference(root, rank=4)
    reference.copy_from(injection)
    objective = StableDiffusionLoRAObjective(
        ours, prediction_type=prediction_type, snr_gamma=snr_gamma
    )
    pixels = torch.rand(2, 3, resolution, resolution, generator=g) * 2 - 1
    ids = torch.tensor([ours.tokenizer("a cat"), ours.tokenizer("a dog in the snow")])
    posterior, noise, timesteps = objective.draw(pixels, g)
    latents = objective.encode_images(pixels, posterior)
    loss = objective.loss(latents, objective.encode_text(ids), noise, timesteps)
    loss.backward()
    expected = reference.loss(
        pixels,
        ids,
        posterior,
        noise,
        timesteps,
        prediction_type=prediction_type,
        snr_gamma=snr_gamma,
    )
    expected.backward()
    if record is not None:
        record("training_first_step_loss", **max_diff(loss.detach(), expected.detach()))
    assert torch.equal(loss, expected)
    for ours_param, theirs in zip(
        injection.parameters(),
        reference.lora_parameters(injection.module_names),
        strict=True,
    ):
        assert torch.equal(ours_param.grad, theirs.grad)


@pytest.mark.parametrize(
    ("prediction_type", "snr_gamma"),
    [("epsilon", None), ("v_prediction", None), ("epsilon", 5.0), ("v_prediction", 5.0)],
)
def test_first_step_loss_and_gradients_match(tmp_path, prediction_type, snr_gamma):
    root = write_tiny_sd(tmp_path / "tiny")
    first_step(root, resolution=32, prediction_type=prediction_type, snr_gamma=snr_gamma)


def test_first_step_matches_on_sd_v1_5(record):
    root = checkpoint_dir(5)
    first_step(root, resolution=256, prediction_type="epsilon", snr_gamma=None, record=record)


class Recording:
    """Wrap the trainer's objective to keep each step's tensors for replay."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __call__(self, pixels, ids, generator):
        posterior, noise, timesteps = self.inner.draw(pixels, generator)
        self.calls.append((pixels.clone(), ids.clone(), posterior, noise, timesteps))
        latents = self.inner.encode_images(pixels, posterior)
        return self.inner.loss(latents, self.inner.encode_text(ids), noise, timesteps)


def write_folder(root):
    root.mkdir(parents=True)
    lines = []
    for index, image in enumerate(random_images(3, ((40, 48), (32, 32)))):
        image.save(root / f"{index}.png")
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


def test_trainer_steps_match_the_reference_loop(tmp_path):
    root = write_tiny_sd(tmp_path / "tiny")
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = TrainingArguments(
        output_dir=tmp_path / "run",
        max_steps=4,
        resolution=32,
        random_flip=True,
        learning_rate=1e-3,
        lr_scheduler="cosine",
        lr_warmup_steps=1,
        max_grad_norm=0.01,
        logging_steps=1,
    )
    trainer = Trainer("sd-v1-5", root, args, dataset)
    trainer._setup()
    reference = Reference(root, rank=args.lora_rank)
    reference.copy_from(trainer.injection)
    trainer.objective = Recording(trainer.objective)
    trainer.train()

    from diffusers.optimization import get_scheduler

    parameters = list(reference.lora_parameters(trainer.injection.module_names))
    optimizer = torch.optim.AdamW(
        parameters, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8
    )
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=1, num_training_steps=4)
    losses = []
    for pixels, ids, posterior, noise, timesteps in trainer.objective.calls:
        loss = reference.loss(
            pixels, ids, posterior, noise, timesteps, prediction_type="epsilon", snr_gamma=None
        )
        loss.backward()
        # The script's clip call receives an exhausted generator; this clips as intended.
        torch.nn.utils.clip_grad_norm_(parameters, 0.01)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        losses.append(float(loss))
    assert [entry["loss"] for entry in trainer.log_history] == losses
    for ours, theirs in zip(trainer.injection.parameters(), parameters, strict=True):
        assert torch.equal(ours, theirs)


def test_saved_adapter_matches_in_diffusers(tmp_path):
    root = write_tiny_sd(tmp_path / "tiny")
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = TrainingArguments(
        output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
    )
    trainer = Trainer("sd-v1-5", root, args, dataset)
    trainer.train()
    path = trainer.save_model(tmp_path / "final")
    ref = diffusers.StableDiffusionPipeline.from_pretrained(
        root,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    ref.set_progress_bar_config(disable=True)
    ref.load_lora_weights(str(path), adapter_name="trained")
    expected = ref(
        "a cat",
        num_inference_steps=3,
        height=32,
        width=32,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    with pipeline("t2i", backend="sd-v1-5", model=root, lora=path) as pipe:
        result = pipe("a cat", steps=3, height=32, width=32, seed=42)
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)
