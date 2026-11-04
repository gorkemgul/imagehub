"""SDXL LoRA training parity against diffusers' SDXL LoRA training script step, CPU fp32."""

import json

import pytest
import torch
from helpers import write_tiny_sdxl

from imagehub import pipeline
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments
from imagehub.training.data import preprocess_image_xl
from imagehub.training.sdxl import StableDiffusionXLFamily

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
transforms = pytest.importorskip("torchvision.transforms")
functional = pytest.importorskip("torchvision.transforms.functional")
PIL = pytest.importorskip("PIL.Image")

CPU = torch.device("cpu")
TARGETS = ["to_k", "to_q", "to_v", "to_out.0"]


def images(count, sizes):
    g = torch.Generator().manual_seed(0)
    result = []
    for index in range(count):
        width, height = sizes[index % len(sizes)]
        pixels = torch.randint(0, 256, (height, width, 3), generator=g, dtype=torch.uint8)
        result.append(PIL.frombytes("RGB", (width, height), bytes(pixels.flatten().tolist())))
    return result


def test_xl_transform_matches_the_script():
    resize = transforms.Resize(64, interpolation=transforms.InterpolationMode.BILINEAR)
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    for image in images(4, ((97, 64), (64, 130), (80, 80), (300, 211))):
        for center in (True, False):
            # The script's preprocess_train without flipping.
            torch.manual_seed(5)
            original = (image.height, image.width)
            resized = resize(image)
            if center:
                y1 = max(0, int(round((resized.height - 64) / 2.0)))
                x1 = max(0, int(round((resized.width - 64) / 2.0)))
                cropped = transforms.CenterCrop(64)(resized)
            else:
                y1, x1, h, w = transforms.RandomCrop(64).get_params(resized, (64, 64))
                cropped = functional.crop(resized, y1, x1, h, w)
            expected = to_tensor(cropped)
            torch.manual_seed(5)
            actual, size, crop = preprocess_image_xl(
                image, resolution=64, center_crop=center, random_flip=False
            )
            assert torch.equal(actual, expected)
            assert size == original and crop == (y1, x1)


class Reference:
    def __init__(self, root):
        self.unet = diffusers.UNet2DConditionModel.from_pretrained(root, subfolder="unet")
        self.vae = diffusers.AutoencoderKL.from_pretrained(root, subfolder="vae")
        self.text_encoder = transformers.CLIPTextModel.from_pretrained(
            root, subfolder="text_encoder"
        )
        self.text_encoder_2 = transformers.CLIPTextModelWithProjection.from_pretrained(
            root, subfolder="text_encoder_2"
        )
        self.scheduler = diffusers.DDPMScheduler.from_pretrained(root, subfolder="scheduler")
        for module in (self.unet, self.vae, self.text_encoder, self.text_encoder_2):
            module.requires_grad_(False)
        self.unet.add_adapter(
            peft.LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=TARGETS)
        )
        self.unet.train()

    def lora_parameters(self, names):
        for name in names:
            layer = self.unet.get_submodule(name)
            yield layer.lora_A["default"].weight
            yield layer.lora_B["default"].weight

    def loss(self, batch, draws, *, prediction_type, snr_gamma):
        with torch.no_grad():
            dist = self.vae.encode(batch["pixel_values"]).latent_dist
            model_input = (
                dist.mean + dist.std * draws["posterior"]
            ) * self.vae.config.scaling_factor
        noise, timesteps = draws["noise"], draws["timesteps"]
        noisy = self.scheduler.add_noise(model_input, noise, timesteps)
        # encode_prompt with text_input_ids_list, as in the script.
        embeds = []
        with torch.no_grad():
            for encoder, ids in (
                (self.text_encoder, batch["input_ids"]),
                (self.text_encoder_2, batch["input_ids_2"]),
            ):
                output = encoder(ids, output_hidden_states=True, return_dict=False)
                pooled = output[0]
                embeds.append(output[-1][-2])
        prompt_embeds = torch.concat(embeds, dim=-1)
        prediction = self.unet(
            noisy,
            timesteps,
            prompt_embeds,
            added_cond_kwargs={"time_ids": batch["time_ids"], "text_embeds": pooled},
            return_dict=False,
        )[0]
        self.scheduler.register_to_config(prediction_type=prediction_type)
        if prediction_type == "epsilon":
            target = noise
        else:
            target = self.scheduler.get_velocity(model_input, noise, timesteps)
        if snr_gamma is None:
            return torch.nn.functional.mse_loss(prediction.float(), target.float())
        from diffusers.training_utils import compute_snr

        snr = compute_snr(self.scheduler, timesteps)
        weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
        weights = weights / snr if prediction_type == "epsilon" else weights / (snr + 1)
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none")
        return (loss.mean(dim=list(range(1, len(loss.shape)))) * weights).mean()


def write_folder(root):
    root.mkdir(parents=True)
    lines = []
    for index, image in enumerate(images(3, ((40, 48), (48, 40)))):
        image.save(root / f"{index}.png")
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


@pytest.mark.parametrize(
    ("prediction_type", "snr_gamma"),
    [("epsilon", None), ("v_prediction", None), ("epsilon", 5.0)],
)
def test_first_sdxl_step_matches(tmp_path, prediction_type, snr_gamma):
    root = write_tiny_sdxl(tmp_path / "tiny", sample_size=4)
    family = StableDiffusionXLFamily()
    ours = family.load(root, device=CPU, dtype=torch.float32, model_id="sdxl-base-1.0")
    for module in [ours.unet, *family.frozen(ours)]:
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
    reference = Reference(root)
    with torch.no_grad():
        for mine, theirs in zip(
            injection.parameters(), reference.lora_parameters(injection.module_names), strict=True
        ):
            theirs.copy_(mine)
    args = TrainingArguments(
        output_dir=tmp_path / "run",
        max_steps=1,
        resolution=32,
        prediction_type=prediction_type,
        snr_gamma=snr_gamma,
    )
    objective = family.objective(ours, args)
    collator = family.collator(ours, args, generator=g, train=True)
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    batch = collator([data[0], data[1]])
    draws = objective.draw(batch, g)
    loss = objective.loss_from(batch, draws)
    loss.backward()
    expected = reference.loss(batch, draws, prediction_type=prediction_type, snr_gamma=snr_gamma)
    expected.backward()
    assert torch.equal(loss, expected)
    # The forward pass and the loss are bit-identical. The SDXL U-Net backward differs from
    # diffusers in float32 summation order: relative to the largest LoRA gradient, the largest
    # difference measured is 6.2e-7 (SD 1.x gradients are identical, with or without linear
    # projections). Self-attention over a single 1x1 token has a zero true gradient, so the
    # comparison is scaled by the largest gradient, not per tensor. Tolerance: next power of ten.
    pairs = list(
        zip(
            injection.parameters(),
            reference.lora_parameters(injection.module_names),
            strict=True,
        )
    )
    scale = max(float(theirs.grad.abs().max()) for _, theirs in pairs)
    worst = max(float((mine.grad - theirs.grad).abs().max()) for mine, theirs in pairs) / scale
    assert worst < 1e-6, worst


def test_saved_sdxl_adapter_matches_in_diffusers(tmp_path):
    root = write_tiny_sdxl(tmp_path / "tiny", sample_size=4)
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = TrainingArguments(
        output_dir=tmp_path / "run", max_steps=3, resolution=32, learning_rate=1e-2
    )
    trainer = Trainer("sdxl-base-1.0", root, args, data)
    trainer.train()
    path = trainer.save_model(tmp_path / "final")
    ref = diffusers.StableDiffusionXLPipeline.from_pretrained(
        root, local_files_only=True, add_watermarker=False
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
    with pipeline("t2i", backend="sdxl-base-1.0", model=root, lora=path) as pipe:
        result = pipe("a cat", steps=3, height=32, width=32, seed=42)
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)
