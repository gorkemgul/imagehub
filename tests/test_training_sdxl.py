import json
import math

import pytest
import torch

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments
from imagehub.training.data import StableDiffusionXLCollator, preprocess_image_xl

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tiny_sdxl(tmp_path_factory):
    from helpers import write_tiny_sdxl

    return write_tiny_sdxl(tmp_path_factory.mktemp("train-sdxl") / "sdxl", sample_size=4)


@pytest.fixture(scope="module")
def tiny_sd(tmp_path_factory):
    from helpers import write_tiny_sd

    return write_tiny_sd(tmp_path_factory.mktemp("train-sd-ckpt") / "sd")


def write_folder(root, sizes=((40, 48), (48, 40), (32, 32))):
    root.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    lines = []
    for index, (width, height) in enumerate(sizes):
        pixels = torch.randint(0, 256, (height, width, 3), generator=g, dtype=torch.uint8)
        image = PIL.frombytes("RGB", (width, height), bytes(pixels.flatten().tolist()))
        image.save(root / f"{index}.png")
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


def arguments(tmp_path, **overrides):
    values = dict(
        output_dir=tmp_path / "run",
        max_steps=3,
        resolution=32,
        logging_steps=1,
        save_steps=100,
    )
    return TrainingArguments(**(values | overrides))


def test_xl_transform_records_size_and_crop():
    g = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (48, 64, 3), generator=g, dtype=torch.uint8)
    image = PIL.frombytes("RGB", (64, 48), bytes(pixels.flatten().tolist()))
    out, size, crop = preprocess_image_xl(image, resolution=32, center_crop=True, random_flip=False)
    assert out.shape == (3, 32, 32)
    assert size == (48, 64)
    assert crop == (0, 5)  # resized to 32x42, centered: round((42 - 32) / 2)

    def draw(seed):
        return preprocess_image_xl(
            image,
            resolution=32,
            center_crop=False,
            random_flip=True,
            generator=torch.Generator().manual_seed(seed),
        )

    first, again = draw(3), draw(3)
    assert torch.equal(first[0], again[0]) and first[2] == again[2]


def test_xl_collator_builds_time_ids(tmp_path):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))

    def tokenizer(text):
        return [len(text)] * 77

    collator = StableDiffusionXLCollator(
        tokenizer, tokenizer, resolution=32, center_crop=True, random_flip=False
    )
    batch = collator([data[0], data[1]])
    assert batch["pixel_values"].shape == (2, 3, 32, 32)
    assert batch["input_ids"].shape == batch["input_ids_2"].shape == (2, 77)
    assert batch["time_ids"].tolist() == [[48, 40, 3, 0, 32, 32], [40, 48, 0, 3, 32, 32]]
    assert batch["time_ids"].dtype == torch.float32


def test_sdxl_training_saves_an_adapter_the_pipeline_loads(tmp_path, tiny_sdxl):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = arguments(
        tmp_path,
        learning_rate=1e-2,
        random_flip=True,
        eval_strategy="steps",
        eval_steps=3,
        validation_prompts=("a cat",),
        validation_num_inference_steps=2,
        validation_guidance_scale=5.0,
        seed=4,
    )
    trainer = Trainer("sdxl-base-1.0", tiny_sdxl, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    assert any("eval_loss" in entry for entry in output.log_history)
    adapter = trainer.save_model(tmp_path / "final")
    config = json.loads((tmp_path / "final" / "adapter_config.json").read_text())
    assert config["base_model"] == "sdxl-base-1.0"
    with pipeline("t2i", backend="sdxl-base-1.0", model=tiny_sdxl) as pipe:
        plain = pipe("a cat", seed=4, steps=2, guidance=5.0, width=32, height=32)
    with pipeline("t2i", backend="sdxl-base-1.0", model=tiny_sdxl, lora=adapter) as pipe:
        tuned = pipe("a cat", seed=4, steps=2, guidance=5.0, width=32, height=32)
    assert not torch.equal(plain.images[0].tensor, tuned.images[0].tensor)
    with PIL.open(args.output_dir / "samples" / "step-3" / "0.png") as image:
        sample = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(sample.view(32, 32, 3), tuned.images[0].tensor)


def test_sdxl_bf16_training_keeps_the_autoencoder_in_float32(tmp_path, tiny_sdxl):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    trainer = Trainer("sdxl-base-1.0", tiny_sdxl, arguments(tmp_path, mixed_precision="bf16"), data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    assert next(trainer.components.vae.parameters()).dtype == torch.float32
    assert next(trainer.components.unet.conv_in.parameters()).dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("model", "fixture"), [("sd-v1-5", "tiny_sd"), ("sdxl-base-1.0", "tiny_sdxl")]
)
def test_gradient_checkpointing_gives_the_same_gradients(tmp_path, request, model, fixture):
    root = request.getfixturevalue(fixture)
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    gradients = []
    for checkpointing in (False, True):
        args = arguments(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing)
        trainer = Trainer(model, root, args, data)
        trainer._setup()
        with torch.no_grad():
            for parameter in trainer.parameters:
                parameter.normal_(0, 0.02, generator=torch.Generator().manual_seed(1))
        batch = trainer._next_batch()
        trainer.objective(batch, torch.Generator().manual_seed(2)).backward()
        gradients.append([p.grad.clone() for p in trainer.parameters])
    assert all(torch.equal(a, b) for a, b in zip(*gradients, strict=True))
