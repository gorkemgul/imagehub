import json
import math

import pytest
import torch
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments, get_training_spec

PIL = pytest.importorskip("PIL.Image")

TARGETS = ("to_k", "to_q", "to_v", "to_out.0")


@pytest.fixture(scope="module")
def tiny_z_image(tmp_path_factory):
    pytest.importorskip("transformers")
    from helpers import write_tiny_z_image

    return write_tiny_z_image(tmp_path_factory.mktemp("train-z-image") / "z-image")


def _folder(root):
    root.mkdir()
    pixels = torch.randint(
        0, 256, (40, 48, 3), generator=torch.Generator().manual_seed(0), dtype=torch.uint8
    )
    PIL.frombytes("RGB", (48, 40), bytes(pixels.flatten().tolist())).save(root / "0.png")
    (root / "metadata.jsonl").write_text(
        json.dumps({"file_name": "0.png", "text": "a tiny cat"}) + "\n"
    )
    return ImageFolderDataset(root)


def _args(tmp_path, **overrides):
    return TrainingArguments(
        output_dir=tmp_path / "run",
        max_steps=1,
        resolution=32,
        max_sequence_length=8,
        logging_steps=1,
        save_steps=2,
        **overrides,
    )


def test_z_image_options_profiles_and_lora_loading(tmp_path, tiny_z_image):
    data = _folder(tmp_path / "data")
    for option, value in (
        ("precondition_outputs", True),
        ("guidance_scale", 4.0),
        ("snr_gamma", 5.0),
        ("complex_human_instruction", ("Enhance: ",)),
    ):
        with pytest.raises(ValueError, match=option):
            Trainer("z-image", tiny_z_image, _args(tmp_path, **{option: value}), data)
    spec = get_training_spec("z-image")
    assert spec.support.value == "native"
    assert spec.target_modules == TARGETS
    assert get_training_spec("z-image-turbo").support.value == "inference_only"

    request = dict(seed=1, steps=2, width=32, height=32, max_sequence_length=8)
    with pipeline("t2i", backend="z-image", model=tiny_z_image) as pipe:
        plain = pipe("a cat", **request).images[0].tensor
        weights = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, std=20.0
        )
    adapter = write_peft(tmp_path / "adapter.safetensors", weights, alpha=2.0)
    with pipeline("t2i", backend="z-image", model=tiny_z_image, lora=adapter) as pipe:
        assert not torch.equal(pipe("a cat", **request).images[0].tensor, plain)


def test_z_image_training_saves_an_adapter_and_validation_matches_pipeline(tmp_path, tiny_z_image):
    data = _folder(tmp_path / "data")
    args = _args(
        tmp_path,
        learning_rate=1e-2,
        random_flip=True,
        eval_strategy="steps",
        eval_steps=1,
        validation_prompts=("a cat",),
        validation_num_inference_steps=2,
        validation_guidance_scale=3.0,
        seed=4,
    )
    trainer = Trainer("z-image", tiny_z_image, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    adapter = trainer.save_model(tmp_path / "final")
    with pipeline("t2i", backend="z-image", model=tiny_z_image, lora=adapter) as pipe:
        tuned = (
            pipe("a cat", seed=4, steps=2, guidance=3.0, width=32, height=32, max_sequence_length=8)
            .images[0]
            .tensor
        )
    with PIL.open(args.output_dir / "samples" / "step-1" / "0.png") as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(pixels.view(32, 32, 3), tuned)


def test_z_image_gradient_checkpointing_gives_the_same_gradients(tmp_path, tiny_z_image):
    data = _folder(tmp_path / "data")
    gradients = []
    for checkpointing in (False, True):
        trainer = Trainer(
            "z-image",
            tiny_z_image,
            _args(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing),
            data,
        )
        trainer._setup()
        with torch.no_grad():
            for parameter in trainer.parameters:
                parameter.normal_(0, 0.02, generator=torch.Generator().manual_seed(1))
        trainer.objective(trainer._next_batch(), torch.Generator().manual_seed(2)).backward()
        gradients.append([parameter.grad.clone() for parameter in trainer.parameters])
    assert all(torch.equal(a, b) for a, b in zip(*gradients, strict=True))
