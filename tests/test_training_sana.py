import json
import math

import pytest
import torch
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments, get_training_spec

PIL = pytest.importorskip("PIL.Image")

TARGETS = ("to_k", "to_q", "to_v")


@pytest.fixture(scope="module")
def tiny_sana(tmp_path_factory):
    from helpers import write_tiny_sana

    return write_tiny_sana(tmp_path_factory.mktemp("train-sana") / "sana")


def write_folder(root):
    root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0)
    lines = []
    for index, (width, height) in enumerate(((40, 48), (48, 40), (32, 32))):
        pixels = torch.randint(0, 256, (height, width, 3), generator=generator, dtype=torch.uint8)
        PIL.frombytes("RGB", (width, height), bytes(pixels.flatten().tolist())).save(
            root / f"{index}.png"
        )
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


def arguments(tmp_path, **overrides):
    return TrainingArguments(
        **(
            dict(
                output_dir=tmp_path / "run",
                max_steps=3,
                resolution=32,
                max_sequence_length=8,
                logging_steps=1,
                save_steps=100,
            )
            | overrides
        )
    )


def test_sana_options_profiles_and_lora_loading(tmp_path, tiny_sana):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    for option, value in (
        ("precondition_outputs", True),
        ("guidance_scale", 4.0),
        ("snr_gamma", 5.0),
    ):
        with pytest.raises(ValueError, match=option):
            Trainer("sana-600m-1024", tiny_sana, arguments(tmp_path, **{option: value}), data)
    for model in ("sana-600m-1024", "sana-1600m-1024", "sana-1.5-4.8b-1024"):
        spec = get_training_spec(model)
        assert spec.support.value == "native"
        assert spec.target_modules == TARGETS
    assert get_training_spec("sana-sprint-1.6b-1024").support.value == "inference_only"

    request = dict(seed=1, steps=2, width=32, height=32, max_sequence_length=8)
    with pipeline("t2i", backend="sana", model=tiny_sana) as pipe:
        plain = pipe("a cat", **request).images[0].tensor
        weights = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, std=0.5
        )
    adapter = write_peft(tmp_path / "adapter.safetensors", weights, alpha=2.0)
    with pipeline("t2i", backend="sana", model=tiny_sana, lora=adapter) as pipe:
        assert not torch.equal(pipe("a cat", **request).images[0].tensor, plain)


def test_sana_training_saves_an_adapter_and_validation_matches_pipeline(tmp_path, tiny_sana):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = arguments(
        tmp_path,
        learning_rate=1e-2,
        random_flip=True,
        eval_strategy="steps",
        eval_steps=3,
        validation_prompts=("a cat",),
        validation_num_inference_steps=2,
        validation_guidance_scale=3.0,
        seed=4,
    )
    trainer = Trainer("sana-600m-1024", tiny_sana, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    adapter = trainer.save_model(tmp_path / "final")
    with pipeline("t2i", backend="sana", model=tiny_sana, lora=adapter) as pipe:
        tuned = (
            pipe(
                "a cat",
                seed=4,
                steps=2,
                guidance=3.0,
                width=32,
                height=32,
                max_sequence_length=8,
                complex_human_instruction=[],
            )
            .images[0]
            .tensor
        )
    with PIL.open(args.output_dir / "samples" / "step-3" / "0.png") as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(pixels.view(32, 32, 3), tuned)


def test_sana_gradient_checkpointing_gives_the_same_gradients(tmp_path, tiny_sana):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    gradients = []
    for checkpointing in (False, True):
        trainer = Trainer(
            "sana-600m-1024",
            tiny_sana,
            arguments(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing),
            data,
        )
        trainer._setup()
        with torch.no_grad():
            for parameter in trainer.parameters:
                parameter.normal_(0, 0.02, generator=torch.Generator().manual_seed(1))
        trainer.objective(trainer._next_batch(), torch.Generator().manual_seed(2)).backward()
        gradients.append([parameter.grad.clone() for parameter in trainer.parameters])
    assert all(torch.equal(a, b) for a, b in zip(*gradients, strict=True))
