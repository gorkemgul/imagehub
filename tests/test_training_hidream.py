import json
import math

import pytest
import torch
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments, get_training_spec

PIL = pytest.importorskip("PIL.Image")

TARGETS = ("to_k", "to_q", "to_v", "to_out")


@pytest.fixture(scope="module")
def tiny_hidream(tmp_path_factory):
    pytest.importorskip("transformers")
    from helpers import write_tiny_hidream

    return write_tiny_hidream(tmp_path_factory.mktemp("train-hidream") / "hidream")


def _folder(root):
    root.mkdir()
    pixels = torch.randint(
        0, 256, (24, 32, 3), generator=torch.Generator().manual_seed(0), dtype=torch.uint8
    )
    PIL.frombytes("RGB", (32, 24), bytes(pixels.flatten().tolist())).save(root / "0.png")
    (root / "metadata.jsonl").write_text(
        json.dumps({"file_name": "0.png", "text": "a tiny cat"}) + "\n"
    )
    return ImageFolderDataset(root)


def _args(tmp_path, **overrides):
    return TrainingArguments(
        output_dir=tmp_path / "run",
        max_steps=1,
        resolution=16,
        max_sequence_length=8,
        logging_steps=1,
        save_steps=2,
        **overrides,
    )


def test_hidream_options_profiles_and_lora_loading(tmp_path, tiny_hidream):
    data = _folder(tmp_path / "data")
    for option, value in (
        ("precondition_outputs", True),
        ("guidance_scale", 4.0),
        ("snr_gamma", 5.0),
        ("complex_human_instruction", ("Enhance: ",)),
        ("system_prompt", "not applicable"),
    ):
        with pytest.raises(ValueError, match=option):
            Trainer("hidream-i1-dev", tiny_hidream, _args(tmp_path, **{option: value}), data)
    spec = get_training_spec("hidream-i1-dev")
    assert spec.support.value == "native"
    assert spec.target_modules == TARGETS
    assert get_training_spec("hidream-i1-full").support.value == "native"
    assert get_training_spec("hidream-i1-fast").support.value == "inference_only"

    request = dict(seed=1, steps=1, width=32, height=32, max_sequence_length=8)
    with pipeline("t2i", backend="hidream", model=tiny_hidream) as pipe:
        plain = pipe("a cat", **request).images[0].tensor
        weights = random_lora(
            {
                "transformer": pipe.backend.components._from_pretrained(
                    HiDreamImageTransformer2DModel, "transformer"
                )
            },
            TARGETS,
            std=20.0,
        )
    adapter = write_peft(tmp_path / "adapter.safetensors", weights, alpha=2.0)
    with pipeline("t2i", backend="hidream", model=tiny_hidream, lora=adapter) as pipe:
        assert not torch.equal(pipe("a cat", **request).images[0].tensor, plain)


def test_hidream_training_save_load_and_checkpointing(tmp_path, tiny_hidream):
    data = _folder(tmp_path / "data")
    args = _args(
        tmp_path,
        learning_rate=1e-2,
        random_flip=True,
        eval_strategy="steps",
        eval_steps=1,
        validation_prompts=("a cat",),
        validation_num_inference_steps=1,
        validation_guidance_scale=0.0,
        seed=4,
    )
    trainer = Trainer("hidream-i1-dev", tiny_hidream, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    adapter = trainer.save_model(tmp_path / "final")
    with pipeline("t2i", backend="hidream", model=tiny_hidream, lora=adapter) as pipe:
        tuned = (
            pipe("a cat", seed=4, steps=1, guidance=0.0, width=16, height=16, max_sequence_length=8)
            .images[0]
            .tensor
        )
    with PIL.open(args.output_dir / "samples" / "step-1" / "0.png") as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(pixels.view_as(tuned), tuned)

    gradients = []
    for checkpointing in (False, True):
        current = Trainer(
            "hidream-i1-dev",
            tiny_hidream,
            _args(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing),
            data,
        )
        current._setup()
        current.objective(current._next_batch(), torch.Generator().manual_seed(2)).backward()
        gradients.append([parameter.grad.clone() for parameter in current.parameters])
    assert all(torch.equal(left, right) for left, right in zip(*gradients, strict=True))
