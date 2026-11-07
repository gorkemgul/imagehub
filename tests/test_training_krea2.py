import json
import math

import pytest
import torch

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments, get_training_spec

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tiny_krea2(tmp_path_factory):
    pytest.importorskip("transformers")
    from helpers import write_tiny_krea2

    root = tmp_path_factory.mktemp("train-krea2")
    return (
        write_tiny_krea2(root / "raw", distilled=False),
        write_tiny_krea2(root / "turbo", distilled=True),
    )


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
        lora_rank=1,
        lora_alpha=1,
        logging_steps=1,
        save_steps=2,
        **overrides,
    )


def test_krea2_profile_and_distilled_snapshot_refusal(tmp_path, tiny_krea2):
    raw, turbo = tiny_krea2
    data = _folder(tmp_path / "data")
    spec = get_training_spec("krea2-turbo")
    assert spec.support.value == "native"
    assert spec.lora_rank == spec.lora_alpha == 32
    assert spec.training_repo == "krea/Krea-2-Raw"
    assert spec.training_revision == "6b0ece7fffb640c5e3bcbe0a7f10f66b8e60a603"
    with pytest.raises(ValueError, match="Krea-2-Raw"):
        Trainer("krea2-turbo", turbo, _args(tmp_path), data)
    assert Trainer("krea2-turbo", raw, _args(tmp_path), data).checkpoint == raw


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("precondition_outputs", True),
        ("snr_gamma", 5.0),
        ("prediction_type", "v_prediction"),
        ("complex_human_instruction", ("not used",)),
        ("system_prompt", "not used"),
        ("guidance_scale", 4.0),
    ],
)
def test_krea2_rejects_unused_options(tmp_path, tiny_krea2, option, value):
    raw, _ = tiny_krea2
    with pytest.raises(ValueError, match=option):
        Trainer("krea2-turbo", raw, _args(tmp_path, **{option: value}), _folder(tmp_path / "data"))


def test_krea2_train_save_load_validation_and_checkpointing(tmp_path, tiny_krea2):
    raw, turbo = tiny_krea2
    data = _folder(tmp_path / "data")
    args = _args(
        tmp_path,
        learning_rate=1e-2,
        eval_strategy="steps",
        eval_steps=1,
        validation_prompts=("a cat",),
        seed=4,
    )
    trainer = Trainer("krea2-turbo", raw, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    adapter = trainer.save_model(tmp_path / "adapter")
    request = dict(seed=4, steps=1, guidance=0.0, width=32, height=32, max_sequence_length=8)
    with pipeline("t2i", backend="krea2", model=turbo, lora=adapter) as tuned:
        assert tuned("a cat", **request).images[0].tensor.shape == (32, 32, 3)
    with pipeline("t2i", backend="krea2", model=raw, lora=adapter) as tuned:
        expected = (
            tuned(
                "a cat", seed=4, steps=8, guidance=0.0, width=32, height=32, max_sequence_length=8
            )
            .images[0]
            .tensor
        )
    with PIL.open(args.output_dir / "samples" / "step-1" / "0.png") as image:
        actual = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(actual.view_as(expected), expected)

    gradients = []
    for checkpointing in (False, True):
        current = Trainer(
            "krea2-turbo",
            raw,
            _args(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing),
            data,
        )
        current._setup()
        current.objective(current._next_batch(), torch.Generator().manual_seed(2)).backward()
        gradients.append([parameter.grad.clone() for parameter in current.parameters])
    assert all(torch.equal(left, right) for left, right in zip(*gradients, strict=True))
