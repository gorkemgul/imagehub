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
def tiny_qwen(tmp_path_factory):
    pytest.importorskip("transformers")
    from helpers import write_tiny_qwen_image

    return write_tiny_qwen_image(tmp_path_factory.mktemp("train-qwen") / "qwen")


@pytest.fixture(scope="module")
def tiny_qwen21(tmp_path_factory):
    pytest.importorskip("transformers")
    from helpers import write_tiny_qwen_image21

    return write_tiny_qwen_image21(tmp_path_factory.mktemp("train-qwen21") / "qwen21")


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
        logging_steps=1,
        save_steps=2,
        **overrides,
    )


@pytest.mark.parametrize(
    "model,fixture", [("qwen-image", "tiny_qwen"), ("qwen-image-2.1", "tiny_qwen21")]
)
def test_qwen_profiles_and_lora_loading(tmp_path, model, fixture, request):
    tiny = request.getfixturevalue(fixture)
    assert get_training_spec(model).target_modules == TARGETS
    plain_request = dict(seed=2, steps=1, width=32, height=32, true_cfg=1.0)
    with pipeline("t2i", backend=model, model=tiny) as pipe:
        plain = pipe("a cat", **plain_request).images[0].tensor
        weights = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, std=20.0
        )
    adapter = write_peft(tmp_path / f"{model}.safetensors", weights, alpha=2.0)
    with pipeline("t2i", backend=model, model=tiny, lora=adapter, lora_scale=1.0) as pipe:
        assert pipe("a cat", **plain_request).images[0].tensor.shape == plain.shape


@pytest.mark.parametrize(
    "model,fixture", [("qwen-image", "tiny_qwen"), ("qwen-image-2.1", "tiny_qwen21")]
)
def test_qwen_training_save_load_and_checkpointing(tmp_path, model, fixture, request):
    tiny = request.getfixturevalue(fixture)
    data = _folder(tmp_path / "data")
    args = _args(
        tmp_path,
        seed=4,
        validation_prompts=("a cat",),
        validation_num_inference_steps=1,
        eval_strategy="steps",
        eval_steps=1,
    )
    trainer = Trainer(model, tiny, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    adapter = trainer.save_model(tmp_path / "adapter")
    with pipeline("t2i", backend=model, model=tiny, lora=adapter) as pipe:
        tuned = pipe("a cat", seed=4, steps=1, width=32, height=32, true_cfg=1.0).images[0].tensor
    with PIL.open(args.output_dir / "samples" / "step-1" / "0.png") as image:
        saved = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(saved.view(32, 32, 3), tuned[:, :, :3])
    gradients = []
    for checkpointing in (False, True):
        current = Trainer(
            model,
            tiny,
            _args(tmp_path / str(checkpointing), seed=1, gradient_checkpointing=checkpointing),
            data,
        )
        current._setup()
        current.objective(current._next_batch(), torch.Generator().manual_seed(2)).backward()
        gradients.append([parameter.grad.clone() for parameter in current.parameters])
    assert all(torch.equal(left, right) for left, right in zip(*gradients, strict=True))


def test_qwen21_rejects_original_only_max_sequence_length(tmp_path, tiny_qwen21):
    with pytest.raises(ValueError, match="max_sequence_length"):
        Trainer(
            "qwen-image-2.1",
            tiny_qwen21,
            _args(tmp_path, max_sequence_length=512),
            _folder(tmp_path / "data"),
        )._setup()


def test_qwen21_uses_its_script_rank_and_alpha_defaults(tmp_path, tiny_qwen21):
    trainer = Trainer("qwen-image-2.1", tiny_qwen21, _args(tmp_path), _folder(tmp_path / "data"))
    trainer._setup()
    assert trainer.injection is not None
    assert (trainer.injection.config.rank, trainer.injection.config.alpha) == (16, 16.0)


def test_qwen_uses_explicit_rank_and_alpha(tmp_path, tiny_qwen21):
    trainer = Trainer(
        "qwen-image-2.1",
        tiny_qwen21,
        _args(tmp_path, lora_rank=4, lora_alpha=4),
        _folder(tmp_path / "data"),
    )
    trainer._setup()
    assert trainer.injection is not None
    assert (trainer.injection.config.rank, trainer.injection.config.alpha) == (4, 4.0)
