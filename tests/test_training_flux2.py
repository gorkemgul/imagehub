import json
import math

import pytest
import torch
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments, get_training_spec

PIL = pytest.importorskip("PIL.Image")

TARGETS = (
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
    *(f"single_transformer_blocks.{index}.attn.to_out" for index in range(24)),
)


@pytest.fixture(scope="module")
def tiny_flux2(tmp_path_factory):
    pytest.importorskip("transformers")
    from helpers import write_tiny_flux2

    return write_tiny_flux2(tmp_path_factory.mktemp("train-flux2") / "flux2")


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


def test_flux2_options_profiles_and_lora_loading(tmp_path, tiny_flux2):
    data = _folder(tmp_path / "data")
    for option, value in (
        ("precondition_outputs", True),
        ("snr_gamma", 5.0),
        ("complex_human_instruction", ("Enhance: ",)),
        ("system_prompt", "not applicable"),
    ):
        with pytest.raises(ValueError, match=option):
            Trainer("flux2-klein-4b", tiny_flux2, _args(tmp_path, **{option: value}), data)
    with pytest.raises(ValueError, match="guidance_scale"):
        trainer = Trainer("flux2-klein-4b", tiny_flux2, _args(tmp_path, guidance_scale=4.0), data)
        trainer._setup()
    spec = get_training_spec("flux2-klein-4b")
    assert spec.support.value == "native"
    assert spec.target_modules == TARGETS
    assert get_training_spec("flux2-klein-9b").support.value == "native"

    request = dict(seed=1, steps=1, width=32, height=32, max_sequence_length=8)
    with pipeline("t2i", backend="flux2", model=tiny_flux2) as pipe:
        plain = pipe("a cat", **request).images[0].tensor
        assert pipe.backend.components is not None
        weights = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, std=20.0
        )
    adapter = write_peft(tmp_path / "adapter.safetensors", weights, alpha=2.0)
    with pipeline("t2i", backend="flux2", model=tiny_flux2, lora=adapter) as pipe:
        assert not torch.equal(pipe("a cat", **request).images[0].tensor, plain)


def test_flux2_training_save_load_and_checkpointing(tmp_path, tiny_flux2):
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
    trainer = Trainer("flux2-klein-4b", tiny_flux2, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    adapter = trainer.save_model(tmp_path / "final")
    with pipeline("t2i", backend="flux2", model=tiny_flux2, lora=adapter) as pipe:
        tuned = (
            pipe("a cat", seed=4, steps=1, guidance=0.0, width=32, height=32, max_sequence_length=8)
            .images[0]
            .tensor
        )
    with PIL.open(args.output_dir / "samples" / "step-1" / "0.png") as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(pixels.view_as(tuned), tuned)

    gradients = []
    for checkpointing in (False, True):
        current = Trainer(
            "flux2-klein-4b",
            tiny_flux2,
            _args(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing),
            data,
        )
        current._setup()
        current.objective(current._next_batch(), torch.Generator().manual_seed(2)).backward()
        gradients.append([parameter.grad.clone() for parameter in current.parameters])
    assert all(torch.equal(left, right) for left, right in zip(*gradients, strict=True))
