import json
import math

import pytest
import torch
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments
from imagehub.training.flow_matching import loss_weighting, sample_density, training_sigmas

PIL = pytest.importorskip("PIL.Image")

TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj")


@pytest.fixture(scope="module")
def tiny_sd3(tmp_path_factory):
    from helpers import write_tiny_sd3

    return write_tiny_sd3(tmp_path_factory.mktemp("train-sd3") / "sd3")


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
        output_dir=tmp_path / "run", max_steps=3, resolution=32, logging_steps=1, save_steps=100
    )
    return TrainingArguments(**(values | overrides))


def test_training_sigmas_follow_the_shifted_schedule():
    timesteps, sigmas = training_sigmas({"num_train_timesteps": 1000, "shift": 3.0})
    assert timesteps.shape == sigmas.shape == (1000,)
    assert sigmas[0] == 1.0 and sigmas[-1] < sigmas[0]
    assert torch.equal(timesteps, sigmas * 1000)
    plain = training_sigmas({"num_train_timesteps": 1000, "shift": 1.0})[1]
    assert torch.allclose(plain, torch.arange(1000, 0, -1) / 1000)


@pytest.mark.parametrize("scheme", ["logit_normal", "mode", "sigma_sqrt", "cosmap", "none"])
def test_density_and_weighting(scheme):
    u = sample_density(
        scheme,
        64,
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        generator=torch.Generator().manual_seed(0),
    )
    assert u.shape == (64,) and float(u.min()) >= 0 and float(u.max()) <= 1
    weights = loss_weighting(scheme, torch.full((2, 1, 1, 1), 0.5))
    assert weights.shape == (2, 1, 1, 1) and bool((weights > 0).all())


def test_options_are_checked_per_family(tmp_path, tiny_sd3):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    with pytest.raises(ValueError, match="weighting_scheme"):
        Trainer("sd-v1-5", tmp_path, arguments(tmp_path, weighting_scheme="mode"), data)
    with pytest.raises(ValueError, match="snr_gamma"):
        Trainer("sd3-medium", tiny_sd3, arguments(tmp_path, snr_gamma=5.0), data)
    with pytest.raises(ValueError, match="weighting_scheme"):
        TrainingArguments(output_dir=tmp_path, max_steps=1, weighting_scheme="uniform")


def test_sd3_lora_loading(tmp_path, tiny_sd3):
    with pipeline("t2i", backend="sd3-medium", model=tiny_sd3) as pipe:
        transformer = pipe.backend.components.transformer
        plain = pipe("a cat", seed=1, steps=2, width=32, height=32).images[0].tensor
    weights = random_lora({"transformer": transformer}, TARGETS, std=0.5)
    zero = {k: (down, torch.zeros_like(up)) for k, (down, up) in weights.items()}
    path = write_peft(tmp_path / "a.safetensors", weights, alpha=2.0)
    empty = write_peft(tmp_path / "zero.safetensors", zero)
    request = dict(seed=1, steps=2, width=32, height=32)
    with pipeline("t2i", backend="sd3-medium", model=tiny_sd3, lora=empty) as pipe:
        assert torch.equal(pipe("a cat", **request).images[0].tensor, plain)
    with pipeline("t2i", backend="sd3-medium", model=tiny_sd3, lora=path) as pipe:
        assert not torch.equal(pipe("a cat", **request).images[0].tensor, plain)


@pytest.mark.parametrize("scheme", ["logit_normal", "mode", "cosmap"])
def test_sd3_training_saves_an_adapter_the_pipeline_loads(tmp_path, tiny_sd3, scheme):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = arguments(
        tmp_path,
        learning_rate=1e-2,
        random_flip=True,
        weighting_scheme=scheme,
        eval_strategy="steps",
        eval_steps=3,
        validation_prompts=("a cat",),
        validation_num_inference_steps=2,
        validation_guidance_scale=3.0,
        seed=4,
    )
    trainer = Trainer("sd3-medium", tiny_sd3, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    assert any("eval_loss" in entry for entry in output.log_history)
    adapter = trainer.save_model(tmp_path / "final")
    request = dict(seed=4, steps=2, guidance=3.0, width=32, height=32)
    with pipeline("t2i", backend="sd3-medium", model=tiny_sd3, lora=adapter) as pipe:
        tuned = pipe("a cat", **request).images[0].tensor
    with PIL.open(args.output_dir / "samples" / "step-3" / "0.png") as image:
        sample = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(sample.view(32, 32, 3), tuned)


def test_sd3_gradient_checkpointing_gives_the_same_gradients(tmp_path, tiny_sd3):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    gradients = []
    for checkpointing in (False, True):
        args = arguments(tmp_path / str(checkpointing), gradient_checkpointing=checkpointing)
        trainer = Trainer("sd3-medium", tiny_sd3, args, data)
        trainer._setup()
        with torch.no_grad():
            for parameter in trainer.parameters:
                parameter.normal_(0, 0.02, generator=torch.Generator().manual_seed(1))
        batch = trainer._next_batch()
        trainer.objective(batch, torch.Generator().manual_seed(2)).backward()
        gradients.append([p.grad.clone() for p in trainer.parameters])
    assert all(torch.equal(a, b) for a, b in zip(*gradients, strict=True))
