import json
import math

import pytest
import torch
from lora_helpers import random_lora, write_peft

from imagehub import pipeline
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments

PIL = pytest.importorskip("PIL.Image")

TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp")


@pytest.fixture(scope="module")
def tiny_flux(tmp_path_factory):
    from helpers import write_tiny_flux1

    return write_tiny_flux1(tmp_path_factory.mktemp("train-flux") / "flux")


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
                logging_steps=1,
                save_steps=100,
            )
            | overrides
        )
    )


def test_flux_options_and_profiles_are_checked(tmp_path, tiny_flux):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    with pytest.raises(ValueError, match="precondition_outputs"):
        Trainer("flux1-dev", tiny_flux, arguments(tmp_path, precondition_outputs=True), data)
    with pytest.raises(ValueError, match="snr_gamma"):
        Trainer("flux1-dev", tiny_flux, arguments(tmp_path, snr_gamma=5.0), data)
    for model in ("sd-v1-5", "sdxl-base-1.0", "sd3-medium"):
        with pytest.raises(ValueError, match="guidance_scale"):
            Trainer(model, tmp_path, arguments(tmp_path, guidance_scale=5.0), data)
    from imagehub.training import get_training_spec

    assert get_training_spec("flux1-schnell").support.value == "inference_only"
    for model in ("flux1-dev", "flux1-krea-dev", "flux-1-lite-8b", "flux1-kontext-dev"):
        assert get_training_spec(model).support.value == "native"


def test_flux_lora_loading(tmp_path, tiny_flux):
    request = dict(seed=1, steps=2, width=32, height=32, max_sequence_length=8)
    with pipeline("t2i", backend="flux", model=tiny_flux) as pipe:
        plain = pipe("a cat", **request).images[0].tensor
        weights = random_lora(
            {"transformer": pipe.backend.components.transformer}, TARGETS, std=0.5
        )
    adapter = write_peft(tmp_path / "adapter.safetensors", weights, alpha=2.0)
    with pipeline("t2i", backend="flux", model=tiny_flux, lora=adapter) as pipe:
        assert not torch.equal(pipe("a cat", **request).images[0].tensor, plain)


@pytest.mark.parametrize("guidance_embeds", [False, True])
def test_flux_training_saves_an_adapter_the_pipeline_loads(tmp_path, guidance_embeds):
    from helpers import write_tiny_flux1

    tiny = write_tiny_flux1(tmp_path / "flux", guidance_embeds=guidance_embeds)
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = arguments(
        tmp_path,
        learning_rate=1e-2,
        random_flip=True,
        weighting_scheme="none",
        eval_strategy="steps",
        eval_steps=3,
        validation_prompts=("a cat",),
        validation_num_inference_steps=2,
        validation_guidance_scale=3.0,
        seed=4,
    )
    trainer = Trainer("flux1-dev", tiny, args, data, eval_dataset=data)
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    assert any("eval_loss" in entry for entry in output.log_history)
    adapter = trainer.save_model(tmp_path / "final")
    request = dict(seed=4, steps=2, guidance=3.0, width=32, height=32)
    with pipeline("t2i", backend="flux", model=tiny, lora=adapter) as pipe:
        tuned = pipe("a cat", **request).images[0].tensor
    with PIL.open(args.output_dir / "samples" / "step-3" / "0.png") as image:
        pixels = torch.frombuffer(bytearray(image.convert("RGB").tobytes()), dtype=torch.uint8)
    assert torch.equal(pixels.view(32, 32, 3), tuned)


def test_flux_gradient_checkpointing_gives_the_same_gradients(tmp_path, tiny_flux):
    data = ImageFolderDataset(write_folder(tmp_path / "data"))
    gradients = []
    for checkpointing in (False, True):
        trainer = Trainer(
            "flux1-dev",
            tiny_flux,
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
