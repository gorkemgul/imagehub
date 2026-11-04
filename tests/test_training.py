import json
import math
import shutil

import pytest
import torch

from imagehub import pipeline
from imagehub.lora import LoRALayer, lora_layers
from imagehub.lora_formats import read_lora_file
from imagehub.training import (
    ImageFolderDataset,
    Trainer,
    TrainingArguments,
    TrainingSupport,
    get_training_spec,
    list_training_specs,
    preprocess_image,
)
from imagehub.training.schedules import lr_lambda

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture(scope="module")
def tiny_sd(tmp_path_factory):
    from helpers import write_tiny_sd

    return write_tiny_sd(tmp_path_factory.mktemp("train-sd") / "sd")


def write_folder(root, sizes=((40, 48), (48, 40), (32, 32))):
    root.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    lines = []
    for index, (width, height) in enumerate(sizes):
        pixels = torch.randint(0, 256, (height, width, 3), generator=g, dtype=torch.uint8)
        PIL.fromarray(pixels.numpy()).save(root / f"{index}.png")
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


def arguments(tmp_path, **overrides):
    values = dict(
        output_dir=tmp_path / "run",
        max_steps=4,
        resolution=32,
        center_crop=True,
        logging_steps=1,
        save_steps=100,
    )
    return TrainingArguments(**(values | overrides))


# Profiles and arguments --------------------------------------------------------------------


def test_training_profiles():
    spec = get_training_spec("sd-v1-5")
    assert spec.support is TrainingSupport.LORA
    assert spec.family == "stable-diffusion"
    assert spec.target_modules == ("to_q", "to_k", "to_v", "to_out.0")
    for distilled in ("sdxs-512-0.9", "sd-turbo", "lcm-dreamshaper-v7"):
        assert get_training_spec(distilled).support is TrainingSupport.INFERENCE_ONLY
    assert get_training_spec("sdxl-base-1.0").support is TrainingSupport.INFERENCE_ONLY
    trainable = list_training_specs(support="lora")
    assert trainable and all(s.family == "stable-diffusion" for s in trainable)
    assert len(list_training_specs()) > len(trainable)
    with pytest.raises(KeyError, match="sd-v1-5"):
        get_training_spec("sd-v1-55")


@pytest.mark.parametrize(
    "bad",
    [
        dict(max_steps=0),
        dict(learning_rate=0),
        dict(lr_scheduler="step"),
        dict(mixed_precision="fp8"),
        dict(adam_beta1=1.0),
        dict(lora_dropout=1.0),
        dict(snr_gamma=0),
        dict(prediction_type="sample"),
        dict(report_to="tensorboard"),
        dict(center_crop="yes"),
        dict(lora_target_modules=()),
        dict(noise_offset=float("nan")),
    ],
)
def test_arguments_reject_invalid_values(tmp_path, bad):
    with pytest.raises(ValueError):
        TrainingArguments(**(dict(output_dir=tmp_path, max_steps=1) | bad))


def test_arguments_round_trip_to_dict(tmp_path):
    args = TrainingArguments(output_dir=tmp_path, max_steps=3, lora_target_modules=["to_q"])
    values = args.to_dict()
    assert values["output_dir"] == str(tmp_path)
    assert values["lora_target_modules"] == ["to_q"]
    json.dumps(values)


# Data --------------------------------------------------------------------------------------


def test_image_folder_dataset(tmp_path):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    assert len(dataset) == 3
    item = dataset[1]
    assert item["text"] == "a cat number 1"
    assert item["image"].size == (48, 40)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "invalid JSON"),
        ('{"file_name": "0.png"}', "text"),
        ('{"file_name": "missing.png", "text": "x"}', "does not exist"),
        ('{"file_name": "bad.png", "text": "x"}', "readable image"),
    ],
)
def test_image_folder_dataset_errors(tmp_path, line, message):
    root = write_folder(tmp_path / "data")
    (root / "bad.png").write_bytes(b"not an image")
    (root / "metadata.jsonl").write_text(line + "\n")
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        ImageFolderDataset(root)


def test_preprocess_image_shapes_and_determinism(tmp_path):
    image = ImageFolderDataset(write_folder(tmp_path / "data"))[0]["image"]
    center = preprocess_image(image, resolution=32, center_crop=True, random_flip=False)
    assert center.shape == (3, 32, 32)
    assert -1 <= center.min() and center.max() <= 1

    def draw(seed):
        return preprocess_image(
            image,
            resolution=32,
            center_crop=False,
            random_flip=True,
            generator=torch.Generator().manual_seed(seed),
        )

    assert torch.equal(draw(1), draw(1))
    tensor = torch.rand(3, 32, 40)
    out = preprocess_image(tensor, resolution=32, center_crop=True, random_flip=False)
    assert torch.equal(out, (tensor[:, :, 4:36] - 0.5) / 0.5)


def test_lr_schedules():
    assert lr_lambda("constant", 5, 10)(0) == 1.0
    assert lr_lambda("constant_with_warmup", 4, 10)(2) == 0.5
    assert lr_lambda("linear", 0, 10)(5) == 0.5
    assert math.isclose(lr_lambda("cosine", 0, 10)(5), 0.5)
    with pytest.raises(ValueError):
        lr_lambda("step", 0, 1)


# Trainer -----------------------------------------------------------------------------------


def test_trainer_refuses_inference_only_models(tmp_path):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    with pytest.raises(ValueError, match="sd-v1-5"):
        Trainer("sdxl-base-1.0", tmp_path, arguments(tmp_path), dataset)


def test_training_changes_only_the_adapter_and_lowers_the_loss(tmp_path, tiny_sd, capsys):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data", sizes=((32, 32),)))
    args = arguments(tmp_path, max_steps=30, learning_rate=1e-2, lora_rank=2, lora_alpha=2)
    trainer = Trainer("sd-v1-5", tiny_sd, args, dataset)
    trainer._setup()
    unet = trainer.components.unet
    frozen = {k: v.clone() for k, v in unet.state_dict().items() if "lora_" not in k}
    probe_pixels, probe_ids = trainer._next_batch()
    trainer.position = 0

    def probe():
        with torch.no_grad():
            return float(
                trainer.objective(probe_pixels, probe_ids, torch.Generator().manual_seed(7))
            )

    before = probe()
    result = trainer.train()
    after = probe()
    assert result.global_step == 30
    assert math.isfinite(result.training_loss)
    assert after < before
    assert len(result.log_history) == 30
    assert "step 30/30" in capsys.readouterr().out
    for key, value in unet.state_dict().items():
        if "lora_" not in key:
            assert torch.equal(value, frozen[key]), key
    layers = lora_layers(unet)
    assert layers and all(isinstance(layer, LoRALayer) for layer in layers.values())
    assert (args.output_dir / "checkpoint-30" / "trainer_state.pt").is_file()


def test_resume_is_exact(tmp_path, tiny_sd):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    common = dict(max_steps=6, save_steps=3, random_flip=True, center_crop=False, noise_offset=0.1)
    straight = Trainer("sd-v1-5", tiny_sd, arguments(tmp_path / "a", **common), dataset)
    straight.train()
    resumed_args = arguments(tmp_path / "b", **common)
    resumed_args.output_dir.mkdir(parents=True)
    shutil.copytree(
        tmp_path / "a" / "run" / "checkpoint-3", resumed_args.output_dir / "checkpoint-3"
    )
    resumed = Trainer("sd-v1-5", tiny_sd, resumed_args, dataset)
    output = resumed.train(resume_from_checkpoint=True)
    assert output.global_step == 6
    expected = straight.injection.adapter_state_dict()
    actual = resumed.injection.adapter_state_dict()
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert resumed.log_history == straight.log_history


def test_gradient_accumulation_consumes_micro_batches(tmp_path, tiny_sd):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = arguments(tmp_path, max_steps=2, gradient_accumulation_steps=2, train_batch_size=2)
    trainer = Trainer("sd-v1-5", tiny_sd, args, dataset)
    output = trainer.train()
    assert output.global_step == 2
    assert trainer.epoch * 3 + trainer.position == 8


def test_bf16_training_runs(tmp_path, tiny_sd):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    trainer = Trainer(
        "sd-v1-5", tiny_sd, arguments(tmp_path, max_steps=2, mixed_precision="bf16"), dataset
    )
    output = trainer.train()
    assert math.isfinite(output.training_loss)
    assert all(p.dtype == torch.float32 for p in trainer.injection.parameters())


def test_saved_adapter_loads_for_inference(tmp_path, tiny_sd):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    args = arguments(tmp_path, max_steps=3, learning_rate=1e-2)
    trainer = Trainer("sd-v1-5", tiny_sd, args, dataset)
    trainer.train()
    path = trainer.save_model(tmp_path / "final")
    config = json.loads((tmp_path / "final" / "adapter_config.json").read_text())
    assert config["base_model"] == "sd-v1-5" and config["rank"] == 4
    with pipeline("t2i", backend="stable-diffusion", model=tiny_sd) as pipe:
        components = {"unet": pipe.backend.components.unet}
        weights = read_lora_file(path, components)
        assert {w.alpha for w in weights} == {4.0}
        plain = pipe("a cat", seed=1, steps=2, width=32, height=32).images[0].tensor
    with pipeline("t2i", backend="stable-diffusion", model=tiny_sd, lora=path) as pipe:
        tuned = pipe("a cat", seed=1, steps=2, width=32, height=32).images[0].tensor
    assert not torch.equal(plain, tuned)


def test_save_model_needs_training(tmp_path, tiny_sd):
    dataset = ImageFolderDataset(write_folder(tmp_path / "data"))
    trainer = Trainer("sd-v1-5", tiny_sd, arguments(tmp_path), dataset)
    with pytest.raises(RuntimeError, match="train"):
        trainer.save_model()


def test_objective_ignores_sampling_scheduler_settings(tiny_sd):
    from imagehub.backends.stable_diffusion import StableDiffusionComponents
    from imagehub.training import StableDiffusionLoRAObjective

    components = StableDiffusionComponents.load(
        tiny_sd, device=torch.device("cpu"), dtype=torch.float32
    )
    components.scheduler_config = components.scheduler_config | {
        "timestep_spacing": "linspace",
        "use_karras_sigmas": True,
        "prediction_type": "v_prediction",
    }
    objective = StableDiffusionLoRAObjective(components)
    assert objective.prediction_type == "v_prediction"
    assert objective.num_train_timesteps == 1000


@pytest.mark.parametrize("page", ["training.md", "training.tr.md"])
def test_training_support_table_matches_the_profiles(page):
    from pathlib import Path

    text = (Path(__file__).parents[1] / "docs" / "guides" / page).read_text(encoding="utf-8")
    table = text.split("<!-- training-support:start -->")[1].split("<!-- training-support:end -->")[
        0
    ]
    rows = [line.split("|")[1:3] for line in table.strip().splitlines()[2:]]
    documented = [(name.strip().strip("`"), support.strip()) for name, support in rows]
    expected = [(s.model_id, s.support.value) for s in list_training_specs(support="lora")]
    assert documented == expected
