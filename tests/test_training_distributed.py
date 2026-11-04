"""Data-parallel training with two CPU processes (gloo), started like ``torchrun`` would."""

import json
import math
import os
import shutil
import socket

import pytest
import torch
import torch.multiprocessing as mp

from imagehub.backends.stable_diffusion import StableDiffusionComponents
from imagehub.lora import LoRAConfig, inject_lora
from imagehub.training import ImageFolderDataset, Trainer, TrainingArguments
from imagehub.training.schedules import lr_lambda

PIL = pytest.importorskip("PIL.Image")
pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is not available"
)

WORLD = 2


class Recording:
    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __call__(self, pixels, ids, generator):
        posterior, noise, timesteps = self.inner.draw(pixels, generator)
        self.calls.append((pixels.clone(), ids.clone(), posterior, noise, timesteps))
        latents = self.inner.encode_images(pixels, posterior)
        return self.inner.loss(latents, self.inner.encode_text(ids), noise, timesteps)


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_folder(root, count=5):
    root.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    lines = []
    for index in range(count):
        pixels = torch.randint(0, 256, (32, 40, 3), generator=g, dtype=torch.uint8)
        PIL.frombytes("RGB", (40, 32), bytes(pixels.flatten().tolist())).save(root / f"{index}.png")
        lines.append(json.dumps({"file_name": f"{index}.png", "text": f"a cat number {index}"}))
    (root / "metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


def _worker(rank, port, job):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD),
    )
    torch.set_num_threads(1)
    data = ImageFolderDataset(job["data"])
    trainer = Trainer(
        "sd-v1-5",
        job["model"],
        TrainingArguments(**job["args"]),
        data,
        eval_dataset=data if job.get("evaluate") else None,
    )
    trainer._setup()
    initial = trainer.injection.adapter_state_dict()
    trainer.objective = Recording(trainer.objective)
    trainer.train(resume_from_checkpoint=job.get("resume", False))
    result = {
        "initial": initial,
        "final": trainer.injection.adapter_state_dict(),
        "calls": trainer.objective.calls,
        "log_history": trainer.log_history,
    }
    if job.get("evaluate"):
        result["eval"] = trainer.evaluate()
    torch.save(result, os.path.join(job["out"], f"rank{rank}.pt"))
    torch.distributed.destroy_process_group()


def run(job):
    os.makedirs(job["out"], exist_ok=True)
    mp.spawn(_worker, args=(free_port(), job), nprocs=WORLD, join=True)
    return [torch.load(os.path.join(job["out"], f"rank{r}.pt")) for r in range(WORLD)]


def job_for(tmp_path, model, name, **args):
    values = dict(
        output_dir=str(tmp_path / name / "run"),
        max_steps=3,
        resolution=32,
        random_flip=True,
        learning_rate=1e-2,
        logging_steps=1,
        save_steps=100,
    )
    return {
        "model": str(model),
        "data": str(write_folder(tmp_path / "data")),
        "out": str(tmp_path / name / "results"),
        "args": values | args,
    }


@pytest.fixture(autouse=True)
def restore_threads():
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def tiny_sd(tmp_path_factory):
    from helpers import write_tiny_sd

    return write_tiny_sd(tmp_path_factory.mktemp("ddp-sd") / "sd")


def test_two_processes_equal_a_replay_with_averaged_gradients(tmp_path, tiny_sd):
    results = run(job_for(tmp_path, tiny_sd, "ddp", max_grad_norm=0.05, lr_scheduler="cosine"))
    first, second = results
    for key in first["final"]:
        assert torch.equal(first["final"][key], second["final"][key]), key
    assert first["log_history"] == second["log_history"]
    # Each process trained on different images.
    assert not torch.equal(first["calls"][0][0], second["calls"][0][0])

    torch.set_num_threads(1)  # the processes ran with one thread each
    components = StableDiffusionComponents.load(
        tiny_sd, device=torch.device("cpu"), dtype=torch.float32
    )
    for module in (components.unet, components.text_encoder, components.vae):
        module.requires_grad_(False)
    injection = inject_lora(
        components.unet,
        LoRAConfig(rank=4, alpha=4, target_modules=("to_q", "to_k", "to_v", "to_out.0")),
    )
    injection.load_adapter_state_dict(first["initial"])
    components.unet.train()
    from imagehub.training import StableDiffusionLoRAObjective

    objective = StableDiffusionLoRAObjective(components)
    parameters = list(injection.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-2, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda("cosine", 0, 3))
    losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        for result in results:
            pixels, ids, posterior, noise, timesteps = result["calls"][step]
            latents = objective.encode_images(pixels, posterior)
            loss = objective.loss(latents, objective.encode_text(ids), noise, timesteps)
            loss.backward()
            step_losses.append(float(loss))
        for parameter in parameters:
            parameter.grad.div_(WORLD)
        torch.nn.utils.clip_grad_norm_(parameters, 0.05)
        optimizer.step()
        scheduler.step()
        losses.append(sum(step_losses) / WORLD)
    replayed = injection.adapter_state_dict()
    for key, value in replayed.items():
        assert torch.equal(first["final"][key], value), key
    logged = [entry["loss"] for entry in first["log_history"]]
    assert all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(logged, losses, strict=True))


def test_distributed_resume_and_world_size_check(tmp_path, tiny_sd):
    straight = run(job_for(tmp_path, tiny_sd, "a", max_steps=4, save_steps=2))
    checkpoints = sorted(p.name for p in (tmp_path / "a" / "run").iterdir())
    assert checkpoints == ["checkpoint-2", "checkpoint-4"]

    resumed_job = job_for(tmp_path, tiny_sd, "b", max_steps=4, save_steps=2)
    resumed_job["resume"] = True
    os.makedirs(resumed_job["args"]["output_dir"])
    shutil.copytree(
        tmp_path / "a" / "run" / "checkpoint-2",
        os.path.join(resumed_job["args"]["output_dir"], "checkpoint-2"),
    )
    resumed = run(resumed_job)
    for key, value in straight[0]["final"].items():
        assert torch.equal(resumed[0]["final"][key], value), key
        assert torch.equal(resumed[1]["final"][key], value), key
    assert resumed[0]["log_history"] == straight[0]["log_history"]

    single = Trainer(
        "sd-v1-5",
        tiny_sd,
        TrainingArguments(output_dir=tmp_path / "a" / "run", max_steps=4, resolution=32),
        ImageFolderDataset(tmp_path / "data"),
    )
    with pytest.raises(ValueError, match="2 processes"):
        single.train(resume_from_checkpoint=True)


def test_distributed_evaluation_matches_one_process(tmp_path, tiny_sd):
    job = job_for(tmp_path, tiny_sd, "eval", max_steps=1)
    job["evaluate"] = True
    results = run(job)
    assert results[0]["eval"] == results[1]["eval"]
    torch.set_num_threads(1)
    single = Trainer(
        "sd-v1-5",
        tiny_sd,
        TrainingArguments(**(job["args"] | {"output_dir": str(tmp_path / "single")})),
        ImageFolderDataset(tmp_path / "data"),
    )
    single._setup()
    single.injection.load_adapter_state_dict(results[0]["final"])
    expected = single.evaluate(ImageFolderDataset(tmp_path / "data"))["eval_loss"]
    assert math.isclose(results[0]["eval"]["eval_loss"], expected, rel_tol=1e-9)
