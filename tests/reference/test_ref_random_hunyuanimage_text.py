"""HunyuanImage 2.1 text side: tokenizers, ByT5 encoder, guider and schedule parity."""

import json
import random
import shutil
import warnings

import pytest
import torch
from fixture_files import fixture_path

from imagehub.backends.hunyuanimage import AdaptiveProjectedMixGuidance, sigmas_for
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization import bytes_to_unicode
from imagehub.tokenization_byt5 import ByT5Tokenizer
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

FIXTURES = fixture_path()
CONFIGS = fixture_path("hunyuanimage_configs")


def _json(name):
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_text_encoder_published_layout():
    config = _json("text_encoder_config.json")
    published = _json("text_encoder_tensors.json")
    with torch.device("meta"):
        model = Qwen25VLTextModel(config)
    shapes = {name: torch.empty(shape, device="meta") for name, shape in published.items()}
    prepared = model.prepare_state_dict(shapes)
    assert {n: list(v.shape) for n, v in model.state_dict().items()} == {
        n: list(v.shape) for n, v in prepared.items()
    }


def test_glyph_encoder_published_layout_and_random_weights_exact():
    config = _json("text_encoder_2_config.json")
    published = _json("text_encoder_2_tensors.json")
    with torch.device("meta"):
        actual = T5EncoderModel(config).state_dict()
    assert {n: list(v.shape) for n, v in actual.items()} == published
    small = config | {"num_layers": 2}
    torch.manual_seed(0)
    reference = transformers.T5EncoderModel(transformers.T5Config(**small)).eval()
    model = T5EncoderModel(small).eval()
    state = {k: v for k, v in reference.state_dict().items() if k != "encoder.embed_tokens.weight"}
    model.load_checkpoint(state)
    ids = torch.randint(3, 260, (2, 128))
    mask = torch.zeros(2, 128, dtype=torch.int64)
    mask[0, :40] = 1
    mask[1, :1] = 1
    with torch.inference_mode():
        expected = reference(input_ids=ids, attention_mask=mask.float())[0]
        actual = model(ids, mask)[0]
    assert torch.equal(actual, expected)


def _byt5_directory(tmp_path):
    for name in ("added_tokens", "special_tokens_map", "tokenizer_config"):
        shutil.copy(CONFIGS / f"tokenizer_2_{name}.json", tmp_path / f"{name}.json")
    return tmp_path


def test_byt5_tokenizer_matches_reference(tmp_path):
    directory = _byt5_directory(tmp_path)
    reference = transformers.ByT5Tokenizer.from_pretrained(directory)
    ours = ByT5Tokenizer(directory)
    pieces = ["a", "b", " ", '"', "é", "中", "</s>", "<pad>", "<unk>", "<extra_id_5>", "\n", "Text"]
    generator = random.Random(0)
    prompts = ['Text "Tencent". ', "", "x" * 400, "中文"]
    for _ in range(1500):
        count = generator.choice([0, 3, 40, 200])
        prompts.append("".join(generator.choice(pieces) for _ in range(count)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for prompt in prompts:
            expected = reference(
                prompt,
                padding="max_length",
                max_length=128,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            ids, mask = ours(prompt, 128)
            assert torch.equal(ids, expected.input_ids), prompt
            assert torch.equal(mask, expected.attention_mask), prompt


def test_qwen_tokenizer_without_tokenizer_json_matches_reference(tmp_path):
    vocab = {c: i for i, c in enumerate(bytes_to_unicode().values())}
    merges = (FIXTURES / "qwen_tokenizer/merges.txt").read_text().splitlines()[1:]
    for merge in merges:
        vocab.setdefault("".join(merge.split()), len(vocab))
    specials = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>")
    for token in specials:
        vocab[token] = len(vocab)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    shutil.copyfile(FIXTURES / "qwen_tokenizer/merges.txt", tmp_path / "merges.txt")
    decoder = {
        str(vocab[token]): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token != "<tool_call>",
        }
        for token in specials
    }
    config = {
        "added_tokens_decoder": decoder,
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "tokenizer_class": "Qwen2Tokenizer",
        "model_max_length": 4096,
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config))
    reference = transformers.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    ours = Qwen2Tokenizer(tmp_path)
    for text in ("a cat", "", "<|im_start|>user\nhi cat<|im_end|>", "x<tool_call>y cat  cat"):
        expected = reference(
            [text], max_length=40, padding="max_length", truncation=True, return_tensors="pt"
        )
        ids, mask = ours(text, 40)
        assert torch.equal(ids, expected.input_ids)
        assert torch.equal(mask, expected.attention_mask.bool())


@pytest.mark.parametrize("steps", [1, 2, 7, 50])
def test_schedule_matches_reference(steps):
    import numpy as np

    config = _json("scheduler_scheduler_config.json")
    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    sigmas = np.linspace(1.0, 0.0, steps + 1)[:-1]
    assert sigmas.tolist() == sigmas_for(steps)
    reference.set_timesteps(sigmas=sigmas)
    ours = FlowMatchEulerDiscreteScheduler(config)
    ours.set_timesteps(steps, sigmas=sigmas_for(steps))
    assert torch.equal(ours.timesteps, reference.timesteps)
    assert torch.equal(ours.sigmas, reference.sigmas)
    reference.set_begin_index(0)
    ours.set_begin_index(0)
    sample = torch.randn(1, 4, 3, 3)
    for timestep in reference.timesteps:
        output = torch.randn(1, 4, 3, 3)
        expected = reference.step(output, timestep, sample, return_dict=False)[0]
        actual = ours.step(output, timestep, sample)
        assert torch.equal(actual, expected)
        sample = expected


@pytest.mark.parametrize("name", ["guider_guider_config.json", "ocr_guider_guider_config.json"])
@pytest.mark.parametrize("guidance", [None, 1.0, 5.0])
def test_guider_matches_reference(name, guidance):
    config = _json(name)
    reference = diffusers.guiders.AdaptiveProjectedMixGuidance(
        **{k: v for k, v in config.items() if not k.startswith("_")}
    )
    if guidance is not None:
        reference = reference.new(guidance_scale=guidance)
    ours = AdaptiveProjectedMixGuidance(config, guidance)
    steps = 12 if name.startswith("guider") else 42
    generator = torch.Generator().manual_seed(0)
    for step in range(steps):
        cond = torch.randn(1, 64, 5, 7, generator=generator)
        uncond = torch.randn(1, 64, 5, 7, generator=generator)
        reference.set_state(step=step, num_inference_steps=steps, timestep=None)
        assert reference.num_conditions == (2 if ours.cfg_enabled else 1)
        batches = reference.prepare_inputs({"x": (cond, uncond)})
        assert len(batches) == reference.num_conditions
        for batch, prediction in zip(batches, (cond, uncond), strict=False):
            batch.noise_pred = prediction
        expected = reference(batches)[0]
        actual = ours.combine(step, cond, uncond if ours.cfg_enabled else None)
        assert torch.equal(actual, expected)
