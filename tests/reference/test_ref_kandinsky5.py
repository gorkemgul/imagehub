"""Kandinsky 5 parity against a local published snapshot on CPU float32.

The published transformer and Qwen encoder are bfloat16.  Both implementations upcast them to
float32.  Each reference component is saved and freed before the corresponding native component
loads; end-to-end runs use four steps rather than the 50-step default.
"""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.kandinsky5 import _clip_text_encoder
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.kandinsky5_transformer import Kandinsky5Transformer3DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")
safetensors = pytest.importorskip("safetensors.torch")

CPU = torch.device("cpu")
MODEL_ID = "kandinsky-5-t2i-lite"
PROMPT = "a photograph of an astronaut riding a horse on the moon"
STEPS = 4
PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a promt engineer. Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>"
)


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is not None:
        files = sorted(
            {directory / name for name in json.loads(index.read_text())["weight_map"].values()}
        )
    else:
        files = sorted(directory.glob("*.safetensors"))
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes.update(
            {
                name: tuple(entry["shape"])
                for name, entry in header.items()
                if name != "__metadata__"
            }
        )
    return shapes


def _header_result(model, shapes):
    prepared = model.prepare_state_dict(
        {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
    )
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    actual = {name: tuple(value.shape) for name, value in prepared.items()}
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(shapes),
    }


def _load_diffusers_component(model_class, directory):
    """Load a float32 Diffusers model without requiring accelerate."""
    model = model_class.from_config(read_json(directory / "config.json"))
    state = safetensors.load_file(directory / "diffusion_pytorch_model.safetensors", device="cpu")
    model.load_state_dict(state)
    del state
    return model.eval()


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    with torch.device("meta"):
        result["text_encoder"] = _header_result(
            Qwen25VLTextModel(read_json(root / "text_encoder" / "config.json")),
            _checkpoint_shapes(root / "text_encoder"),
        )
        clip_config = read_json(root / "text_encoder_2" / "config.json")["text_config"]
        from imagehub.models.clip_text import CLIPTextModel

        result["text_encoder_2"] = _header_result(
            CLIPTextModel(clip_config),
            {
                name: shape
                for name, shape in _checkpoint_shapes(root / "text_encoder_2").items()
                if name.startswith("text_model.")
            },
        )
        result["transformer"] = _header_result(
            Kandinsky5Transformer3DModel(read_json(root / "transformer" / "config.json")),
            _checkpoint_shapes(root / "transformer"),
        )
        result["vae"] = _header_result(
            AutoencoderKL(read_json(root / "vae" / "config.json")), _checkpoint_shapes(root / "vae")
        )
    record("header_kandinsky5", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizers(record):
    root = snapshot_dir(MODEL_ID)
    qwen_reference = transformers.AutoProcessor.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    qwen_actual = Qwen2Tokenizer(root / "tokenizer")
    clip_reference = transformers.CLIPTokenizer.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    clip_actual = CLIPTokenizer(root / "tokenizer_2")
    qwen_text = PROMPT_TEMPLATE.format(PROMPT)
    expected_qwen = qwen_reference(text=[qwen_text], images=None, videos=None, return_tensors="pt")[
        "input_ids"
    ][0]
    expected_clip = clip_reference(
        PROMPT, max_length=77, truncation=True, padding="max_length", return_tensors="pt"
    ).input_ids[0]
    actual_clip = torch.tensor(clip_actual(PROMPT))
    exact = torch.equal(
        torch.tensor(qwen_actual.tokenize(qwen_text)), expected_qwen
    ) and torch.equal(actual_clip, expected_clip)
    record("kandinsky5_tokenizers", exact=exact, tolerance="exact")
    assert exact


def _reference_conditioning(root, tmp_path):
    processor = transformers.AutoProcessor.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    text = PROMPT_TEMPLATE.format(PROMPT)
    inputs = processor(text=[text], images=None, videos=None, return_tensors="pt")
    qwen = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        qwen_hidden = qwen(
            input_ids=inputs.input_ids, output_hidden_states=True, return_dict=True
        ).hidden_states[-1]
    del qwen
    gc.collect()
    clip_tokenizer = transformers.CLIPTokenizer.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    clip_inputs = clip_tokenizer(
        PROMPT, max_length=77, truncation=True, padding="max_length", return_tensors="pt"
    )
    clip = transformers.CLIPTextModel.from_pretrained(
        root / "text_encoder_2", local_files_only=True, torch_dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        clip_pooled = clip(**clip_inputs).pooler_output
    del clip
    gc.collect()
    path = tmp_path / "conditioning.pt"
    torch.save(
        (inputs.input_ids, inputs.attention_mask, qwen_hidden, clip_inputs.input_ids, clip_pooled),
        path,
    )
    return path


def test_text_encoders(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    saved = _reference_conditioning(root, tmp_path)
    qwen_ids, qwen_mask, expected_qwen, clip_ids, expected_clip = torch.load(
        saved, weights_only=True
    )
    qwen = Qwen25VLTextModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_qwen = qwen(qwen_ids, qwen_mask.bool())[-1]
    _check(record, "kandinsky5_qwen", actual_qwen, expected_qwen)
    del qwen
    gc.collect()
    clip = _clip_text_encoder(root / "text_encoder_2", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        hidden = clip(clip_ids)
        eos = int(clip.config["eos_token_id"])
        positions = (
            clip_ids.to(torch.int).argmax(dim=-1)
            if eos == 2
            else (clip_ids.to(torch.int) == eos).int().argmax(dim=-1)
        )
        actual_clip = hidden[torch.arange(hidden.shape[0]), positions]
    _check(record, "kandinsky5_clip", actual_clip, expected_clip)


def test_transformer_and_vae(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(0)
    latents = torch.randn(1, 1, 128, 128, 16, generator=generator)
    conditioning = _reference_conditioning(root, tmp_path)
    qwen_ids, qwen_mask, qwen_hidden, clip_ids, clip_pooled = torch.load(
        conditioning, weights_only=True
    )
    del qwen_ids, qwen_mask, clip_ids
    qwen_embed = qwen_hidden[:, 41:]
    reference = _load_diffusers_component(
        diffusers.Kandinsky5Transformer3DModel, root / "transformer"
    )
    with torch.inference_mode():
        expected_transformer = reference(
            latents,
            qwen_embed,
            torch.tensor([500.0]),
            clip_pooled,
            [torch.arange(1), torch.arange(64), torch.arange(64)],
            torch.arange(qwen_embed.shape[1]),
        ).sample
    saved = tmp_path / "components.pt"
    torch.save((qwen_embed, clip_pooled, expected_transformer), saved)
    del reference, qwen_hidden, qwen_embed, clip_pooled, expected_transformer
    gc.collect()
    transformer = Kandinsky5Transformer3DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    qwen_embed, clip_pooled, expected_transformer = torch.load(saved, weights_only=True)
    with torch.inference_mode():
        actual_transformer = transformer(
            latents,
            qwen_embed,
            torch.tensor([500.0]),
            clip_pooled,
            [torch.arange(1), torch.arange(64), torch.arange(64)],
            torch.arange(qwen_embed.shape[1]),
        )
    _check(record, "kandinsky5_transformer", actual_transformer, expected_transformer)
    del transformer
    gc.collect()
    reference = _load_diffusers_component(diffusers.AutoencoderKL, root / "vae")
    with torch.inference_mode():
        expected_vae = reference.decode(
            latents.permute(0, 4, 1, 2, 3).reshape(1, 16, 128, 128)
        ).sample
    del reference
    gc.collect()
    vae = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_vae = vae.decode(latents.permute(0, 4, 1, 2, 3).reshape(1, 16, 128, 128))
    _check(record, "kandinsky5_vae_decode", actual_vae, expected_vae)
    gc.collect()


@pytest.mark.parametrize("height,width,seed", [(1024, 1024, 0), (640, 1408, 7)])
def test_end_to_end(record, tmp_path, height, width, seed):
    root = snapshot_dir(MODEL_ID)
    transformer = _load_diffusers_component(
        diffusers.Kandinsky5Transformer3DModel, root / "transformer"
    )
    vae = _load_diffusers_component(diffusers.AutoencoderKL, root / "vae")
    reference = diffusers.Kandinsky5T2IPipeline.from_pretrained(
        root,
        transformer=transformer,
        vae=vae,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    reference.set_progress_bar_config(disable=True)
    expected = reference(
        PROMPT,
        height=height,
        width=width,
        num_inference_steps=STEPS,
        guidance_scale=3.5,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    ).image
    saved = tmp_path / f"e2e-{height}-{width}-{seed}.pt"
    torch.save(torch.from_numpy((expected * 255).round().astype("uint8")), saved)
    del reference, transformer, vae, expected
    gc.collect()
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, height=height, width=width, steps=STEPS, guidance=3.5, seed=seed
    )
    expected = torch.load(saved, weights_only=True)
    pixels = torch.stack([item.tensor for item in actual.images])
    passed = torch.equal(pixels, expected)
    name = "e2e_kandinsky5" if seed == 0 else "seed_e2e_kandinsky5"
    record(
        name, passed=passed, height=height, width=width, seed=seed, steps=STEPS, tolerance="exact"
    )
    assert passed
