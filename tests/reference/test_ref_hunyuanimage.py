"""HunyuanImage 2.1 parity on CPU float32 against diffusers and transformers.

The transformer alone is about 68 GB in float32, so every reference stage runs in a
subprocess that saves its result and exits before the native model is loaded; two large
models are never held at once. The end-to-end runs use shortened step counts.
"""

import gc
import json
import struct
import textwrap

import pytest
import torch
from helpers import max_diff, reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.backends.hunyuanimage import (
    _TEMPLATE,
    AdaptiveProjectedMixGuidance,
    HunyuanImageBackend,
    encode_glyph,
    encode_qwen,
    sigmas_for,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_hunyuanimage import AutoencoderKLHunyuanImage
from imagehub.models.hunyuanimage_transformer import HunyuanImageTransformer2DModel
from imagehub.models.qwen2_5_vl_text import Qwen25VLTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler
from imagehub.tokenization_byt5 import ByT5Tokenizer
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "hunyuanimage-2-1"
PROMPT = "A cute cartoon penguin plush toy standing in a painting studio, photorealistic"
GLYPH_PROMPT = "A red beret with the word \"Tencent\" on it, and a sign that reads 'Hello'"
E2E_STEPS = 8
E2E_SIZE = 512

_PRELUDE = textwrap.dedent(
    """
    import json, sys, torch, diffusers, transformers
    from pathlib import Path
    args = json.loads(sys.argv[1])
    root = Path(args["root"])
    out = args["out"]
    torch.set_num_threads(args["threads"])
    CPU = torch.device("cpu")

    def pipe_with(**modules):
        names = ["scheduler", "vae", "text_encoder", "tokenizer", "text_encoder_2",
                 "tokenizer_2", "transformer", "guider", "ocr_guider"]
        return diffusers.HunyuanImagePipeline(**{n: modules.get(n) for n in names})

    def guiders():
        return [
            diffusers.guiders.AdaptiveProjectedMixGuidance(
                **{k: v for k, v in json.loads(
                    (root / name / "guider_config.json").read_text()).items() if k[0] != "_"})
            for name in ("guider", "ocr_guider")
        ]
    """
)
_QWEN = textwrap.dedent(
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True)
    text_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, dtype=torch.float32)
    pipe = pipe_with(text_encoder=text_encoder, tokenizer=tokenizer)
    result = {}
    with torch.inference_mode():
        for key, prompt in (("prompt", args["prompt"]), ("negative", "")):
            result[key] = pipe._get_qwen_prompt_embeds(
                tokenizer=tokenizer, text_encoder=text_encoder, prompt=[prompt], device=CPU,
                tokenizer_max_length=1000, template=pipe.prompt_template_encode, drop_idx=34)
        ids = tokenizer([pipe.prompt_template_encode.format(args["prompt"])], max_length=1034,
                        padding="max_length", truncation=True, return_tensors="pt")
        result["ids"] = ids.input_ids
        result["token_mask"] = ids.attention_mask
    torch.save(result, out)
    """
)
_GLYPH = textwrap.dedent(
    """
    tokenizer_2 = transformers.ByT5Tokenizer.from_pretrained(root / "tokenizer_2")
    text_encoder_2 = transformers.T5EncoderModel.from_pretrained(
        root / "text_encoder_2", local_files_only=True, dtype=torch.float32)
    pipe = pipe_with(text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2)
    with torch.inference_mode():
        result = pipe._get_byt5_prompt_embeds(
            tokenizer=tokenizer_2, text_encoder=text_encoder_2, prompt=args["glyph"], device=CPU,
            tokenizer_max_length=128)
    torch.save(result, out)
    """
)
_TRANSFORMER_STEP = textwrap.dedent(
    """
    transformer = diffusers.HunyuanImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32).to(torch.float32)
    inputs = torch.load(args["inputs"], weights_only=True)
    with torch.inference_mode():
        result = transformer(**inputs, return_dict=False)[0]
    torch.save(result, out)
    """
)
_DENOISE = textwrap.dedent(
    """
    transformer = diffusers.HunyuanImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32).to(torch.float32)
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    guider, ocr_guider = guiders()
    pipe = pipe_with(transformer=transformer, scheduler=scheduler, guider=guider,
                     ocr_guider=ocr_guider)
    pipe.set_progress_bar_config(disable=True)
    embeds = torch.load(args["embeds"], weights_only=True)
    kwargs = dict(
        prompt_embeds=embeds["prompt"][0], prompt_embeds_mask=embeds["prompt"][1],
        negative_prompt_embeds=embeds["negative"][0],
        negative_prompt_embeds_mask=embeds["negative"][1],
        prompt_embeds_2=embeds["glyph"][0], prompt_embeds_mask_2=embeds["glyph"][1],
        negative_prompt_embeds_2=embeds["glyph"][0],
        negative_prompt_embeds_mask_2=embeds["glyph"][1],
        num_inference_steps=args["steps"], height=args["size"], width=args["size"],
        output_type="latent")
    if args.get("latents"):
        kwargs["latents"] = torch.load(args["latents"], weights_only=True)
    else:
        kwargs["generator"] = torch.Generator("cpu").manual_seed(args["seed"])
    with torch.inference_mode():
        result = pipe(**kwargs).images
    torch.save(result, out)
    """
)
_DECODE = textwrap.dedent(
    """
    from diffusers.image_processor import VaeImageProcessor
    vae = diffusers.AutoencoderKLHunyuanImage.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32)
    latents = torch.load(args["latents"], weights_only=True)
    with torch.inference_mode():
        image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        pixels = VaeImageProcessor(vae_scale_factor=32).postprocess(image, output_type="np")
    pixels = torch.from_numpy((pixels * 255).round().astype("uint8"))
    torch.save({"image": image, "pixels": pixels}, out)
    """
)
_VAE_CHECK = textwrap.dedent(
    """
    vae = diffusers.AutoencoderKLHunyuanImage.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32)
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 64, 32, 32, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    with torch.inference_mode():
        result = {"decode": vae.decode(latents, return_dict=False)[0],
                  "encode": vae.encode(image).latent_dist.mode()}
    torch.save(result, out)
    """
)


@pytest.fixture(scope="module")
def root():
    return snapshot_dir(MODEL_ID)


def _reference(tmp_path, name, body, **args):
    """Run a reference stage in a fresh process whose exit frees its memory."""
    out = tmp_path / f"{name}.pt"
    args |= {
        "root": str(args.pop("snapshot")),
        "threads": torch.get_num_threads(),
    }
    return reference_in_subprocess(
        "reference.hunyuanimage.run", {"body": _PRELUDE + body, "args": args}, out
    )


def _check(record, name, actual, expected):
    record(name, **max_diff(actual, expected), tolerance=0.0)
    assert torch.equal(actual, expected)


def _shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    files = (
        [directory / name for name in sorted(set(read_json(index)["weight_map"].values()))]
        if index
        else [next(directory.glob("*.safetensors"))]
    )
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    return shapes


def _compare(expected, published):
    return {
        "missing": sorted(expected.keys() - published.keys()),
        "extra": sorted(published.keys() - expected.keys()),
        "mismatch": sorted(
            n for n in expected.keys() & published.keys() if expected[n] != published[n]
        ),
        "tensors": len(published),
    }


def test_header(record, root):
    result = {}
    for component, cls in {
        "transformer": HunyuanImageTransformer2DModel,
        "vae": AutoencoderKLHunyuanImage,
        "text_encoder_2": T5EncoderModel,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        expected = {n: tuple(v.shape) for n, v in model.state_dict().items()}
        result[component] = _compare(expected, _shapes(root / component))
    with torch.device("meta"):
        text = Qwen25VLTextModel(read_json(root / "text_encoder" / "config.json"))
    published = {
        n: torch.empty(s, device="meta") for n, s in _shapes(root / "text_encoder").items()
    }
    prepared = {n: tuple(v.shape) for n, v in text.prepare_state_dict(published).items()}
    result["text_encoder"] = _compare(
        {n: tuple(v.shape) for n, v in text.state_dict().items()},
        prepared,
    )
    result["text_encoder"]["tensors"] = len(published)
    record("header_hunyuanimage_2_1", **result)
    assert all(not e[k] for e in result.values() for k in ("missing", "extra", "mismatch"))


def test_tokenizers(record, root):
    qwen_reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    byt5_reference = transformers.ByT5Tokenizer.from_pretrained(root / "tokenizer_2")
    qwen, byt5 = Qwen2Tokenizer(root / "tokenizer"), ByT5Tokenizer(root / "tokenizer_2")

    prompts = [PROMPT, GLYPH_PROMPT, "", "日本語のテキスト <|im_end|>x"]
    for prompt in prompts:
        text = _TEMPLATE.format(prompt)
        expected = qwen_reference(
            [text], max_length=1034, padding="max_length", truncation=True, return_tensors="pt"
        )
        ids, mask = qwen(text, 1034)
        assert torch.equal(ids, expected.input_ids)
        assert torch.equal(mask, expected.attention_mask.bool())
        expected = byt5_reference(
            prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt"
        )
        ids, mask = byt5(prompt, 128)
        assert torch.equal(ids, expected.input_ids)
        assert torch.equal(mask, expected.attention_mask)
    record("hunyuanimage_tokenizers", exact=True, prompts=len(prompts), tolerance="exact")


def test_text_encoder_and_glyph_encoder(record, root, tmp_path):
    expected = _reference(tmp_path, "qwen", _QWEN, snapshot=root, prompt=PROMPT)
    (actual_prompt, actual_negative) = encode_qwen(
        root, [PROMPT, ""], device=CPU, dtype=torch.float32
    )
    _check(record, "hunyuanimage_text_encoder", actual_prompt[0], expected["prompt"][0])
    _check(
        record, "hunyuanimage_text_encoder_negative", actual_negative[0], expected["negative"][0]
    )
    assert torch.equal(actual_prompt[1].long(), expected["prompt"][1].long())
    gc.collect()
    glyph = "Text \"Tencent\". Text 'Hello'. "
    expected_glyph = _reference(tmp_path, "glyph", _GLYPH, snapshot=root, glyph=glyph)
    actual_glyph = encode_glyph(root, glyph, device=CPU, dtype=torch.float32)
    _check(record, "hunyuanimage_glyph_encoder", actual_glyph[0], expected_glyph[0])
    assert torch.equal(actual_glyph[1].long(), expected_glyph[1].long())


def test_transformer_single_step(record, root, tmp_path):
    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    mask = torch.zeros(1, 1000, dtype=torch.int64)
    mask[:, :37] = 1
    mask_2 = torch.zeros(1, 128, dtype=torch.int64)
    mask_2[:, :20] = 1
    inputs = {
        "hidden_states": torch.randn(1, 64, 16, 16, generator=generator),
        "timestep": torch.tensor([612.5]),
        "encoder_hidden_states": torch.randn(
            1, 1000, config["text_embed_dim"], generator=generator
        ),
        "encoder_attention_mask": mask,
        "encoder_hidden_states_2": torch.randn(
            1, 128, config["text_embed_2_dim"], generator=generator
        ),
        "encoder_attention_mask_2": mask_2,
    }
    torch.save(inputs, tmp_path / "inputs.pt")
    expected = _reference(
        tmp_path, "step", _TRANSFORMER_STEP, snapshot=root, inputs=str(tmp_path / "inputs.pt")
    )
    native = HunyuanImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = native(**inputs)
    _check(record, "hunyuanimage_transformer", actual, expected)
    del native
    gc.collect()


def test_vae_decode_and_encode(record, root, tmp_path):
    expected = _reference(tmp_path, "vae", _VAE_CHECK, snapshot=root)
    generator = torch.Generator("cpu").manual_seed(1)
    latents = torch.randn(1, 64, 32, 32, generator=generator)
    image = torch.rand(1, 3, 1024, 1024, generator=generator) * 2 - 1
    native = AutoencoderKLHunyuanImage.from_pretrained(
        root / "vae", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        decoded, encoded = native.decode(latents), native.encode(image)
    _check(record, "hunyuanimage_vae_decode", decoded, expected["decode"])
    _check(record, "hunyuanimage_vae_encode", encoded, expected["encode"])


def test_scheduler(record, root):
    config = read_json(root / "scheduler" / "scheduler_config.json")
    import numpy as np

    reference = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    reference.set_timesteps(sigmas=np.linspace(1.0, 0.0, 51)[:-1])
    native = FlowMatchEulerDiscreteScheduler(config)
    native.set_timesteps(50, sigmas=sigmas_for(50))
    exact = torch.equal(native.sigmas, reference.sigmas) and torch.equal(
        native.timesteps, reference.timesteps
    )
    record("hunyuanimage_scheduler", exact=exact, tolerance="exact")
    assert exact


def _reference_embeds(root, tmp_path, prompt):
    qwen = _reference(tmp_path, "qwen_e2e", _QWEN, snapshot=root, prompt=prompt)
    d_model = int(read_json(root / "text_encoder_2" / "config.json")["d_model"])
    embeds = {
        "prompt": qwen["prompt"],
        "negative": qwen["negative"],
        "glyph": (torch.zeros(1, 128, d_model), torch.zeros(1, 128, dtype=torch.int64)),
    }
    torch.save(embeds, tmp_path / "embeds.pt")
    return str(tmp_path / "embeds.pt")


def _reference_image(root, tmp_path, embeds, **args):
    latents = _reference(tmp_path, "denoise", _DENOISE, snapshot=root, embeds=embeds, **args)
    torch.save(latents, tmp_path / "final.pt")
    decoded = _reference(
        tmp_path, "decode", _DECODE, snapshot=root, latents=str(tmp_path / "final.pt")
    )
    return latents, decoded["pixels"]


def test_end_to_end_with_fixed_noise(record, root, tmp_path):
    """Shortened to 8 steps at 512x512; covers classifier-free and projected guidance."""
    noise = torch.randn(
        1, 64, E2E_SIZE // 32, E2E_SIZE // 32, generator=torch.Generator("cpu").manual_seed(0)
    )
    torch.save(noise, tmp_path / "noise.pt")
    embeds = _reference_embeds(root, tmp_path, PROMPT)
    expected_latents, expected_pixels = _reference_image(
        root,
        tmp_path,
        embeds,
        steps=E2E_STEPS,
        size=E2E_SIZE,
        latents=str(tmp_path / "noise.pt"),
    )
    backend = HunyuanImageBackend(root, model_id=MODEL_ID)
    reference_embeds = torch.load(embeds, weights_only=True)
    native_embeds = encode_qwen(root, [PROMPT, ""], device=CPU, dtype=torch.float32)
    _check(record, "hunyuanimage_e2e_text", native_embeds[0][0], reference_embeds["prompt"][0])
    guider = AdaptiveProjectedMixGuidance(read_json(root / "guider" / "guider_config.json"))
    actual_latents = backend._denoise(
        pos=reference_embeds["prompt"],
        neg=reference_embeds["negative"],
        glyph_pos=reference_embeds["glyph"],
        glyph_neg=reference_embeds["glyph"],
        guider=guider,
        steps=E2E_STEPS,
        latents=noise.clone(),
    )
    _check(record, "hunyuanimage_e2e_latents", actual_latents, expected_latents)
    gc.collect()
    result = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, steps=E2E_STEPS, width=E2E_SIZE, height=E2E_SIZE, latents=noise
    )
    pixels = torch.stack([item.tensor for item in result.images])
    passed = torch.equal(pixels, expected_pixels)
    record(
        "e2e_hunyuanimage_2_1",
        passed=passed,
        max_pixel_diff=int((pixels.int() - expected_pixels.int()).abs().max()),
        steps=E2E_STEPS,
        size=E2E_SIZE,
    )
    assert passed


def test_seed_end_to_end_42(record, root, tmp_path):
    steps, size = 3, 256
    embeds = _reference_embeds(root, tmp_path, PROMPT)
    _, expected_pixels = _reference_image(root, tmp_path, embeds, steps=steps, size=size, seed=42)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, steps=steps, width=size, height=size, seed=42
    )
    pixels = torch.stack([item.tensor for item in result.images])
    passed = torch.equal(pixels, expected_pixels)
    record("seed_e2e_hunyuanimage_2_1", passed=passed, seed=42, steps=steps, size=size)
    assert passed
