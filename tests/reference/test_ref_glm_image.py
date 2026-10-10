"""GLM-Image real-checkpoint parity on CPU float32 (skips without a local snapshot).

The model has about 16 billion parameters, so the reference runs stage by stage in fresh
processes, each saves its result, and the native code is loaded only afterwards. The end-to-end
test shortens the diffusion stage to a few steps; the autoregressive stage runs in full.
"""

import json
import multiprocessing
import random
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.glm_image import GlmImageBackend, glyph_embeds
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.glm_image_ar import GlmImageARModel
from imagehub.models.glm_image_transformer import GlmImageTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization_byt5 import ByT5Tokenizer
from imagehub.tokenization_glm_image import GlmImageTokenizer, build_prior_prompt

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
MODEL_ID = "glm-image"
PROMPT = (
    'A close-up photograph of a wooden shop sign that reads "Open Late" above a small chalkboard '
    "saying 'Fresh Bread', warm evening light"
)
END_TO_END_STEPS = 4


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _headers(directory: Path):
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = [next(directory.glob("*.safetensors"))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {k: tuple(v["shape"]) for k, v in header.items() if k != "__metadata__"}
    return shapes


def _in_subprocess(function, *args):
    """Run a reference stage in a fresh process so its memory is returned afterwards.

    The intra-op thread count is the one of this process: CPU kernels split their sums by thread.
    """
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        1, initializer=torch.set_num_threads, initargs=(torch.get_num_threads(),)
    ) as pool:
        return pool.apply(function, args)


def _reference_pipeline(root, **components):
    byt5 = transformers.ByT5Tokenizer.from_pretrained(str(root / "tokenizer"))
    processor = transformers.AutoProcessor.from_pretrained(str(root / "processor"))
    return diffusers.GlmImagePipeline(
        tokenizer=byt5,
        processor=processor,
        text_encoder=components.get("text_encoder"),
        vision_language_encoder=components.get("vision_language_encoder"),
        vae=components.get("vae"),
        transformer=components.get("transformer"),
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(root / "scheduler")
        ),
    )


def _reference_prior(root, prompt, height, width, seed, output):
    model = transformers.GlmImageForConditionalGeneration.from_pretrained(
        str(root / "vision_language_encoder"), dtype=torch.float32
    ).eval()
    reference = _reference_pipeline(root, vision_language_encoder=model)
    with torch.inference_mode():
        tokens, _, _ = reference.generate_prior_tokens(
            prompt=prompt,
            height=height,
            width=width,
            image=None,
            device=CPU,
            generator=torch.Generator("cpu").manual_seed(seed),
        )
    torch.save(tokens, output)


def _reference_glyphs(root, prompt, output):
    encoder = transformers.T5EncoderModel.from_pretrained(
        str(root / "text_encoder"), dtype=torch.float32
    ).eval()
    reference = _reference_pipeline(root, text_encoder=encoder)
    with torch.inference_mode():
        torch.save(
            {
                "cond": reference._get_glyph_embeds(prompt, 2048, CPU, torch.float32),
                "uncond": reference._get_glyph_embeds("", 2048, CPU, torch.float32),
            },
            output,
        )


def _reference_step(root, inputs, output):
    transformer = diffusers.GlmImageTransformer2DModel.from_pretrained(
        str(root / "transformer"), torch_dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        result = {
            name: transformer(
                hidden_states=inputs["latents"],
                encoder_hidden_states=inputs[name],
                prior_token_id=inputs["prior"],
                prior_token_drop=torch.full_like(
                    inputs["prior"], name == "uncond", dtype=torch.bool
                ),
                timestep=inputs["timestep"],
                target_size=inputs["target_size"],
                crop_coords=inputs["crop_coords"],
                return_dict=False,
            )[0]
            for name in ("cond", "uncond")
        }
    torch.save(result, output)


def _reference_image(root, prompt, prior, height, width, steps, seed, output):
    text_encoder = transformers.T5EncoderModel.from_pretrained(
        str(root / "text_encoder"), dtype=torch.float32
    ).eval()
    transformer = diffusers.GlmImageTransformer2DModel.from_pretrained(
        str(root / "transformer"), torch_dtype=torch.float32
    ).eval()
    vae = diffusers.AutoencoderKL.from_pretrained(str(root / "vae"), torch_dtype=torch.float32)
    reference = _reference_pipeline(
        root, text_encoder=text_encoder, transformer=transformer, vae=vae.eval()
    )
    with torch.inference_mode():
        image = reference(
            prompt,
            prior_token_ids=prior,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images[0]
    torch.save(torch.from_numpy((image * 255).round().astype("uint8")), output)


def test_glm_image_headers_match_native_models(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for name, cls, config_name in (
        ("vision_language_encoder", GlmImageARModel, "config.json"),
        ("text_encoder", T5EncoderModel, "config.json"),
        ("transformer", GlmImageTransformer2DModel, "config.json"),
        ("vae", AutoencoderKL, "config.json"),
    ):
        config = json.loads((root / name / config_name).read_text(encoding="utf-8"))
        with torch.device("meta"):
            model = cls(config)
        header = _headers(root / name)
        prepared = model.prepare_state_dict({k: torch.empty(0) for k in header})
        expected = model.state_dict()
        missing = sorted(set(expected) - set(prepared))
        extra = sorted(set(prepared) - set(expected))
        mismatch = [
            k for k in expected if k in prepared and tuple(header[k]) != tuple(expected[k].shape)
        ]
        result[name] = {
            "missing": missing,
            "extra": extra,
            "mismatch": mismatch,
            "tensors": len(header),
        }
        assert not missing and not extra and not mismatch, name
    record("header_glm-image", **result)


def test_glm_image_tokenizers_match_the_reference(record):
    root = snapshot_dir(MODEL_ID)
    processor = transformers.AutoProcessor.from_pretrained(str(root / "processor"))
    ours = GlmImageTokenizer(root / "processor")
    prompts = [
        PROMPT,
        "a cat",
        "",
        "日本語 中文 한국어 \U0001f431 <|image|> [gMASK] <think>",
        "numbers 1234567 3.14  spaces\n\nnewline",
    ]
    for prompt in prompts:
        for height, width in ((1024, 1024), (1152, 768), (768, 1344)):
            expected = processor.apply_chat_template(
                [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]],
                tokenize=True,
                padding=False,
                target_h=height,
                target_w=width,
                return_dict=True,
                return_tensors="pt",
            )
            ids, grids = build_prior_prompt(ours, prompt, height, width)
            assert torch.equal(ids, expected["input_ids"])
            assert torch.equal(
                torch.tensor([[1, *grid] for grid in grids]), expected["image_grid_thw"]
            )
    rng = random.Random(0)
    pieces = [
        "a",
        "Cat",
        "日本",
        "\U0001f431",
        "1234567",
        "'s",
        " ",
        "\n",
        "!?.",
        "<sop>",
        "<|image|>",
    ]
    for _ in range(300):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 10)))
        assert ours.tokenize(text) == processor.tokenizer(text)["input_ids"], text
    byt5 = transformers.ByT5Tokenizer.from_pretrained(str(root / "tokenizer"))
    native = ByT5Tokenizer(root / "tokenizer")
    for text in ("Open Late", "Fresh Bread", "", "你好", "x" * 3000, "<extra_id_3> a"):
        assert (
            native.tokenize(text, 2048)
            == byt5([text], max_length=2048, truncation=True).input_ids[0]
        )
    record("glm_image_tokenizers", prompts=len(prompts), fuzz=300)


def test_glm_image_glyph_embeddings(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    _in_subprocess(_reference_glyphs, root, PROMPT, tmp_path / "glyphs.pt")
    saved = torch.load(tmp_path / "glyphs.pt", weights_only=True)
    encoder = T5EncoderModel.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    tokenizer = ByT5Tokenizer(root / "tokenizer")
    for name, prompt in (("cond", PROMPT), ("uncond", "")):
        actual = glyph_embeds(tokenizer, encoder, prompt, 2048, CPU, torch.float32)
        _check(record, f"glm_image_glyph_{name}", actual, saved[name])


def test_glm_image_prior_tokens_512(record, tmp_path):
    # Sampling with the published generation settings; the global seed drives the reference.
    root = snapshot_dir(MODEL_ID)
    _in_subprocess(_reference_prior, root, PROMPT, 512, 512, 0, tmp_path / "prior.pt")
    saved = torch.load(tmp_path / "prior.pt", weights_only=True)
    backend = GlmImageBackend(root)
    backend.load()
    actual = backend._prior_tokens(PROMPT, 512, 512, 0)
    record("glm_image_prior_tokens_512", equal=bool(torch.equal(actual, saved)))
    assert torch.equal(actual, saved)


def test_glm_image_transformer_conditional_and_unconditional_step(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(0)
    height = width = 512
    glyphs = torch.load(tmp_path / "glyphs.pt") if (tmp_path / "glyphs.pt").exists() else None
    if glyphs is None:
        _in_subprocess(_reference_glyphs, root, PROMPT, tmp_path / "glyphs.pt")
        glyphs = torch.load(tmp_path / "glyphs.pt", weights_only=True)
    inputs = {
        "latents": torch.randn(1, 16, height // 8, width // 8, generator=generator),
        "cond": glyphs["cond"],
        "uncond": glyphs["uncond"],
        "prior": torch.randint(0, 16384, (1, (height // 16) * (width // 16)), generator=generator),
        "timestep": torch.tensor([749.0]),
        "target_size": torch.tensor([[float(height), float(width)]]),
        "crop_coords": torch.zeros(1, 2),
    }
    _in_subprocess(_reference_step, root, inputs, tmp_path / "step.pt")
    saved = torch.load(tmp_path / "step.pt", weights_only=True)
    model = GlmImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        for name in ("cond", "uncond"):
            actual = model(
                inputs["latents"],
                inputs[name],
                inputs["prior"],
                torch.full_like(inputs["prior"], name == "uncond", dtype=torch.bool),
                inputs["timestep"],
                inputs["target_size"],
                inputs["crop_coords"],
            )
            _check(record, f"glm_image_transformer_{name}", actual, saved[name])


def test_glm_image_vae_decode(record):
    root = snapshot_dir(MODEL_ID)
    reference = diffusers.AutoencoderKL.from_pretrained(
        str(root / "vae"), torch_dtype=torch.float32
    ).eval()
    native = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    latents = torch.randn(1, 16, 32, 32, generator=torch.Generator("cpu").manual_seed(0))
    with torch.inference_mode():
        _check(
            record, "glm_image_vae_decode", native.decode(latents), reference.decode(latents).sample
        )


@pytest.mark.parametrize(
    ("seed", "size", "steps"),
    [(0, 1024, END_TO_END_STEPS), (1, 512, 2)],
    ids=["defaults-seed-0", "seed-1-512"],
)
def test_glm_image_end_to_end(record, tmp_path, seed, size, steps):
    root = snapshot_dir(MODEL_ID)
    _in_subprocess(_reference_prior, root, PROMPT, size, size, seed, tmp_path / "prior.pt")
    prior = torch.load(tmp_path / "prior.pt", weights_only=True)
    _in_subprocess(
        _reference_image, root, PROMPT, prior, size, size, steps, seed, tmp_path / "image.pt"
    )
    expected = torch.load(tmp_path / "image.pt", weights_only=True)
    result = pipeline("t2i", backend="glm-image", model=root)(
        PROMPT, seed=seed, width=size, height=size, steps=steps
    )
    actual = result.images[0].tensor
    differing = int((actual != expected).sum())
    record(f"glm_image_e2e_seed{seed}_{size}", steps=steps, differing_values=differing)
    assert torch.equal(actual, expected)
