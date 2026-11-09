"""Tiny random HunyuanImage 2.1 snapshot: end-to-end parity with the diffusers pipeline."""

import json

import pytest
import torch
from helpers import write_tiny_hunyuanimage

from imagehub import pipeline
from imagehub.checkpoints import load_weights

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")


def _reference_pipeline(root):
    def config(name):
        return json.loads((root / name).read_text())

    text_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, dtype=torch.float32
    )
    text_encoder_2 = transformers.T5EncoderModel(
        transformers.T5Config(**config("text_encoder_2/config.json"))
    )
    text_encoder_2.load_state_dict(load_weights(root / "text_encoder_2"), strict=False)
    transformer = diffusers.HunyuanImageTransformer2DModel(**config("transformer/config.json"))
    transformer.load_state_dict(load_weights(root / "transformer"))
    vae = diffusers.AutoencoderKLHunyuanImage(**config("vae/config.json"))
    vae.load_state_dict(load_weights(root / "vae"))
    guiders = {
        name: diffusers.guiders.AdaptiveProjectedMixGuidance(
            **{k: v for k, v in config(f"{name}/guider_config.json").items() if k[0] != "_"}
        )
        for name in ("guider", "ocr_guider")
    }
    reference = diffusers.HunyuanImagePipeline(
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_config(
            config("scheduler/scheduler_config.json")
        ),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True
        ),
        text_encoder_2=text_encoder_2,
        tokenizer_2=transformers.ByT5Tokenizer.from_pretrained(root / "tokenizer_2"),
        transformer=transformer,
        guider=guiders["guider"],
        ocr_guider=guiders["ocr_guider"],
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _noise(height, width, seed):
    return torch.randn(
        1, 8, height // 4, width // 4, generator=torch.Generator("cpu").manual_seed(seed)
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_hunyuanimage(tmp_path_factory.mktemp("tiny"))
    return root, _reference_pipeline(root)


@pytest.mark.parametrize(
    ("prompt", "guidance", "height", "width", "seed"),
    [
        ("a cat", None, 32, 32, 1),
        ("a cat", None, 24, 40, 2),
        ("a cat", 1.0, 32, 32, 3),
        ("a sign that says 'hi cat'", None, 32, 32, 4),
        ("a sign that says “hi”", 2.5, 28, 36, 5),
    ],
    ids=["default", "non-square", "unguided", "glyph", "glyph-scale-non-square"],
)
def test_generation_matches_diffusers_exactly(tiny, prompt, guidance, height, width, seed):
    root, reference = tiny
    noise = _noise(height, width, seed)
    steps = 8
    scale = 3.5 if guidance is None else guidance
    reference.guider = reference.guider.new(guidance_scale=scale)
    reference.ocr_guider = reference.ocr_guider.new(guidance_scale=scale)
    expected = reference(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        latents=noise.clone(),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="hunyuanimage", model=root)(
        prompt,
        height=height,
        width=width,
        steps=steps,
        latents=noise,
        **({} if guidance is None else {"guidance": guidance}),
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)


@pytest.mark.parametrize("seed", [0, 1234])
def test_seed_matches_diffusers_generator(tiny, seed):
    root, reference = tiny
    reference.guider = reference.guider.new(guidance_scale=3.5)
    expected = reference(
        "a cat",
        height=32,
        width=32,
        num_inference_steps=3,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="hunyuanimage", model=root)(
        "a cat", height=32, width=32, steps=3, seed=seed
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)
    assert result.metadata["seed"] == seed
