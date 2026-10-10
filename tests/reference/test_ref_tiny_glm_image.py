"""Tiny random-weight GLM-Image pipeline compared with the diffusers pipeline."""

import numpy as np
import pytest
import torch
from glm_image_helpers import (
    AR_CONFIG,
    GENERATION,
    SCHEDULER,
    T5_CONFIG,
    TRANSFORMER_CONFIG,
    VAE_CONFIG,
    write_tiny_glm_image,
)

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CHAT_TEMPLATE = (
    "{%- for m in messages -%}{%- for item in m.content -%}"
    "{%- if item.type == 'text' -%}{{ item.text }}{%- endif -%}{%- endfor -%}{%- endfor -%}"
)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    torch.manual_seed(0)
    ar_config = dict(AR_CONFIG)
    ar_config["vision_config"] = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "image_size": 64,
        "patch_size": 16,
    }
    ar_config["vq_config"] = {"embed_dim": 16, "num_embeddings": 64, "latent_channels": 32}
    vlm = transformers.GlmImageForConditionalGeneration(
        transformers.GlmImageConfig(**ar_config)
    ).eval()
    for parameter in vlm.parameters():
        if parameter.ndim > 1:
            torch.nn.init.normal_(parameter, std=0.2)
    vlm.generation_config.update(**GENERATION)
    text_encoder = transformers.T5EncoderModel(
        transformers.T5Config(**T5_CONFIG, feed_forward_proj="gated-gelu")
    ).eval()
    transformer = diffusers.GlmImageTransformer2DModel(**TRANSFORMER_CONFIG).eval()
    vae = diffusers.AutoencoderKL(**VAE_CONFIG).eval()
    states = {
        "ar": vlm.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "transformer": transformer.state_dict(),
        "vae": vae.state_dict(),
    }
    # The text encoder state of transformers also holds the tied encoder embedding.
    states["text_encoder"] = {
        k: v for k, v in states["text_encoder"].items() if k != "encoder.embed_tokens.weight"
    }
    root = write_tiny_glm_image(tmp_path_factory.mktemp("tiny-glm-image-ref"), states)
    glm_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(root / "processor" / "tokenizer.json"),
        pad_token="<end>",
        bos_token="<begin>",
        eos_token="<end>",
        image_token="<image>",
        grid_bos_token="<sop>",
        grid_eos_token="<eop>",
        extra_special_tokens={
            "image_token": "<image>",
            "grid_bos_token": "<sop>",
            "grid_eos_token": "<eop>",
        },
    )
    processor = transformers.GlmImageProcessor(
        image_processor=transformers.GlmImageImageProcessor(),
        tokenizer=glm_tokenizer,
        chat_template=CHAT_TEMPLATE,
    )
    byt5 = transformers.ByT5Tokenizer.from_pretrained(str(root / "tokenizer"))
    reference = diffusers.GlmImagePipeline(
        tokenizer=byt5,
        processor=processor,
        text_encoder=text_encoder,
        vision_language_encoder=vlm,
        vae=vae,
        transformer=transformer,
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER),
    )
    return root, reference


CASES = [
    ("a cat", 1, 64, 64, 3, 1.5),
    ("a sign with 'hello' and \"two\" words", 2, 64, 64, 2, 1.0),
    ("a wide 'photo'", 3, 96, 64, 4, 2.5),
    ("“quoted” tall", 4, 64, 96, 1, 1.5),
]


@pytest.mark.parametrize(("prompt", "seed", "width", "height", "steps", "guidance"), CASES)
def test_tiny_pipeline_matches_diffusers_exactly(
    tiny, prompt, seed, width, height, steps, guidance
):
    root, reference = tiny
    with torch.inference_mode():
        expected = reference(
            prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
    actual = pipeline("t2i", backend="glm-image", model=root)(
        prompt, seed=seed, width=width, height=height, steps=steps, guidance=guidance
    ).images[0]
    assert np.array_equal(np.asarray(expected), actual.tensor.numpy())
