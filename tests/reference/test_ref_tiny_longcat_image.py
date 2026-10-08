import json

import pytest
import torch
from helpers import TINY_SD3_VAE

from imagehub import pipeline
from imagehub.backends import longcat_image as native_pipeline
from imagehub.backends.longcat_image import LongCatImageComponents, denoise, rewrite_prompt
from imagehub.checkpoints import save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.longcat_image_transformer import LongCatImageTransformer2DModel
from imagehub.tokenization_qwen import bytes_to_unicode

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

TEXT_CONFIG = {
    "vocab_size": 259,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000.0,
    "rope_scaling": {"type": "default", "mrope_section": [2, 1, 1]},
    "attention_bias": True,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "use_sliding_window": False,
    "tie_word_embeddings": False,
    "bos_token_id": 256,
    "eos_token_id": [258],
    "pad_token_id": 256,
}
TRANSFORMER_CONFIG = {
    "_class_name": "LongCatImageTransformer2DModel",
    "patch_size": 1,
    "in_channels": 64,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 16,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "pooled_projection_dim": 32,
    "guidance_embeds": False,
    "axes_dims_rope": [4, 6, 6],
}
VAE_CONFIG = TINY_SD3_VAE | {
    "latent_channels": 16,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "sample_size": 32,
}
SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 4,
    "max_image_seq_len": 16,
}
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% for content in message['content'] %}{{ content['text'] }}{% endfor %}"
    "<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _write_tokenizer(directory):
    vocab = {character: index for index, character in enumerate(bytes_to_unicode().values())}
    specials = ("<|endoftext|>", "<|im_start|>", "<|im_end|>")
    for token in specials:
        vocab[token] = len(vocab)
    tokenizer = transformers.Qwen2Tokenizer(
        vocab=vocab,
        merges=[],
        pad_token=specials[0],
        eos_token=specials[2],
        additional_special_tokens=list(specials[1:]),
    )
    tokenizer.save_pretrained(directory)
    return tokenizer


def _write_tiny_snapshot(directory):
    root = directory
    root.mkdir(parents=True)
    tokenizer = _write_tokenizer(root / "tokenizer")
    _write_tokenizer(root / "text_processor")
    processor = transformers.Qwen2VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(),
        video_processor=transformers.Qwen2VLVideoProcessor(),
        tokenizer=tokenizer,
        chat_template=CHAT_TEMPLATE,
    )
    processor.save_pretrained(root / "text_processor")

    config = {
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "model_type": "qwen2_5_vl",
        "text_config": TEXT_CONFIG,
        "vision_config": {
            "depth": 1,
            "embed_dim": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
        },
        "image_token_id": 255,
        "video_token_id": 255,
        "bos_token_id": 256,
        "eos_token_id": [258],
        "pad_token_id": 256,
    }
    generation_config = {
        "do_sample": True,
        "top_k": 1,
        "top_p": 0.001,
        "temperature": 0.1,
        "repetition_penalty": 1.05,
        "eos_token_id": [258],
        "pad_token_id": 256,
    }
    torch.manual_seed(0)
    text_encoder = transformers.Qwen2_5_VLForConditionalGeneration(
        transformers.Qwen2_5_VLConfig(**config)
    )
    (root / "text_encoder").mkdir()
    (root / "text_encoder/config.json").write_text(json.dumps(config))
    (root / "text_encoder/generation_config.json").write_text(json.dumps(generation_config))
    save_safetensors(text_encoder.state_dict(), root / "text_encoder/model.safetensors")

    for name, cls, model_config in (
        ("transformer", LongCatImageTransformer2DModel, TRANSFORMER_CONFIG),
        ("vae", AutoencoderKL, VAE_CONFIG),
    ):
        model = cls(model_config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(model_config))
        save_safetensors(model.state_dict(), root / name / "diffusion_pytorch_model.safetensors")
    (root / "scheduler").mkdir()
    (root / "scheduler/scheduler_config.json").write_text(json.dumps(SCHEDULER_CONFIG))
    return root, tokenizer, processor, text_encoder


def _reference_pipeline(root, tokenizer, processor, text_encoder):
    reference = diffusers.LongCatImagePipeline(
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler"),
        vae=diffusers.AutoencoderKL.from_pretrained(root / "vae", torch_dtype=torch.float32),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_processor=processor,
        transformer=diffusers.LongCatImageTransformer2DModel.from_pretrained(
            root / "transformer", torch_dtype=torch.float32
        ),
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def test_tiny_rewritten_prompt_matches_exactly(tmp_path, monkeypatch):
    root, tokenizer, processor, text_encoder = _write_tiny_snapshot(tmp_path / "tiny")
    reference = _reference_pipeline(root, tokenizer, processor, text_encoder)
    monkeypatch.setattr(native_pipeline, "SYSTEM_PROMPT_EN", "Rewrite this prompt.")
    monkeypatch.setattr(native_pipeline, "TOKENIZER_MAX_LENGTH", 1)
    reference.tokenizer_max_length = 1
    reference.rewire_prompt.__globals__["SYSTEM_PROMPT_EN"] = "Rewrite this prompt."
    expected = reference.rewire_prompt("a cat", torch.device("cpu"))[0]
    components = LongCatImageComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual = rewrite_prompt(components, "a cat", torch.device("cpu"))
    assert actual == expected


@pytest.mark.parametrize(
    ("enable_prompt_rewrite", "enable_cfg_renorm"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_tiny_generation_matches_diffusers_exactly(
    tmp_path, monkeypatch, enable_prompt_rewrite, enable_cfg_renorm
):
    root, tokenizer, processor, text_encoder = _write_tiny_snapshot(tmp_path / "tiny")
    reference = _reference_pipeline(root, tokenizer, processor, text_encoder)
    monkeypatch.setattr(native_pipeline, "SYSTEM_PROMPT_EN", "Rewrite this prompt.")
    monkeypatch.setattr(native_pipeline, "TOKENIZER_MAX_LENGTH", 1)
    reference.tokenizer_max_length = 1
    reference.rewire_prompt.__globals__["SYSTEM_PROMPT_EN"] = "Rewrite this prompt."
    noise = torch.randn(1, 16, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    packed = reference._pack_latents(noise.clone(), 1, 16, 4, 4)
    common = {
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "num_inference_steps": 2,
        "guidance_scale": 4.5,
        "height": 32,
        "width": 32,
        "enable_prompt_rewrite": enable_prompt_rewrite,
        "enable_cfg_renorm": enable_cfg_renorm,
        "cfg_renorm_min": 0.2,
    }
    expected_latents = reference(**common, latents=packed.clone(), output_type="latent").images
    expected_pixels = reference(**common, latents=packed.clone(), output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = LongCatImageComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt="blurry",
        steps=2,
        guidance=4.5,
        width=32,
        height=32,
        latents=noise.clone(),
        enable_cfg_renorm=enable_cfg_renorm,
        cfg_renorm_min=0.2,
        enable_prompt_rewrite=enable_prompt_rewrite,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    result = pipeline("t2i", backend="longcat-image", model=root)(
        "a cat",
        negative_prompt="blurry",
        steps=2,
        guidance=4.5,
        width=32,
        height=32,
        latents=noise,
        enable_prompt_rewrite=enable_prompt_rewrite,
        enable_cfg_renorm=enable_cfg_renorm,
        cfg_renorm_min=0.2,
    )
    actual_pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual_pixels, expected_pixels)


def test_tiny_seed_matches_diffusers_exactly(tmp_path):
    root, tokenizer, processor, text_encoder = _write_tiny_snapshot(tmp_path / "tiny")
    reference = _reference_pipeline(root, tokenizer, processor, text_encoder)
    expected = reference(
        "a cat",
        num_inference_steps=2,
        guidance_scale=4.5,
        height=32,
        width=32,
        enable_prompt_rewrite=False,
        generator=torch.Generator("cpu").manual_seed(42),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    actual = pipeline("t2i", backend="longcat-image", model=root)(
        "a cat",
        steps=2,
        guidance=4.5,
        width=32,
        height=32,
        enable_prompt_rewrite=False,
        seed=42,
    )
    assert torch.equal(torch.stack([image.tensor for image in actual.images]), expected)
