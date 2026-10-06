import json
import shutil

import pytest
import torch
from helpers import (
    FIXTURES,
    TINY_SD3_T5,
    TINY_SD3_VAE,
    TINY_TEXT_FILE_CONFIG,
    write_tiny_tokenizer,
)

from imagehub import pipeline
from imagehub.backends.flux import FluxComponents, _pack_latents, denoise
from imagehub.checkpoints import save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.flux_transformer import FluxTransformer2DModel
from imagehub.models.t5 import T5EncoderModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


def _write_tiny_flux1(directory, *, guidance_embeds, dynamic_shifting):
    root = directory
    root.mkdir(parents=True)
    torch.manual_seed(0)
    write_tiny_tokenizer(root / "tokenizer")
    tokenizer_2 = root / "tokenizer_2"
    tokenizer_2.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer_2 / "tokenizer.json")
    tokenizer_2_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (tokenizer_2 / "tokenizer_config.json").write_text(json.dumps(tokenizer_2_config))
    (tokenizer_2 / "special_tokens_map.json").write_text(
        json.dumps(
            {key: tokenizer_2_config[key] for key in ("pad_token", "eos_token", "unk_token")}
        )
    )
    scheduler = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": dynamic_shifting,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 4,
        "max_image_seq_len": 16,
    }
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    transformer_config = {
        "_class_name": "FluxTransformer2DModel",
        "patch_size": 1,
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "guidance_embeds": guidance_embeds,
        "axes_dims_rope": [2, 2, 4],
    }
    vae_config = TINY_SD3_VAE | {
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "sample_size": 16,
    }
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        ("text_encoder_2", T5EncoderModel, TINY_SD3_T5, "model.safetensors"),
        (
            "transformer",
            FluxTransformer2DModel,
            transformer_config,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, vae_config, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


@pytest.mark.parametrize(
    ("guidance_embeds", "dynamic_shifting", "use_seed", "negative_prompt", "true_cfg_scale"),
    [
        (False, False, False, None, 1.0),
        (False, False, True, None, 1.0),
        (True, True, False, None, 1.0),
        (True, True, True, None, 1.0),
        (True, True, False, "blurry", 2.0),
    ],
    ids=["schnell-latents", "schnell-seed", "dev-latents", "dev-seed", "dev-true-cfg"],
)
def test_tiny_flux1_matches_diffusers_exactly(
    tmp_path, guidance_embeds, dynamic_shifting, use_seed, negative_prompt, true_cfg_scale
):
    root = _write_tiny_flux1(
        tmp_path / "tiny",
        guidance_embeds=guidance_embeds,
        dynamic_shifting=dynamic_shifting,
    )
    reference = diffusers.FluxPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    common = {
        "prompt": "a cat",
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "num_inference_steps": 2,
        "guidance_scale": 3.5 if guidance_embeds else 0.0,
        "height": 32,
        "width": 32,
        "max_sequence_length": 8,
    }
    source = (
        {"generator": torch.Generator("cpu").manual_seed(42)}
        if use_seed
        else {"latents": _pack_latents(noise.clone(), 1, 4, 4, 4)}
    )
    expected_latents = reference(**common, **source, output_type="latent").images
    if use_seed:
        source = {"generator": torch.Generator("cpu").manual_seed(42)}
    expected_pixels = reference(**common, **source, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = FluxComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_noise = (
        torch.randn(1, 4, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
        if use_seed
        else noise.clone()
    )
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        steps=2,
        guidance=3.5 if guidance_embeds else 0.0,
        width=32,
        height=32,
        latents=actual_noise,
        max_sequence_length=8,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    call = {
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "steps": 2,
        "guidance": 3.5 if guidance_embeds else 0.0,
        "width": 32,
        "height": 32,
        "max_sequence_length": 8,
    }
    if use_seed:
        call["seed"] = 42
    else:
        call["latents"] = noise
    result = pipeline("t2i", backend="flux", model=root)("a cat", **call)
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
