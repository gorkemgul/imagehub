import json

import pytest
import torch
from helpers import TINY_VOCAB_SIZE, write_tiny_tokenizer

from imagehub.backends.kandinsky22 import (
    Kandinsky22Components,
    decode_latents,
    denoise_decoder,
    run_prior,
)
from imagehub.checkpoints import save_safetensors

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

TEXT_CONFIG = {
    "vocab_size": TINY_VOCAB_SIZE,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 77,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 16,
    "bos_token_id": TINY_VOCAB_SIZE - 2,
    "eos_token_id": TINY_VOCAB_SIZE - 1,
    "pad_token_id": TINY_VOCAB_SIZE - 1,
}
VISION_CONFIG = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 8,
    "patch_size": 4,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "projection_dim": 16,
}
PRIOR_CONFIG = {
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "num_layers": 2,
    "embedding_dim": 16,
    "num_embeddings": 77,
    "additional_embeddings": 4,
    "time_embed_dim": 16,
    "embedding_proj_dim": 16,
    "clip_embed_dim": 16,
}
UNET_CONFIG = {
    "sample_size": 8,
    "in_channels": 4,
    "out_channels": 8,
    "down_block_types": ["CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D"],
    "block_out_channels": [32],
    "layers_per_block": 1,
    "attention_head_dim": 4,
    "cross_attention_dim": 16,
    "encoder_hid_dim": 16,
    "encoder_hid_dim_type": "image_proj",
    "addition_embed_type": "image",
    "norm_num_groups": 8,
}
MOVQ_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D"],
    "block_out_channels": [32],
    "layers_per_block": 1,
    "latent_channels": 4,
    "num_vq_embeddings": 16,
    "norm_num_groups": 32,
    "vq_embed_dim": 4,
    "norm_type": "spatial",
}
PRIOR_SCHEDULER_CONFIG = {
    "num_train_timesteps": 10,
    "variance_type": "fixed_small_log",
    "clip_sample": True,
    "clip_sample_range": 10.0,
    "prediction_type": "sample",
    "beta_schedule": "squaredcos_cap_v2",
}
SCHEDULER_CONFIG = {
    "num_train_timesteps": 10,
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "beta_schedule": "linear",
    "variance_type": "learned_range",
    "clip_sample": True,
    "clip_sample_range": 1.0,
    "prediction_type": "epsilon",
    "timestep_spacing": "leading",
    "steps_offset": 0,
}


def _write_component(root, name, model, config, weights):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_safetensors(model.state_dict(), directory / weights)


def _tiny_pipeline(root):
    torch.manual_seed(0)
    text_encoder = transformers.CLIPTextModelWithProjection(
        transformers.CLIPTextConfig(**TEXT_CONFIG)
    ).eval()
    image_encoder = transformers.CLIPVisionModelWithProjection(
        transformers.CLIPVisionConfig(**VISION_CONFIG)
    ).eval()
    prior = diffusers.PriorTransformer(**PRIOR_CONFIG).eval()
    unet = diffusers.UNet2DConditionModel(**UNET_CONFIG).eval()
    movq = diffusers.VQModel(**MOVQ_CONFIG).eval()
    tokenizer_path = write_tiny_tokenizer(root / "prior" / "tokenizer")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(tokenizer_path)
    _write_component(root / "prior", "text_encoder", text_encoder, TEXT_CONFIG, "model.safetensors")
    _write_component(
        root / "prior", "image_encoder", image_encoder, VISION_CONFIG, "model.safetensors"
    )
    _write_component(
        root / "prior", "prior", prior, PRIOR_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    (root / "prior" / "scheduler").mkdir()
    (root / "prior" / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(PRIOR_SCHEDULER_CONFIG), encoding="utf-8"
    )
    _write_component(root, "unet", unet, UNET_CONFIG, "diffusion_pytorch_model.safetensors")
    _write_component(root, "movq", movq, MOVQ_CONFIG, "diffusion_pytorch_model.safetensors")
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(SCHEDULER_CONFIG), encoding="utf-8"
    )
    reference = diffusers.KandinskyV22CombinedPipeline(
        unet=unet,
        scheduler=diffusers.DDPMScheduler(**SCHEDULER_CONFIG),
        movq=movq,
        prior_prior=prior,
        prior_image_encoder=image_encoder,
        prior_text_encoder=text_encoder,
        prior_tokenizer=tokenizer,
        prior_scheduler=diffusers.UnCLIPScheduler(**PRIOR_SCHEDULER_CONFIG),
        prior_image_processor=transformers.CLIPImageProcessor(
            crop_size=8, size=8, image_mean=[0.5] * 3, image_std=[0.5] * 3
        ),
    )
    reference.set_progress_bar_config(disable=True)
    return reference


@pytest.mark.parametrize("negative_prompt", [None, "bad"], ids=["zero-embed", "negative"])
def test_tiny_combined_pipeline_matches_exactly(tmp_path, negative_prompt):
    root = tmp_path / "tiny"
    reference = _tiny_pipeline(root)
    batch = 2 if negative_prompt is not None else 1
    prior_latents = torch.randn(batch, 16, generator=torch.Generator().manual_seed(3))
    arguments = {
        "prompt": "a cat",
        "negative_prompt": negative_prompt,
        "prior_num_inference_steps": 2,
        "prior_guidance_scale": 4.0,
        "num_inference_steps": 2,
        "guidance_scale": 4.0,
        "height": 8,
        "width": 8,
        "latents": prior_latents.clone(),
    }
    expected_latents = reference(
        **arguments,
        generator=torch.Generator().manual_seed(11),
        output_type="latent",
    ).images
    expected_pixels = reference(
        **arguments,
        generator=torch.Generator().manual_seed(11),
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = Kandinsky22Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    generator = torch.Generator().manual_seed(11)
    image_embeds, negative_image_embeds = run_prior(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=4.0,
        latents=prior_latents.clone(),
        generator=generator,
        device=torch.device("cpu"),
    )
    actual_latents = denoise_decoder(
        components,
        image_embeds,
        negative_image_embeds,
        steps=2,
        guidance=4.0,
        height=8,
        width=8,
        generator=generator,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    assert torch.equal(decode_latents(components.movq, actual_latents), expected_pixels)
