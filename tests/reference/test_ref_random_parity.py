"""Structural parity with random weights: no checkpoint download needed."""

import json
from pathlib import Path

import pytest
import torch
from fixture_files import restricted_fixture
from helpers import (
    LCM_SCHEDULER,
    SD1_SCHEDULER,
    SD1_UNET,
    SDXL_SCHEDULER,
    SDXL_UNET,
    SSD1B_UNET,
    TINY_SSD1B_UNET,
    TINY_TEXT_CONFIG,
    TINY_UNET_CONFIG,
    TINY_VAE_CONFIG,
    legacy_vae_state,
    text_state_for_ours,
)

from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIP_TEXT_DEFAULTS, CLIPTextModel, CLIPTextModelWithProjection
from imagehub.models.janus_vq import JanusGenerationModel, postprocess_janus
from imagehub.models.llama import LlamaCache, LlamaModel
from imagehub.models.qwen3 import Qwen3Model
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.z_image_transformer import ZImageTransformer2DModel
from imagehub.schedulers.edm_dpmsolver import EDMDPMSolverMultistepScheduler
from imagehub.schedulers.euler import EulerDiscreteScheduler
from imagehub.schedulers.flow_match_euler import (
    FlowMatchEulerDiscreteScheduler,
    z_image_sigmas,
)
from imagehub.schedulers.lcm import LCMScheduler
from imagehub.schedulers.pndm import PNDMScheduler

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("config", [TINY_TEXT_CONFIG, {}], ids=["tiny", "sd1"])
def test_clip_text_random_weights(config):
    torch.manual_seed(0)
    ref = transformers.CLIPTextModel(
        transformers.CLIPTextConfig(**(CLIP_TEXT_DEFAULTS | config))
    ).eval()
    ours = CLIPTextModel(config).eval()
    ours.load_checkpoint(text_state_for_ours(ref.state_dict()))
    ids = torch.randint(0, ours.config["vocab_size"], (3, 77))
    with torch.inference_mode():
        actual_last, actual_hidden = ours(ids, output_hidden_states=True)
        expected = ref(ids, output_hidden_states=True)
        exact(actual_last, expected.last_hidden_state)
        exact(actual_hidden[-2], expected.hidden_states[-2])


SDXL_TEXT_2 = {
    "vocab_size": 49408,
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_hidden_layers": 32,
    "num_attention_heads": 20,
    "max_position_embeddings": 77,
    "layer_norm_eps": 1e-5,
    "hidden_act": "gelu",
    "projection_dim": 1280,
    "eos_token_id": 2,
}
TINY_SDXL_TEXT_2 = TINY_TEXT_CONFIG | {
    "hidden_act": "gelu",
    "projection_dim": 24,
    "eos_token_id": 2,
}


@pytest.mark.parametrize("config", [TINY_SDXL_TEXT_2, SDXL_TEXT_2], ids=["tiny", "sdxl"])
def test_clip_text_projection_random_weights(config):
    torch.manual_seed(0)
    ref = transformers.CLIPTextModelWithProjection(transformers.CLIPTextConfig(**config)).eval()
    ours = CLIPTextModelWithProjection(config).eval()
    ours.load_checkpoint(text_state_for_ours(ref.state_dict()))
    ids = torch.randint(3, config["vocab_size"], (3, 77))
    ids[:, 30] = 2
    with torch.inference_mode():
        actual_embeds, actual_last, actual_hidden = ours(ids, output_hidden_states=True)
        expected = ref(ids, output_hidden_states=True)
    exact(actual_embeds, expected.text_embeds)
    exact(actual_last, expected.last_hidden_state)
    exact(actual_hidden[-2], expected.hidden_states[-2])


def test_qwen3_random_weights():
    config = {
        "vocab_size": 97,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "max_position_embeddings": 32,
        "attention_dropout": 0.0,
        "attention_bias": False,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "use_cache": False,
        "use_sliding_window": False,
    }
    torch.manual_seed(0)
    ref = transformers.Qwen3Model(transformers.Qwen3Config(**config)).eval()
    ours = Qwen3Model(config).eval()
    ours.load_checkpoint({"model." + name: value for name, value in ref.state_dict().items()})
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    with torch.inference_mode():
        actual_last, actual_hidden = ours(input_ids, attention_mask, output_hidden_states=True)
        expected = ref(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    exact(actual_hidden[-2], expected.hidden_states[-2])
    exact(actual_last, expected.last_hidden_state)


LLAMA_TINY = {
    "vocab_size": 128,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 1024,
    "attention_dropout": 0.0,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "use_cache": True,
}
LLAMA_MEDIUM = LLAMA_TINY | {
    "hidden_size": 512,
    "intermediate_size": 768,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
}


@pytest.mark.parametrize("config", [LLAMA_TINY, LLAMA_MEDIUM], ids=["tiny", "medium"])
def test_llama_static_cache_random_weights(config):
    torch.manual_seed(0)
    ref_config = transformers.LlamaConfig(**config)
    ref_config._attn_implementation = "sdpa"
    ref = transformers.LlamaModel(ref_config).eval()
    ours = LlamaModel(config).eval()
    ours.load_checkpoint(ref.state_dict())

    prompt = torch.tensor(
        [
            [0, 0, 1, 13, 14, 15, 2],
            [1, 21, 22, 23, 2, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
        ]
    )
    input_tokens = prompt.repeat(2, 1)
    unconditional = input_tokens[2:]
    unconditional[(unconditional != 1) & (unconditional != 2)] = 0
    inputs_embeds = ref.embed_tokens(input_tokens)
    attention_mask = attention_mask.repeat(2, 1)
    max_cache_len = max(20, 576 + prompt.shape[1])
    ref_cache = transformers.StaticCache(ref_config, max_cache_len=max_cache_len)
    cache = LlamaCache(config, max_cache_len=max_cache_len)
    cache_position = torch.arange(prompt.shape[1])

    with torch.inference_mode():
        for step in range(21):
            expected = ref(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=ref_cache,
                position_ids=cache_position.unsqueeze(0),
                use_cache=True,
            ).last_hidden_state
            actual = ours(inputs_embeds, attention_mask, cache, cache_position)
            exact(actual, expected)

            if step == 20:
                break
            inputs_embeds = torch.randn(4, 1, config["hidden_size"])
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            cache_position = cache_position.new_tensor([prompt.shape[1] + step])


TINY_JANUS_VQ = {
    "base_channels": 32,
    "channel_multiplier": [1, 2],
    "depth": 2,
    "double_latent": False,
    "dropout": 0.0,
    "embed_dim": 4,
    "hidden_act": "gelu",
    "image_token_embed_dim": 32,
    "in_channels": 3,
    "initializer_range": 0.02,
    "latent_channels": 32,
    "model_type": "janus_vqgan",
    "num_embeddings": 64,
    "num_hidden_layers": 2,
    "num_patches": 2,
    "num_res_blocks": 1,
    "out_channels": 3,
    "projection_dim": 32,
}


def published_janus_vq():
    return json.loads(restricted_fixture("janus_pro_1b_config.json").read_text())["vq_config"]


@pytest.mark.parametrize("config", [TINY_JANUS_VQ, None], ids=["tiny", "janus-1b"])
def test_janus_vq_decode_random_weights(config):
    from transformers.models.janus.configuration_janus import JanusVQVAEConfig
    from transformers.models.janus.modeling_janus import JanusVQVAE

    config = published_janus_vq() if config is None else config
    torch.manual_seed(0)
    ref = JanusVQVAE(JanusVQVAEConfig(**config)).eval()
    ours = JanusGenerationModel(config).eval()
    ours.vqmodel.load_checkpoint(
        {
            name: value
            for name, value in ref.state_dict().items()
            if name.startswith(("quantize.", "post_quant_conv.", "decoder."))
        }
    )
    tokens = torch.randint(0, config["num_embeddings"], (2, config["num_patches"] ** 2))
    with torch.inference_mode():
        actual = ours.decode_image_tokens(tokens)
        expected = ref.decode(tokens).permute(0, 2, 3, 1)
    exact(actual, expected)


def test_janus_generation_modules_random_weights():
    from transformers.models.janus.configuration_janus import JanusVQVAEConfig
    from transformers.models.janus.modeling_janus import JanusVQVAEAlignerMLP, JanusVQVAEHead

    config = JanusVQVAEConfig(**TINY_JANUS_VQ)
    torch.manual_seed(0)
    ref_embeddings = torch.nn.Embedding(config.num_embeddings, config.embed_dim)
    ref_aligner = JanusVQVAEAlignerMLP(config).eval()
    ref_head = JanusVQVAEHead(config).eval()
    ours = JanusGenerationModel(TINY_JANUS_VQ).eval()
    ours.generation_embeddings.load_state_dict(ref_embeddings.state_dict())
    ours.generation_aligner.load_state_dict(ref_aligner.state_dict())
    ours.generation_head.load_state_dict(ref_head.state_dict())
    tokens = torch.randint(0, config.num_embeddings, (2, 7))
    hidden_states = torch.randn(2, 7, config.image_token_embed_dim)
    with torch.inference_mode():
        exact(ours.generation_embeddings(tokens), ref_embeddings(tokens))
        exact(ours.generation_aligner(ref_embeddings(tokens)), ref_aligner(ref_embeddings(tokens)))
        exact(ours.generation_head(hidden_states), ref_head(hidden_states))


def test_janus_postprocess_pixels():
    # AutoProcessor picks the torchvision-backed JanusImageProcessor when torchvision is installed.
    pytest.importorskip("torchvision")
    from transformers.models.janus.image_processing_janus import JanusImageProcessor

    processor = JanusImageProcessor(
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5], rescale_factor=1 / 255
    )
    # decode_image_tokens returns (batch, height, width, channels) in roughly [-1, 1].
    decoded = (torch.randn(2, 24, 24, 3) * 0.7).clamp(-1.2, 1.2)
    decoded[0, 0, 0] = 1.0
    decoded[0, 0, 1] = -1.0
    expected = processor.postprocess(list(decoded.float()), return_tensors="pt")["pixel_values"]
    expected = torch.as_tensor(expected).permute(0, 2, 3, 1)
    exact(postprocess_janus(decoded), expected)


SD1_VAE = {
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "latent_channels": 4,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "sample_size": 512,
}


@pytest.mark.parametrize(
    ("config", "latent_hw"),
    [(SD1_VAE | TINY_VAE_CONFIG, (8, 6)), (SD1_VAE, (64, 64))],
    ids=["tiny", "sd1"],
)
def test_vae_random_weights(config, latent_hw):
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKL.from_config(config).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    ours = AutoencoderKL(config).eval()
    ours.load_checkpoint(legacy_vae_state(ref.state_dict()))
    z = torch.randn(1, 4, *latent_hw)
    image = torch.rand(1, 3, latent_hw[0] * 8, latent_hw[1] * 8) * 2 - 1
    with torch.inference_mode():
        exact(ours.decode(z), ref.decode(z).sample)
        exact(ours.encode(image), ref.encode(image).latent_dist.mean)


Z_IMAGE_VAE = json.loads(
    (Path(__file__).parents[1] / "fixtures/z_image_turbo_configs/vae_config.json").read_text(
        encoding="utf-8"
    )
)
TINY_Z_IMAGE_VAE = Z_IMAGE_VAE | {
    "block_out_channels": [32, 32, 64, 64],
    "norm_num_groups": 8,
    "sample_size": 64,
}


def test_z_image_vae_random_weights():
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKL.from_config(TINY_Z_IMAGE_VAE).eval()
    for parameter in ref.parameters():
        parameter.data.normal_(0, 0.02)
    ours = AutoencoderKL(TINY_Z_IMAGE_VAE).eval()
    ours.load_checkpoint(legacy_vae_state(ref.state_dict()))
    latents = torch.randn(1, 16, 8, 6)
    image = torch.rand(1, 3, 64, 48) * 2 - 1
    with torch.inference_mode():
        exact(ours.decode(latents), ref.decode(latents).sample)
        exact(ours.encode(image), ref.encode(image).latent_dist.mean)


def test_z_image_vae_full_config_layout():
    with torch.device("meta"):
        ref = diffusers.AutoencoderKL.from_config(Z_IMAGE_VAE)
        ours = AutoencoderKL(Z_IMAGE_VAE)
    expected = {
        name: list(value.shape) for name, value in legacy_vae_state(ref.state_dict()).items()
    }
    actual = {name: list(value.shape) for name, value in ours.state_dict().items()}
    assert actual == expected


COGVIEW4_VAE = {
    "block_out_channels": [128, 512, 1024, 1024],
    "down_block_types": ["DownEncoderBlock2D"] * 4,
    "up_block_types": ["UpDecoderBlock2D"] * 4,
    "latent_channels": 16,
    "layers_per_block": 3,
    "norm_num_groups": 32,
    "mid_block_add_attention": False,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "scaling_factor": 1.0,
    "shift_factor": 0.0,
    "sample_size": 1024,
}
TINY_COGVIEW4_VAE = COGVIEW4_VAE | {
    "block_out_channels": [32, 32, 64, 64],
    "norm_num_groups": 8,
    "sample_size": 64,
}


def test_cogview4_vae_random_weights():
    torch.manual_seed(0)
    ref = diffusers.AutoencoderKL.from_config(TINY_COGVIEW4_VAE).eval()
    for parameter in ref.parameters():
        parameter.data.normal_(0, 0.02)
    ours = AutoencoderKL(TINY_COGVIEW4_VAE).eval()
    ours.load_checkpoint(legacy_vae_state(ref.state_dict()))
    latents = torch.randn(1, 16, 8, 6)
    image = torch.rand(1, 3, 64, 48) * 2 - 1
    with torch.inference_mode():
        exact(ours.decode(latents), ref.decode(latents).sample)
        exact(ours.encode(image), ref.encode(image).latent_dist.mean)


def test_cogview4_vae_full_config_layout():
    with torch.device("meta"):
        ref = diffusers.AutoencoderKL.from_config(COGVIEW4_VAE)
        ours = AutoencoderKL(COGVIEW4_VAE)
    expected = {
        name: list(value.shape) for name, value in legacy_vae_state(ref.state_dict()).items()
    }
    actual = {name: list(value.shape) for name, value in ours.state_dict().items()}
    assert actual == expected


def test_z_image_transformer_random_weights():
    config = {
        "all_patch_size": [2],
        "all_f_patch_size": [1],
        "in_channels": 8,
        "dim": 32,
        "n_layers": 3,
        "n_refiner_layers": 2,
        "n_heads": 2,
        "n_kv_heads": 2,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 24,
        "siglip_feat_dim": None,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": [4, 4, 8],
        "axes_lens": [64, 32, 32],
    }
    torch.manual_seed(0)
    ref = diffusers.ZImageTransformer2DModel.from_config(config).eval()
    ours = ZImageTransformer2DModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    x = [torch.randn(8, 1, 10, 14), torch.randn(8, 1, 8, 12)]
    timesteps = torch.tensor([0.875, 0.25])
    cap_feats = [torch.randn(5, 24), torch.randn(9, 24)]
    with torch.inference_mode():
        expected = ref(x, timesteps, cap_feats).sample
        actual = ours(x, timesteps, cap_feats)
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected, strict=True):
        exact(actual_item, expected_item)


@pytest.mark.parametrize(
    ("config", "sizes"),
    [(TINY_UNET_CONFIG, [(16, 16), (13, 9)]), (SD1_UNET, [(64, 64), (65, 47)])],
    ids=["tiny", "sd1"],
)
def test_unet_random_weights(config, sizes):
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(config).eval()
    for p in ref.parameters():
        p.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    context = torch.randn(2, 77, config["cross_attention_dim"])
    for hw in sizes:
        sample = torch.randn(2, 4, *hw)
        for t in (torch.tensor(981), torch.tensor(1)):
            with torch.inference_mode():
                exact(ours(sample, t, context), ref(sample, t, context).sample)


def test_lcm_unet_random_weights():
    config = TINY_UNET_CONFIG | {"time_cond_proj_dim": 16}
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in ref.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    sample = torch.randn(2, 4, 13, 9)
    context = torch.randn(2, 7, config["cross_attention_dim"])
    timestep_cond = torch.randn(2, config["time_cond_proj_dim"])
    with torch.inference_mode():
        actual = ours(sample, torch.tensor(999), context, timestep_cond=timestep_cond)
        expected = ref(
            sample,
            torch.tensor(999),
            context,
            timestep_cond=timestep_cond,
        ).sample
    exact(actual, expected)


TINY_SDXL_UNET = SDXL_UNET | {
    "block_out_channels": [32, 64, 64],
    "transformer_layers_per_block": [1, 2, 3],
    "attention_head_dim": [2, 4, 4],
    "cross_attention_dim": 32,
    "addition_time_embed_dim": 8,
    "projection_class_embeddings_input_dim": 80,
    "layers_per_block": 1,
    "norm_num_groups": 8,
}


@pytest.mark.parametrize(
    ("config", "sizes"),
    [(TINY_SDXL_UNET, [(16, 16), (13, 9)]), (SDXL_UNET, [(128, 128), (129, 127)])],
    ids=["tiny", "sdxl"],
)
def test_sdxl_unet_random_weights(config, sizes):
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in ref.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    context = torch.randn(2, 7, config["cross_attention_dim"])
    text_dim = (
        config["projection_class_embeddings_input_dim"] - 6 * config["addition_time_embed_dim"]
    )
    text_embeds = torch.randn(2, text_dim)
    time_ids = torch.randn(2, 6)
    for hw in sizes:
        sample = torch.randn(2, 4, *hw)
        for timestep in (torch.tensor(981), torch.tensor(1)):
            with torch.inference_mode():
                actual = ours(
                    sample,
                    timestep,
                    context,
                    text_embeds=text_embeds,
                    time_ids=time_ids,
                )
                expected = ref(
                    sample,
                    timestep,
                    context,
                    added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                ).sample
            exact(actual, expected)


@pytest.mark.parametrize(
    ("config", "sizes"),
    [(TINY_SSD1B_UNET, [(16, 16), (13, 9)]), (SSD1B_UNET, [(128, 128)])],
    ids=["tiny", "ssd1b"],
)
def test_ssd1b_unet_random_weights(config, sizes):
    torch.manual_seed(0)
    ref = diffusers.UNet2DConditionModel.from_config(config).eval()
    for parameter in ref.parameters():
        parameter.data.normal_(0, 0.02)
    ours = UNet2DConditionModel(config).eval()
    ours.load_checkpoint(ref.state_dict())
    context = torch.randn(2, 7, config["cross_attention_dim"])
    text_dim = (
        config["projection_class_embeddings_input_dim"] - 6 * config["addition_time_embed_dim"]
    )
    text_embeds = torch.randn(2, text_dim)
    time_ids = torch.randn(2, 6)
    for hw in sizes:
        sample = torch.randn(2, 4, *hw)
        for timestep in (torch.tensor(999), torch.tensor(500), torch.tensor(1)):
            with torch.inference_mode():
                actual = ours(
                    sample,
                    timestep,
                    context,
                    text_embeds=text_embeds,
                    time_ids=time_ids,
                )
                expected = ref(
                    sample,
                    timestep,
                    context,
                    added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                ).sample
            exact(actual, expected)


def test_pndm_matches_diffusers():
    ref = diffusers.PNDMScheduler.from_config(SD1_SCHEDULER)
    ours = PNDMScheduler(SD1_SCHEDULER)
    for steps in (50, 25, 7, 1):
        ref.set_timesteps(steps)
        ours.set_timesteps(steps)
        assert torch.equal(ref.timesteps, ours.timesteps), steps
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, 8)
    x_ref = x.clone()
    ref.set_timesteps(50)
    ours.set_timesteps(50)
    for t in ref.timesteps:
        eps = torch.randn(2, 4, 8, 8)
        x_ref = ref.step(eps, t, x_ref).prev_sample
        x = ours.step(eps, t, x)
    exact(x, x_ref)


def test_lcm_scheduler_matches_diffusers():
    ref = diffusers.LCMScheduler.from_config(LCM_SCHEDULER)
    ours = LCMScheduler(LCM_SCHEDULER)
    for steps in (1, 2, 4, 10, 50):
        ref.set_timesteps(steps)
        ours.set_timesteps(steps)
        exact(ours.timesteps, ref.timesteps)

    sample = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    expected = sample.clone()
    ref.set_timesteps(4)
    ours.set_timesteps(4)
    ref_generator = torch.Generator().manual_seed(42)
    our_generator = torch.Generator().manual_seed(42)
    for timestep in ref.timesteps:
        model_output = torch.randn(
            sample.shape, generator=torch.Generator().manual_seed(int(timestep))
        )
        expected, expected_denoised = ref.step(
            model_output,
            timestep,
            expected,
            generator=ref_generator,
            return_dict=False,
        )
        sample, denoised = ours.step(
            model_output,
            timestep,
            sample,
            generator=our_generator,
        )
        exact(sample, expected)
        exact(denoised, expected_denoised)


def test_euler_matches_diffusers():
    ref = diffusers.EulerDiscreteScheduler.from_config(SDXL_SCHEDULER)
    ours = EulerDiscreteScheduler(SDXL_SCHEDULER)
    for steps in (1000, 50, 30, 7, 1):
        ref.set_timesteps(steps)
        ours.set_timesteps(steps)
        exact(ours.timesteps, ref.timesteps)
        exact(ours.sigmas, ref.sigmas)
        exact(torch.as_tensor(ours.init_noise_sigma), torch.as_tensor(ref.init_noise_sigma))
    torch.manual_seed(0)
    sample = torch.randn(2, 4, 8, 8)
    expected = sample.clone()
    ref.set_timesteps(30)
    ours.set_timesteps(30)
    for timestep in ref.timesteps:
        model_output = torch.randn_like(sample)
        exact(
            ours.scale_model_input(sample, timestep),
            ref.scale_model_input(expected, timestep),
        )
        expected = ref.step(model_output, timestep, expected).prev_sample
        sample = ours.step(model_output, timestep, sample)
    exact(sample, expected)


def test_flow_match_euler_matches_diffusers():
    fixture = Path(__file__).parents[1] / (
        "fixtures/z_image_turbo_configs/scheduler_scheduler_config.json"
    )
    config = json.loads(fixture.read_text(encoding="utf-8"))
    for steps in (1, 8, 9, 50):
        for sigmas in (z_image_sigmas(steps), None):
            ref = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
            ours = FlowMatchEulerDiscreteScheduler(config)
            ref.set_timesteps(steps, sigmas=sigmas, mu=0.9)
            ours.set_timesteps(steps, sigmas=sigmas, mu=0.9)
            exact(ours.timesteps, ref.timesteps)
            exact(ours.sigmas, ref.sigmas)

    torch.manual_seed(0)
    sample = torch.randn(2, 16, 8, 8)
    expected = sample.clone()
    ref = diffusers.FlowMatchEulerDiscreteScheduler.from_config(config)
    ours = FlowMatchEulerDiscreteScheduler(config)
    sigmas = z_image_sigmas(9)
    ref.set_timesteps(9, sigmas=sigmas, mu=0.9)
    ours.set_timesteps(9, sigmas=sigmas, mu=0.9)
    ref.set_begin_index(0)
    ours.set_begin_index(0)
    for timestep in ref.timesteps:
        model_output = torch.randn_like(sample)
        expected = ref.step(model_output, timestep, expected).prev_sample
        sample = ours.step(model_output, timestep, sample)
        exact(sample, expected)


def test_edm_dpmsolver_matches_diffusers():
    fixture = restricted_fixture("playground_v25_configs/scheduler_config.json")
    config = json.loads(fixture.read_text(encoding="utf-8"))
    ref = diffusers.EDMDPMSolverMultistepScheduler.from_config(config)
    ours = EDMDPMSolverMultistepScheduler(config)
    for steps in (1, 2, 25, 50):
        ref.set_timesteps(steps)
        ours.set_timesteps(steps)
        exact(ours.timesteps, ref.timesteps)
        exact(ours.sigmas, ref.sigmas)
        exact(torch.as_tensor(ours.init_noise_sigma), torch.as_tensor(ref.init_noise_sigma))
    torch.manual_seed(0)
    sample = torch.randn(2, 4, 8, 8)
    expected = sample.clone()
    ref.set_timesteps(25)
    ours.set_timesteps(25)
    for timestep in ref.timesteps:
        model_output = torch.randn_like(sample)
        exact(
            ours.scale_model_input(sample, timestep),
            ref.scale_model_input(expected, timestep),
        )
        expected = ref.step(model_output, timestep, expected).prev_sample
        sample = ours.step(model_output, timestep, sample)
    exact(sample, expected)
