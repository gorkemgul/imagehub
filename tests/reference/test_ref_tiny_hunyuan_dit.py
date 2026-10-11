import json
import shutil

import pytest
import torch
from helpers import TINY_SD3_VAE, snapshot_dir

from imagehub import pipeline
from imagehub.backends.hunyuan_dit import HunyuanDiTComponents, decode_latents, denoise
from imagehub.checkpoints import save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.bert import BertModel
from imagehub.models.hunyuan_dit_transformer import HunyuanDiT2DModel
from imagehub.models.t5 import T5EncoderModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")

BERT_CONFIG = {
    "architectures": ["BertModel"],
    "vocab_size": 47020,
    "hidden_size": 24,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 48,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "max_position_embeddings": 512,
    "type_vocab_size": 2,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
}
T5_CONFIG = {
    "architectures": ["T5EncoderModel"],
    "vocab_size": 250112,
    "d_model": 32,
    "d_kv": 8,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "relative_attention_num_buckets": 8,
    "relative_attention_max_distance": 16,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "dense_act_fn": "gelu_new",
    "is_gated_act": True,
    "is_encoder_decoder": True,
    "use_cache": False,
}
TRANSFORMER_CONFIG = {
    "_class_name": "HunyuanDiT2DModel",
    "sample_size": 4,
    "patch_size": 2,
    "in_channels": 4,
    "num_layers": 4,
    "attention_head_dim": 8,
    "num_attention_heads": 4,
    "hidden_size": 32,
    "mlp_ratio": 2.0,
    "learn_sigma": True,
    "cross_attention_dim": 24,
    "cross_attention_dim_t5": 32,
    "pooled_projection_dim": 32,
    "text_len": 77,
    "text_len_t5": 256,
    "activation_fn": "gelu-approximate",
    "norm_type": "layer_norm",
}
VAE_CONFIG = TINY_SD3_VAE | {"scaling_factor": 0.13025, "shift_factor": None}
SCHEDULER_CONFIG = {
    "_class_name": "DDPMScheduler",
    "beta_end": 0.03,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "clip_sample_range": 1.0,
    "dynamic_thresholding_ratio": 0.995,
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "set_alpha_to_one": False,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "thresholding": False,
    "timestep_spacing": "leading",
    "trained_betas": None,
    "variance_type": "fixed_small",
}


def _write_tiny_hunyuan_dit(directory, *, release="v1.1"):
    v12 = release == "v1.2"
    transformer_config = TRANSFORMER_CONFIG | (
        {"use_style_cond_and_image_meta_size": False} if v12 else {}
    )
    scheduler_config = SCHEDULER_CONFIG | ({"beta_end": 0.018} if v12 else {})
    tokenizer_source = snapshot_dir("hunyuan-dit-v1.1-distilled")
    directory.mkdir()
    for name in ("tokenizer", "tokenizer_2"):
        shutil.copytree(tokenizer_source / name, directory / name)
    (directory / "scheduler").mkdir()
    (directory / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(scheduler_config), encoding="utf-8"
    )
    (directory / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "HunyuanDiTPipeline",
                "feature_extractor": [None, None],
                "requires_safety_checker": True,
                "safety_checker": [None, None],
                "scheduler": ["diffusers", "DDPMScheduler"],
                "text_encoder": ["transformers", "BertModel"],
                "text_encoder_2": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "BertTokenizer"],
                "tokenizer_2": ["transformers", "T5Tokenizer"],
                "transformer": ["diffusers", "HunyuanDiT2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    torch.manual_seed(0)
    for name, cls, config, weights in (
        ("text_encoder", BertModel, BERT_CONFIG, "model.safetensors"),
        ("text_encoder_2", T5EncoderModel, T5_CONFIG, "model.safetensors"),
        (
            "transformer",
            HunyuanDiT2DModel,
            transformer_config,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, VAE_CONFIG, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        (directory / name).mkdir()
        (directory / name / "config.json").write_text(json.dumps(config), encoding="utf-8")
        save_safetensors(model.state_dict(), directory / name / weights)
    return directory


@pytest.mark.parametrize(
    ("release", "width", "height", "noise_seed", "seed", "guidance"),
    [
        ("v1.1", 32, 32, 7, 42, 5.0),
        ("v1.2", 32, 32, 7, 42, 5.0),
        ("v1.2", 48, 32, 11, 3, 5.0),
        ("v1.2", 32, 32, 7, 42, 1.0),
    ],
    ids=["v1.1", "v1.2", "v1.2-non-square", "v1.2-no-guidance"],
)
@pytest.mark.parametrize(
    ("negative_prompt", "guidance_rescale"),
    [(None, 0.0), ("blurry", 0.0), ("blurry", 0.7)],
    ids=["empty-negative", "negative", "guidance-rescale"],
)
def test_tiny_hunyuan_dit_matches_diffusers_exactly(
    tmp_path, negative_prompt, guidance_rescale, release, width, height, noise_seed, seed, guidance
):
    root = _write_tiny_hunyuan_dit(tmp_path / "tiny", release=release)
    reference = diffusers.HunyuanDiTPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    noise = torch.randn(
        1, 4, height // 8, width // 8, generator=torch.Generator("cpu").manual_seed(noise_seed)
    )
    expected_latents = reference(
        "a cat",
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=guidance,
        guidance_rescale=guidance_rescale,
        height=height,
        width=width,
        use_resolution_binning=False,
        latents=noise.clone(),
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
    ).images
    expected_pixels = reference(
        "a cat",
        negative_prompt=negative_prompt,
        num_inference_steps=2,
        guidance_scale=guidance,
        guidance_rescale=guidance_rescale,
        height=height,
        width=width,
        use_resolution_binning=False,
        latents=noise.clone(),
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    components = HunyuanDiTComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        guidance_rescale=guidance_rescale,
        width=width,
        height=height,
        latents=noise.clone(),
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    assert torch.equal(decode_latents(components.vae, actual_latents), expected_pixels)
    actual = pipeline("t2i", backend="hunyuan-dit", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        guidance_rescale=guidance_rescale,
        width=width,
        height=height,
        use_resolution_binning=False,
        latents=noise,
        seed=seed,
    )
    assert torch.equal(torch.stack([image.tensor for image in actual.images]), expected_pixels)
