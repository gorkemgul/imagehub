"""Tiny-snapshot PixArt Alpha and Sigma end-to-end parity."""

import json
from pathlib import Path

import pytest
import torch
from helpers import TINY_SD3_VAE

from imagehub import pipeline
from imagehub.backends.pixart import PixArtComponents, decode_latents, denoise
from imagehub.checkpoints import save_safetensors
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.pixart_transformer import PixArtTransformer2DModel
from imagehub.models.t5 import T5EncoderModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
sentencepiece = pytest.importorskip("sentencepiece")

T5_CONFIG = {
    "architectures": ["T5EncoderModel"],
    "vocab_size": 164,
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
    "num_attention_heads": 4,
    "attention_head_dim": 8,
    "in_channels": 4,
    "out_channels": 8,
    "num_layers": 2,
    "dropout": 0.0,
    "norm_num_groups": 8,
    "cross_attention_dim": 32,
    "attention_bias": True,
    "sample_size": 32,
    "patch_size": 2,
    "activation_fn": "gelu-approximate",
    "num_embeds_ada_norm": 1000,
    "upcast_attention": False,
    "norm_type": "ada_norm_single",
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "caption_channels": 32,
    "attention_type": "default",
}
VAE_CONFIG = TINY_SD3_VAE | {"scaling_factor": 0.13025, "shift_factor": None}
SCHEDULER_CONFIG = {
    "_class_name": "DPMSolverMultistepScheduler",
    "algorithm_type": "dpmsolver++",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "dynamic_thresholding_ratio": 0.995,
    "euler_at_final": False,
    "final_sigmas_type": "zero",
    "flow_shift": 1.0,
    "lambda_min_clipped": -float("inf"),
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "prediction_type": "epsilon",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_type": "midpoint",
    "steps_offset": 0,
    "thresholding": False,
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": False,
    "use_karras_sigmas": False,
    "use_lu_lambdas": False,
    "time_shift_type": "exponential",
    "variance_type": None,
}


def _write_tokenizer(directory: Path) -> None:
    directory.mkdir()
    corpus = directory / "corpus.txt"
    corpus.write_text(
        "a cat on a mat\na blurry photograph\nan astronaut riding a horse\n",
        encoding="utf-8",
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(directory / "spiece"),
        vocab_size=64,
        model_type="unigram",
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    (directory / "spiece.vocab").unlink()
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "add_prefix_space": True,
                "eos_token": "</s>",
                "extra_ids": 100,
                "legacy": True,
                "model_max_length": 300,
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
        ),
        encoding="utf-8",
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"}),
        encoding="utf-8",
    )


def _write_tiny_pixart(directory: Path, class_name: str) -> Path:
    directory.mkdir()
    _write_tokenizer(directory / "tokenizer")
    (directory / "scheduler").mkdir()
    (directory / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(SCHEDULER_CONFIG), encoding="utf-8"
    )
    (directory / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": class_name,
                "scheduler": ["diffusers", "DPMSolverMultistepScheduler"],
                "text_encoder": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "T5Tokenizer"],
                "transformer": ["diffusers", "PixArtTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        ),
        encoding="utf-8",
    )
    transformer_config = TRANSFORMER_CONFIG | {
        "_class_name": "PixArtTransformer2DModel",
        "use_additional_conditions": False,
    }
    torch.manual_seed(0)
    for name, cls, config, weights in (
        ("text_encoder", T5EncoderModel, T5_CONFIG, "model.safetensors"),
        (
            "transformer",
            PixArtTransformer2DModel,
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
    ("class_name", "max_sequence_length"),
    [("PixArtSigmaPipeline", 300), ("PixArtAlphaPipeline", 120)],
    ids=["sigma", "alpha"],
)
@pytest.mark.parametrize("negative_prompt", [None, "blurry"], ids=["empty-negative", "negative"])
def test_tiny_pixart_matches_diffusers_exactly(
    tmp_path, class_name, max_sequence_length, negative_prompt
):
    root = _write_tiny_pixart(tmp_path / "tiny", class_name)
    pipeline_class = getattr(diffusers, class_name)
    reference = pipeline_class.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    height, width = 40, 32
    binned_height, binned_width = reference.image_processor.classify_height_width_bin(
        height,
        width,
        ratios=diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha.ASPECT_RATIO_256_BIN,
    )
    noise = torch.randn(
        1,
        4,
        binned_height // 8,
        binned_width // 8,
        generator=torch.Generator("cpu").manual_seed(7),
    )
    kwargs = {
        "prompt": "a cat",
        "negative_prompt": negative_prompt or "",
        "num_inference_steps": 2,
        "guidance_scale": 4.5,
        "height": height,
        "width": width,
        "use_resolution_binning": True,
        "max_sequence_length": max_sequence_length,
        "latents": noise.clone(),
        "generator": torch.Generator("cpu").manual_seed(42),
        "clean_caption": False,
    }
    expected_latents = reference(**kwargs, output_type="latent").images
    expected_pixels = reference(**kwargs, output_type="np").images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    components = PixArtComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=4.5,
        width=binned_width,
        height=binned_height,
        max_sequence_length=max_sequence_length,
        latents=noise.clone(),
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    actual_pixels = decode_latents(
        components.vae,
        actual_latents,
        output_width=width,
        output_height=height,
        resize=True,
    )
    assert torch.equal(actual_pixels, expected_pixels)
    actual = pipeline("t2i", backend="pixart", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=4.5,
        width=width,
        height=height,
        use_resolution_binning=True,
        max_sequence_length=max_sequence_length,
        latents=noise,
        seed=42,
    )
    assert torch.equal(torch.stack([image.tensor for image in actual.images]), expected_pixels)
