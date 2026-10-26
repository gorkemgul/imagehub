"""Tiny random-weight parity for Kandinsky 2.1 against the diffusers combined pipeline."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest
import torch
from helpers import max_diff, write_tiny_tokenizer

from imagehub.backends.kandinsky21 import (
    Kandinsky21Components,
    _decode_latents,
    _denoise_decoder,
    _run_prior,
)
from imagehub.checkpoints import save_safetensors
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.multilingual_clip import MultilingualCLIP
from imagehub.models.prior_transformer import PriorTransformer
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.models.vq_model import VQModel
from imagehub.schedulers.ddim import DDIMScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
FIXTURES = Path(__file__).parents[1] / "fixtures/kandinsky21_configs"

DECODER_TEXT_CONFIG = {
    "vocab_size": 250002,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "max_position_embeddings": 64,
    "type_vocab_size": 1,
    "layer_norm_eps": 1e-5,
    "pad_token_id": 1,
    "position_embedding_type": "absolute",
    "bos_token_id": 0,
    "eos_token_id": 2,
    "transformerDimensions": 16,
    "numDims": 16,
    "architectures": ["MultilingualCLIP"],
    "model_type": "xlm-roberta",
}
UNET_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "sample_size": 16,
    "in_channels": 4,
    "out_channels": 8,
    "down_block_types": ["ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"],
    "up_block_types": ["SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"],
    "block_out_channels": [32, 64],
    "layers_per_block": 1,
    "attention_head_dim": 8,
    "cross_attention_dim": 16,
    "encoder_hid_dim": 16,
    "encoder_hid_dim_type": "text_image_proj",
    "addition_embed_type": "text_image",
    "resnet_time_scale_shift": "scale_shift",
    "norm_num_groups": 8,
}
MOVQ_CONFIG = {
    "_class_name": "VQModel",
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
    "sample_size": 32,
}
PRIOR_TEXT_CONFIG = {
    "vocab_size": 517,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "max_position_embeddings": 77,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 16,
    "bos_token_id": 515,
    "eos_token_id": 516,
    "pad_token_id": 516,
}
PRIOR_IMAGE_CONFIG = {
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
PRIOR_SCHEDULER_CONFIG = {
    "num_train_timesteps": 10,
    "variance_type": "fixed_small_log",
    "clip_sample": True,
    "clip_sample_range": 10.0,
    "prediction_type": "sample",
    "beta_schedule": "squaredcos_cap_v2",
}
SCHEDULER_CONFIG = {
    "_class_name": "DDIMScheduler",
    "num_train_timesteps": 10,
    "beta_schedule": "linear",
    "beta_start": 0.00085,
    "beta_end": 0.012,
    "clip_sample": False,
    "set_alpha_to_one": False,
    "steps_offset": 0,
    "prediction_type": "epsilon",
    "thresholding": False,
}


def _write_component(root: Path, name: str, model, config: dict, weights: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_safetensors(dict(model.state_dict()), directory / weights)


def _tiny_snapshot(root: Path) -> None:
    torch.manual_seed(0)
    # Decoder uses the published XLM-Roberta tokenizer (copied from fixtures).
    tokenizer_src = FIXTURES / "decoder/tokenizer"
    tokenizer_dst = root / "tokenizer"
    tokenizer_dst.mkdir(parents=True)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
    ):
        shutil.copy2(tokenizer_src / name, tokenizer_dst / name)

    text_encoder = MultilingualCLIP(DECODER_TEXT_CONFIG)
    _write_component(root, "text_encoder", text_encoder, DECODER_TEXT_CONFIG, "model.safetensors")

    unet = UNet2DConditionModel(UNET_CONFIG)
    _write_component(root, "unet", unet, UNET_CONFIG, "diffusion_pytorch_model.safetensors")

    movq = VQModel(MOVQ_CONFIG)
    _write_component(root, "movq", movq, MOVQ_CONFIG, "diffusion_pytorch_model.safetensors")

    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(SCHEDULER_CONFIG), encoding="utf-8"
    )
    (root / "ddpm_scheduler").mkdir()
    (root / "ddpm_scheduler" / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "DDPMScheduler",
                "num_train_timesteps": 10,
                "beta_schedule": "linear",
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "clip_sample": False,
                "prediction_type": "epsilon",
                "variance_type": "fixed_small",
            }
        ),
        encoding="utf-8",
    )
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "KandinskyPipeline",
                "_diffusers_version": "0.17.0.dev0",
                "text_encoder": ["kandinsky", "MultilingualCLIP"],
                "tokenizer": ["transformers", "XLMRobertaTokenizerFast"],
                "scheduler": ["diffusers", "DDIMScheduler"],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "movq": ["diffusers", "VQModel"],
            }
        ),
        encoding="utf-8",
    )

    # Prior
    prior_root = root / "prior"
    write_tiny_tokenizer(prior_root / "tokenizer")
    prior_text = CLIPTextModelWithProjection(PRIOR_TEXT_CONFIG)
    _write_component(prior_root, "text_encoder", prior_text, PRIOR_TEXT_CONFIG, "model.safetensors")
    prior_image = CLIPVisionModelWithProjection(PRIOR_IMAGE_CONFIG)
    _write_component(
        prior_root, "image_encoder", prior_image, PRIOR_IMAGE_CONFIG, "model.safetensors"
    )
    prior = PriorTransformer(PRIOR_CONFIG)
    _write_component(
        prior_root, "prior", prior, PRIOR_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    (prior_root / "scheduler").mkdir()
    (prior_root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(PRIOR_SCHEDULER_CONFIG), encoding="utf-8"
    )
    (prior_root / "image_processor").mkdir()
    (prior_root / "image_processor" / "preprocessor_config.json").write_text(
        json.dumps({"crop_size": 8, "size": 8, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}),
        encoding="utf-8",
    )
    (prior_root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "KandinskyPriorPipeline",
                "prior": ["diffusers", "PriorTransformer"],
                "image_encoder": ["transformers", "CLIPVisionModelWithProjection"],
                "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "scheduler": ["diffusers", "UnCLIPScheduler"],
                "image_processor": ["transformers", "CLIPImageProcessor"],
            }
        ),
        encoding="utf-8",
    )


def _reference(root: Path):
    prior = diffusers.KandinskyPriorPipeline.from_pretrained(
        root / "prior", local_files_only=True, torch_dtype=torch.float32
    )
    decoder = diffusers.KandinskyPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    combined = diffusers.KandinskyCombinedPipeline(
        text_encoder=decoder.text_encoder,
        tokenizer=decoder.tokenizer,
        unet=decoder.unet,
        scheduler=decoder.scheduler,
        movq=decoder.movq,
        prior_prior=prior.prior,
        prior_image_encoder=prior.image_encoder,
        prior_text_encoder=prior.text_encoder,
        prior_tokenizer=prior.tokenizer,
        prior_scheduler=prior.scheduler,
        prior_image_processor=prior.image_processor,
    )
    combined.set_progress_bar_config(disable=True)
    return combined


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tiny_tokenizer_matches_reference(tmp_path, record):
    root = tmp_path / "tiny"
    _tiny_snapshot(root)
    ours = Kandinsky21Components.load(root, device=CPU, dtype=torch.float32).tokenizer
    reference = transformers.XLMRobertaTokenizerFast.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    prompts = [
        "",
        "a cat",
        "hello   world",
        " leading and trailing ",
        "café naïve",
        "İstanbul’da güzel bir gün",
        "Привет, мир!",
        "こんにちは世界",
        "你好，世界",
        "مرحبا بالعالم",
        "emoji 👩🏽‍🚀✨",
        "line one\nline two\ttab",
        "e\N{COMBINING ACUTE ACCENT} vs é",
        "a" * 1000,
        "Türkçe English русский 日本語 العربية हिन्दी",
    ]
    rng = random.Random(21)
    alphabet = "abc XYZ\t\néİЖ界مह🚀"
    prompts.extend(
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 180))) for _ in range(100)
    )
    exact = True
    for text in prompts:
        expected = reference(
            text, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
        )
        our_ids, our_mask = ours(text, max_length=77)
        exact &= torch.equal(expected["input_ids"][0], our_ids)
        exact &= torch.equal(expected["attention_mask"][0].bool(), our_mask)
    record("kandinsky21_tiny_tokenizer", exact=exact, prompts=len(prompts))
    assert exact


def test_tiny_multilingual_clip_matches_exactly(tmp_path, record):
    root = tmp_path / "tiny"
    _tiny_snapshot(root)
    ours = MultilingualCLIP.from_pretrained(root / "text_encoder", device=CPU, dtype=torch.float32)
    reference = diffusers.pipelines.kandinsky.text_encoder.MultilingualCLIP.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    ids = torch.tensor([[0, 42, 17, 2, 1], [0, 9, 2, 1, 1]])
    mask = ids.ne(1)
    with torch.inference_mode():
        actual_pooled, actual_hidden = ours(ids, mask)
        expected_pooled, expected_hidden = reference(ids, mask)
    _check(record, "kandinsky21_tiny_multilingual_clip_pooled", actual_pooled, expected_pooled)
    _check(record, "kandinsky21_tiny_multilingual_clip_hidden", actual_hidden, expected_hidden)


def test_tiny_text_image_unet_and_ddim_match_exactly(tmp_path, record):
    root = tmp_path / "tiny"
    _tiny_snapshot(root)
    ours = UNet2DConditionModel.from_pretrained(root / "unet", device=CPU, dtype=torch.float32)
    reference = diffusers.UNet2DConditionModel.from_pretrained(
        root / "unet", local_files_only=True, torch_dtype=torch.float32
    )
    sample = torch.randn(2, 4, 8, 16, generator=torch.Generator().manual_seed(1))
    hidden = torch.randn(2, 77, 16, generator=torch.Generator().manual_seed(2))
    text = torch.randn(2, 16, generator=torch.Generator().manual_seed(3))
    image = torch.randn(2, 16, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        actual = ours(sample, torch.tensor(5), hidden, text_embeds=text, image_embeds=image)
        expected = reference(
            sample,
            5,
            encoder_hidden_states=hidden,
            added_cond_kwargs={"text_embeds": text, "image_embeds": image},
        ).sample
    _check(record, "kandinsky21_tiny_text_image_unet", actual, expected)

    our_scheduler = DDIMScheduler(SCHEDULER_CONFIG)
    ref_scheduler = diffusers.DDIMScheduler.from_config(SCHEDULER_CONFIG)
    our_scheduler.set_timesteps(4)
    ref_scheduler.set_timesteps(4)
    exact_timesteps = torch.equal(our_scheduler.timesteps, ref_scheduler.timesteps)
    record("kandinsky21_tiny_ddim_timesteps", exact=exact_timesteps, tolerance="exact")
    assert exact_timesteps
    actual = torch.randn(1, 4, 3, 5, generator=torch.Generator().manual_seed(5))
    expected = actual.clone()
    outputs = [
        torch.randn(actual.shape, generator=torch.Generator().manual_seed(seed))
        for seed in range(6, 10)
    ]
    for timestep, output in zip(our_scheduler.timesteps, outputs, strict=True):
        actual = our_scheduler.step(output, timestep, actual)
        expected = ref_scheduler.step(output, timestep, expected).prev_sample
    _check(record, "kandinsky21_tiny_ddim_trajectory", actual, expected)


@pytest.mark.parametrize(
    "negative_prompt,guidance,width,height,seed",
    [(None, 4.0, 8, 8, 11), ("bad", 4.0, 8, 8, 11), (None, 1.0, 16, 8, 29)],
    ids=["guided", "negative", "unguided-nonsquare-second-seed"],
)
def test_tiny_combined_pipeline_matches_exactly(
    tmp_path, record, negative_prompt, guidance, width, height, seed
):
    root = tmp_path / "tiny"
    _tiny_snapshot(root)
    reference = _reference(root)
    components = Kandinsky21Components.load(root, device=CPU, dtype=torch.float32)

    batch = 2 if negative_prompt is not None else 1
    prior_latents = torch.randn(batch, 16, generator=torch.Generator().manual_seed(3))
    generator = torch.Generator().manual_seed(seed)

    our_pos, our_neg = _run_prior(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        guidance=guidance,
        latents=prior_latents.clone(),
        generator=generator,
        device=CPU,
    )
    our_latents = _denoise_decoder(
        components,
        our_pos,
        our_neg,
        prompt="a cat",
        steps=2,
        guidance=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    our_pixels = _decode_latents(components.movq, our_latents)

    expected = reference(
        "a cat",
        negative_prompt=negative_prompt,
        latents=prior_latents.clone(),
        prior_num_inference_steps=2,
        num_inference_steps=2,
        height=height,
        width=width,
        guidance_scale=guidance,
        prior_guidance_scale=guidance,
        generator=torch.Generator().manual_seed(seed),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))

    passed = torch.equal(our_pixels, expected)
    record(
        f"e2e_tiny_kandinsky21_{guidance}_{width}x{height}_{seed}_{negative_prompt}",
        passed=passed,
        pixels_max_abs=int((our_pixels.int() - expected.int()).abs().max()),
    )
    assert passed
