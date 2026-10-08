"""Tiny random-component Ideogram 4 end-to-end parity."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import imagehub.backends.ideogram4 as native_pipeline
from imagehub.backends.ideogram4 import Ideogram4Components, decode_latents, denoise
from imagehub.models.fp8 import load_fp8_state_dict, swap_linears_to_fp8
from imagehub.models.ideogram4_transformer import Ideogram4Transformer

pytestmark = pytest.mark.reference
ideogram4 = pytest.importorskip("ideogram4")
reference_pipeline = pytest.importorskip("ideogram4.pipeline_ideogram4")
quantized_loading = pytest.importorskip("ideogram4.quantized_loading")


CONFIG = {
    "emb_dim": 12,
    "num_layers": 1,
    "num_heads": 2,
    "attention_head_dim": 6,
    "intermediate_size": 16,
    "adanln_dim": 4,
    "in_channels": 8,
    "llm_features_dim": 8,
    "rope_theta": 10_000,
    "mrope_section": [1, 1, 1],
    "norm_eps": 1e-5,
}


class _Tokenizer:
    def chat_prompt(self, prompt, *, enable_thinking=True):
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        return self.chat_prompt(messages[0]["content"][0]["text"])

    def tokenize(self, text):
        return [1, 2]

    def __call__(self, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and not add_special_tokens
        return {"input_ids": torch.tensor([self.tokenize(text)])}


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(2, 2, 1)

    def forward(self, latents):
        image = torch.cat((latents[:, :1], latents[:, 1:2], latents[:, :1]), dim=1)
        return torch.nn.functional.interpolate(image, scale_factor=8, mode="nearest")


class _Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _Decoder()

    def decode(self, latents):
        return self.decoder(latents)


def _fp8_state(model):
    state = dict(model.state_dict())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = f"{name}.weight"
            state[key], state[f"{key}_scale"] = quantized_loading.quantize_weight_to_fp8(state[key])
    return state


def _native_transformer(state):
    model = Ideogram4Transformer(CONFIG)
    swap_linears_to_fp8(model, state, torch.float32)
    load_fp8_state_dict(model, state, torch.device("cpu"), torch.float32, assign=True)
    return model.eval()


def _features(_model, token_ids, _positions, indicator):
    result = torch.zeros(*token_ids.shape, CONFIG["llm_features_dim"])
    values = torch.arange(CONFIG["llm_features_dim"], dtype=torch.float32)
    return result + values * (indicator == 3).unsqueeze(-1)


def test_tiny_ideogram4_fixed_noise_and_seed_match_official(monkeypatch):
    torch.manual_seed(0)
    reference_config = ideogram4.Ideogram4Config(
        **{key: value for key, value in CONFIG.items() if key != "attention_head_dim"}
    )
    conditional_state = _fp8_state(ideogram4.Ideogram4Transformer(reference_config))
    unconditional_state = _fp8_state(ideogram4.Ideogram4Transformer(reference_config))
    reference_conditional = reference_pipeline._build_transformer(
        reference_config, conditional_state, torch.device("cpu"), torch.float32
    )
    reference_unconditional = reference_pipeline._build_transformer(
        reference_config, unconditional_state, torch.device("cpu"), torch.float32
    )
    autoencoder = _Autoencoder().eval()
    tokenizer = _Tokenizer()
    reference = reference_pipeline.Ideogram4Pipeline(
        reference_conditional,
        reference_unconditional,
        SimpleNamespace(),
        tokenizer,
        autoencoder,
        reference_pipeline.Ideogram4PipelineConfig(
            patch_size=2, ae_scale_factor=8, max_text_tokens=32
        ),
        torch.device("cpu"),
        torch.float32,
    )
    reference.latent_shift = torch.zeros(8)
    reference.latent_scale = torch.ones(8)
    reference._verify_prompts = MethodType(lambda self, prompts, **kwargs: None, reference)
    reference._encode_text = MethodType(
        lambda self, token_ids, positions, indicator: _features(
            None, token_ids, positions, indicator
        ),
        reference,
    )
    captured = {}
    original_decode = reference._decode

    def capture(self, latents, **kwargs):
        captured["latents"] = latents.clone()
        return original_decode(latents, **kwargs)

    reference._decode = MethodType(capture, reference)
    guidance_schedule = (3.0, 7.0)
    expected_images = reference(
        "tiny prompt",
        height=32,
        width=32,
        num_steps=2,
        guidance_schedule=guidance_schedule,
        mu=0.5,
        std=1.75,
        seed=42,
    )

    components = Ideogram4Components(
        tokenizer=tokenizer,
        text_encoder=SimpleNamespace(),
        conditional_transformer=_native_transformer(conditional_state),
        unconditional_transformer=_native_transformer(unconditional_state),
        vae=autoencoder,
        latent_shift=torch.zeros(8),
        latent_scale=torch.ones(8),
        max_text_tokens=32,
    )
    monkeypatch.setattr(native_pipeline, "_verify_prompt", lambda prompt: None)
    monkeypatch.setattr(native_pipeline, "encode_text", _features)
    noise = torch.randn(
        (1, 4, 8), generator=torch.Generator("cpu").manual_seed(42), dtype=torch.float32
    )
    actual_latents = denoise(
        components,
        prompt="tiny prompt",
        steps=2,
        guidance_schedule=guidance_schedule,
        mu=0.5,
        std=1.75,
        width=32,
        height=32,
        latents=noise,
    )
    actual_pixels = decode_latents(components, actual_latents, 32, 32)
    expected_pixels = torch.from_numpy(__import__("numpy").array(expected_images[0]))
    torch.testing.assert_close(actual_latents, captured["latents"], atol=0, rtol=0)
    torch.testing.assert_close(actual_pixels[0], expected_pixels, atol=0, rtol=0)
