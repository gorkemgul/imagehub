import pytest
import torch
from helpers import write_tiny_flux2

from imagehub import pipeline
from imagehub.backends.flux2 import Flux2Components, denoise, encode_prompt

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")


def test_tiny_flux2_prompt_and_generation_match_diffusers_exactly(tmp_path):
    root = write_tiny_flux2(tmp_path / "tiny")
    reference = diffusers.Flux2KleinPipeline.from_pretrained(root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    components = Flux2Components.load(root, device=torch.device("cpu"), dtype=torch.float32)

    prompt = "a cat"
    expected_text = reference.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    actual_text = components.tokenizer.chat_prompt(prompt, enable_thinking=False)
    assert actual_text == expected_text
    actual_tokens, actual_mask = components.tokenizer(actual_text)
    expected_tokens = reference.tokenizer(
        expected_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    assert torch.equal(actual_tokens, expected_tokens.input_ids)
    assert torch.equal(actual_mask, expected_tokens.attention_mask.bool())
    native_prompt, native_ids = encode_prompt(components, prompt, torch.device("cpu"), 1, 512)
    expected_prompt, expected_ids = reference.encode_prompt(prompt)
    torch.testing.assert_close(native_prompt, expected_prompt, rtol=0, atol=0)
    assert torch.equal(native_ids, expected_ids)

    noise = torch.randn(1, 16, 4, 4, generator=torch.Generator("cpu").manual_seed(42))
    expected_latents = reference(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=1.0,
        height=16,
        width=16,
        latents=noise.clone(),
        output_type="latent",
    ).images
    actual_latents = denoise(
        components,
        prompt=prompt,
        steps=2,
        guidance=1.0,
        width=16,
        height=16,
        latents=noise.clone(),
        max_sequence_length=512,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)

    expected = reference(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=1.0,
        height=16,
        width=16,
        latents=noise.clone(),
        output_type="np",
    ).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="flux2", model=root)(
        prompt, steps=2, width=16, height=16, latents=noise
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)
