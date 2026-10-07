import json

import pytest
import torch
from helpers import write_tiny_krea2

from imagehub import pipeline
from imagehub.backends.krea2 import Krea2Components, denoise, encode_prompt
from imagehub.checkpoints import load_weights

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")


def _reference_pipeline(root):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    text_encoder = transformers.Qwen3VLModel.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    transformer_config = json.loads((root / "transformer/config.json").read_text())
    transformer = diffusers.Krea2Transformer2DModel(**transformer_config)
    transformer.load_state_dict(load_weights(root / "transformer"))
    vae_config = json.loads((root / "vae/config.json").read_text())
    vae = diffusers.AutoencoderKLQwenImage(**vae_config)
    vae.load_state_dict(load_weights(root / "vae"))
    scheduler_config = json.loads((root / "scheduler/scheduler_config.json").read_text())
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    reference = diffusers.Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        text_encoder_select_layers=tuple(range(1, 13)),
        is_distilled=True,
        patch_size=2,
    )
    reference.set_progress_bar_config(disable=True)
    return reference


@pytest.mark.parametrize("guidance", [0.0, 2.0], ids=["unguided", "guided"])
def test_tiny_krea2_generation_matches_diffusers_exactly(tmp_path, guidance):
    root = write_tiny_krea2(tmp_path / "tiny")
    reference = _reference_pipeline(root)
    components = Krea2Components.load(root, device=torch.device("cpu"), dtype=torch.float32)

    prompt = "a cat"
    expected_prompt, expected_mask = reference.encode_prompt(prompt, max_sequence_length=16)
    actual_prompt, actual_mask = encode_prompt(
        components, prompt, torch.device("cpu"), max_sequence_length=16
    )
    torch.testing.assert_close(actual_prompt, expected_prompt, rtol=0, atol=0)
    assert torch.equal(actual_mask, expected_mask)

    raw = torch.randn(1, 4, 2, 2, generator=torch.Generator("cpu").manual_seed(42))
    noise = reference._pack_latents(raw, 1, 4, 2, 2)
    expected_latents = reference(
        prompt,
        negative_prompt="blurry",
        num_inference_steps=2,
        guidance_scale=guidance,
        height=16,
        width=16,
        latents=noise.clone(),
        max_sequence_length=16,
        output_type="latent",
    ).images
    actual_latents = denoise(
        components,
        prompt=prompt,
        negative_prompt="blurry",
        steps=2,
        guidance=guidance,
        width=16,
        height=16,
        latents=noise.clone(),
        max_sequence_length=16,
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)

    expected_pixels = reference(
        prompt,
        negative_prompt="blurry",
        num_inference_steps=2,
        guidance_scale=guidance,
        height=16,
        width=16,
        latents=noise.clone(),
        max_sequence_length=16,
        output_type="np",
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))
    result = pipeline("t2i", backend="krea2", model=root)(
        prompt,
        negative_prompt="blurry",
        guidance=guidance,
        steps=2,
        width=16,
        height=16,
        latents=noise,
        max_sequence_length=16,
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
