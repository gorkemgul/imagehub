"""Tiny end-to-end DreamLite parity against both diffusers pipelines."""

from types import MethodType

import pytest
import torch
from test_ref_random_dreamlite import TINY_UNET, TINY_VAE

import imagehub.backends.dreamlite as native_pipeline
from imagehub.backends.dreamlite import DreamLiteComponents, decode_latents, denoise
from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.dreamlite import DreamLiteUNetModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

SCHEDULER = {
    "base_image_seq_len": 1,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 16,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class _TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    @property
    def dtype(self):
        return self.anchor.dtype


def _embeddings(count, device, dtype):
    generator = torch.Generator("cpu").manual_seed(99 + count)
    embeds = torch.randn(count, 5, 32, generator=generator, dtype=dtype).to(device)
    return embeds, torch.ones(count, 5, dtype=torch.long, device=device)


@pytest.mark.parametrize(
    ("mobile", "guidance", "seed"),
    [(False, 0.0, 3), (False, 3.5, 7), (True, None, 3), (True, None, 7)],
)
def test_tiny_dreamlite_pipeline_exact(monkeypatch, mobile, guidance, seed):
    torch.manual_seed(0)
    ours_unet = DreamLiteUNetModel(TINY_UNET).eval()
    ours_vae = AutoencoderTiny(TINY_VAE).eval()
    reference_unet = diffusers.DreamLiteUNetModel(**TINY_UNET).eval()
    reference_vae = diffusers.AutoencoderTiny(**TINY_VAE).eval()
    reference_unet.load_state_dict(ours_unet.state_dict(), strict=True)
    reference_vae.load_state_dict(ours_vae.state_dict(), strict=True)
    reference_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER)
    pipeline_class = diffusers.DreamLiteMobilePipeline if mobile else diffusers.DreamLitePipeline
    monkeypatch.setattr(
        pipeline_class, "_execution_device", property(lambda _self: torch.device("cpu"))
    )
    reference = pipeline_class(
        text_encoder=_TextEncoder(),
        tokenizer=None,
        processor=None,
        vae=reference_vae,
        unet=reference_unet,
        scheduler=reference_scheduler,
    )
    reference.to("cpu")
    reference.set_progress_bar_config(disable=True)

    def reference_encode(_self, mode, prompts, device, dtype, **_kwargs):
        return _embeddings(len(prompts), device, dtype)

    reference.encode_prompt = MethodType(reference_encode, reference)

    def native_encode(_components, prompts, device, _max_length):
        embeds, mask = _embeddings(len(prompts), device, torch.float32)
        return embeds, mask.bool()

    monkeypatch.setattr(native_pipeline, "encode_prompt", native_encode)
    components = DreamLiteComponents(
        tokenizer=None,
        text_encoder=None,
        unet=ours_unet,
        vae=ours_vae,
        scheduler_config=SCHEDULER,
    )
    noise = torch.randn(1, 4, 4, 6, generator=torch.Generator("cpu").manual_seed(seed))
    kwargs = {
        "prompt": "a cat",
        "num_inference_steps": 2,
        "height": 32,
        "width": 48,
        "generator": torch.Generator("cpu").manual_seed(seed),
        "output_type": "np",
    }
    if not mobile:
        kwargs["guidance_scale"] = guidance
    expected = reference(**kwargs).images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=None,
        steps=2,
        guidance=guidance,
        width=48,
        height=32,
        latents=noise,
        mobile=mobile,
        max_sequence_length=16,
    )
    actual = decode_latents(ours_vae, actual_latents)
    assert torch.equal(actual, expected)
