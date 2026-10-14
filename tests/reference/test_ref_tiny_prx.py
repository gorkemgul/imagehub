"""Tiny PRX end-to-end parity against diffusers PRXPipeline."""

import json
from types import SimpleNamespace

import pytest
import torch
from helpers import write_tiny_prx

from imagehub.backends.prx import PRXComponents, clean_text, decode_latents, denoise
from imagehub.checkpoints import load_weights

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
modeling = pytest.importorskip("transformers.models.t5gemma.modeling_t5gemma")
np = pytest.importorskip("numpy")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import diffusers.pipelines.prx.pipeline_prx as pipeline_module

    if not hasattr(pipeline_module, "ftfy"):
        pipeline_module.ftfy = SimpleNamespace(fix_text=lambda value: value)
    root = write_tiny_prx(tmp_path_factory.mktemp("prx") / "tiny")
    text_cfg = json.loads((root / "text_encoder/config.json").read_text())
    transformer_cfg = json.loads((root / "transformer/config.json").read_text())
    vae_cfg = json.loads((root / "vae/config.json").read_text())
    text = modeling.T5GemmaEncoder(transformers.T5GemmaModuleConfig(**text_cfg)).eval()
    text.load_state_dict(load_weights(root / "text_encoder"))
    transformer = diffusers.PRXTransformer2DModel(**transformer_cfg).eval()
    transformer.load_state_dict(load_weights(root / "transformer"))
    vae = diffusers.AutoencoderKL.from_config(vae_cfg).eval()
    vae_state = {}
    for name, value in load_weights(root / "vae").items():
        for legacy, current in (
            ("query", "to_q"),
            ("key", "to_k"),
            ("value", "to_v"),
            ("proj_attn", "to_out.0"),
        ):
            name = name.replace(f".{legacy}.", f".{current}.")
        vae_state[name] = value
    vae.load_state_dict(vae_state)
    reference = diffusers.PRXPipeline(
        tokenizer=transformers.GemmaTokenizerFast.from_pretrained(root / "tokenizer"),
        text_encoder=text,
        transformer=transformer,
        vae=vae,
        scheduler=diffusers.FlowMatchEulerDiscreteScheduler(shift=3.0),
    )
    reference.register_to_config(default_sample_size=64)
    reference.set_progress_bar_config(disable=True)
    ours = PRXComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return reference, ours


@pytest.mark.parametrize(
    ("width", "height", "guidance", "seed"),
    [(64, 64, 4.0, 0), (96, 48, 1.0, 17)],
)
def test_tiny_end_to_end_exact(tiny, record, width, height, guidance, seed) -> None:
    reference, components = tiny
    prompt = "A photo of a cat on the mat."
    expected = reference(
        prompt,
        negative_prompt="",
        height=height,
        width=width,
        num_inference_steps=2,
        guidance_scale=guidance,
        generator=torch.Generator().manual_seed(seed),
        output_type="np",
        use_resolution_binning=False,
        tokenizer_max_length=24,
    ).images
    noise = torch.randn(
        1, 4, height // 8, width // 8, generator=torch.Generator().manual_seed(seed)
    )
    latents = denoise(
        components,
        prompt=prompt,
        negative_prompt="",
        steps=2,
        guidance=guidance,
        max_sequence_length=24,
        skip_text_cleaning=False,
        latents=noise,
    )
    actual = decode_latents(components.vae, latents)
    expected_pixels = torch.from_numpy(np.asarray(expected)).mul(255).round().to(torch.uint8)
    pixels_equal = torch.equal(actual, expected_pixels)
    record(
        f"prx_tiny_{width}x{height}_guidance_{guidance}_seed_{seed}",
        pixels_equal=pixels_equal,
        tolerance="exact",
    )
    assert pixels_equal


def test_prompt_cleaning_matches_reference(tiny, record) -> None:
    reference, _ = tiny
    prompts = (
        "A CAT &amp; a dog",
        "https://example.com @someone product.jpg",
        "東京 free shipping page 12",
        "this-is-a-many-part-file-name",
        '"quoted prompt"',
        "192.168.1.1 jc6640 800x600",
    )
    matches = [clean_text(prompt) for prompt in prompts] == [
        reference.text_preprocessor.clean_text(prompt) for prompt in prompts
    ]
    record("prx_prompt_cleaning", prompts=len(prompts), matches=matches, tolerance="exact")
    assert matches
