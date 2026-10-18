"""Tiny-snapshot end-to-end parity with the pinned official Meissonic pipeline."""

import sys

import pytest
import torch
from helpers import write_tiny_meissonic

from imagehub import pipeline
from imagehub.backends.meissonic import MeissonicComponents, decode_tokens, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

REFERENCE = "/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/meissonic"
PROMPT = "A photo of a cat, sitting on the mat in the sun."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = write_tiny_meissonic(tmp_path_factory.mktemp("meissonic") / "tiny")
    sys.path.insert(0, REFERENCE)
    try:
        from src.pipeline import Pipeline
        from src.scheduler import Scheduler
        from src.transformer import Transformer2DModel

        transformer = Transformer2DModel.from_pretrained(
            root / "transformer", local_files_only=True
        )
        for block in [*transformer.transformer_blocks, *transformer.single_transformer_blocks]:
            block.attn.head_dim = 8
        reference = Pipeline(
            diffusers.VQModel.from_pretrained(root / "vqvae", local_files_only=True),
            transformers.CLIPTokenizer.from_pretrained(root / "tokenizer", local_files_only=True),
            transformers.CLIPTextModelWithProjection.from_pretrained(
                root / "text_encoder", local_files_only=True
            ),
            transformer,
            Scheduler(mask_token_id=23, masking_schedule="cosine"),
        )
        reference.set_progress_bar_config(disable=True)
        components = MeissonicComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
        yield root, reference, components
    finally:
        sys.path.remove(REFERENCE)


@pytest.mark.parametrize(
    ("seed", "guidance", "height", "width", "name"),
    [
        (0, 1.0, 32, 32, "default_grid"),
        (42, 5.0, 32, 32, "guidance"),
        (7, 3.0, 32, 64, "non_square"),
    ],
)
def test_tiny_pipeline_exact(tiny, record, seed, guidance, height, width, name):
    root, reference, components = tiny
    kwargs = {
        "height": height,
        "width": width,
        "num_inference_steps": 3,
        "guidance_scale": guidance,
    }
    expected_tokens = reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
        **kwargs,
    ).images
    expected_images = reference(
        PROMPT,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="pil",
        **kwargs,
    ).images
    actual_tokens = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=3,
        guidance=guidance,
        height=height,
        width=width,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    expected_pixels = torch.from_numpy(np.stack([np.array(image) for image in expected_images]))
    actual_pixels = decode_tokens(components.vqvae, actual_tokens)
    assert torch.equal(actual_tokens, expected_tokens)
    assert torch.equal(actual_pixels, expected_pixels)
    result = pipeline("t2i", backend="meissonic", model=root)(
        PROMPT,
        seed=seed,
        steps=3,
        guidance=guidance,
        height=height,
        width=width,
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected_pixels)
    record(f"meissonic_tiny_{name}", tokens_equal=True, pixels_equal=True)
