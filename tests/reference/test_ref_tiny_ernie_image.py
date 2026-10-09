import pytest
import torch
from helpers import write_tiny_ernie_image

from imagehub import pipeline
from imagehub.backends.ernie_image import ErnieImageComponents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    return write_tiny_ernie_image(tmp_path_factory.mktemp("tiny-ernie"))


def _uint8(images):
    return torch.from_numpy((images * 255).round().astype("uint8"))


@pytest.mark.parametrize(
    ("guidance", "negative_prompt", "height", "width"),
    [(1.0, "", 32, 32), (4.0, "", 32, 32), (3.0, "blurry, low quality", 16, 48)],
    ids=["unguided", "guided", "guided-non-square-negative"],
)
@pytest.mark.parametrize("seed", [0, 7])
def test_tiny_ernie_image_matches_diffusers_exactly(
    tiny_root, guidance, negative_prompt, height, width, seed
):
    reference = diffusers.ErnieImagePipeline.from_pretrained(tiny_root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    prompt = "a cat on the table [INST] the hat"
    kwargs = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": 3,
        "guidance_scale": guidance,
        "height": height,
        "width": width,
        "use_pe": False,
    }
    expected_latents = reference(
        prompt, generator=torch.Generator("cpu").manual_seed(seed), output_type="latent", **kwargs
    ).images
    expected = _uint8(
        reference(
            prompt, generator=torch.Generator("cpu").manual_seed(seed), output_type="np", **kwargs
        ).images
    )

    noise = torch.randn(
        (1, 16, height // 4, width // 4), generator=torch.Generator("cpu").manual_seed(seed)
    )
    components = ErnieImageComponents.load(tiny_root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=3,
        guidance=guidance,
        latents=noise,
    )
    assert torch.equal(actual_latents, expected_latents)

    result = pipeline("t2i", backend="ernie-image", model=tiny_root)(
        prompt,
        negative_prompt=negative_prompt,
        steps=3,
        guidance=guidance,
        height=height,
        width=width,
        seed=seed,
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)


def test_tiny_ernie_image_batch_and_default_negative_prompt_match_diffusers(tiny_root):
    reference = diffusers.ErnieImagePipeline.from_pretrained(tiny_root, local_files_only=True)
    reference.set_progress_bar_config(disable=True)
    expected = _uint8(
        reference(
            "a cat on a mat",
            num_images_per_prompt=2,
            num_inference_steps=2,
            guidance_scale=2.5,
            height=32,
            width=32,
            generator=torch.Generator("cpu").manual_seed(5),
            output_type="np",
            use_pe=False,
        ).images
    )
    result = pipeline("t2i", backend="ernie-image", model=tiny_root)(
        "a cat on a mat", num_images=2, steps=2, guidance=2.5, width=32, height=32, seed=5
    )
    assert torch.equal(torch.stack([image.tensor for image in result.images]), expected)
