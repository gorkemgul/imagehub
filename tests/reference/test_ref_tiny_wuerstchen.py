"""Tiny random Würstchen v2 snapshot against ``WuerstchenCombinedPipeline`` (CPU float32)."""

import pytest
import torch
from helpers import text_state_for_ours, write_tiny_tokenizer

from imagehub import pipeline
from imagehub.backends.wuerstchen import (
    WuerstchenComponents,
    decode_latents,
    denoise_decoder,
    run_prior,
)
from tests.test_wuerstchen import (
    DECODER_CONFIG,
    DECODER_TEXT_CONFIG,
    PRIOR_CONFIG,
    PRIOR_TEXT_CONFIG,
    SCHEDULER_CONFIG,
    VQGAN_CONFIG,
    write_tiny_snapshot,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
wuerstchen = pytest.importorskip("diffusers.pipelines.deprecated.wuerstchen")
paella_module = pytest.importorskip(
    "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model"
)


def _randomize(model):
    with torch.no_grad():
        for value in model.parameters():
            if not value.any():  # zero-initialised output layers and modulation
                value.normal_(std=0.2)
        for name, tensor in model.state_dict().items():
            if name.endswith("gammas"):
                tensor.copy_(torch.randn_like(tensor) * 0.5)
            elif name.endswith("running_mean"):
                tensor.copy_(torch.randn_like(tensor))
            elif name.endswith("running_var"):
                tensor.copy_(torch.rand_like(tensor) + 0.5)
    return model.eval()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-wuerstchen")
    torch.manual_seed(0)
    prior_text = transformers.CLIPTextModel(transformers.CLIPTextConfig(**PRIOR_TEXT_CONFIG))
    text = transformers.CLIPTextModel(transformers.CLIPTextConfig(**DECODER_TEXT_CONFIG))
    prior = _randomize(wuerstchen.WuerstchenPrior(**PRIOR_CONFIG))
    decoder = _randomize(wuerstchen.WuerstchenDiffNeXt(**DECODER_CONFIG))
    vqgan = _randomize(paella_module.PaellaVQModel(**VQGAN_CONFIG))
    prior_text.eval()
    text.eval()
    write_tiny_snapshot(
        root,
        {
            "prior_text": text_state_for_ours(prior_text.state_dict()),
            "prior": prior.state_dict(),
            "text": text_state_for_ours(text.state_dict()),
            "decoder": decoder.state_dict(),
            "vqgan": vqgan.state_dict(),
        },
    )
    tokenizer = transformers.CLIPTokenizer.from_pretrained(write_tiny_tokenizer(root / "tokenizer"))
    reference = diffusers.WuerstchenCombinedPipeline(
        tokenizer=tokenizer,
        text_encoder=text,
        decoder=decoder,
        scheduler=diffusers.DDPMWuerstchenScheduler(**SCHEDULER_CONFIG),
        vqgan=vqgan,
        prior_tokenizer=tokenizer,
        prior_text_encoder=prior_text,
        prior_prior=prior,
        prior_scheduler=diffusers.DDPMWuerstchenScheduler(**SCHEDULER_CONFIG),
    )
    for pipe in (reference, reference.prior_pipe, reference.decoder_pipe):
        pipe.set_progress_bar_config(disable=True)
    components = WuerstchenComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, reference, components


CASES = {
    "default": {
        "seed": 5,
        "guidance": 0.0,
        "prior_guidance": 4.0,
        "negative": None,
        "size": (128, 128),
    },
    "decoder_guidance": {
        "seed": 6,
        "guidance": 3.0,
        "prior_guidance": 4.0,
        "negative": "cat",
        "size": (128, 128),
    },
    "decoder_guidance_only": {
        "seed": 8,
        "guidance": 3.0,
        "prior_guidance": 1.0,
        "negative": "hi",
        "size": (128, 128),
    },
    "non_square": {
        "seed": 7,
        "guidance": 0.0,
        "prior_guidance": 4.0,
        "negative": None,
        "size": (128, 256),
    },
    "other_seed": {
        "seed": 1234,
        "guidance": 0.0,
        "prior_guidance": 4.0,
        "negative": None,
        "size": (128, 128),
    },
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_tiny_combined_pipeline_matches_exactly(record, tiny, case):
    root, reference, components = tiny
    seed, guidance, prior_guidance, negative, (height, width) = CASES[case].values()
    arguments = {
        "prompt": "a cat",
        "negative_prompt": negative,
        "height": height,
        "width": width,
        "prior_num_inference_steps": 3,
        "prior_guidance_scale": prior_guidance,
        "num_inference_steps": 2,
        "decoder_guidance_scale": guidance,
    }
    expected_prior = reference.prior_pipe(
        prompt="a cat",
        negative_prompt=negative,
        height=height,
        width=width,
        num_inference_steps=3,
        guidance_scale=prior_guidance,
        generator=torch.Generator().manual_seed(seed),
        output_type="pt",
    ).image_embeddings
    expected_latents = reference(
        **arguments, generator=torch.Generator().manual_seed(seed), output_type="latent"
    ).images
    expected_pixels = reference(
        **arguments, generator=torch.Generator().manual_seed(seed), output_type="np"
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    cpu = torch.device("cpu")
    negative_text = "" if negative is None else negative
    generator = torch.Generator().manual_seed(seed)
    embeddings = run_prior(
        components,
        prompt="a cat",
        negative_prompt=negative_text,
        steps=3,
        guidance=prior_guidance,
        height=height,
        width=width,
        generator=generator,
        device=cpu,
    )
    latents = denoise_decoder(
        components,
        embeddings,
        prompt="a cat",
        negative_prompt=negative_text,
        steps=2,
        guidance=guidance,
        generator=generator,
        device=cpu,
    )
    pixels = decode_latents(components.vqgan, latents)
    record(
        f"wuerstchen_tiny_{case}",
        prior_max_abs_difference=float((embeddings - expected_prior).abs().max()),
        decoder_max_abs_difference=float((latents - expected_latents).abs().max()),
        pixels_equal=bool(torch.equal(pixels, expected_pixels)),
        tolerance=0.0,
    )
    assert torch.equal(embeddings, expected_prior)
    assert torch.equal(latents, expected_latents)
    assert torch.equal(pixels, expected_pixels)
    assert pixels.shape[1:3] == (height, width)

    result = pipeline("t2i", backend="wuerstchen", model=root)(
        "a cat",
        seed=seed,
        steps=2,
        prior_steps=3,
        guidance=guidance,
        prior_guidance=prior_guidance,
        negative_prompt=negative,
        height=height,
        width=width,
    )
    assert torch.equal(result.images[0].tensor, expected_pixels[0])


def test_long_prompt_is_truncated_like_the_reference(tiny):
    _, reference, components = tiny
    prompt = "cat hi " * 60
    expected = reference.prior_pipe(
        prompt=prompt,
        height=128,
        width=128,
        num_inference_steps=2,
        guidance_scale=4.0,
        generator=torch.Generator().manual_seed(3),
        output_type="pt",
    ).image_embeddings
    actual = run_prior(
        components,
        prompt=prompt,
        negative_prompt="",
        steps=2,
        guidance=4.0,
        height=128,
        width=128,
        generator=torch.Generator().manual_seed(3),
        device=torch.device("cpu"),
    )
    assert torch.equal(actual, expected)
