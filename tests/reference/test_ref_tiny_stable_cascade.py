import json

import pytest
import torch
from helpers import write_tiny_tokenizer

from imagehub.backends.stable_cascade import (
    StableCascadeComponents,
    decode_latents,
    denoise_decoder,
    run_prior,
)
from imagehub.checkpoints import save_safetensors
from tests.test_stable_cascade import (
    DECODER_CONFIG,
    DECODER_INDEX,
    PRIOR_CONFIG,
    PRIOR_INDEX,
    SCHEDULER_CONFIG,
    TEXT_CONFIG,
    VQGAN_CONFIG,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
paella_module = pytest.importorskip(
    "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model"
)


def _write_component(root, name, model, config, weights):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_safetensors(model.state_dict(), directory / weights)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _randomize_unet(model):
    with torch.no_grad():
        for name, value in model.named_parameters():
            if name.endswith(("gamma", "beta")):
                value.normal_(std=0.5)


def _randomize_vqgan(model):
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith("gammas"):
                tensor.copy_(torch.randn_like(tensor) * 0.5)
            elif name.endswith("running_mean"):
                tensor.copy_(torch.randn_like(tensor))
            elif name.endswith("running_var"):
                tensor.copy_(torch.rand_like(tensor) + 0.5)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-cascade")
    torch.manual_seed(0)
    text_encoder = transformers.CLIPTextModelWithProjection(
        transformers.CLIPTextConfig(**TEXT_CONFIG)
    ).eval()
    prior = diffusers.StableCascadeUNet(**PRIOR_CONFIG).eval()
    decoder = diffusers.StableCascadeUNet(**DECODER_CONFIG).eval()
    _randomize_unet(prior)
    _randomize_unet(decoder)
    vqgan = paella_module.PaellaVQModel(**VQGAN_CONFIG)
    _randomize_vqgan(vqgan)
    vqgan.eval()
    tokenizer_path = write_tiny_tokenizer(root / "prior" / "tokenizer")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(tokenizer_path)

    _write_component(root / "prior", "text_encoder", text_encoder, TEXT_CONFIG, "model.safetensors")
    _write_component(
        root / "prior", "prior", prior, PRIOR_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    _write_json(root / "prior" / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    _write_json(root / "prior" / "model_index.json", PRIOR_INDEX)
    _write_component(
        root, "decoder", decoder, DECODER_CONFIG, "diffusion_pytorch_model.safetensors"
    )
    _write_component(root, "vqgan", vqgan, VQGAN_CONFIG, "diffusion_pytorch_model.safetensors")
    _write_json(root / "scheduler" / "scheduler_config.json", SCHEDULER_CONFIG)
    _write_json(root / "model_index.json", DECODER_INDEX)

    reference = diffusers.StableCascadeCombinedPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        decoder=decoder,
        scheduler=diffusers.DDPMWuerstchenScheduler(**SCHEDULER_CONFIG),
        vqgan=vqgan,
        prior_prior=prior,
        prior_text_encoder=text_encoder,
        prior_tokenizer=tokenizer,
        prior_scheduler=diffusers.DDPMWuerstchenScheduler(**SCHEDULER_CONFIG),
    )
    reference.set_progress_bar_config(disable=True)
    reference.prior_pipe.set_progress_bar_config(disable=True)
    reference.decoder_pipe.set_progress_bar_config(disable=True)
    components = StableCascadeComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return reference, components


CASES = {
    "default": {"seed": 5, "guidance": 0.0, "negative": None, "size": (128, 128)},
    "decoder_guidance": {"seed": 6, "guidance": 3.0, "negative": "cat", "size": (128, 128)},
    "non_square": {"seed": 7, "guidance": 0.0, "negative": None, "size": (128, 256)},
    "other_seed": {"seed": 1234, "guidance": 0.0, "negative": None, "size": (128, 128)},
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_tiny_combined_pipeline_matches_exactly(record, tiny, case):
    reference, components = tiny
    seed, guidance, negative, (height, width) = CASES[case].values()
    arguments = {
        "prompt": "a cat",
        "negative_prompt": negative,
        "height": height,
        "width": width,
        "prior_num_inference_steps": 3,
        "prior_guidance_scale": 4.0,
        "num_inference_steps": 2,
        "decoder_guidance_scale": guidance,
    }
    expected_prior = reference.prior_pipe(
        prompt="a cat",
        negative_prompt=negative,
        height=height,
        width=width,
        num_inference_steps=3,
        guidance_scale=4.0,
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

    generator = torch.Generator().manual_seed(seed)
    embeddings, pooled, negative_pooled = run_prior(
        components,
        prompt="a cat",
        negative_prompt="" if negative is None else negative,
        steps=3,
        guidance=4.0,
        height=height,
        width=width,
        generator=generator,
        device=torch.device("cpu"),
    )
    latents = denoise_decoder(
        components,
        embeddings,
        pooled,
        negative_pooled,
        steps=2,
        guidance=guidance,
        generator=generator,
        device=torch.device("cpu"),
    )
    pixels = decode_latents(components.vqgan, latents)
    record(
        f"stable_cascade_tiny_{case}",
        prior_max_abs_difference=float((embeddings - expected_prior).abs().max()),
        decoder_max_abs_difference=float((latents - expected_latents).abs().max()),
        pixels_equal=bool(torch.equal(pixels, expected_pixels)),
        tolerance=0.0,
    )
    assert torch.equal(embeddings, expected_prior)
    assert torch.equal(latents, expected_latents)
    assert torch.equal(pixels, expected_pixels)
    assert pixels.shape[1:3] == (height, width)


def test_different_seeds_give_different_images(tiny):
    _, components = tiny
    images = []
    for seed in (1, 2):
        generator = torch.Generator().manual_seed(seed)
        embeddings, pooled, negative_pooled = run_prior(
            components,
            prompt="a cat",
            negative_prompt="",
            steps=2,
            guidance=4.0,
            height=128,
            width=128,
            generator=generator,
            device=torch.device("cpu"),
        )
        latents = denoise_decoder(
            components,
            embeddings,
            pooled,
            negative_pooled,
            steps=2,
            guidance=0.0,
            generator=generator,
            device=torch.device("cpu"),
        )
        images.append(decode_latents(components.vqgan, latents))
    assert not torch.equal(images[0], images[1])
