"""Stable Cascade parity against the local two-repository snapshots on CPU float32.

Snapshot layout: the decoder repository at the root (``decoder`` or ``decoder_lite``, ``vqgan``,
``scheduler``) and the prior repository under ``prior/`` (``prior`` or ``prior_lite``,
``text_encoder``, ``tokenizer``, ``scheduler``). The reference is assembled from components
because the snapshot has neither the prior's image encoder nor the decoder's text encoder.
"""

import gc
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.stable_cascade import (
    StableCascadeBackend,
    StableCascadeComponents,
    decode_latents,
    denoise_decoder,
    encode_text,
    run_prior,
)
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.stable_cascade_unet import StableCascadeUNet

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
paella_module = pytest.importorskip(
    "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model"
)

CPU = torch.device("cpu")
MODEL_IDS = ["stable-cascade-lite", "stable-cascade"]
PROMPT = "a photograph of an astronaut riding a horse"
SEEDS = (0, 1234)
SIZE = 512
PRIOR_STEPS = 60
PRIOR_GUIDANCE = 4.0
DECODER_STEPS = 12
DECODER_GUIDANCE = 0.0


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _free(*_objects):
    gc.collect()


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    shapes = {}
    for path in directory.glob("*.safetensors"):
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _components_of(model_id: str) -> tuple[str, str]:
    config = MODEL_CATALOG[model_id].config
    return str(config["prior_component"]), str(config["decoder_component"])


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _to_uint8(images) -> torch.Tensor:
    return torch.from_numpy((images * 255).round().astype("uint8"))


@pytest.fixture(scope="module", params=MODEL_IDS)
def model_id(request):
    return request.param


@pytest.fixture(scope="module")
def root(model_id):
    return snapshot_dir(model_id)


@pytest.fixture(scope="module")
def ours(model_id, root):
    """Our pipeline; its backend holds the only copy of our weights for this model id."""
    backend = StableCascadeBackend(root, model_id=model_id)
    pipe = pipeline("t2i", backend=backend)
    backend.load()
    assert isinstance(backend.components, StableCascadeComponents)
    yield pipe, backend.components
    backend.unload()
    _free()


@pytest.fixture(scope="module")
def text_reference(root):
    """Reference prompt embeddings: (prompt, pooled, negative, negative pooled)."""
    tokenizer = transformers.CLIPTokenizer.from_pretrained(root / "prior" / "tokenizer")
    text_encoder = transformers.CLIPTextModelWithProjection.from_pretrained(
        root / "prior" / "text_encoder", torch_dtype=torch.float32
    ).eval()
    stub = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
    with torch.inference_mode():
        result = diffusers.StableCascadePriorPipeline.encode_prompt(
            stub, CPU, 1, 1, True, prompt=PROMPT, negative_prompt=None
        )
    del text_encoder, stub
    _free()
    return result


def test_headers(record, model_id, root):
    prior_name, decoder_name = _components_of(model_id)
    result = {}
    for directory, cls in {
        root / "prior" / "text_encoder": CLIPTextModelWithProjection,
        root / "prior" / prior_name: StableCascadeUNet,
        root / decoder_name: StableCascadeUNet,
        root / "vqgan": PaellaVQModel,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(directory / "config.json"))
        shapes = _checkpoint_shapes(directory)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[str(directory.relative_to(root))] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    record(f"header_{model_id}", **result)
    assert all(
        shapes["tensors"] > 0 and not shapes[key]
        for shapes in result.values()
        for key in ("missing", "extra", "mismatch")
    )


def test_text_encoder(record, model_id, ours, text_reference):
    _, components = ours
    expected_embeds, expected_pooled, expected_negative, expected_negative_pooled = text_reference
    embeds, pooled = encode_text(components, PROMPT, CPU)
    negative, negative_pooled = encode_text(components, "", CPU)
    _check(record, f"{model_id}_text_embeds", embeds, expected_embeds)
    _check(record, f"{model_id}_text_pooled", pooled, expected_pooled)
    _check(record, f"{model_id}_negative_embeds", negative, expected_negative)
    _check(record, f"{model_id}_negative_pooled", negative_pooled, expected_negative_pooled)


def test_prior_unet(record, model_id, root, ours, text_reference):
    _, components = ours
    prior_name, _ = _components_of(model_id)
    embeds, pooled, negative, negative_pooled = text_reference
    text = torch.cat([embeds, negative])
    text_pooled = torch.cat([pooled, negative_pooled])
    clip_img = torch.zeros(2, 1, int(components.prior.config["clip_image_in_channels"]))
    sample = _randn(2, 16, 12, 12, seed=11)
    reference = diffusers.StableCascadeUNet.from_pretrained(
        root / "prior" / prior_name, torch_dtype=torch.float32
    ).eval()
    for label, value in (("1", 1.0), ("0.5", 0.5), ("1_60", 1 / 60)):
        ratio = torch.full((2,), value)
        with torch.inference_mode():
            expected = reference(
                sample=sample,
                timestep_ratio=ratio,
                clip_text_pooled=text_pooled,
                clip_text=text,
                clip_img=clip_img,
                return_dict=False,
            )[0]
            actual = components.prior(sample, ratio, text_pooled, clip_text=text, clip_img=clip_img)
        _check(record, f"{model_id}_prior_unet_ratio_{label}", actual, expected)
    del reference
    _free()


def test_decoder_unet(record, model_id, root, ours, text_reference):
    _, components = ours
    _, decoder_name = _components_of(model_id)
    pooled = text_reference[1]
    sample = _randn(1, 4, 128, 128, seed=21)
    effnet = _randn(1, 16, 12, 12, seed=22)
    reference = diffusers.StableCascadeUNet.from_pretrained(
        root / decoder_name, torch_dtype=torch.float32
    ).eval()
    for label, value in (("1", 1.0), ("0.5", 0.5)):
        ratio = torch.full((1,), value)
        with torch.inference_mode():
            expected = reference(
                sample=sample,
                timestep_ratio=ratio,
                clip_text_pooled=pooled,
                effnet=effnet,
                return_dict=False,
            )[0]
            actual = components.decoder(sample, ratio, pooled, effnet=effnet)
        _check(record, f"{model_id}_decoder_unet_ratio_{label}", actual, expected)
    del reference
    _free()


def test_stage_a(record, model_id, root, ours):
    _, components = ours
    reference = paella_module.PaellaVQModel.from_pretrained(
        root / "vqgan", torch_dtype=torch.float32
    ).eval()
    latents = _randn(1, 4, 128, 128, seed=31)
    with torch.inference_mode():
        expected = reference.decode(reference.config.scale_factor * latents).sample
        actual = components.vqgan.decode(components.vqgan.scale_factor * latents)
        expected_pixels = (
            (expected.clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy() * 255)
            .round()
            .astype("uint8")
        )
    _check(record, f"{model_id}_stage_a", actual, expected)
    pixels = decode_latents(components.vqgan, latents)
    assert torch.equal(pixels, torch.from_numpy(expected_pixels))
    del reference
    _free()


def _build_reference(root: Path, model_id: str):
    prior_name, decoder_name = _components_of(model_id)
    tokenizer = transformers.CLIPTokenizer.from_pretrained(root / "prior" / "tokenizer")
    text_encoder = transformers.CLIPTextModelWithProjection.from_pretrained(
        root / "prior" / "text_encoder", torch_dtype=torch.float32
    ).eval()
    prior = diffusers.StableCascadeUNet.from_pretrained(
        root / "prior" / prior_name, torch_dtype=torch.float32
    ).eval()
    decoder = diffusers.StableCascadeUNet.from_pretrained(
        root / decoder_name, torch_dtype=torch.float32
    ).eval()
    vqgan = paella_module.PaellaVQModel.from_pretrained(
        root / "vqgan", torch_dtype=torch.float32
    ).eval()
    prior_scheduler = diffusers.DDPMWuerstchenScheduler.from_pretrained(
        root / "prior", subfolder="scheduler"
    )
    scheduler = diffusers.DDPMWuerstchenScheduler.from_pretrained(root, subfolder="scheduler")
    combined = diffusers.StableCascadeCombinedPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        decoder=decoder,
        scheduler=scheduler,
        vqgan=vqgan,
        prior_prior=prior,
        prior_text_encoder=text_encoder,
        prior_tokenizer=tokenizer,
        prior_scheduler=prior_scheduler,
    )
    for pipe in (combined, combined.prior_pipe, combined.decoder_pipe):
        pipe.set_progress_bar_config(disable=True)
    # The combined pipeline builds its stages with the library defaults for these two values.
    resolution_multiple = read_json(root / "prior" / "model_index.json").get(
        "resolution_multiple", 42.67
    )
    latent_dim_scale = read_json(root / "model_index.json").get("latent_dim_scale", 10.67)
    assert combined.prior_pipe.config.resolution_multiple == resolution_multiple
    assert combined.decoder_pipe.config.latent_dim_scale == latent_dim_scale
    return combined


@pytest.fixture(scope="module")
def reference_runs(model_id, root):
    """Reference results per seed; the reference models are freed before returning."""
    combined = _build_reference(root, model_id)
    runs = {}
    with torch.inference_mode():
        # Seed 0 stage by stage, exactly as the combined pipeline chains its stages.
        generator = torch.Generator("cpu").manual_seed(SEEDS[0])
        prior_out = combined.prior_pipe(
            prompt=PROMPT,
            negative_prompt=None,
            height=SIZE,
            width=SIZE,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE,
            generator=generator,
            output_type="pt",
            return_dict=True,
        )
        captured = {}

        def capture(pipe, step, timestep, callback_kwargs):
            captured["latents"] = callback_kwargs["latents"].clone()
            return callback_kwargs

        images = combined.decoder_pipe(
            image_embeddings=prior_out.image_embeddings,
            prompt=None,
            num_inference_steps=DECODER_STEPS,
            guidance_scale=DECODER_GUIDANCE,
            negative_prompt=None,
            prompt_embeds=prior_out.prompt_embeds,
            prompt_embeds_pooled=prior_out.prompt_embeds_pooled,
            negative_prompt_embeds=prior_out.negative_prompt_embeds,
            negative_prompt_embeds_pooled=prior_out.negative_prompt_embeds_pooled,
            generator=generator,
            output_type="np",
            callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
        runs[SEEDS[0]] = {
            "embeddings": prior_out.image_embeddings,
            "latents": captured["latents"],
            "images": _to_uint8(images),
        }
        # The second seed through the combined pipeline itself.
        images = combined(
            prompt=PROMPT,
            height=SIZE,
            width=SIZE,
            prior_num_inference_steps=PRIOR_STEPS,
            prior_guidance_scale=PRIOR_GUIDANCE,
            num_inference_steps=DECODER_STEPS,
            decoder_guidance_scale=DECODER_GUIDANCE,
            generator=torch.Generator("cpu").manual_seed(SEEDS[1]),
            output_type="np",
        ).images
        runs[SEEDS[1]] = {"images": _to_uint8(images)}
    del combined, prior_out
    _free()
    return runs


def test_end_to_end_defaults(record, model_id, ours, reference_runs):
    pipe, components = ours
    expected = reference_runs[SEEDS[0]]
    generator = torch.Generator("cpu").manual_seed(SEEDS[0])
    embeddings, pooled, negative_pooled = run_prior(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=PRIOR_STEPS,
        guidance=PRIOR_GUIDANCE,
        height=SIZE,
        width=SIZE,
        generator=generator,
        device=CPU,
    )
    _check(record, f"{model_id}_prior_latents", embeddings, expected["embeddings"])
    latents = denoise_decoder(
        components,
        embeddings,
        pooled,
        negative_pooled,
        steps=DECODER_STEPS,
        guidance=DECODER_GUIDANCE,
        generator=generator,
        device=CPU,
    )
    _check(record, f"{model_id}_decoder_latents", latents, expected["latents"])
    staged = decode_latents(components.vqgan, latents)
    assert torch.equal(staged, expected["images"])
    result = pipe(PROMPT, seed=SEEDS[0])
    actual = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(actual, expected["images"])
    record(
        f"e2e_{model_id}",
        passed=passed,
        pixels_max_abs=int((actual.int() - expected["images"].int()).abs().max()),
        seed=SEEDS[0],
    )
    assert passed


def test_second_seed(record, model_id, ours, reference_runs):
    pipe, _ = ours
    expected = reference_runs[SEEDS[1]]["images"]
    result = pipe(PROMPT, seed=SEEDS[1])
    actual = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(actual, expected)
    record(f"seed_e2e_{model_id}", passed=passed, seed=SEEDS[1])
    assert passed
    assert not torch.equal(actual, reference_runs[SEEDS[0]]["images"])
