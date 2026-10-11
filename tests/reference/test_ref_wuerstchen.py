"""Würstchen v2 parity against the local two-repository snapshot on CPU float32.

Snapshot layout: the decoder repository at the root (``decoder``, ``text_encoder``,
``tokenizer``, ``vqgan``, ``scheduler``) and the prior repository under ``prior/`` (``prior``,
``text_encoder``, ``tokenizer``, ``scheduler``). The reference is
``WuerstchenCombinedPipeline`` assembled from the same files.
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
from imagehub.backends.wuerstchen import (
    WuerstchenBackend,
    WuerstchenComponents,
    decode_latents,
    denoise_decoder,
    encode_text,
    run_prior,
)
from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.paella_vq import PaellaVQModel
from imagehub.models.wuerstchen import WuerstchenDiffNeXt, WuerstchenPrior

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
wuerstchen = pytest.importorskip("diffusers.pipelines.deprecated.wuerstchen")
paella_module = pytest.importorskip(
    "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model"
)

CPU = torch.device("cpu")
MODEL_ID = "wuerstchen-v2"
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


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _to_uint8(images) -> torch.Tensor:
    return torch.from_numpy((images * 255).round().astype("uint8"))


@pytest.fixture(scope="module")
def root():
    return snapshot_dir(MODEL_ID)


@pytest.fixture(scope="module")
def ours(root):
    """Our pipeline; its backend holds the only copy of our weights."""
    backend = WuerstchenBackend(root, model_id=MODEL_ID)
    pipe = pipeline("t2i", backend=backend)
    backend.load()
    assert isinstance(backend.components, WuerstchenComponents)
    yield pipe, backend.components
    backend.unload()
    _free()


def _reference_text(root: Path, stage_root: Path, decoder: bool):
    tokenizer = transformers.CLIPTokenizer.from_pretrained(stage_root / "tokenizer")
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        stage_root / "text_encoder", torch_dtype=torch.float32
    ).eval()
    stub = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
    with torch.inference_mode():
        if decoder:
            result = wuerstchen.WuerstchenDecoderPipeline.encode_prompt(
                stub, [PROMPT], CPU, 1, True, negative_prompt=None
            )
        else:
            result = wuerstchen.WuerstchenPriorPipeline.encode_prompt(
                stub, CPU, 1, True, prompt=[PROMPT], negative_prompt=None
            )
    del text_encoder, stub
    _free()
    return result


@pytest.fixture(scope="module")
def prior_text_reference(root):
    """Reference prior text states: (prompt, negative)."""
    return _reference_text(root, root / "prior", decoder=False)


@pytest.fixture(scope="module")
def decoder_text_reference(root):
    """Reference decoder text states: (prompt, negative)."""
    return _reference_text(root, root, decoder=True)


def test_headers(record, root):
    result = {}
    for directory, cls in {
        root / "prior" / "text_encoder": CLIPTextModel,
        root / "text_encoder": CLIPTextModel,
        root / "prior" / "prior": WuerstchenPrior,
        root / "decoder": WuerstchenDiffNeXt,
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
    record(f"header_{MODEL_ID}", **result)
    assert all(
        shapes["tensors"] > 0 and not shapes[key]
        for shapes in result.values()
        for key in ("missing", "extra", "mismatch")
    )


def test_text_encoders(record, ours, prior_text_reference, decoder_text_reference):
    _, components = ours
    prior_prompt, prior_negative = prior_text_reference
    decoder_prompt, decoder_negative = decoder_text_reference
    for label, tokenizer, encoder, expected_prompt, expected_negative in (
        (
            "prior",
            components.prior_tokenizer,
            components.prior_text_encoder,
            prior_prompt,
            prior_negative,
        ),
        (
            "decoder",
            components.tokenizer,
            components.text_encoder,
            decoder_prompt,
            decoder_negative,
        ),
    ):
        _check(
            record,
            f"{MODEL_ID}_{label}_text_states",
            encode_text(tokenizer, encoder, PROMPT, CPU),
            expected_prompt,
        )
        _check(
            record,
            f"{MODEL_ID}_{label}_negative_states",
            encode_text(tokenizer, encoder, "", CPU),
            expected_negative,
        )


def test_prior_step(record, root, ours, prior_text_reference):
    _, components = ours
    prompt, negative = prior_text_reference
    text = torch.cat([prompt, negative])
    sample = _randn(2, 16, 12, 12, seed=11)
    reference = wuerstchen.WuerstchenPrior.from_pretrained(
        root / "prior" / "prior", torch_dtype=torch.float32
    ).eval()
    for label, value in (("1", 1.0), ("0.5", 0.5), ("1_60", 1 / 60)):
        ratio = torch.full((2,), value)
        with torch.inference_mode():
            expected = reference(sample, r=ratio, c=text)
            actual = components.prior(sample, ratio, text)
        _check(record, f"{MODEL_ID}_prior_step_ratio_{label}", actual, expected)
    del reference
    _free()


def test_decoder_step(record, root, ours, decoder_text_reference):
    _, components = ours
    prompt, _ = decoder_text_reference
    sample = _randn(1, 4, 128, 128, seed=21)
    effnet = _randn(1, 16, 12, 12, seed=22)
    reference = wuerstchen.WuerstchenDiffNeXt.from_pretrained(
        root / "decoder", torch_dtype=torch.float32
    ).eval()
    for label, value in (("1", 1.0), ("0.5", 0.5)):
        ratio = torch.full((1,), value)
        with torch.inference_mode():
            expected = reference(sample, r=ratio, effnet=effnet, clip=prompt)
            actual = components.decoder(sample, ratio, effnet, prompt)
        _check(record, f"{MODEL_ID}_decoder_step_ratio_{label}", actual, expected)
    del reference
    _free()


def test_vq_decode(record, root, ours):
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
    _check(record, f"{MODEL_ID}_vq_decode", actual, expected)
    pixels = decode_latents(components.vqgan, latents)
    assert torch.equal(pixels, torch.from_numpy(expected_pixels))
    del reference
    _free()


def _build_reference(root: Path):
    prior_root = root / "prior"
    tokenizer = transformers.CLIPTokenizer.from_pretrained(root / "tokenizer")
    prior_tokenizer = transformers.CLIPTokenizer.from_pretrained(prior_root / "tokenizer")
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        root / "text_encoder", torch_dtype=torch.float32
    ).eval()
    prior_text_encoder = transformers.CLIPTextModel.from_pretrained(
        prior_root / "text_encoder", torch_dtype=torch.float32
    ).eval()
    prior = wuerstchen.WuerstchenPrior.from_pretrained(
        prior_root / "prior", torch_dtype=torch.float32
    ).eval()
    decoder = wuerstchen.WuerstchenDiffNeXt.from_pretrained(
        root / "decoder", torch_dtype=torch.float32
    ).eval()
    vqgan = paella_module.PaellaVQModel.from_pretrained(
        root / "vqgan", torch_dtype=torch.float32
    ).eval()
    combined = diffusers.WuerstchenCombinedPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        decoder=decoder,
        scheduler=diffusers.DDPMWuerstchenScheduler.from_pretrained(root, subfolder="scheduler"),
        vqgan=vqgan,
        prior_tokenizer=prior_tokenizer,
        prior_text_encoder=prior_text_encoder,
        prior_prior=prior,
        prior_scheduler=diffusers.DDPMWuerstchenScheduler.from_pretrained(
            prior_root, subfolder="scheduler"
        ),
    )
    for pipe in (combined, combined.prior_pipe, combined.decoder_pipe):
        pipe.set_progress_bar_config(disable=True)
    # The combined pipeline builds its stages with the library defaults for these values.
    prior_index = read_json(prior_root / "model_index.json")
    assert combined.prior_pipe.config.resolution_multiple == prior_index["resolution_multiple"]
    assert combined.prior_pipe.config.latent_mean == prior_index["latent_mean"]
    assert combined.prior_pipe.config.latent_std == prior_index["latent_std"]
    assert (
        combined.decoder_pipe.config.latent_dim_scale
        == read_json(root / "model_index.json")["latent_dim_scale"]
    )
    return combined


@pytest.fixture(scope="module")
def reference_runs(root):
    """Reference results per seed; the reference models are freed before returning."""
    combined = _build_reference(root)
    runs = {}
    with torch.inference_mode():
        # Seed 0 stage by stage, exactly as the combined pipeline chains its stages.
        generator = torch.Generator("cpu").manual_seed(SEEDS[0])
        embeddings = combined.prior_pipe(
            prompt=PROMPT,
            height=SIZE,
            width=SIZE,
            num_inference_steps=PRIOR_STEPS,
            guidance_scale=PRIOR_GUIDANCE,
            negative_prompt=None,
            generator=generator,
            output_type="pt",
            return_dict=False,
        )[0]
        captured = {}

        def capture(pipe, step, timestep, callback_kwargs):
            captured["latents"] = callback_kwargs["latents"].clone()
            return callback_kwargs

        images = combined.decoder_pipe(
            image_embeddings=embeddings,
            prompt=PROMPT,
            num_inference_steps=DECODER_STEPS,
            guidance_scale=DECODER_GUIDANCE,
            negative_prompt=None,
            generator=generator,
            output_type="np",
            callback_on_step_end=capture,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images
        runs[SEEDS[0]] = {
            "embeddings": embeddings,
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
    del combined, embeddings
    _free()
    return runs


def test_end_to_end_defaults(record, ours, reference_runs):
    pipe, components = ours
    expected = reference_runs[SEEDS[0]]
    generator = torch.Generator("cpu").manual_seed(SEEDS[0])
    embeddings = run_prior(
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
    _check(record, f"{MODEL_ID}_prior_latents", embeddings, expected["embeddings"])
    latents = denoise_decoder(
        components,
        embeddings,
        prompt=PROMPT,
        negative_prompt="",
        steps=DECODER_STEPS,
        guidance=DECODER_GUIDANCE,
        generator=generator,
        device=CPU,
    )
    _check(record, f"{MODEL_ID}_decoder_latents", latents, expected["latents"])
    staged = decode_latents(components.vqgan, latents)
    assert torch.equal(staged, expected["images"])
    result = pipe(PROMPT, seed=SEEDS[0])
    actual = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(actual, expected["images"])
    record(
        f"e2e_{MODEL_ID}",
        passed=passed,
        pixels_max_abs=int((actual.int() - expected["images"].int()).abs().max()),
        seed=SEEDS[0],
    )
    assert passed


def test_second_seed(record, ours, reference_runs):
    pipe, _ = ours
    expected = reference_runs[SEEDS[1]]["images"]
    result = pipe(PROMPT, seed=SEEDS[1])
    actual = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(actual, expected)
    record(f"seed_e2e_{MODEL_ID}", passed=passed, seed=SEEDS[1])
    assert passed
    assert not torch.equal(actual, reference_runs[SEEDS[0]]["images"])
