"""aMUSEd parity against local published snapshots on CPU float32 (both published sizes)."""

import gc
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.amused import AmusedComponents, decode_tokens, denoise
from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.uvit_2d import UVit2DModel
from imagehub.models.vq_model import VQModel
from imagehub.tokenization import CLIPTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_IDS = ["amused-256", "amused-512"]
SIZES = {"amused-256": 256, "amused-512": 512}
# A different, non-square size per model for the seeded pipeline run.
SEED_SIZES = {"amused-256": (192, 320), "amused-512": (384, 640)}
PROMPT = "a photograph of an astronaut riding a horse"
STEPS = 12
GUIDANCE = 10.0


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is None:
        files = [directory / "model.safetensors"]
        if not files[0].exists():
            files = [directory / "diffusion_pytorch_model.safetensors"]
    else:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _reference_pipeline(root, **modules):
    reference = diffusers.AmusedPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32, **modules
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _tokens(ours, text):
    return torch.tensor([ours(text)])


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_headers(record, model_id):
    root = snapshot_dir(model_id)
    result = {}
    for component, cls in {
        "text_encoder": CLIPTextModelWithProjection,
        "transformer": UVit2DModel,
        "vqvae": VQModel,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    record(f"header_{model_id}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_tokenizer(record, model_id):
    root = snapshot_dir(model_id)
    ours = CLIPTokenizer(root / "tokenizer")
    reference = transformers.CLIPTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    cases = [
        PROMPT,
        "",
        "A Cat, sitting on THE mat!  Quoted: 'it' costs 5 euros.",
        " ".join([PROMPT] * 20),
    ]
    matches = []
    for case in cases:
        expected = reference(
            case,
            padding="max_length",
            truncation=True,
            max_length=reference.model_max_length,
            return_tensors="pt",
        ).input_ids
        matches.append(torch.equal(_tokens(ours, case), expected))
    record(f"{model_id}_tokenizer", cases=matches, tolerance="exact")
    assert all(matches)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_text_encoder(record, model_id):
    root = snapshot_dir(model_id)
    ours = CLIPTextModelWithProjection.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    reference = transformers.CLIPTextModelWithProjection.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    ).eval()
    tokenizer = CLIPTokenizer(root / "tokenizer")
    for label, text in (("prompt", PROMPT), ("negative", "")):
        ids = _tokens(tokenizer, text)
        with torch.inference_mode():
            embeds, _, hidden = ours(ids, output_hidden_states=True)
            expected = reference(ids, return_dict=True, output_hidden_states=True)
        _check(record, f"{model_id}_text_embeds_{label}", embeds, expected.text_embeds)
        _check(record, f"{model_id}_hidden_state_{label}", hidden[-2], expected.hidden_states[-2])
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_transformer_step(record, model_id):
    root = snapshot_dir(model_id)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    tokenizer = CLIPTokenizer(root / "tokenizer")
    with torch.inference_mode():
        pooled, _, hidden = text_encoder(_tokens(tokenizer, PROMPT), output_hidden_states=True)
    states = hidden[-2]
    del text_encoder
    ours = UVit2DModel.from_pretrained(root / "transformer", device=CPU, dtype=torch.float32)
    reference = diffusers.UVit2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    mask_id = 8255
    grid = SIZES[model_id] // 16
    generator = torch.Generator("cpu").manual_seed(1)
    ids = torch.randint(0, 8192, (2, grid, grid), generator=generator)
    ids[torch.rand(ids.shape, generator=generator) < 0.6] = mask_id
    size = SIZES[model_id]
    micro = torch.tensor([[size, size, 0, 0, 6]], dtype=torch.float32).expand(2, -1)
    with torch.inference_mode():
        actual = ours(ids, states.expand(2, -1, -1), pooled.expand(2, -1), micro)
        expected = reference(
            ids,
            micro_conds=micro,
            pooled_text_emb=pooled.expand(2, -1),
            encoder_hidden_states=states.expand(2, -1, -1),
        )
    _check(record, f"{model_id}_transformer_step", actual, expected)
    del ours, reference
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_vq_decode_of_fixed_ids(record, model_id):
    root = snapshot_dir(model_id)
    ours = VQModel.from_pretrained(root / "vqvae", device=CPU, dtype=torch.float32)
    reference = diffusers.VQModel.from_pretrained(
        root / "vqvae", local_files_only=True, torch_dtype=torch.float32
    )
    channels = int(ours.config["latent_channels"])
    ids = torch.randint(0, 8192, (1, 16, 16), generator=torch.Generator("cpu").manual_seed(1))
    with torch.inference_mode():
        actual = ours.decode(ids, force_not_quantize=True, shape=(1, 16, 16, channels))
        expected = reference.decode(
            ids, force_not_quantize=True, shape=(1, 16, 16, channels)
        ).sample
    _check(record, f"{model_id}_vq_decode", actual, expected)
    del ours, reference
    gc.collect()


def _reference_output(root, seed, **kwargs):
    reference = _reference_pipeline(root)
    tokens = reference(
        PROMPT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="latent",
        **kwargs,
    ).images
    # The pipeline's own decode and post-processing, applied to the tokens of the same run.
    with torch.inference_mode():
        height, width = tokens.shape[1:]
        image = reference.vqvae.decode(
            tokens,
            force_not_quantize=True,
            shape=(1, height, width, reference.vqvae.config.latent_channels),
        ).sample.clip(0, 1)
    images = reference.image_processor.postprocess(image, output_type="pil")
    pixels = torch.from_numpy(np.stack([np.array(item) for item in images]))
    del reference
    gc.collect()
    return tokens, pixels


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(record, model_id):
    root = snapshot_dir(model_id)
    expected_tokens, expected_pixels = _reference_output(root, 0)
    components = AmusedComponents.load(root, device=CPU, dtype=torch.float32, model_id=model_id)
    size = SIZES[model_id]
    actual_tokens = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=STEPS,
        guidance=GUIDANCE,
        height=size,
        width=size,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    actual_pixels = decode_tokens(components.vqvae, actual_tokens)
    tokens_equal = torch.equal(actual_tokens, expected_tokens)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = tokens_equal and pixel_diff == 0
    record(
        f"e2e_{model_id}",
        tokens_equal=tokens_equal,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=STEPS,
        guidance=GUIDANCE,
        size=[size, size],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(record, model_id):
    root = snapshot_dir(model_id)
    height, width = SEED_SIZES[model_id]
    _, expected = _reference_output(root, 1234, height=height, width=width)
    result = pipeline("t2i", backend=model_id, model=root)(
        PROMPT, seed=1234, height=height, width=width
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        size=[width, height],
    )
    assert passed
