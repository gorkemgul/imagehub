"""OmniGen parity against a local published snapshot on CPU float32.

The published transformer is stored in bfloat16; both implementations convert it to float32 on
load. The end-to-end runs use shortened step counts (a full 1024x1024, 50-step run takes hours
on CPU); the sampler itself is covered at every step count by the scheduler tests.
"""

import gc
import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.omnigen import OmniGenComponents, decode_latents, denoise
from imagehub.backends.omnigen_inputs import (
    IMAGE_PLACEHOLDER,
    build_inputs,
    encode_input_images,
    preprocess_input_image,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.omnigen_transformer import OmniGenTransformer2DModel
from imagehub.tokenization_omnigen import OmniGenTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL.Image")

CPU = torch.device("cpu")
MODEL_ID = "omnigen-v1"
PROMPT = "A photograph of an astronaut riding a horse on the moon."
EDIT_PROMPT = f"Turn {IMAGE_PLACEHOLDER} into a watercolor painting."
FIXTURES = Path(__file__).parents[1] / "fixtures"
# Defaults are 50 steps at 1024x1024; the real runs are shortened (see the module docstring).
E2E_STEPS = 4
EDIT_STEPS = 4
EDIT_SIZE = 512
SEED_STEPS = 2


def _checkpoint_shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    files = list(directory.glob("*.safetensors"))
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


def _picture(size=(320, 256)):
    """A deterministic synthetic photograph-like image (gradients plus noise)."""
    generator = torch.Generator().manual_seed(11)
    width, height = size
    x = torch.linspace(0, 255, width).expand(height, width)
    y = torch.linspace(255, 0, height)[:, None].expand(height, width)
    noise = torch.randint(0, 40, (height, width), generator=generator).float()
    data = torch.stack([x, y, (x + y) / 2 + noise], dim=-1).clamp(0, 255).to(torch.uint8)
    return PIL.fromarray(data.numpy())


def test_headers(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in {"transformer": OmniGenTransformer2DModel, "vae": AutoencoderKL}.items():
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
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer(record):
    root = snapshot_dir(MODEL_ID)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        published = hashlib.sha256((root / "tokenizer" / name).read_bytes()).hexdigest()
        fixture = hashlib.sha256((FIXTURES / "omnigen_tokenizer" / name).read_bytes()).hexdigest()
        assert published == fixture, name
    ours = OmniGenTokenizer(root / "tokenizer")
    reference = transformers.LlamaTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    cases = [PROMPT, EDIT_PROMPT, "", " ".join([PROMPT] * 40)]
    matches = [ours.tokenize(case) == reference(case).input_ids for case in cases]
    record("omnigen_tokenizer_real", cases=len(cases), matches=matches, tolerance="exact")
    assert all(matches)


def _transformer_inputs(components_tokenizer, with_image):
    image = _picture((128, 96)) if with_image else None
    pixels = None if image is None else preprocess_input_image(image)
    prompt = EDIT_PROMPT if with_image else PROMPT
    return build_inputs(
        components_tokenizer,
        prompt,
        None if pixels is None else [pixels],
        height=128,
        width=128,
        use_img_cfg=with_image,
    )


@pytest.mark.parametrize("with_image", [False, True])
@pytest.mark.parametrize("timestep", [0.0, 0.31, 0.87])
def test_transformer_step(record, with_image, timestep):
    root = snapshot_dir(MODEL_ID)
    ours = OmniGenTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    reference = diffusers.OmniGenTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    ).eval()
    processed = _transformer_inputs(OmniGenTokenizer(root / "tokenizer"), with_image)
    generator = torch.Generator("cpu").manual_seed(1)
    copies = 3 if with_image else 2
    latents = torch.randn(1, 4, 16, 16, generator=generator).repeat(copies, 1, 1, 1)
    image_latents = [
        torch.randn(1, 4, x.shape[-2] // 8, x.shape[-1] // 8, generator=generator)
        for x in processed.input_pixel_values
    ]
    steps = torch.full((copies,), timestep)
    args = (
        processed.input_ids,
        image_latents,
        processed.input_image_sizes,
        processed.attention_mask,
        processed.position_ids,
    )
    with torch.inference_mode():
        actual = ours(latents, steps, *args)
        expected = reference(latents, steps, *args, return_dict=False)[0]
    _check(record, f"omnigen_transformer_{with_image}_{timestep}", actual, expected)
    del ours, reference
    gc.collect()


def test_vae(record):
    root = snapshot_dir(MODEL_ID)
    ours = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    assert reference.config.scaling_factor == ours.scaling_factor
    pixels = preprocess_input_image(_picture((256, 192))).unsqueeze(0)
    seed = 4
    with torch.inference_mode():
        torch.manual_seed(seed)
        expected = (
            reference.encode(pixels).latent_dist.sample().mul_(reference.config.scaling_factor)
        )
        actual = encode_input_images(ours, [pixels], torch.Generator("cpu").manual_seed(seed))[0]
        _check(record, "omnigen_vae_encode", actual, expected)
        latents = torch.randn(1, 4, 64, 64, generator=torch.Generator("cpu").manual_seed(1))
        scaled = latents / reference.config.scaling_factor
        _check(
            record,
            "omnigen_vae_decode",
            ours.decode(scaled),
            reference.decode(scaled, return_dict=False)[0],
        )
    del ours, reference
    gc.collect()


def _reference_run(root, seed, *, image=None, **kwargs):
    reference = diffusers.OmniGenPipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    reference.set_progress_bar_config(disable=True)
    outputs = []
    for output_type in ("latent", "pil"):
        torch.manual_seed(seed)
        outputs.append(
            reference(
                EDIT_PROMPT if image is not None else PROMPT,
                input_images=None if image is None else [image],
                generator=torch.Generator("cpu").manual_seed(seed),
                output_type=output_type,
                **kwargs,
            ).images
        )
    latents, images = outputs
    pixels = torch.from_numpy(np.stack([np.array(item) for item in images]))
    del reference
    gc.collect()
    return latents, pixels


def test_end_to_end_text_to_image(record):
    root = snapshot_dir(MODEL_ID)
    expected_latents, expected_pixels = _reference_run(
        root, 0, height=1024, width=1024, num_inference_steps=E2E_STEPS
    )
    components = OmniGenComponents.load(root, device=CPU, dtype=torch.float32, model_id=MODEL_ID)
    processed = build_inputs(
        components.tokenizer, PROMPT, None, height=1024, width=1024, use_img_cfg=False
    )
    noise = torch.randn(1, 4, 128, 128, generator=torch.Generator("cpu").manual_seed(0))
    actual_latents = denoise(
        components, processed, [], steps=E2E_STEPS, guidance=2.5, image_guidance=1.6, latents=noise
    )
    actual_pixels = decode_latents(components.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        f"e2e_{MODEL_ID}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        steps=E2E_STEPS,
        guidance=2.5,
        size=[1024, 1024],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


def test_end_to_end_image_input(record):
    root = snapshot_dir(MODEL_ID)
    image = _picture()
    kwargs = {
        "height": EDIT_SIZE,
        "width": EDIT_SIZE,
        "num_inference_steps": EDIT_STEPS,
    }
    expected_latents, expected_pixels = _reference_run(root, 0, image=image, **kwargs)
    components = OmniGenComponents.load(root, device=CPU, dtype=torch.float32, model_id=MODEL_ID)
    pixels = preprocess_input_image(image)
    processed = build_inputs(
        components.tokenizer,
        EDIT_PROMPT,
        [pixels],
        height=EDIT_SIZE,
        width=EDIT_SIZE,
        use_img_cfg=True,
    )
    image_latents = encode_input_images(
        components.vae, processed.input_pixel_values, torch.Generator("cpu").manual_seed(0)
    )
    noise = torch.randn(
        1, 4, EDIT_SIZE // 8, EDIT_SIZE // 8, generator=torch.Generator("cpu").manual_seed(0)
    )
    actual_latents = denoise(
        components,
        processed,
        image_latents,
        steps=EDIT_STEPS,
        guidance=2.5,
        image_guidance=1.6,
        latents=noise,
    )
    actual_pixels = decode_latents(components.vae, actual_latents)
    latent_diff = max_diff(actual_latents, expected_latents)
    pixel_diff = int((actual_pixels.int() - expected_pixels.int()).abs().max())
    passed = latent_diff["max_abs"] == 0 and pixel_diff == 0
    record(
        f"e2e_image_{MODEL_ID}",
        latents=latent_diff,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=EDIT_PROMPT,
        steps=EDIT_STEPS,
        guidance=2.5,
        image_guidance=1.6,
        size=[EDIT_SIZE, EDIT_SIZE],
        input_size=[256, 320],
        seed=0,
    )
    assert passed
    del components
    gc.collect()


def test_seed_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    _, expected = _reference_run(root, 1234, height=768, width=1360, num_inference_steps=SEED_STEPS)
    result = pipeline("t2i", backend=MODEL_ID, model=root)(
        PROMPT, seed=1234, height=768, width=1360, steps=SEED_STEPS
    )
    pixels = torch.stack([image.tensor for image in result.images])
    passed = torch.equal(pixels, expected)
    pixel_diff = int((pixels.int() - expected.int()).abs().max())
    record(
        f"seed_e2e_{MODEL_ID}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        steps=SEED_STEPS,
        size=[1360, 768],
    )
    assert passed
