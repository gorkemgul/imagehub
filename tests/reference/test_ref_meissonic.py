"""Meissonic parity against the local published snapshot on CPU float32."""

import gc
import json
import struct
import sys
from pathlib import Path

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.checkpoints import read_json
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.meissonic_transformer import MeissonicTransformer2DModel
from imagehub.models.vq_model import VQModel
from imagehub.tokenization import CLIPTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
np = pytest.importorskip("numpy")

REFERENCE = "/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/meissonic"
PROMPT = "a photograph of an astronaut riding a horse"
CPU = torch.device("cpu")
HEIGHT, WIDTH, STEPS, GUIDANCE = 1024, 32, 2, 9.0


def _checkpoint_shapes(directory: Path):
    path = next(directory.glob("*.safetensors"))
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(value["shape"]) for name, value in header.items() if name != "__metadata__"}


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _reference_classes():
    if not Path(REFERENCE).is_dir():
        pytest.skip(f"Meissonic reference checkout not found at {REFERENCE}")
    sys.path.insert(0, REFERENCE)
    from src.pipeline import Pipeline
    from src.scheduler import Scheduler
    from src.transformer import Transformer2DModel

    return Pipeline, Scheduler, Transformer2DModel


def _fix_attention(transformer):
    head_dim = transformer.config.attention_head_dim
    for block in [*transformer.transformer_blocks, *transformer.single_transformer_blocks]:
        block.attn.head_dim = head_dim


def _reference_pipeline(root):
    Pipeline, Scheduler, Transformer = _reference_classes()
    transformer = Transformer.from_pretrained(root / "transformer", local_files_only=True)
    _fix_attention(transformer)
    pipe = Pipeline(
        diffusers.VQModel.from_pretrained(root / "vqvae", local_files_only=True),
        transformers.CLIPTokenizer.from_pretrained(root / "tokenizer", local_files_only=True),
        transformers.CLIPTextModelWithProjection.from_pretrained(
            root / "text_encoder", local_files_only=True
        ),
        transformer,
        Scheduler.from_pretrained(root / "scheduler", local_files_only=True),
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def test_headers(record):
    root = snapshot_dir("meissonic")
    result = {}
    for component, cls in {
        "text_encoder": CLIPTextModelWithProjection,
        "transformer": MeissonicTransformer2DModel,
        "vqvae": VQModel,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        actual = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = model.state_dict()
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name
                for name in expected.keys() & actual.keys()
                if expected[name].shape != actual[name].shape
            ),
            "tensors": len(shapes),
        }
    record("header_meissonic", **result)
    assert all(
        not value[key] for value in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizer_and_text_encoder(record):
    root = snapshot_dir("meissonic")
    tokenizer = CLIPTokenizer(root / "tokenizer")
    reference_tokenizer = transformers.CLIPTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    cases = [PROMPT, "", "A Cat!  Quoted: 'it' costs 5 euros.", " ".join([PROMPT] * 20)]
    matches = []
    for text in cases:
        expected = reference_tokenizer(
            text, padding="max_length", truncation=True, max_length=77, return_tensors="pt"
        ).input_ids
        matches.append(torch.equal(torch.tensor([tokenizer(text)]), expected))
    assert all(matches)
    record("meissonic_tokenizer", cases=matches, tolerance="exact")

    ids = torch.tensor([tokenizer(PROMPT)])
    reference = transformers.CLIPTextModelWithProjection.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    ).eval()
    with torch.inference_mode():
        expected = reference(ids, return_dict=True, output_hidden_states=True)
    expected_embeds, expected_hidden = expected.text_embeds, expected.hidden_states[-2]
    del reference, expected
    gc.collect()
    ours = CLIPTextModelWithProjection.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual_embeds, _, actual_hidden = ours(ids, output_hidden_states=True)
    _check(record, "meissonic_text_embeds", actual_embeds, expected_embeds)
    _check(record, "meissonic_text_hidden", actual_hidden[-2], expected_hidden)


def test_transformer_step(record):
    root = snapshot_dir("meissonic")
    tokenizer = CLIPTokenizer(root / "tokenizer")
    encoder = CLIPTextModelWithProjection.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        pooled, _, hidden = encoder(torch.tensor([tokenizer(PROMPT)]), output_hidden_states=True)
    hidden = hidden[-2]
    del encoder
    _, _, Transformer = _reference_classes()
    reference = Transformer.from_pretrained(root / "transformer", local_files_only=True)
    _fix_attention(reference)
    ids = torch.full((1, HEIGHT // 16, WIDTH // 16), 8255, dtype=torch.long)
    img_ids = torch.zeros((HEIGHT // 32) * (WIDTH // 32), 3)
    img_ids[:, 1] = torch.arange(HEIGHT // 32).repeat_interleave(WIDTH // 32)
    txt_ids = torch.zeros(77, 3)
    micro = torch.tensor([[WIDTH, HEIGHT, 0, 0, 6]], dtype=torch.float32)
    timestep = torch.tensor([1])
    with torch.inference_mode():
        expected = reference(
            hidden_states=ids,
            encoder_hidden_states=hidden,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            micro_conds=micro,
        )
    del reference
    gc.collect()
    ours = MeissonicTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(ids, hidden, pooled, timestep, img_ids, txt_ids, micro)
    _check(record, "meissonic_transformer_step", actual, expected)


def test_vq_decode(record):
    root = snapshot_dir("meissonic")
    ids = torch.randint(0, 8192, (1, 2, 2), generator=torch.Generator().manual_seed(1))
    reference = diffusers.VQModel.from_pretrained(root / "vqvae", local_files_only=True)
    with torch.inference_mode():
        expected = reference.decode(ids, force_not_quantize=True, shape=(1, 2, 2, 64)).sample
    del reference
    gc.collect()
    ours = VQModel.from_pretrained(root / "vqvae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = ours.decode(ids, force_not_quantize=True, shape=(1, 2, 2, 64))
    _check(record, "meissonic_vq_decode", actual, expected)


def _reference_output(root, seed):
    reference = _reference_pipeline(root)
    result = reference(
        PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(seed),
        output_type="pil",
    ).images
    pixels = torch.from_numpy(np.stack([np.array(image) for image in result]))
    del reference
    gc.collect()
    return pixels


def _run_e2e(record, seed, key):
    root = snapshot_dir("meissonic")
    expected = _reference_output(root, seed)
    result = pipeline("t2i", backend="meissonic", model=root)(
        PROMPT,
        seed=seed,
        height=HEIGHT,
        width=WIDTH,
        steps=STEPS,
        guidance=GUIDANCE,
    )
    actual = torch.stack([image.tensor for image in result.images])
    difference = int((actual.int() - expected.int()).abs().max())
    passed = torch.equal(actual, expected)
    record(
        key,
        passed=passed,
        max_pixel_diff=difference,
        seed=seed,
        steps=STEPS,
        guidance=GUIDANCE,
        size=[WIDTH, HEIGHT],
    )
    assert passed


def test_end_to_end(record):
    _run_e2e(record, 0, "e2e_meissonic")


def test_seed_end_to_end(record):
    _run_e2e(record, 42, "seed_e2e_meissonic")
