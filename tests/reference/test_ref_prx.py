"""PRX parity against local published snapshots on CPU float32."""

import gc
import json
import random
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import snapshot_dir

from imagehub.backends.prx import PRXComponents, decode_latents, denoise, encode_prompt
from imagehub.checkpoints import load_weights, read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.prx_transformer import PRXTransformer2DModel
from imagehub.models.t5gemma import T5GemmaEncoder
from imagehub.tokenization_gemma import GemmaTokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
modeling = pytest.importorskip("transformers.models.t5gemma.modeling_t5gemma")
np = pytest.importorskip("numpy")

MODELS = ("prx-1024-t2i-beta", "prx-512-t2i-sft")
PROMPT = "a photograph of an astronaut riding a horse"
FUZZ_PIECES = (
    "a",
    "Cat",
    "İstanbul",
    "日本",
    "한국어",
    "🐱",
    "1234567",
    "é",
    "é",
    " ",
    "\t",
    "\n",
    "!?.",
    "▁",
    "<eos>",
    "<bos>",
    "<pad>",
    "<start_of_turn>",
    "<unused3>",
    "<mask>",
)


def _reference_pipeline(root: Path):
    import diffusers.pipelines.prx.pipeline_prx as pipeline_module

    if not hasattr(pipeline_module, "ftfy"):
        pipeline_module.ftfy = SimpleNamespace(fix_text=lambda value: value)
    text_config = read_json(root / "text_encoder/config.json")
    text_encoder = modeling.T5GemmaEncoder(transformers.T5GemmaModuleConfig(**text_config)).eval()
    text_encoder.load_state_dict(load_weights(root / "text_encoder"))
    transformer = diffusers.PRXTransformer2DModel.from_config(
        read_json(root / "transformer/config.json")
    ).eval()
    transformer.load_state_dict(load_weights(root / "transformer"))
    vae = diffusers.AutoencoderKL.from_config(read_json(root / "vae/config.json")).eval()
    vae.load_state_dict(load_weights(root / "vae"))
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(
        read_json(root / "scheduler/scheduler_config.json")
    )
    reference = diffusers.PRXPipeline(
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            root / "tokenizer", local_files_only=True
        ),
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    reference.register_to_config(
        default_sample_size=read_json(root / "model_index.json")["default_sample_size"]
    )
    reference.set_progress_bar_config(disable=True)
    return reference


def _shapes(directory: Path) -> dict[str, tuple[int, ...]]:
    index = next(directory.glob("*.safetensors.index.json"), None)
    files = sorted(directory.glob("*.safetensors"))
    if index:
        weight_map = json.loads(index.read_text())["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    result = {}
    for path in files:
        with path.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(length))
        result |= {
            name: tuple(value["shape"]) for name, value in header.items() if name != "__metadata__"
        }
    return result


@pytest.mark.parametrize("model_id", MODELS)
def test_headers(model_id, record) -> None:
    root = snapshot_dir(model_id)
    outcome = {}
    for component, cls in {
        "text_encoder": T5GemmaEncoder,
        "transformer": PRXTransformer2DModel,
        "vae": AutoencoderKL,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = _shapes(root / component)
        outcome[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(actual),
        }
    record(f"header_{model_id}", **outcome)
    assert all(
        not value[key] for value in outcome.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODELS)
def test_tokenizer(model_id, record) -> None:
    root = snapshot_dir(model_id)
    ours = GemmaTokenizer(root / "tokenizer")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    rng = random.Random(0)
    cases = [PROMPT, "", "Café — 東京", "  spaced   text  ", "<bos> literal"] + [
        "".join(rng.choice(FUZZ_PIECES) for _ in range(rng.randint(0, 18))) for _ in range(600)
    ]
    matches = []
    for text in cases:
        ids, mask = ours([text], max_length=256, padding_side="right")
        expected = reference(
            [text], padding="max_length", max_length=256, truncation=True, return_tensors="pt"
        )
        matches.append(
            torch.equal(ids, expected.input_ids)
            and torch.equal(mask, expected.attention_mask.bool())
        )
    record(f"prx_{model_id}_tokenizer", cases=matches, tolerance="exact")
    assert all(matches)


@pytest.mark.parametrize("model_id", MODELS)
def test_components_and_end_to_end(model_id, record) -> None:
    root = snapshot_dir(model_id)
    reference = _reference_pipeline(root)
    expected_prompt, expected_mask, expected_negative, expected_negative_mask = (
        reference.encode_prompt(
            PROMPT, device=torch.device("cpu"), negative_prompt="", tokenizer_max_length=256
        )
    )
    expected_embeddings = torch.cat((expected_negative, expected_prompt))
    expected_masks = torch.cat((expected_negative_mask, expected_mask))

    generator = torch.Generator().manual_seed(3)
    hidden_states = torch.randn(2, 16, 16, 24, generator=generator)
    timestep = torch.tensor([0.8, 0.3])
    with torch.inference_mode():
        expected_noise = reference.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=expected_embeddings,
            attention_mask=expected_masks,
            return_dict=False,
        )[0]

    vae_latents = torch.randn(1, 16, 8, 12, generator=generator)
    vae_input = vae_latents / float(reference.vae.config.scaling_factor) + float(
        reference.vae.config.shift_factor or 0.0
    )
    with torch.inference_mode():
        expected_decode = reference.vae.decode(vae_input, return_dict=False)[0]
    size = 1024 if "1024" in model_id else 512
    noise = torch.randn(1, 16, size // 8, size // 8, generator=torch.Generator().manual_seed(0))
    expected = reference(
        PROMPT,
        negative_prompt="",
        height=size,
        width=size,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
        output_type="np",
        use_resolution_binning=False,
    ).images
    del reference, expected_prompt, expected_negative
    gc.collect()

    components = PRXComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    ours_embeddings, ours_mask = encode_prompt(
        components, PROMPT, "", 256, torch.device("cpu"), True, False
    )
    embeddings_diff = float((ours_embeddings - expected_embeddings).abs().max())
    masks_equal = torch.equal(ours_mask, expected_masks)
    record(
        f"prx_{model_id}_text_encoder",
        max_abs=embeddings_diff,
        masks_equal=masks_equal,
        tolerance="exact",
    )
    assert torch.equal(ours_embeddings, expected_embeddings)
    assert masks_equal

    with torch.inference_mode():
        actual_noise = components.transformer(hidden_states, timestep, ours_embeddings, ours_mask)
        actual_decode = components.vae.decode(vae_input)
    noise_diff = float((actual_noise - expected_noise).abs().max())
    decode_diff = float((actual_decode - expected_decode).abs().max())
    record(f"prx_{model_id}_transformer", max_abs=noise_diff, tolerance="exact")
    assert torch.equal(actual_noise, expected_noise)
    record(f"prx_{model_id}_vae_decode", max_abs=decode_diff, tolerance="exact")
    assert torch.equal(actual_decode, expected_decode)

    actual_latents = denoise(
        components,
        prompt=PROMPT,
        negative_prompt="",
        steps=2,
        guidance=4.0,
        max_sequence_length=256,
        skip_text_cleaning=False,
        latents=noise,
    )
    actual = decode_latents(components.vae, actual_latents)
    expected_pixels = torch.from_numpy(expected).mul(255).round().to(torch.uint8)
    pixels_equal = torch.equal(actual, expected_pixels)
    record(f"e2e_{model_id}", pixels_equal=pixels_equal, tolerance="exact")
    record(f"seed_e2e_{model_id}", seed=0, pixels_equal=pixels_equal, tolerance="exact")
    assert pixels_equal
