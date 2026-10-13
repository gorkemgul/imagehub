"""LDM real-checkpoint parity checks; all tests skip until its snapshot is supplied."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.ldm import LDMComponents, LDMUNet, encode_prompt
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.ldm_bert import LDMBertModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")

MODEL_ID = "ldm-text2im-large-256"
CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"
COMPONENTS = {"bert": LDMBertModel, "unet": LDMUNet, "vqvae": AutoencoderKL}


def _check(record, name, actual, expected):
    result = max_diff(actual, expected)
    record(name, **result, tolerance=0.0)
    assert torch.equal(actual, expected)


def test_random_ldm_bert(record):
    from diffusers.pipelines.latent_diffusion import pipeline_latent_diffusion as reference_ldm

    config = {
        "vocab_size": 31,
        "max_position_embeddings": 7,
        "encoder_layers": 2,
        "encoder_ffn_dim": 32,
        "encoder_attention_heads": 2,
        "head_dim": 4,
        "d_model": 8,
        "dropout": 0.0,
    }
    reference = reference_ldm.LDMBertModel(reference_ldm.LDMBertConfig(**config)).eval()
    ours = LDMBertModel(config).eval()
    ours.load_state_dict(reference.state_dict(), strict=True)
    ids = torch.tensor([[1, 2, 3, 0, 0]])
    _check(record, "ldm_random_bert", ours(ids), reference(ids)[0])


def test_ldm_header(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component, cls in COMPONENTS.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        path = next((root / component).glob("*.safetensors"))
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes = {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
        actual = {name: tuple(tensor.shape) for name, tensor in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(actual),
        }
    record(f"header_{MODEL_ID}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_ldm_components(record):
    root = snapshot_dir(MODEL_ID)
    ours = LDMComponents.load(root, CPU, torch.float32)
    reference = diffusers.LDMTextToImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    ids, _ = ours.tokenizer([PROMPT], 77)
    expected_ids = reference.tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids
    record("ldm_tokenizer", ids_match=torch.equal(ids, expected_ids), tolerance="exact")
    assert torch.equal(ids, expected_ids)
    _check(
        record,
        "ldm_text_encoder",
        encode_prompt(ours, PROMPT, CPU),
        reference.bert(expected_ids)[0],
    )
    latents = torch.randn(1, 4, 32, 32, generator=torch.Generator().manual_seed(1))
    context = encode_prompt(ours, PROMPT, CPU)
    _check(
        record,
        "ldm_unet",
        ours.unet(latents, torch.tensor(981), context),
        reference.unet(latents, torch.tensor(981), encoder_hidden_states=context).sample,
    )
    _check(
        record,
        "ldm_vqvae_decode",
        ours.vqvae.decode(latents),
        reference.vqvae.decode(latents).sample,
    )
    del ours, reference
    gc.collect()


def test_ldm_end_to_end(record):
    root = snapshot_dir(MODEL_ID)
    generator = torch.Generator("cpu").manual_seed(0)
    reference = diffusers.LDMTextToImagePipeline.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    )
    expected = reference(
        PROMPT,
        num_inference_steps=50,
        guidance_scale=1.0,
        eta=0.0,
        generator=generator,
        output_type="np",
    ).images
    del reference
    gc.collect()
    actual = pipeline("t2i", backend=MODEL_ID, model=root)(PROMPT, seed=0).images[0].tensor
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))[0]
    _check(record, f"e2e_{MODEL_ID}", actual, expected)
    _check(record, f"seed_e2e_{MODEL_ID}", actual, expected)
