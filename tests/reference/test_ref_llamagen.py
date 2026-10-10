"""Parity of the LlamaGen GPT and VQ-16 decoder against the official code on CPU float32."""

import gc
import json
import struct
from types import SimpleNamespace

import pytest
import torch
from helpers import max_diff, snapshot_dir
from llamagen_driver import left_pad, reference_captions, reference_sample

from imagehub import pipeline
from imagehub.backends.llamagen import LlamaGenComponents, encode_prompts, generate, to_uint8
from imagehub.checkpoints import load_weights, read_json
from imagehub.models.llamagen import LlamaGenGPT, gpt_xl_config
from imagehub.models.llamagen_vq import LlamaGenVQ
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference


def _randomize_gpt(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.output.weight.normal_(0.0, 0.02, generator=generator)
        for module in model.modules():
            if type(module).__name__ == "RMSNorm":
                module.weight.uniform_(0.5, 1.5, generator=generator)
    return model.eval()


def _ours_from(reference, config):
    with torch.device("meta"):
        model = LlamaGenGPT(config)
    model.load_checkpoint(reference.state_dict())
    model.init_non_persistent_buffers()
    return model.eval()


def _compare_gpt(gpt, reference_config, config, steps, batch=2):
    torch.manual_seed(0)
    reference = _randomize_gpt(
        gpt.Transformer(gpt.ModelArgs(**reference_config, model_type="t2i")), seed=1
    )
    ours = _ours_from(reference, config)
    cls_tokens = config["cls_token_num"]
    caption_dim = config["caption_dim"]
    vocab = config["vocab_size"]
    generator = torch.Generator().manual_seed(2)
    caption = torch.randn(batch, cls_tokens, caption_dim, generator=generator)
    emb_masks = torch.ones(batch, cls_tokens, dtype=torch.int64)
    emb_masks[0, : cls_tokens // 3] = 0  # left padding as in the sampling script
    caption = caption * emb_masks[:, :, None]
    uncond = torch.zeros_like(caption) + reference.cls_embedding.uncond_embedding
    assert torch.equal(uncond, torch.zeros_like(caption) + ours.cls_embedding.uncond_embedding)
    combined = torch.cat([caption, uncond])
    tokens = torch.randint(0, vocab, (2 * batch, steps), generator=generator)

    total = cls_tokens + steps + 1
    with torch.device("cpu"):
        reference.setup_caches(2 * batch, total, torch.float32)
    ours.setup_caches(2 * batch, total, torch.float32)
    reference.causal_mask[:, :, :cls_tokens] = reference.causal_mask[:, :, :cls_tokens] * torch.cat(
        [emb_masks, emb_masks]
    ).unsqueeze(1)
    eye = torch.eye(reference.causal_mask.size(1), reference.causal_mask.size(2))
    reference.causal_mask[:] = reference.causal_mask * (1 - eye) + eye
    ours.apply_caption_mask(torch.cat([emb_masks, emb_masks]))
    assert torch.equal(reference.causal_mask, ours.causal_mask)

    with torch.no_grad():
        positions = torch.arange(0, cls_tokens)
        expected, _ = reference(None, combined, positions)
        actual = ours(None, combined, positions)
        assert torch.equal(actual, expected)
        input_pos = torch.tensor([cls_tokens], dtype=torch.int32)
        for step in range(steps):
            expected, _ = reference(tokens[:, step : step + 1], None, input_pos)
            actual = ours(tokens[:, step : step + 1], None, input_pos.clone())
            assert torch.equal(actual, expected), f"decode step {step}"
            input_pos += 1


def test_tiny_gpt_prefill_and_cached_decode_are_exact(llamagen_reference):
    gpt, _, _ = llamagen_reference
    # The reference embedder always allocates 120 unconditional caption rows.
    config = {
        "dim": 64,
        "n_layer": 3,
        "n_head": 4,
        "multiple_of": 16,
        "caption_dim": 24,
        "vocab_size": 128,
        "cls_token_num": 120,
        "block_size": 16,
    }
    reference_config = config
    _compare_gpt(gpt, reference_config, config, steps=6)


def test_published_xl_gpt_prefill_and_cached_decode_are_exact(llamagen_reference, record):
    gpt, _, _ = llamagen_reference
    config = gpt_xl_config(16)
    reference_config = {
        key: value
        for key, value in config.items()
        if key not in ("n_kv_head", "ffn_dim_multiplier")
    }
    _compare_gpt(gpt, reference_config, config, steps=3, batch=1)
    record("llamagen_gpt_xl_random_weights", prefill_equal=True, decode_steps=3)


def test_freqs_cis_matches_reference_at_both_sizes(llamagen_reference):
    gpt, _, _ = llamagen_reference
    for block_size in (256, 1024):
        expected = gpt.precompute_freqs_cis_2d(int(block_size**0.5), 64, 10000, 120)
        with torch.device("meta"):
            model = LlamaGenGPT(gpt_xl_config(block_size))
        model.init_non_persistent_buffers()
        assert torch.equal(model.freqs_cis, expected)


def _randomize_vq(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name and name.endswith("weight"):
                parameter.uniform_(0.5, 1.5, generator=generator)
            elif "norm" in name:
                parameter.normal_(0.0, 0.1, generator=generator)
        model.quantize.embedding.weight.normal_(generator=generator)
    return model.eval()


def _compare_vq(vq, reference_kwargs, config, grid):
    torch.manual_seed(0)
    reference = _randomize_vq(vq.VQ_16(**reference_kwargs), seed=1)
    with torch.device("meta"):
        ours = LlamaGenVQ(config)
    ours.load_checkpoint(reference.state_dict())
    ours.eval()
    embed_dim = ours.config["codebook_embed_dim"]
    generator = torch.Generator().manual_seed(3)
    ids = torch.randint(0, ours.config["codebook_size"], (2, grid * grid), generator=generator)
    shape = (2, embed_dim, grid, grid)
    with torch.no_grad():
        expected = reference.decode_code(ids, shape)
        actual = ours.decode_code(ids, shape)
    assert actual.shape == (2, 3, 16 * grid, 16 * grid)
    assert torch.equal(actual, expected)


def test_tiny_vq_decode_code_is_exact(llamagen_reference):
    _, _, vq = llamagen_reference
    kwargs = {"codebook_size": 64, "codebook_embed_dim": 4, "z_channels": 32}
    _compare_vq(vq, kwargs, {"codebook_size": 64, "codebook_embed_dim": 4, "z_channels": 32}, 4)


def test_published_vq_decode_code_is_exact(llamagen_reference, record):
    _, _, vq = llamagen_reference
    kwargs = {"codebook_size": 16384, "codebook_embed_dim": 8}
    _compare_vq(vq, kwargs, {}, 16)
    record("llamagen_vq16_random_weights", decode_code_equal=True, ids="16x16")


# Real checkpoints. Each model id needs a local snapshot (see ``python -m imagehub.pretrained``)
# named by IMAGEHUB_SNAPSHOT_<MODEL_ID>; the tests skip without it.

MODEL_IDS = ["llamagen-t2i-xl-256", "llamagen-t2i-xl-512"]
SIZES = {"llamagen-t2i-xl-256": 256, "llamagen-t2i-xl-512": 512}
GPT_FILES = {
    "llamagen-t2i-xl-256": "t2i_XL_stage1_256.pt",
    "llamagen-t2i-xl-512": "t2i_XL_stage2_512.pt",
}
PROMPT = "A blue Porsche 356 parked in front of a yellow brick wall."
OTHER_PROMPT = "A photo of an astronaut riding a horse in the forest."
CPU = torch.device("cpu")
T5_SKIPPED = ("decoder.", "lm_head.", "encoder.embed_tokens.")


def _shapes(path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"]
    return {name: tuple(tensor.shape) for name, tensor in state.items()}


def _safetensors_shapes(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shapes = {}
    for shard in sorted(set(index["weight_map"].values())):
        with (directory / shard).open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _load_reference_gpt(gpt, path, latent_size):
    model = gpt.GPT_models["GPT-XL"](block_size=latent_size**2, cls_token_num=120, model_type="t2i")
    incompatible = model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)["model"], strict=False
    )
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) <= {"freqs_cis"}
    return model.eval()


def _load_reference_vq(vq, path):
    model = vq.VQ_16(codebook_size=16384, codebook_embed_dim=8)
    incompatible = model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)["model"], strict=False
    )
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    return model.eval()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_headers(record, model_id):
    root = snapshot_dir(model_id)
    result = {}
    gpt_shapes = _shapes(root / GPT_FILES[model_id])
    latent = SIZES[model_id] // 16
    with torch.device("meta"):
        gpt_model = LlamaGenGPT(gpt_xl_config(latent**2))
        vq_model = LlamaGenVQ({})
        text_model = T5EncoderModel(read_json(root / "text_encoder" / "config.json"))
    t5_shapes = _safetensors_shapes(root / "text_encoder")
    checks = {
        "gpt": (gpt_model, gpt_shapes),
        "vq": (vq_model, _shapes(root / "vq_ds16_t2i.pt")),
        "text_encoder": (
            text_model,
            {name: shape for name, shape in t5_shapes.items() if not name.startswith(T5_SKIPPED)},
        ),
    }
    for component, (model, shapes) in checks.items():
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
    # Left out: the decoder, the output head and the tied copy of the embedding, if stored.
    result["text_encoder"]["skipped"] = sum(name.startswith(T5_SKIPPED) for name in t5_shapes)
    record(f"header_{model_id}", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.fixture(scope="module")
def captions():
    """Reference caption embeddings and the driver's left-padded inputs for two prompts."""
    # Both models share the text encoder, so the first available snapshot supplies it.
    for model_id in MODEL_IDS:
        try:
            root = snapshot_dir(model_id)
        except pytest.skip.Exception:
            continue
        ids, embeddings, masks = reference_captions(root / "text_encoder", [PROMPT, OTHER_PROMPT])
        c_indices, c_masks = left_pad(embeddings, masks)
        return root / "text_encoder", ids, embeddings, masks, c_indices, c_masks
    pytest.skip("No LlamaGen snapshot is configured")


def test_tokenizer_and_t5_embeddings(record, captions):
    text_dir, ids, _, masks, c_indices, c_masks = captions
    transformers = pytest.importorskip("transformers")
    tokenizer = T5Tokenizer(text_dir)
    reference = transformers.AutoTokenizer.from_pretrained(text_dir)
    cases = [
        PROMPT,
        OTHER_PROMPT,
        "",
        "A Cat, sitting on THE mat!  Quoted: 'it' costs 5 euros.",
        " ".join([PROMPT] * 12),
    ]
    matches = []
    for case in cases:
        text = case.lower().strip()
        expected = reference(
            [text],
            max_length=120,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        actual_ids, actual_mask = tokenizer([text], 120)
        matches.append(
            torch.equal(actual_ids, expected["input_ids"])
            and torch.equal(actual_mask.long(), expected["attention_mask"])
        )
    record("llamagen_tokenizer", cases=matches, tolerance="exact")
    assert all(matches)

    components = SimpleNamespace(
        tokenizer=tokenizer,
        text_encoder=T5EncoderModel.from_pretrained(text_dir, device=CPU, dtype=torch.float32),
    )
    actual_indices, actual_masks = encode_prompts(components, [PROMPT, OTHER_PROMPT], CPU)
    measured = max_diff(actual_indices, c_indices)
    record("llamagen_t5_caption_embeddings", **measured, tolerance=0.0)
    assert torch.equal(actual_masks, c_masks)
    assert torch.equal(actual_indices, c_indices)
    assert torch.equal(
        ids, tokenizer([PROMPT.lower().strip(), OTHER_PROMPT.lower().strip()], 120)[0]
    )
    del components
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_gpt_prefill_logits(record, llamagen_reference, captions, model_id):
    gpt, _, _ = llamagen_reference
    root = snapshot_dir(model_id)
    latent = SIZES[model_id] // 16
    *_, c_indices, c_masks = captions
    reference = _load_reference_gpt(gpt, root / GPT_FILES[model_id], latent)
    components = LlamaGenComponents.load(root, device=CPU, dtype=torch.float32, model_id=model_id)
    ours = components.gpt
    cond = c_indices[:1]
    masks = c_masks[:1]
    uncond = torch.zeros_like(cond) + reference.cls_embedding.uncond_embedding
    assert torch.equal(uncond, torch.zeros_like(cond) + ours.cls_embedding.uncond_embedding)
    combined = torch.cat([cond, uncond])
    total = 120 + latent**2
    with torch.no_grad():
        reference.setup_caches(2, total, torch.float32)
        ours.setup_caches(2, total, torch.float32)
        reference.causal_mask[:, :, :120] = reference.causal_mask[:, :, :120] * torch.cat(
            [masks, masks]
        ).unsqueeze(1)
        eye = torch.eye(reference.causal_mask.size(1), reference.causal_mask.size(2))
        reference.causal_mask[:] = reference.causal_mask * (1 - eye) + eye
        ours.apply_caption_mask(torch.cat([masks, masks]))
        assert torch.equal(reference.causal_mask, ours.causal_mask)
        positions = torch.arange(0, 120)
        expected, _ = reference(None, combined, positions)
        actual = ours(None, combined, positions)
    measured = max_diff(actual, expected)
    record(f"{model_id}_gpt_prefill", **measured, tolerance=0.0)
    assert torch.equal(actual, expected)
    del reference, components, ours
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_vq_decode_of_fixed_ids(record, llamagen_reference, model_id):
    _, _, vq = llamagen_reference
    root = snapshot_dir(model_id)
    latent = SIZES[model_id] // 16
    reference = _load_reference_vq(vq, root / "vq_ds16_t2i.pt")
    ours = LlamaGenVQ({})
    state = {name: tensor.clone() for name, tensor in load_weights(root / "vq_ds16_t2i.pt").items()}
    ours.load_checkpoint(state)
    ours.eval()
    ids = torch.randint(0, 16384, (1, latent**2), generator=torch.Generator().manual_seed(5))
    shape = (1, 8, latent, latent)
    with torch.no_grad():
        expected = reference.decode_code(ids, shape)
        actual = ours.decode_code(ids, shape)
    measured = max_diff(actual, expected)
    record(f"{model_id}_vq_decode", **measured, tolerance=0.0)
    assert torch.equal(actual, expected)
    del reference, ours
    gc.collect()


def _reference_run(llamagen_reference, root, model_id, prompt, seed, tmp_path, **options):
    gpt, generate, vq = llamagen_reference
    latent = SIZES[model_id] // 16
    _, embeddings, masks = reference_captions(root / "text_encoder", [prompt])
    c_indices, c_masks = left_pad(embeddings, masks)
    gc.collect()
    reference_gpt = _load_reference_gpt(gpt, root / GPT_FILES[model_id], latent)
    reference_vq = _load_reference_vq(vq, root / "vq_ds16_t2i.pt")
    result = reference_sample(
        generate,
        reference_gpt,
        reference_vq,
        c_indices,
        c_masks,
        seed=seed,
        latent_size=latent,
        embed_dim=8,
        output=tmp_path / "sample.png",
        **options,
    )
    del reference_gpt, reference_vq
    gc.collect()
    return result


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_end_to_end(record, llamagen_reference, model_id, tmp_path):
    root = snapshot_dir(model_id)
    expected_tokens, expected_pixels = _reference_run(
        llamagen_reference, root, model_id, PROMPT, 0, tmp_path
    )
    components = LlamaGenComponents.load(root, device=CPU, dtype=torch.float32, model_id=model_id)
    latent = SIZES[model_id] // 16
    embeddings, masks = encode_prompts(components, [PROMPT], CPU)
    tokens = generate(
        components.gpt,
        embeddings,
        latent**2,
        masks,
        cfg_scale=7.5,
        temperature=1.0,
        top_k=1000,
        top_p=1.0,
        generator=torch.Generator("cpu").manual_seed(0),
    )
    with torch.inference_mode():
        decoded = components.vq.decode_code(tokens, (1, 8, latent, latent))
    pixels = to_uint8(decoded)[0]
    tokens_equal = torch.equal(tokens, expected_tokens)
    pixel_diff = int((pixels.int() - expected_pixels.int()).abs().max())
    passed = tokens_equal and pixel_diff == 0
    record(
        f"e2e_{model_id}",
        tokens_equal=tokens_equal,
        pixels_max_abs=pixel_diff,
        passed=passed,
        prompt=PROMPT,
        cfg_scale=7.5,
        top_k=1000,
        size=[SIZES[model_id]] * 2,
        seed=0,
    )
    assert passed
    del components
    gc.collect()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_seed_end_to_end(record, llamagen_reference, model_id, tmp_path):
    root = snapshot_dir(model_id)
    options = {"cfg_scale": 4.0, "temperature": 0.9, "top_k": 500, "top_p": 0.95}
    expected_tokens, expected_pixels = _reference_run(
        llamagen_reference, root, model_id, OTHER_PROMPT, 1234, tmp_path, **options
    )
    result = pipeline("t2i", backend=model_id, model=root)(OTHER_PROMPT, seed=1234, **options)
    pixels = result.images[0].tensor
    passed = torch.equal(pixels, expected_pixels)
    pixel_diff = int((pixels.int() - expected_pixels.int()).abs().max())
    record(
        f"seed_e2e_{model_id}",
        passed=passed,
        max_pixel_diff=pixel_diff,
        seed=1234,
        size=[SIZES[model_id]] * 2,
        **options,
    )
    assert passed
