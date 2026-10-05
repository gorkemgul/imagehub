"""Parity of native Janus-Pro-1B generation against transformers on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.janus import JanusComponents, generate_image_tokens
from imagehub.checkpoints import read_json
from imagehub.models.janus import JanusModel
from imagehub.models.janus_vq import postprocess_janus

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
PROMPT = "a photograph of an astronaut riding a horse"


def _reference(root, num_images=1):
    processor = transformers.AutoProcessor.from_pretrained(root, local_files_only=True)
    model = transformers.JanusForConditionalGeneration.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    return model, processor


def _inputs(processor):
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], generation_mode="image", return_tensors="pt")
    assert inputs.input_ids.tolist() == [
        [
            100000,
            100588,
            25,
            245,
            14537,
            280,
            274,
            62202,
            17040,
            245,
            7569,
            185,
            185,
            100589,
            25,
            100003,
        ]
    ]
    return inputs


def test_published_tensor_headers_match(record):
    root = snapshot_dir("janus-pro-1b")
    config = read_json(root / "config.json")
    with torch.device("meta"):
        model = JanusModel(config)
    with (root / "model.safetensors").open("rb") as source:
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
    result = {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(shapes),
    }
    record("header_janus-pro-1b", **result)
    assert not result["missing"] and not result["extra"] and not result["mismatch"]


def test_language_model_prefill_and_decode(record):
    root = snapshot_dir("janus-pro-1b")
    reference, processor = _reference(root)
    components = JanusComponents.load(root, device=CPU, dtype=torch.float32)
    inputs = _inputs(processor)
    input_ids = inputs.input_ids.repeat(2, 1)
    mask = inputs.attention_mask.repeat(2, 1)
    input_ids[1, 1:-1] = reference.generation_config.pad_token_id
    reference_inputs = reference.get_input_embeddings()(input_ids)
    native_inputs = components.model.language_model.embed_tokens(input_ids)
    cache_length = max(reference.generation_config.max_length or 20, input_ids.shape[1] + 576)
    assert cache_length == 592
    native_cache = __import__("imagehub.models.llama", fromlist=["LlamaCache"]).LlamaCache(
        components.model.language_model.config, cache_length
    )
    pairs = []
    with torch.inference_mode():
        ref_cache = transformers.StaticCache(
            reference.model.language_model.config,
            max_batch_size=2,
            max_cache_len=cache_length,
            device=CPU,
            dtype=torch.float32,
        )
        expected = reference.model.language_model(
            inputs_embeds=reference_inputs,
            attention_mask=mask,
            past_key_values=ref_cache,
            cache_position=torch.arange(input_ids.shape[1]),
        ).last_hidden_state
        actual = components.model.language_model(
            native_inputs, mask, native_cache, torch.arange(input_ids.shape[1])
        )
        pairs.append((actual, expected))
        for index in range(8):
            token = torch.tensor([[index % 16], [index % 16]])
            mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
            position = torch.tensor([input_ids.shape[1] + index])
            expected = reference.model.language_model(
                inputs_embeds=reference.prepare_embeddings_for_image_generation(token),
                attention_mask=mask,
                past_key_values=ref_cache,
                cache_position=position,
            ).last_hidden_state
            actual = components.model.language_model(
                components.model.prepare_embeddings_for_image_generation(token),
                mask,
                native_cache,
                position,
            )
            pairs.append((actual, expected))
    worst = max(
        (max_diff(actual, expected) for actual, expected in pairs),
        key=lambda value: value["max_abs"],
    )
    record("janus_language_model", **worst, steps=8, tolerance=0.0)
    for actual, expected in pairs:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    del reference, components
    gc.collect()


def test_vq_decode(record):
    root = snapshot_dir("janus-pro-1b")
    reference, _ = _reference(root)
    components = JanusComponents.load(root, device=CPU, dtype=torch.float32)
    tokens = (torch.arange(576) % 16384).reshape(1, 576)
    with torch.inference_mode():
        expected = reference.decode_image_tokens(tokens)
        actual = components.model.decode_image_tokens(tokens)
    measured = max_diff(actual, expected)
    record("janus_vq_decode", **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_images", [1, 2])
def test_end_to_end(record, num_images):
    root = snapshot_dir("janus-pro-1b")
    reference, processor = _reference(root, num_images)
    inputs = _inputs(processor)
    torch.manual_seed(42)
    with torch.inference_mode():
        # Setting model.generation_config.num_return_sequences (the model card's way) fails
        # validation in transformers 5.18.0; the generate() argument is equivalent.
        expected_tokens = reference.generate(
            **inputs,
            generation_mode="image",
            do_sample=True,
            use_cache=True,
            num_return_sequences=num_images,
        )
        expected_pixels = processor.postprocess(
            reference.decode_image_tokens(expected_tokens), return_tensors="pt"
        )["pixel_values"].permute(0, 2, 3, 1)
    del reference
    gc.collect()
    components = JanusComponents.load(root, device=CPU, dtype=torch.float32)
    actual_tokens = generate_image_tokens(
        components,
        prompt=PROMPT,
        guidance=5.0,
        temperature=1.0,
        do_sample=True,
        num_images=num_images,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    with torch.inference_mode():
        actual_pixels = postprocess_janus(components.model.decode_image_tokens(actual_tokens))
    passed = torch.equal(actual_tokens, expected_tokens) and torch.equal(
        actual_pixels, expected_pixels
    )
    record(
        "e2e_janus-pro-1b" if num_images == 1 else f"e2e_batch{num_images}_janus-pro-1b",
        num_images=num_images,
        tokens_match=torch.equal(actual_tokens, expected_tokens),
        pixels_match=torch.equal(actual_pixels, expected_pixels),
        passed=passed,
    )
    assert passed


def test_seed_end_to_end(record):
    root = snapshot_dir("janus-pro-1b")
    reference, processor = _reference(root)
    inputs = _inputs(processor)
    torch.manual_seed(42)
    with torch.inference_mode():
        tokens = reference.generate(
            **inputs, generation_mode="image", do_sample=True, use_cache=True
        )
        expected = processor.postprocess(
            reference.decode_image_tokens(tokens), return_tensors="pt"
        )["pixel_values"][0].permute(1, 2, 0)
    del reference
    gc.collect()
    actual = pipeline("t2i", backend="janus-pro-1b", model=root)(PROMPT, seed=42).images[0].tensor
    passed = torch.equal(actual, expected)
    record("seed_e2e_janus-pro-1b", seed=42, pixels_match=passed, passed=passed)
    assert passed
