import json
import shutil

import pytest
import torch
from helpers import FIXTURES, TINY_JANUS_CONFIG

from imagehub.backends.janus import JanusComponents, generate_image_tokens
from imagehub.models.janus_vq import postprocess_janus

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

CHAT_TEMPLATE = """{% for message in messages %}<|User|>: {{ message['content'].strip() }}

{% endfor %}{% if add_generation_prompt %}<|Assistant|>:{% endif %}"""


class _NoGuidance:
    def __call__(self, input_ids, scores):
        return scores[: input_ids.shape[0]]


def _write_reference(root):
    root.mkdir(parents=True)
    config = transformers.JanusConfig(**TINY_JANUS_CONFIG)
    torch.manual_seed(0)
    model = transformers.JanusForConditionalGeneration(config).eval()
    tokenizer_root = FIXTURES / "deepseek_tokenizer"
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tokenizer_root / name, root / name)
    tokenizer_data = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer_data["model"]["vocab"][":"] = 26
    (root / "tokenizer.json").write_text(json.dumps(tokenizer_data), encoding="utf-8")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        extra_special_tokens={
            "image_token": "<image_placeholder>",
            "boi_token": "<begin_of_image>",
            "eoi_token": "<end_of_image>",
        },
    )
    processor = transformers.JanusProcessor(
        transformers.JanusImageProcessor(image_mean=[0.5] * 3, image_std=[0.5] * 3),
        tokenizer,
        chat_template=CHAT_TEMPLATE,
        num_image_tokens=4,
    )
    model.save_pretrained(root, safe_serialization=True)
    processor.save_pretrained(root)
    model = transformers.JanusForConditionalGeneration.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    processor = transformers.AutoProcessor.from_pretrained(root, local_files_only=True)
    return model, processor


@pytest.mark.parametrize(
    ("guidance", "do_sample", "temperature"),
    [(5.0, True, 0.7), (1.0, True, 1.0), (5.0, False, 1.0)],
    ids=["cfg-sampling", "no-cfg", "greedy"],
)
def test_tiny_janus_matches_transformers_exactly(tmp_path, guidance, do_sample, temperature):
    root = tmp_path / "tiny"
    model, processor = _write_reference(root)
    text = processor.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], generation_mode="image", return_tensors="pt")
    num_images = 2 if do_sample else 1
    model.generation_config.num_return_sequences = num_images
    model.generation_config.guidance_scale = guidance
    model.generation_config.max_length = 20
    model.generation_config.generation_kwargs = {"boi_token_id": 23}
    model.generation_config.do_sample = do_sample
    model.generation_config.temperature = temperature
    model.generation_config.save_pretrained(root)
    torch.manual_seed(42)
    logits_processor = None
    if guidance == 1:
        logits_processor = transformers.LogitsProcessorList([_NoGuidance()])
    expected_tokens = model.generate(
        **inputs,
        generation_mode="image",
        logits_processor=logits_processor,
        use_cache=True,
    )
    with torch.inference_mode():
        expected_pixels = processor.postprocess(
            model.decode_image_tokens(expected_tokens), return_tensors="pt"
        )["pixel_values"].permute(0, 2, 3, 1)

    components = JanusComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_tokens = generate_image_tokens(
        components,
        prompt="hi",
        guidance=guidance,
        temperature=temperature,
        do_sample=do_sample,
        num_images=num_images,
        generator=torch.Generator("cpu").manual_seed(42),
    )
    with torch.inference_mode():
        actual_pixels = postprocess_janus(components.model.decode_image_tokens(actual_tokens))
    torch.testing.assert_close(actual_tokens, expected_tokens, rtol=0, atol=0)
    torch.testing.assert_close(actual_pixels, expected_pixels, rtol=0, atol=0)
