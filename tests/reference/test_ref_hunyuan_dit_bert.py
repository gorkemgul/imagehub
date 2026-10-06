"""Published HunyuanDiT BERT checkpoint parity on CPU float32."""

import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub.checkpoints import read_json
from imagehub.models.bert import BertModel
from imagehub.tokenization_bert import BertTokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

PROMPTS = [
    "a photograph of an astronaut riding a horse",
    "一只可爱的猫在草地上",
    "",
]


def test_hunyuan_dit_bert_published_tensor_headers_match(record):
    root = snapshot_dir("hunyuan-dit-v1.1-distilled") / "text_encoder"
    with torch.device("meta"):
        model = BertModel(read_json(root / "config.json"))
    with (root / "model.safetensors").open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    actual = {
        name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
    }
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    result = {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual),
    }
    record("header_hunyuan_dit_bert", **result)
    assert not result["missing"]
    assert not result["extra"]
    assert not result["mismatch"]


def test_hunyuan_dit_bert_checkpoint_matches_reference(record):
    root = snapshot_dir("hunyuan-dit-v1.1-distilled")
    tokenizer = BertTokenizer(root / "tokenizer")
    input_ids, attention_mask = tokenizer(PROMPTS, max_length=77)
    ours = BertModel.from_pretrained(root / "text_encoder", dtype=torch.float32)
    reference = transformers.BertModel.from_pretrained(
        root / "text_encoder", local_files_only=True
    ).eval()

    with torch.inference_mode():
        expected = reference(input_ids, attention_mask=attention_mask).last_hidden_state
        actual = ours(input_ids, attention_mask)
    measured = max_diff(actual, expected)
    record("hunyuan_dit_bert", **measured, prompts=len(PROMPTS), tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
