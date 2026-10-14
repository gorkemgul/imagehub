"""Published DeepFloyd IF snapshot checks (opt in with IMAGEHUB_REFERENCE_TESTS=1)."""

import json

import pytest
import torch
from helpers import FIXTURES, snapshot_dir

from imagehub.models.t5 import T5EncoderModel
from imagehub.models.unet_2d import UNet2DConditionModel
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
def test_published_unet_header(stage: str, record) -> None:
    root = snapshot_dir("deepfloyd-if-i-xl")
    model_root = root if stage == "stage_i" else root / "stage_ii"
    path = FIXTURES / "deepfloyd_if_configs" / stage
    with torch.device("meta"):
        model = UNet2DConditionModel(json.loads((path / "unet_config.json").read_text()))
    expected = json.loads((path / "unet.safetensors.header.json").read_text())
    actual = {name: list(value.shape) for name, value in model.state_dict().items()}
    assert (model_root / "unet").is_dir()
    record(
        f"header_deepfloyd_if_{stage}",
        missing=len(expected.keys() - actual.keys()),
        extra=len(actual.keys() - expected.keys()),
        mismatch=sum(expected[k] != actual[k] for k in expected.keys() & actual.keys()),
    )
    assert actual == expected


def test_published_tokenizer_and_shared_t5_output() -> None:
    """Stage II deliberately reuses stage I's encoder, as the published configs do."""
    root = snapshot_dir("deepfloyd-if-i-xl")
    tokenizer = T5Tokenizer(root / "tokenizer")
    ids, mask = tokenizer(["a quiet landscape"], max_length=77)
    encoder = T5EncoderModel.from_pretrained(root / "text_encoder", device=torch.device("cpu"))
    with torch.inference_mode():
        stage_i = encoder(ids, mask)[0]
        stage_ii = encoder(ids, mask)[0]
    assert torch.equal(stage_i, stage_ii)


@pytest.mark.parametrize("stage", ("stage_i", "stage_ii"))
def test_published_unet_one_step(stage: str) -> None:
    root = snapshot_dir("deepfloyd-if-i-xl")
    model_root = root if stage == "stage_i" else root / "stage_ii"
    model = UNet2DConditionModel.from_pretrained(model_root / "unet", device=torch.device("cpu"))
    config = json.loads((model_root / "unet" / "config.json").read_text())
    encoder = T5EncoderModel.from_pretrained(root / "text_encoder", device=torch.device("cpu"))
    tokenizer = T5Tokenizer(root / "tokenizer")
    ids, mask = tokenizer(["a quiet landscape"], max_length=77)
    context = encoder(ids, mask)[0]
    size = 64 if stage == "stage_i" else 256
    sample = torch.randn(
        1, config["in_channels"], size, size, generator=torch.Generator().manual_seed(0)
    )
    labels = torch.tensor([250]) if stage == "stage_ii" else None
    output = model(sample, torch.tensor(999), context, class_labels=labels)
    assert output.shape == (1, config["out_channels"], size, size)
