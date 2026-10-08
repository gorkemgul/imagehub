import json
import struct
from pathlib import Path

import torch

from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.prior_transformer import PriorTransformer

FIXTURES = Path(__file__).parent / "fixtures/kandinsky22_prior_configs"
CHECKPOINT = Path(__file__).parents[1] / "checkpoints/kandinsky22-prior"


def _config(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _header_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(length))
    return {name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"}


def test_clip_vision_published_layout():
    config = _config("image_encoder_config.json")
    with torch.device("meta"):
        model = CLIPVisionModelWithProjection(config)
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert shapes["vision_model.embeddings.class_embedding"] == (1664,)
    assert shapes["vision_model.embeddings.patch_embedding.weight"] == (1664, 3, 14, 14)
    assert shapes["vision_model.embeddings.position_embedding.weight"] == (257, 1664)
    assert shapes["visual_projection.weight"] == (1280, 1664)
    assert len(shapes) == 776
    weights = CHECKPOINT / "image_encoder/model.safetensors"
    if weights.is_file():
        published = _header_shapes(weights)
        published.pop("vision_model.embeddings.position_ids")
        assert shapes == published


def test_prior_published_layout():
    with torch.device("meta"):
        model = PriorTransformer(_config("prior_config.json"))
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    assert shapes["positional_embedding"] == (1, 81, 2048)
    assert shapes["clip_mean"] == (1, 1280)
    assert shapes["clip_std"] == (1, 1280)
    assert shapes["transformer_blocks.19.ff.net.2.weight"] == (2048, 8192)
    assert len(shapes) == 338


def test_prior_post_process_latents():
    model = PriorTransformer(
        {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 1,
            "embedding_dim": 8,
            "num_embeddings": 5,
            "clip_embed_dim": 6,
        }
    )
    latents = torch.randn(2, 6)
    expected = latents * model.clip_std + model.clip_mean
    torch.testing.assert_close(model.post_process_latents(latents), expected, rtol=0, atol=0)
