"""Random-weight parity for the Stable Cascade stage A decoder."""

import json
from pathlib import Path

import pytest
import torch

from imagehub.models.paella_vq import PaellaVQModel

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
paella_module = pytest.importorskip(
    "diffusers.pipelines.deprecated.wuerstchen.modeling_paella_vq_model"
)

PUBLISHED = json.loads(
    Path("tests/fixtures/stable_cascade_configs/vqgan_config.json").read_text(encoding="utf-8")
)
TINY = {
    "levels": 2,
    "bottleneck_blocks": 2,
    "embed_dim": 16,
    "latent_channels": 4,
    "num_vq_embeddings": 8,
}


def build(config):
    torch.manual_seed(0)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    reference = paella_module.PaellaVQModel(**config)
    with torch.no_grad():
        for name, tensor in reference.state_dict().items():
            if name.endswith("gammas"):
                tensor.copy_(torch.randn_like(tensor) * 0.5)
            elif name.endswith("running_mean"):
                tensor.copy_(torch.randn_like(tensor))
            elif name.endswith("running_var"):
                tensor.copy_(torch.rand_like(tensor) + 0.5)
    reference.eval()
    ours = PaellaVQModel(config).eval()
    ours.load_checkpoint(reference.state_dict())
    return reference, ours


@pytest.mark.parametrize(
    ("config", "shape"),
    [(TINY, (2, 4, 6, 7)), (PUBLISHED, (2, 4, 32, 32)), (PUBLISHED, (1, 4, 17, 23))],
    ids=["tiny", "published_32", "published_odd"],
)
def test_decode_random_weights(record, request, config, shape):
    reference, ours = build(config)
    latents = torch.randn(*shape)
    with torch.inference_mode():
        expected = reference.decode(latents, force_not_quantize=True).sample
        actual = ours.decode(latents)
    assert torch.equal(actual, expected)
    name = request.node.callspec.id
    record(f"paella_vq_decode_{name}", shape=list(shape), max_abs_diff=0.0, tolerance="exact")


def test_encode_random_weights(record):
    reference, ours = build(PUBLISHED)
    images = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = reference.encode(images, return_dict=False)[0]
        actual = ours.encode(images)
    assert torch.equal(actual, expected)
    record("paella_vq_encode", shape=[1, 3, 64, 64], max_abs_diff=0.0, tolerance="exact")
