import json

import pytest
import torch
from fixture_files import fixture_path
from helpers import write_tiny_hidream

from imagehub import pipeline
from imagehub.backends.hidream import HiDreamBackend, effective_size
from imagehub.catalog import model_info
from imagehub.exceptions import CheckpointError
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel
from imagehub.models.t5 import T5EncoderModel

FIXTURES = fixture_path("hidream_configs")


@pytest.mark.parametrize("variant", ["dev", "full"])
def test_published_variant_headers_match_model_layouts(variant):
    directory = FIXTURES / variant
    for component, cls in {
        "text_encoder": CLIPTextModelWithProjection,
        "text_encoder_2": CLIPTextModelWithProjection,
        "text_encoder_3": T5EncoderModel,
        "transformer": HiDreamImageTransformer2DModel,
        "vae": AutoencoderKL,
    }.items():
        config = json.loads((directory / f"{component}_config.json").read_text(encoding="utf-8"))
        # Headers are stored with component-prefixed file names; their tensor keys are disjoint.
        stem = "model" if component.startswith("text_encoder") else "diffusion_pytorch_model"
        files = list(directory.glob(f"{component}_{stem}*.safetensors.json"))
        shapes = {}
        for path in files:
            header = json.loads(path.read_text(encoding="utf-8"))
            shapes |= {
                name: tuple(shape) for name, shape in header.items() if name != "__metadata__"
            }
        with torch.device("meta"):
            model = cls(config)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        assert {name: tuple(value.shape) for name, value in prepared.items()} == {
            name: tuple(value.shape) for name, value in model.state_dict().items()
        }


@pytest.fixture(scope="module")
def tiny_hidream(tmp_path_factory):
    return write_tiny_hidream(tmp_path_factory.mktemp("tiny-hidream"))


def test_generates_images_with_metadata(tiny_hidream):
    state = torch.get_rng_state()
    result = pipeline("t2i", backend="hidream", model=tiny_hidream)(
        "a cat", seed=3, steps=2, guidance=0.0, width=1024, height=1024, max_sequence_length=8
    )
    # The tiny autoencoder has two stages, so the 1024 x 1024 request becomes 256 x 256.
    assert (result.images[0].width, result.images[0].height) == (256, 256)
    assert result.metadata["seed"] == 3 and result.metadata["steps"] == 2
    assert result.metadata["scheduler"] == "flow_match_lcm"
    assert (result.metadata["width"], result.metadata["height"]) == (256, 256)
    assert torch.equal(torch.get_rng_state(), state)


def test_seed_makes_noise_and_steps_reproducible(tiny_hidream):
    pipe = pipeline("t2i", backend="hidream-i1-fast", model=tiny_hidream)
    options = dict(steps=2, max_sequence_length=8)
    a = pipe("a cat", seed=11, **options)
    b = pipe("a cat", seed=11, **options)
    c = pipe("a cat", seed=12, **options)
    assert torch.equal(a.images[0].tensor, b.images[0].tensor)
    assert not torch.equal(a.images[0].tensor, c.images[0].tensor)
    # Explicit latents plus a seed reproduce the seeded run: the seed then only drives step noise.
    latents = torch.randn(1, 4, 128, 128, generator=torch.Generator("cpu").manual_seed(11))
    d = pipe("a cat", latents=latents, seed=11, **options)
    assert torch.equal(a.images[0].tensor, d.images[0].tensor)


def test_guidance_above_one_uses_the_negative_prompt(tiny_hidream):
    pipe = pipeline("t2i", backend="hidream", model=tiny_hidream)
    options = dict(seed=1, steps=1, guidance=3.0, max_sequence_length=8)
    plain = pipe("a cat", **options)
    negative = pipe("a cat", negative_prompt="hi", **options)
    assert not torch.equal(plain.images[0].tensor, negative.images[0].tensor)
    assert negative.metadata["negative_prompt"] == "hi"


def test_effective_size_follows_the_pipeline_rescaling():
    assert effective_size(1024, 1024, 8) == (1024, 1024)
    assert effective_size(1024, 512, 8) == (1440, 720)
    assert effective_size(512, 512, 8) == (1024, 1024)
    assert effective_size(768, 1344, 8) == (768, 1344)
    assert effective_size(192, 128, 2) == (312, 208)


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 0}, ValueError, "width"),
        ({"width": 40000, "height": 2}, ValueError, "budget"),
        ({"steps": 0}, ValueError, "steps"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"seed": -1}, ValueError, "seed"),
        ({"latents": torch.zeros(1, 4, 8, 8)}, ValueError, "shape"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"max_sequence_length": 1024}, ValueError, "max_sequence_length"),
    ],
)
def test_invalid_options_fail_before_loading(tiny_hidream, options, error, match):
    backend = HiDreamBackend(tiny_hidream)
    with pytest.raises(error, match=match):
        pipeline("t2i", backend=backend)("a cat", **options)
    assert not backend.is_loaded


def test_catalog_defaults_and_snapshot_hint(tmp_path):
    spec = model_info("hidream-i1-fast")
    assert spec.config["defaults"] == {"steps": 16, "guidance": 0.0, "width": 1024, "height": 1024}
    pipe = pipeline("t2i", backend="hidream-i1-fast", model=tmp_path / "nowhere")
    with pytest.raises(CheckpointError, match=r"imagehub\.pretrained hidream-i1-fast "):
        pipe("a cat")


def test_snapshot_without_the_llama_folder_is_rejected(tmp_path, tiny_hidream):
    import shutil

    broken = tmp_path / "broken"
    shutil.copytree(tiny_hidream, broken)
    shutil.rmtree(broken / "text_encoder_4")
    with pytest.raises(CheckpointError, match="text_encoder_4"):
        HiDreamBackend(broken).load()
