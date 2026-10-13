from pathlib import Path

import pytest
import torch

from imagehub.backends.pixart import PixArtBackend, classify_height_width_bin
from imagehub.catalog import model_info
from imagehub.types import T2IRequest


def test_catalog_defaults_and_backend_defaults():
    sigma = model_info("pixart-sigma-xl-1024")
    alpha = model_info("pixart-alpha-xl-1024")
    assert sigma.family == alpha.family == "pixart"
    assert sigma.config == {"defaults": {"steps": 20, "guidance": 4.5, "max_sequence_length": 300}}
    assert alpha.config == {"defaults": {"steps": 20, "guidance": 4.5, "max_sequence_length": 120}}
    sigma_options = PixArtBackend(Path("missing"), model_id=sigma.id).validate_options(
        T2IRequest("a cat"), {}
    )
    alpha_options = PixArtBackend(Path("missing"), model_id=alpha.id).validate_options(
        T2IRequest("a cat"), {}
    )
    assert sigma_options["max_sequence_length"] == 300
    assert alpha_options["max_sequence_length"] == 120
    assert sigma_options["steps"] == alpha_options["steps"] == 20
    assert sigma_options["guidance"] == alpha_options["guidance"] == 4.5
    assert sigma_options["width"] == sigma_options["height"] == 1024
    assert sigma_options["use_resolution_binning"] is True


def test_lcm_catalog_and_backend_defaults():
    spec = model_info("pixart-lcm-xl-1024")
    assert spec.family == "pixart"
    assert spec.config == {"defaults": {"steps": 4, "guidance": 0.0, "max_sequence_length": 120}}
    options = PixArtBackend(Path("missing"), model_id=spec.id).validate_options(
        T2IRequest("a cat"), {}
    )
    assert options["steps"] == 4
    assert options["guidance"] == 0.0
    assert options["width"] == options["height"] == 1024


@pytest.mark.parametrize(
    ("options", "error", "match"),
    [
        ({"stepz": 3}, TypeError, "stepz"),
        ({"width": 1001, "use_resolution_binning": False}, ValueError, "multiple of 8"),
        ({"steps": 0}, ValueError, "steps"),
        ({"steps": 1001}, ValueError, "at most 1000"),
        ({"guidance": float("nan")}, ValueError, "guidance"),
        ({"negative_prompt": 3}, TypeError, "negative_prompt"),
        ({"seed": -1}, ValueError, "seed"),
        ({"use_resolution_binning": 1}, TypeError, "use_resolution_binning"),
        ({"max_sequence_length": 0}, ValueError, "max_sequence_length"),
        ({"latents": torch.zeros(1, 4, 64, 64)}, ValueError, "shape"),
    ],
)
def test_invalid_options_fail_before_loading(options, error, match):
    backend = PixArtBackend(Path("missing"))
    with pytest.raises(error, match=match):
        backend.validate_options(T2IRequest("a cat"), options)
    assert not backend.is_loaded


def test_resolution_binning_matches_reference_shapes():
    assert classify_height_width_bin(1000, 800, 128) == (1088, 896)
    assert classify_height_width_bin(800, 1000, 128) == (896, 1088)
    backend = PixArtBackend(Path("missing"))
    options = backend.validate_options(T2IRequest("a cat"), {"width": 800, "height": 1000})
    assert (options["height"], options["width"]) == (1088, 896)
    assert (options["output_height"], options["output_width"]) == (1000, 800)


def test_alpha_size_conditions():
    backend = PixArtBackend(Path("missing"))
    options = backend.validate_options(
        T2IRequest("a cat"),
        {"width": 1280, "height": 768, "use_resolution_binning": False},
    )
    resolution, aspect_ratio = backend.size_conditions(options, batch_size=2, do_cfg=True)
    assert resolution.tolist() == [[768.0, 1280.0]] * 4
    torch.testing.assert_close(aspect_ratio, torch.full((4, 1), 0.6), rtol=0, atol=0)


FIXTURES = Path(__file__).parent / "fixtures" / "pixart_configs"
NEW_PIXART_MODELS = {
    "pixart-alpha-xl-512": ("alpha_512", 64, False),
    "pixart-sigma-xl-256": ("sigma_256", 32, False),
    "pixart-sigma-xl-512": ("sigma_512", 64, False),
    "pixart-sigma-xl-2k": ("sigma_2k", 256, False),
    "pixart-lcm-xl-1024": ("lcm_1024", 128, False),
}


def _fixture_shapes(*names: Path) -> dict[str, tuple[int, ...]]:
    import json

    shapes: dict[str, tuple[int, ...]] = {}
    for name in names:
        shapes |= {key: tuple(value) for key, value in json.loads(name.read_text()).items()}
    return shapes


@pytest.mark.parametrize("model_id", sorted(NEW_PIXART_MODELS))
def test_published_transformer_headers_load_strictly_on_meta(model_id):
    import json

    directory, sample_size, _ = NEW_PIXART_MODELS[model_id]
    from imagehub.models.pixart_transformer import PixArtTransformer2DModel

    config = json.loads((FIXTURES / directory / "transformer_config.json").read_text())
    assert config["sample_size"] == sample_size
    with torch.device("meta"):
        model = PixArtTransformer2DModel(config)
    published = _fixture_shapes(FIXTURES / directory / "transformer_tensors.json")
    prepared = model.prepare_state_dict(
        {name: torch.empty(shape, device="meta") for name, shape in published.items()}
    )
    assert {name: tuple(value.shape) for name, value in prepared.items()} == {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }
    assert ("caption_projection.y_embedding" in published) == (
        model_id in {"pixart-alpha-xl-512", "pixart-lcm-xl-1024"}
    )
    assert model.use_additional_conditions is (model_id == "pixart-lcm-xl-1024")


@pytest.mark.parametrize("directory", ["alpha_512", "lcm_1024"])
def test_published_alpha_vae_and_text_encoder_headers_load_on_meta(directory):
    import json

    from imagehub.models.autoencoder_kl import AutoencoderKL
    from imagehub.models.t5 import T5EncoderModel

    root = FIXTURES / directory
    for cls, config_name, tensors in (
        (AutoencoderKL, "vae_config.json", [root / "vae_tensors.json"]),
        (
            T5EncoderModel,
            "text_encoder_config.json",
            sorted(root.glob("text_encoder_tensors_*.json")),
        ),
    ):
        with torch.device("meta"):
            model = cls(json.loads((root / config_name).read_text()))
        published = _fixture_shapes(*tensors)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in published.items()}
        )
        assert {name: tuple(value.shape) for name, value in prepared.items()} == {
            name: tuple(value.shape) for name, value in model.state_dict().items()
        }


def test_published_scheduler_and_pipeline_classes():
    import json

    alpha = json.loads((FIXTURES / "alpha_512" / "model_index.json").read_text())
    sigma = json.loads((FIXTURES / "sigma_base" / "model_index.json").read_text())
    assert alpha["_class_name"] == "PixArtAlphaPipeline"
    assert sigma["_class_name"] == "PixArtSigmaPipeline"
    assert (FIXTURES / "alpha_512" / "scheduler_scheduler_config.json").read_text() == (
        FIXTURES / "sigma_base" / "scheduler_scheduler_config.json"
    ).read_text()
    lcm = json.loads((FIXTURES / "lcm_1024" / "model_index.json").read_text())
    scheduler = json.loads((FIXTURES / "lcm_1024" / "scheduler_scheduler_config.json").read_text())
    assert lcm["_class_name"] == "PixArtAlphaPipeline"
    assert scheduler["_class_name"] == "LCMScheduler"


@pytest.mark.parametrize(
    ("model_id", "native"),
    [
        ("pixart-alpha-xl-1024", 1024),
        ("pixart-sigma-xl-1024", 1024),
        ("pixart-alpha-xl-512", 512),
        ("pixart-sigma-xl-512", 512),
        ("pixart-sigma-xl-256", 256),
        ("pixart-sigma-xl-2k", 2048),
    ],
)
def test_native_size_defaults_and_binning_table(model_id, native, tmp_path):
    import json

    spec = model_info(model_id)
    sample_size = {256: 32, 512: 64, 1024: 128, 2048: 256}[native]
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"sample_size": sample_size, "in_channels": 4})
    )
    backend = PixArtBackend(tmp_path, model_id=model_id)
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert spec.config["defaults"]["max_sequence_length"] == (120 if "alpha" in model_id else 300)
    assert (options["output_width"], options["output_height"]) == (native, native)
    assert (options["width"], options["height"]) == (native, native)
    portrait = backend.validate_options(
        T2IRequest("a cat"), {"width": native // 2, "height": native}
    )
    assert portrait["height"] > portrait["width"]
    assert portrait["height"] % 8 == portrait["width"] % 8 == 0


def test_2048_binning_table_selected_for_2k_transformers():
    from imagehub.backends.pixart import ASPECT_RATIO_2048_BIN, ASPECT_RATIO_BINS

    assert ASPECT_RATIO_BINS[256] is ASPECT_RATIO_2048_BIN
    assert classify_height_width_bin(2000, 2000, 256) == (2048, 2048)
    assert classify_height_width_bin(1000, 3900, 256) == (1024, 3968)
    with pytest.raises(ValueError, match="Invalid sample size"):
        classify_height_width_bin(1000, 1000, 100)


def test_binning_tables_match_diffusers():
    pipelines = pytest.importorskip("diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma")
    from imagehub.backends.pixart import ASPECT_RATIO_BINS

    assert ASPECT_RATIO_BINS[256] == pipelines.ASPECT_RATIO_2048_BIN
    assert ASPECT_RATIO_BINS[128] == pipelines.ASPECT_RATIO_1024_BIN
    assert ASPECT_RATIO_BINS[64] == pipelines.ASPECT_RATIO_512_BIN
    assert ASPECT_RATIO_BINS[32] == pipelines.ASPECT_RATIO_256_BIN
