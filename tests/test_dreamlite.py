import json

import pytest
import torch
from fixture_files import fixture_path

from imagehub.backends.dreamlite import DreamLiteBackend
from imagehub.models.autoencoder_tiny import AutoencoderTiny
from imagehub.models.dreamlite import DreamLiteUNetModel
from imagehub.types import T2IRequest

FIXTURES = fixture_path("dreamlite_configs")


@pytest.mark.parametrize("variant", ["base", "mobile"])
@pytest.mark.parametrize(
    ("component", "model_class"),
    [("unet", DreamLiteUNetModel), ("vae", AutoencoderTiny)],
)
def test_dreamlite_published_headers(variant, component, model_class):
    config = json.loads((FIXTURES / f"{variant}-{component}_config.json").read_text())
    expected = json.loads((FIXTURES / f"{variant}-{component}-header.json").read_text())
    with torch.device("meta"):
        model = model_class(config)
    actual = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    assert actual == expected


def test_dreamlite_options_are_validated_before_loading(tmp_path):
    backend = DreamLiteBackend(tmp_path, model_id="dreamlite-base")
    request = T2IRequest("a cat")
    with pytest.raises(TypeError, match="Unknown DreamLite options"):
        backend.infer(request, unknown=True)
    with pytest.raises(ValueError, match="multiple of 16"):
        backend.infer(request, width=31)
    assert not backend.is_loaded


def test_dreamlite_mobile_defaults():
    backend = DreamLiteBackend("unused", model_id="dreamlite-mobile")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert options["steps"] == 4
    assert options["guidance"] is None
