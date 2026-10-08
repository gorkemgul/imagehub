import pytest
import torch

from imagehub.backends.kandinsky22 import Kandinsky22Backend
from imagehub.types import T2IRequest


def test_options_are_validated_before_loading(tmp_path):
    backend = Kandinsky22Backend(tmp_path)
    request = T2IRequest("a cat")
    with pytest.raises(TypeError, match="Unknown Kandinsky 2.2 options"):
        backend.infer(request, unknown=True)
    with pytest.raises(ValueError, match="steps"):
        backend.infer(request, steps=0)
    with pytest.raises(ValueError, match="width"):
        backend.infer(request, width=511)
    with pytest.raises(ValueError, match="prior_guidance"):
        backend.infer(request, prior_guidance=float("nan"))
    with pytest.raises(TypeError, match="negative_prompt"):
        backend.infer(request, negative_prompt=1)
    assert not backend.is_loaded


def test_latent_shape_validation(tmp_path):
    backend = Kandinsky22Backend(tmp_path)
    request = T2IRequest("a cat")
    with pytest.raises(ValueError, match=r"latents.*\(1, 1280\)"):
        backend.validate_options(request, {"latents": torch.zeros(1, 4, 64, 64)})
    options = backend.validate_options(request, {"latents": torch.zeros(1, 1280)})
    assert options["latents"].shape == (1, 1280)
    options = backend.validate_options(
        request, {"negative_prompt": "bad", "latents": torch.zeros(2, 1280)}
    )
    assert options["latents"].shape == (2, 1280)


def test_catalog_defaults_are_used(tmp_path):
    backend = Kandinsky22Backend(tmp_path, model_id="kandinsky-2.2")
    options = backend.validate_options(T2IRequest("a cat"), {})
    assert options["steps"] == 100
    assert options["guidance"] == 4.0
    assert options["prior_steps"] == 25
    assert options["prior_guidance"] == 4.0
    assert options["width"] == options["height"] == 512
