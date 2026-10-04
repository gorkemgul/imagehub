import pytest

from imagehub.registry import BackendRegistry, available_backends


def test_family_and_model_aliases_are_registered():
    names = available_backends("t2i")["t2i"]
    assert "stable-diffusion" in names
    assert {"sd-v1-1", "sd-v1-5"} <= set(names)


def test_registration_rules():
    registry = BackendRegistry()
    registry.register("t2i", "x", "some.module:Factory")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("t2i", "x", "some.module:Factory")
    with pytest.raises(ValueError, match="module:class"):
        registry.register("t2i", "y", "no_colon")
    with pytest.raises(TypeError):
        registry.register("t2i", "z", 42)
    with pytest.raises(ValueError, match="Available: x"):
        registry.create("t2i", "missing")
