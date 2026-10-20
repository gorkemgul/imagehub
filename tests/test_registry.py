import pytest

from imagehub.backends.stable_diffusion import StableDiffusionBackend
from imagehub.backends.stable_diffusion_xl import StableDiffusionXLBackend
from imagehub.registry import BackendRegistry, available_backends, backend_registry


def test_family_and_model_aliases_are_registered():
    names = available_backends("t2i")["t2i"]
    assert "stable-diffusion" in names
    assert "stable-diffusion-xl" in names
    assert "stable-diffusion-3" in names
    assert {"sd-v1-1", "sd-v1-5"} <= set(names)
    assert "sd-v2-1" in names
    assert "lcm-dreamshaper-v7" in names
    assert "sdxl-base-1.0" in names
    assert "segmind-vega" in names
    assert {"koala-700m", "koala-1b"} <= set(names)
    assert "janus" in names
    assert "janus-pro-1b" in names
    assert {"sd3.5-medium", "sd3.5-large"} <= set(names)
    assert {"qwen-image", "qwen-image-2.1"} <= set(names)
    assert {"flux1-dev", "flux1-kontext-dev", "flux1-krea-dev"} <= set(names)
    assert {"flux2", "flux2-klein-4b", "flux2-klein-9b"} <= set(names)
    assert {
        "hunyuan-dit",
        "hunyuan-dit-v1.1-distilled",
        "hunyuan-dit-v1.2",
        "hunyuan-dit-v1.2-distilled",
    } <= set(names)
    assert {"pixart", "pixart-sigma-xl-1024", "pixart-alpha-xl-1024", "pixart-lcm-xl-1024"} <= set(
        names
    )
    assert {"krea2", "krea2-turbo"} <= set(names)
    assert "ernie-image" in names
    assert "longcat-image" in names
    assert {"cogview3", "cogview3-plus-3b"} <= set(names)
    assert {"cogview4", "cogview4-6b"} <= set(names)
    assert {"ideogram4", "ideogram4-fp8"} <= set(names)
    assert {"kandinsky21", "kandinsky-2-1"} <= set(names)
    assert {"kandinsky22", "kandinsky-2.2"} <= set(names)
    assert {"kandinsky3", "kandinsky-3"} <= set(names)
    assert {"kandinsky5", "kandinsky-5-t2i-lite"} <= set(names)
    assert "kolors" in names
    assert {"hidream", "hidream-i1-fast"} <= set(names)
    assert {"hidream-o1", "hidream-o1-image"} <= set(names)
    assert {"auraflow", "auraflow-v0.3"} <= set(names)
    assert {"prx", "prx-1024-t2i-beta", "prx-512-t2i-sft"} <= set(names)
    assert "pixeldit" in names
    assert {
        "sana",
        "sana-1600m-1024",
        "sana-600m-1024",
        "sana-sprint-1.6b-1024",
        "sana-1.5-4.8b-1024",
    } <= set(names)
    assert {"lumina2", "lumina-image-2.0"} <= set(names)
    assert {"amused", "amused-256", "amused-512"} <= set(names)
    assert "meissonic" in names
    assert {"llamagen", "llamagen-t2i-xl-256", "llamagen-t2i-xl-512"} <= set(names)
    assert {"stable_cascade", "stable-cascade", "stable-cascade-lite"} <= set(names)
    assert {"wuerstchen", "wuerstchen-v2"} <= set(names)
    assert {"omnigen", "omnigen-v1"} <= set(names)


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


def test_catalog_alias_passes_model_id_but_family_does_not(tmp_path):
    alias = backend_registry.create("t2i", "sd-v1-4", model=tmp_path)
    family = backend_registry.create("t2i", "stable-diffusion", model=tmp_path)
    assert isinstance(alias, StableDiffusionBackend) and alias.model_id == "sd-v1-4"
    assert family.model_id is None

    lcm_alias = backend_registry.create("t2i", "lcm-dreamshaper-v7", model=tmp_path)
    assert isinstance(lcm_alias, StableDiffusionBackend)
    assert lcm_alias.model_id == "lcm-dreamshaper-v7"

    xl_alias = backend_registry.create("t2i", "sdxl-base-1.0", model=tmp_path)
    xl_family = backend_registry.create("t2i", "stable-diffusion-xl", model=tmp_path)
    assert isinstance(xl_alias, StableDiffusionXLBackend)
    assert xl_alias.model_id == "sdxl-base-1.0"
    assert xl_family.model_id is None


def test_alias_registration_passes_model_id_to_any_factory(tmp_path):
    seen = []

    def factory(**options):
        seen.append(options)
        return StableDiffusionBackend(tmp_path)

    registry = BackendRegistry()
    registry.register("t2i", "family", factory)
    registry.register("t2i", "alias", factory, model_id="alias")
    registry.create("t2i", "family", device="cpu")
    registry.create("t2i", "alias", device="cpu")
    assert seen == [{"device": "cpu"}, {"model_id": "alias", "device": "cpu"}]
