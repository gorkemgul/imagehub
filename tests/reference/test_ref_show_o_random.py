"""Random-weight parity with the pinned official Show-o implementation."""

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from reference_sources import reference_source

from imagehub.backends.show_o import generate_tokens
from imagehub.models.magvit_v2 import MAGVITv2
from imagehub.models.show_o import ShowOModel

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
REFERENCE = reference_source("Show-o")


def _official_phi():
    spec = importlib.util.spec_from_file_location(
        "show_o_official_phi", REFERENCE / "models/phi.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _official_magvit():
    if "einops" not in sys.modules:
        einops = types.ModuleType("einops")
        einops.rearrange = lambda tensor, *_args, **_kwargs: tensor
        einops.repeat = lambda tensor, *_args, **_kwargs: tensor
        layers = types.ModuleType("einops.layers")
        layers_torch = types.ModuleType("einops.layers.torch")
        layers_torch.Rearrange = torch.nn.Identity
        sys.modules.update(
            {"einops": einops, "einops.layers": layers, "einops.layers.torch": layers_torch}
        )
    package_name = "show_o_reference.models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REFERENCE / "models")]
    sys.modules[package_name] = package
    utilities = types.ModuleType(package_name + ".modeling_utils")
    utilities.ConfigMixin = object
    utilities.ModelMixin = torch.nn.Module
    utilities.register_to_config = lambda function: function
    sys.modules[utilities.__name__] = utilities
    sys.modules[package_name + ".misc"] = types.ModuleType(package_name + ".misc")
    for name in ("common_modules", "modeling_magvitv2"):
        spec = importlib.util.spec_from_file_location(
            package_name + "." + name, REFERENCE / f"models/{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        module.List = list
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[package_name + ".modeling_magvitv2"]


def _official_showo():
    package_name = "show_o_sampler.models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REFERENCE / "models")]
    sys.modules[package_name] = package
    utilities = types.ModuleType(package_name + ".modeling_utils")
    utilities.ConfigMixin = object
    utilities.ModelMixin = torch.nn.Module
    utilities.register_to_config = lambda function: function
    sys.modules[utilities.__name__] = utilities
    sys.modules[package_name + ".phi"] = _official_phi()
    for name in ("sampling", "modeling_showo"):
        spec = importlib.util.spec_from_file_location(
            package_name + "." + name, REFERENCE / f"models/{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[package_name + ".modeling_showo"]


def test_tiny_phi_attention_mlp_and_layer_are_exact(record):
    official = _official_phi()
    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.5,
        "max_position_embeddings": 32,
        "layer_norm_eps": 1e-5,
        "attention_dropout": 0.0,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "rope_theta": 10000.0,
    }
    config = transformers.PhiConfig(**values, qk_layernorm=True, use_cache=False)
    config.rope_theta = 10000.0
    config.rope_scaling = None
    config._attn_implementation = "eager"
    reference = official.PhiForCausalLM(config).eval()
    native = ShowOModel(values).eval()
    native.load_state_dict(
        {"showo." + name: value for name, value in reference.state_dict().items()}
    )
    hidden = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(4))
    mask = torch.zeros(2, 1, 7, 7)
    positions = torch.arange(7).unsqueeze(0)
    with torch.inference_mode():
        expected_attention = reference.model.layers[0].self_attn(
            hidden, attention_mask=mask, position_ids=positions
        )[0]
        actual_attention = native.showo.model.layers[0].self_attn(hidden, mask)
        expected_mlp = reference.model.layers[0].mlp(hidden)
        actual_mlp = native.showo.model.layers[0].mlp(hidden)
        expected_layer = reference.model.layers[0](
            hidden, attention_mask=mask, position_ids=positions
        )[0]
        actual_layer = native.showo.model.layers[0](hidden, mask)
        ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(5))
        expected_logits = reference(ids, attention_mask=mask, use_cache=False).logits
        actual_logits = native(ids, mask)
    assert torch.equal(actual_attention, expected_attention)
    assert torch.equal(actual_mlp, expected_mlp)
    assert torch.equal(actual_layer, expected_layer)
    assert torch.equal(actual_logits, expected_logits)
    record(
        "show_o_phi_random_weights",
        attention_equal=True,
        mlp_equal=True,
        layer_equal=True,
        logits_equal=True,
    )


def test_magvit_v2_decode_is_exact(record):
    official = _official_magvit()
    torch.manual_seed(2)
    reference = official.MAGVITv2().eval()
    native = MAGVITv2({}).eval()
    native.load_state_dict(reference.state_dict())
    tokens = torch.tensor([[0]])
    with torch.inference_mode():
        expected = reference.decode_code(tokens)
        actual = native.decode_code(tokens)
    assert torch.equal(actual, expected)
    record("show_o_magvit_v2_random_weights", decode_equal=True, tokens="1x1")


def test_tiny_discrete_denoising_is_exact(record, monkeypatch):
    official = _official_showo()
    values = {
        "vocab_size": 24,
        "hidden_size": 32,
        "intermediate_size": 48,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32,
    }
    model = ShowOModel(values).eval()

    class Adapter:
        config = SimpleNamespace(mask_token_id=23)

        def __call__(self, input_ids, attention_mask=None):
            return model(input_ids, attention_mask)

    config = SimpleNamespace(
        model=SimpleNamespace(
            showo=SimpleNamespace(num_vq_tokens=3, num_new_special_tokens=2, llm_vocab_size=8)
        ),
        dataset=SimpleNamespace(preprocessing=SimpleNamespace(max_seq_length=1)),
    )
    ids = torch.tensor([[1, 9, 23, 23, 23, 10]])
    mask = torch.zeros(1, 1, 6, 6)
    expected = official.Showo.t2i_generate(
        Adapter(),
        input_ids=ids.clone(),
        attention_mask=mask,
        temperature=1.0,
        timesteps=2,
        guidance_scale=0,
        generator=torch.Generator("cpu").manual_seed(8),
        config=config,
    )
    import imagehub.backends.show_o as backend

    monkeypatch.setattr(backend, "LLM_VOCAB_SIZE", 8)
    monkeypatch.setattr(backend, "NUM_NEW_SPECIAL_TOKENS", 2)
    monkeypatch.setattr(backend, "MASK_TOKEN_ID", 23)
    actual = generate_tokens(
        model,
        ids.clone(),
        None,
        mask,
        guidance=0,
        temperature=1.0,
        steps=2,
        generator=torch.Generator("cpu").manual_seed(8),
        image_tokens=3,
    )
    assert torch.equal(actual, expected)
    record("show_o_tiny_end_to_end", token_ids_equal=True, steps=2)
