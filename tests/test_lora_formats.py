import pytest
import torch
from helpers import TINY_TEXT_FILE_CONFIG, TINY_UNET_FILE_CONFIG
from lora_helpers import (
    TEXT_TARGETS,
    UNET_TARGETS,
    random_lora,
    write_kohya,
    write_legacy,
    write_peft,
)

from imagehub.checkpoints import save_safetensors
from imagehub.lora import LoRAConv2d, LoRALayer, LoRALinear, lora_layers, remove_lora
from imagehub.lora_formats import (
    LoRASpec,
    apply_lora,
    load_lora_file,
    parse_lora_option,
    read_lora_file,
)
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.unet_2d import UNet2DConditionModel

SDXL_UNET = {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [32, 64, 64],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
    "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
    "transformer_layers_per_block": [1, 1, 1],
    "attention_head_dim": [2, 4, 4],
    "cross_attention_dim": 64,
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 8,
    "projection_class_embeddings_input_dim": 80,
    "layers_per_block": 2,
    "norm_num_groups": 8,
    "sample_size": 32,
}


@pytest.fixture(scope="module")
def sd1():
    torch.manual_seed(0)
    return {
        "unet": UNet2DConditionModel(TINY_UNET_FILE_CONFIG),
        "text_encoder": CLIPTextModel(TINY_TEXT_FILE_CONFIG),
    }


def fresh(components):
    for model in components.values():
        remove_lora(model)
    return components


def assert_loaded(components, weights, alpha, adapter="lora_0", scale=1.0):
    for (component, module), (down, up) in weights.items():
        layer = components[component].get_submodule(module)
        assert isinstance(layer, LoRALayer), module
        a, b = layer.weights(adapter)
        assert torch.equal(a, down.reshape(a.shape))
        assert torch.equal(b, up.reshape(b.shape))
        assert layer.scaling[adapter] == scale * alpha / down.shape[0]


@pytest.mark.parametrize(
    ("layout", "alpha", "expected_alpha"),
    [
        ("peft", 3.0, 3.0),
        ("peft", None, 2.0),
        ("kohya", 1.0, 1.0),
        ("kohya", None, 2.0),
        ("legacy", None, 2.0),
    ],
)
def test_each_layout_loads_the_same_weights(tmp_path, sd1, layout, alpha, expected_alpha):
    components = fresh(sd1)
    weights = random_lora({"unet": components["unet"]}, UNET_TARGETS) | random_lora(
        {"text_encoder": components["text_encoder"]}, TEXT_TARGETS, seed=1
    )
    if layout == "legacy":
        # The legacy U-Net layout only names attention projections and plain ``.lora`` layers.
        weights = {
            k: v for k, v in weights.items() if "proj_in" not in k[1] and "proj_out" not in k[1]
        }
    path = tmp_path / "adapter.safetensors"
    if layout == "peft":
        write_peft(path, weights, alpha=alpha)
    elif layout == "kohya":
        write_kohya(path, weights, alpha=alpha)
    else:
        write_legacy(path, weights)
    names = load_lora_file(components, LoRASpec(path, 0.5), adapter="lora_0")
    assert len(names) == len(weights)
    assert_loaded(components, weights, expected_alpha, scale=0.5)


def test_kohya_proj_in_loads_into_convolutions(tmp_path, sd1):
    components = fresh(sd1)
    weights = random_lora({"unet": components["unet"]}, ("proj_in",))
    assert weights
    load_lora_file(
        components, LoRASpec(write_kohya(tmp_path / "a.safetensors", weights)), adapter="a"
    )
    module = next(iter(weights))[1]
    assert isinstance(components["unet"].get_submodule(module), LoRAConv2d)


def test_kohya_sgm_names_resolve_on_sdxl_unets(tmp_path):
    torch.manual_seed(0)
    components = {"unet": UNet2DConditionModel(SDXL_UNET)}
    weights = random_lora(components, ("to_q", "to_out.0", "proj_in"))
    assert any(m.startswith("up_blocks.1") for _, m in weights)
    assert any(m.startswith("mid_block") for _, m in weights)
    path = write_kohya(tmp_path / "a.safetensors", weights, alpha=1.0, layers_per_block=2)
    load_lora_file(components, LoRASpec(path), adapter="lora_0", unet_layers_per_block=2)
    assert_loaded(components, weights, 1.0)
    for _, module in weights:
        assert isinstance(components["unet"].get_submodule(module), LoRALinear)


def test_parse_lora_option(tmp_path):
    first = tmp_path / "a.safetensors"
    second = tmp_path / "b.safetensors"
    for path in (first, second):
        save_safetensors({"x": torch.zeros(1)}, path)
    assert parse_lora_option(None) == ()
    assert parse_lora_option(first, 0.8) == (LoRASpec(first, 0.8),)
    assert parse_lora_option([str(first), (second, 0.5)]) == (
        LoRASpec(first, 1.0),
        LoRASpec(second, 0.5),
    )
    with pytest.raises(ValueError, match="pairs"):
        parse_lora_option([first], 0.5)
    with pytest.raises(FileNotFoundError):
        parse_lora_option(tmp_path / "missing.safetensors")
    with pytest.raises(ValueError, match="safetensors"):
        parse_lora_option(tmp_path / "a.bin")
    with pytest.raises(ValueError, match="finite"):
        parse_lora_option(first, float("inf"))
    with pytest.raises(ValueError, match="finite"):
        parse_lora_option([(first, True)])
    with pytest.raises(ValueError, match="empty"):
        parse_lora_option([])


@pytest.mark.parametrize(
    ("keys", "message"),
    [
        ({"lora_unet_x.dora_scale": torch.ones(1)}, "DoRA"),
        ({"lora_unet_down_blocks_0_x.hada_w1_a": torch.ones(1)}, "LyCORIS"),
        ({"vae.encoder.conv_in.lora_A.weight": torch.ones(1, 1)}, "unrecognized"),
        ({"lora_unet_does_not_exist.lora_down.weight": torch.ones(1, 1)}, "match no module"),
        (
            {
                "text_encoder_2.text_model.x.lora_A.weight": torch.ones(1, 1),
                "text_encoder_2.text_model.x.lora_B.weight": torch.ones(1, 1),
            },
            "match no module",
        ),
    ],
)
def test_rejected_files(tmp_path, sd1, keys, message):
    components = fresh(sd1)
    path = tmp_path / "bad.safetensors"
    save_safetensors(keys, path)
    with pytest.raises(ValueError, match=message):
        read_lora_file(path, components)


def test_unpaired_weights_are_rejected_and_nothing_is_loaded(tmp_path, sd1):
    components = fresh(sd1)
    weights = random_lora({"unet": components["unet"]}, ("to_q",))
    (component, module), (down, _) = next(iter(weights.items()))
    path = tmp_path / "bad.safetensors"
    save_safetensors({f"{component}.{module}.lora_A.weight": down}, path)
    with pytest.raises(ValueError, match="both down and up"):
        load_lora_file(components, LoRASpec(path), adapter="a")
    assert not lora_layers(components["unet"])


def test_text_encoder_2_needs_that_component(tmp_path, sd1):
    components = fresh(sd1)
    weights = random_lora({"text_encoder_2": components["text_encoder"]}, ("q_proj",))
    path = write_kohya(tmp_path / "a.safetensors", weights)
    with pytest.raises(ValueError, match="match no module"):
        read_lora_file(path, components)


def test_stacked_and_fused_adapters_match_unfused(tmp_path, sd1):
    components = fresh(sd1)
    text = components["text_encoder"]
    ids = torch.tensor([[0, 5, 9, 2]])
    first = write_peft(
        tmp_path / "a.safetensors", random_lora({"text_encoder": text}, ("q_proj", "fc1"))
    )
    second = write_kohya(
        tmp_path / "b.safetensors",
        random_lora({"text_encoder": text}, ("q_proj",), seed=3),
        alpha=1,
    )
    specs = parse_lora_option([(first, 0.7), (second, 0.4)])
    apply_lora(components, specs, fuse=False)
    with torch.no_grad():
        unfused = text(ids)[0]
    fresh(components)
    apply_lora(components, specs, fuse=True)
    layers = lora_layers(text)
    assert layers and all(
        layer.merged_adapters == list(layer.adapters) for layer in layers.values()
    )
    with torch.no_grad():
        torch.testing.assert_close(text(ids)[0], unfused, atol=1e-5, rtol=1e-5)
    fresh(components)


def test_transformers_5_text_encoder_spelling_loads(tmp_path, sd1):
    components = fresh(sd1)
    weights = random_lora({"text_encoder": components["text_encoder"]}, ("q_proj",))
    renamed = {(c, m.removeprefix("text_model.")): v for (c, m), v in weights.items()}
    path = write_peft(tmp_path / "a.safetensors", renamed)
    load_lora_file(components, LoRASpec(path), adapter="lora_0")
    assert_loaded(components, weights, 2.0)
