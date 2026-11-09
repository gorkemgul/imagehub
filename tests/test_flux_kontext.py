import json
import shutil
from pathlib import Path

import pytest
import torch
from helpers import (
    FIXTURES,
    TINY_SD3_T5,
    TINY_SD3_VAE,
    TINY_TEXT_FILE_CONFIG,
    write_tiny_tokenizer,
)

from imagehub import pipeline
from imagehub.backends.flux import FluxBackend, kontext_output_size
from imagehub.catalog import MODEL_CATALOG, model_info
from imagehub.checkpoints import save_safetensors
from imagehub.image import Image
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.flux_transformer import FluxTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.types import T2IRequest

PILImage = pytest.importorskip("PIL.Image")

KONTEXT = "flux1-kontext-dev"


def write_tiny_flux_snapshot(root: Path, *, kontext: bool = True, guidance_embeds: bool = True):
    """A random-weight FLUX.1 (Kontext) snapshot in the published diffusers layout."""
    root.mkdir(parents=True)
    torch.manual_seed(0)
    write_tiny_tokenizer(root / "tokenizer")
    tokenizer_2 = root / "tokenizer_2"
    tokenizer_2.mkdir()
    shutil.copy2(FIXTURES / "t5_tokenizer" / "tokenizer.json", tokenizer_2 / "tokenizer.json")
    tokenizer_2_config = {
        "tokenizer_class": "T5TokenizerFast",
        "model_max_length": 512,
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "extra_ids": 1,
    }
    (tokenizer_2 / "tokenizer_config.json").write_text(json.dumps(tokenizer_2_config))
    (tokenizer_2 / "special_tokens_map.json").write_text(
        json.dumps(
            {key: tokenizer_2_config[key] for key in ("pad_token", "eos_token", "unk_token")}
        )
    )
    scheduler = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 1.15,
        "base_image_seq_len": 4,
        "max_image_seq_len": 16,
    }
    (root / "scheduler").mkdir()
    (root / "scheduler" / "scheduler_config.json").write_text(json.dumps(scheduler))
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "FluxKontextPipeline" if kontext else "FluxPipeline",
                "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
                "text_encoder": ["transformers", "CLIPTextModel"],
                "text_encoder_2": ["transformers", "T5EncoderModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "tokenizer_2": ["transformers", "T5TokenizerFast"],
                "transformer": ["diffusers", "FluxTransformer2DModel"],
                "vae": ["diffusers", "AutoencoderKL"],
            }
        )
    )
    transformer_config = {
        "_class_name": "FluxTransformer2DModel",
        "patch_size": 1,
        "in_channels": 16,
        "out_channels": 16,
        "num_layers": 1,
        "num_single_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 2,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 32,
        "guidance_embeds": guidance_embeds,
        "axes_dims_rope": [2, 2, 4],
    }
    vae_config = TINY_SD3_VAE | {
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "sample_size": 16,
    }
    for name, cls, config, weights in (
        ("text_encoder", CLIPTextModel, TINY_TEXT_FILE_CONFIG, "model.safetensors"),
        ("text_encoder_2", T5EncoderModel, TINY_SD3_T5, "model.safetensors"),
        (
            "transformer",
            FluxTransformer2DModel,
            transformer_config,
            "diffusion_pytorch_model.safetensors",
        ),
        ("vae", AutoencoderKL, vae_config, "diffusion_pytorch_model.safetensors"),
    ):
        model = cls(config)
        for parameter in model.parameters():
            parameter.data.normal_(0, 0.05)
        state = dict(model.state_dict())
        if name == "text_encoder":
            state["text_model.embeddings.position_ids"] = torch.arange(77)[None]
        (root / name).mkdir()
        (root / name / "config.json").write_text(json.dumps(config))
        save_safetensors(state, root / name / weights)
    return root


def make_image(width, height, mode="RGB", seed=0):
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(0, 256, (height * width * channels,), generator=g, dtype=torch.uint8)
    return PILImage.frombytes(mode, (width, height), bytes(data.tolist()))


# Tiny runs keep the generated image at 32x32 and the input image at its default snap size.
SMALL = {"steps": 2, "max_area": 32 * 32, "width": 32, "height": 32, "max_sequence_length": 8}


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    return write_tiny_flux_snapshot(tmp_path_factory.mktemp("kontext") / "snap")


def test_catalog_entry():
    spec = model_info(KONTEXT)
    assert spec.family == "flux" and spec.catalog_id == "flux-1-kontext"
    assert spec.repo_id == "black-forest-labs/FLUX.1-Kontext-dev"
    assert spec.revision == "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"
    assert spec.provenance == "author" and spec.gated and spec.format == "diffusers"
    assert spec.license == "FLUX.1-dev Non-Commercial License"
    assert spec.image_input == "optional"
    assert spec.config == {
        "pipeline": "kontext",
        "defaults": {"steps": 28, "guidance": 3.5, "max_sequence_length": 512},
    }
    assert len(set(spec.files)) == len(spec.files) == 23
    assert model_info("flux1-dev").files == tuple(
        name for name in model_info("flux1-dev").files if name in spec.files
    )
    assert MODEL_CATALOG[KONTEXT].image_input == "optional"
    assert {m for m, s in MODEL_CATALOG.items() if s.image_input} == {
        KONTEXT,
        "joyai-image-edit",
        "joyai-image-edit-plus",
        "sdxl-refiner-1.0",
        "omnigen-v1",
        "visualcloze-384",
    }


def test_defaults_and_size_rescale():
    backend = FluxBackend(Path("missing"), model_id=KONTEXT)
    options = backend.validate_options(T2IRequest("edit"), {})
    assert (options["steps"], options["guidance"], options["max_sequence_length"]) == (28, 3.5, 512)
    assert (options["height"], options["width"]) == (1024, 1024)
    assert options["image"] is None
    assert kontext_output_size(1024, 1024) == (1024, 1024)
    assert kontext_output_size(1000, 1500) == (832, 1248)
    assert kontext_output_size(512, 2048) == (512, 2048)
    odd = backend.validate_options(
        T2IRequest("edit"), {"width": 1000, "height": 1000, "max_area": 10**6}
    )
    assert (odd["height"], odd["width"]) == (992, 992)


def test_image_rejected_before_loading_elsewhere():
    for backend in (FluxBackend(Path("missing"), model_id="flux1-dev"), FluxBackend(Path("m"))):
        with pytest.raises(TypeError, match="image"):
            backend.validate_options(T2IRequest("edit"), {"image": make_image(4, 4)})
        with pytest.raises(TypeError, match="max_area"):
            backend.validate_options(T2IRequest("edit"), {"max_area": 4096})
        assert not backend.is_loaded
    with pytest.raises(TypeError):
        pipeline("t2i", backend="stable-diffusion", model=Path("missing"))(
            "edit", image=make_image(4, 4)
        )


def test_invalid_kontext_options_fail_before_loading():
    backend = FluxBackend(Path("missing"), model_id=KONTEXT)
    for options, error in (
        ({"image": "cat.png"}, TypeError),
        ({"image": 3}, TypeError),
        ({"max_area": 0}, ValueError),
        ({"max_area": 1.5}, ValueError),
    ):
        with pytest.raises(error):
            backend.validate_options(T2IRequest("edit"), options)
    assert not backend.is_loaded


def run(snapshot, **options):
    edit = pipeline("t2i", backend=KONTEXT, model=snapshot)
    return edit("a cat", **(SMALL | options))


def test_without_image(snapshot):
    result = run(snapshot, seed=1)
    assert result.images[0].tensor.shape == (32, 32, 3)
    assert result.metadata["image_input"] is False
    assert "input_height" not in result.metadata
    assert torch.equal(result.images[0].tensor, run(snapshot, seed=1).images[0].tensor)
    assert not torch.equal(result.images[0].tensor, run(snapshot, seed=2).images[0].tensor)


def test_with_pil_and_imagehub_image(snapshot):
    source = make_image(96, 64)
    first = run(snapshot, seed=1, image=source)
    assert first.metadata["image_input"] is True
    assert (first.metadata["input_height"], first.metadata["input_width"]) == (832, 1248)
    assert first.images[0].tensor.shape == (32, 32, 3)
    assert torch.equal(first.images[0].tensor, run(snapshot, seed=1, image=source).images[0].tensor)
    wrapped = Image(
        torch.frombuffer(bytearray(source.tobytes()), dtype=torch.uint8).view(64, 96, 3)
    )
    assert torch.equal(
        first.images[0].tensor, run(snapshot, seed=1, image=wrapped).images[0].tensor
    )
    assert not torch.equal(first.images[0].tensor, run(snapshot, seed=1).images[0].tensor)
    rgba = make_image(96, 64, "RGBA")
    assert torch.equal(
        run(snapshot, seed=1, image=rgba).images[0].tensor,
        run(snapshot, seed=1, image=rgba.convert("RGB")).images[0].tensor,
    )


def test_non_square_output_sizes(snapshot):
    result = run(snapshot, seed=1, width=48, height=32, max_area=48 * 32)
    assert result.images[0].tensor.shape == (32, 48, 3)
    assert (result.metadata["height"], result.metadata["width"]) == (32, 48)
    rescaled = run(snapshot, seed=1, width=64, height=32, max_area=48 * 32)
    assert (rescaled.metadata["height"], rescaled.metadata["width"]) == (16, 48)
    assert rescaled.images[0].tensor.shape == (16, 48, 3)


def test_plain_flux_snapshot_still_works_unchanged(tmp_path):
    root = write_tiny_flux_snapshot(tmp_path / "plain", kontext=False)
    result = pipeline("t2i", backend="flux1-dev", model=root)(
        "a cat", steps=2, width=32, height=32, max_sequence_length=8, seed=1
    )
    assert "image_input" not in result.metadata
    assert result.images[0].tensor.shape == (32, 32, 3)
