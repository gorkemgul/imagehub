"""Reference parity coverage for JoyAI-Image; real checks require local snapshots."""

from __future__ import annotations

import gc
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.joyai_image import (
    _MULTI_TEMPLATE,
    JoyAIImageBackend,
    _Components,
    _vision_pixels,
)
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl_wan import AutoencoderKLWan
from imagehub.models.joyimage_transformer import JoyImageEditTransformer3DModel
from imagehub.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLVisionModel
from imagehub.tokenization_qwen import Qwen2Tokenizer
from imagehub.types import T2IRequest

pytestmark = pytest.mark.reference

_TINY = {
    "patch_size": [1, 2, 2],
    "in_channels": 4,
    "out_channels": 4,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "text_dim": 12,
    "mlp_width_ratio": 2.0,
    "num_layers": 2,
    "rope_dim_list": [2, 2, 0],
    "rope_type": "rope",
    "theta": 10000,
}


def test_random_joy_transformer_matches_diffusers() -> None:
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    reference = diffusers.JoyImageEditTransformer3DModel(**_TINY).eval()
    actual = JoyImageEditTransformer3DModel(_TINY).eval()
    actual.load_state_dict(reference.state_dict())
    sample, timestep, text = torch.randn(1, 4, 1, 4, 4), torch.tensor([1.0]), torch.randn(1, 3, 12)
    assert torch.equal(reference(sample, timestep, text).sample, actual(sample, timestep, text))


def test_random_joy_plus_transformer_matches_diffusers() -> None:
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(1)
    reference = diffusers.JoyImageEditPlusTransformer3DModel(**_TINY).eval()
    actual = JoyImageEditTransformer3DModel(_TINY | {"plus": True}).eval()
    actual.load_state_dict(reference.state_dict())
    sample = torch.randn(2, 5, 4, 1, 2, 2)
    timestep = torch.tensor([1.0, 2.0])
    text = torch.randn(2, 3, 12)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    shapes = [[(1, 1, 2), (1, 1, 1)], [(1, 1, 3)]]
    expected = reference(sample, timestep, text, mask, shapes).sample
    actual_output = actual(sample, timestep, text, mask, shapes)
    assert torch.equal(expected, actual_output)


def test_random_qwen3_vl_vision_matches_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    config = {
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 12,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
        "hidden_act": "gelu_pytorch_tanh",
    }
    reference_config = transformers.Qwen3VLVisionConfig(**config)
    torch.manual_seed(2)
    reference = transformers.Qwen3VLVisionModel(reference_config).eval()
    actual = Qwen3VLVisionModel(config).eval()
    actual.load_state_dict(reference.state_dict())
    grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
    pixels = torch.randn(24, 24)
    expected = reference(pixels, grid_thw=grid)
    output, deep = actual(pixels, grid)
    assert torch.equal(expected.pooler_output, output)
    assert torch.equal(expected.deepstack_features[0], deep[0])


def test_random_qwen3_vl_image_token_conditioning_matches_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    text = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
            "rope_type": "default",
        },
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
    }
    vision = {
        "depth": 2,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 16,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
        "hidden_act": "gelu_pytorch_tanh",
    }
    config = transformers.Qwen3VLConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=10,
        video_token_id=11,
        vision_start_token_id=12,
        vision_end_token_id=13,
    )
    torch.manual_seed(3)
    reference = transformers.Qwen3VLForConditionalGeneration(config).eval()
    actual = Qwen3VLForConditionalGeneration(config.to_dict()).eval()
    actual.load_state_dict(reference.state_dict())
    ids = torch.tensor([[1, 12, 10, 10, 10, 10, 13, 2]])
    mask = torch.ones_like(ids)
    types = ids.eq(10).long()
    grid = torch.tensor([[1, 4, 4]])
    pixels = torch.randn(16, 24)
    captured = []
    reference_inputs = []
    actual_inputs = []
    reference_layer = []
    actual_layer = []
    reference_rope = []
    actual_rope = []
    reference.model.language_model.rotary_emb.register_forward_hook(
        lambda _module, _args, output: reference_rope.append(output)
    )
    actual.model.language_model.rotary_emb.register_forward_hook(
        lambda _module, _args, output: actual_rope.append(output)
    )
    reference.model.language_model.layers[0].register_forward_pre_hook(
        lambda _module, args: reference_inputs.append(args[0].clone())
    )
    actual.model.language_model.layers[0].register_forward_pre_hook(
        lambda _module, args: actual_inputs.append(args[0].clone())
    )
    reference.model.language_model.layers[0].register_forward_hook(
        lambda _module, _args, output: reference_layer.append(output.clone())
    )
    actual.model.language_model.layers[0].register_forward_hook(
        lambda _module, _args, output: actual_layer.append(output.clone())
    )
    handle = reference.model.language_model.layers[-1].register_forward_hook(
        lambda _module, _args, output: captured.append(
            output[0] if isinstance(output, tuple) else output
        )
    )
    with torch.inference_mode():
        reference(
            input_ids=ids,
            attention_mask=mask,
            pixel_values=pixels,
            image_grid_thw=grid,
            mm_token_type_ids=types,
        )
        output = actual(
            ids,
            mask,
            pixel_values=pixels,
            image_grid_thw=grid,
            mm_token_type_ids=types,
        )[-1]
    handle.remove()
    assert torch.equal(reference_inputs[0], actual_inputs[0])
    assert all(
        torch.equal(left, right)
        for left, right in zip(reference_rope[0], actual_rope[0], strict=True)
    )
    assert torch.equal(reference_layer[0], actual_layer[0])
    assert torch.equal(captured[0], output)


def test_random_wan_vae_matches_diffusers() -> None:
    diffusers = pytest.importorskip("diffusers")
    config = {
        "base_dim": 4,
        "decoder_base_dim": 4,
        "z_dim": 2,
        "dim_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_scales": [],
        "temperal_downsample": [False],
        "dropout": 0.0,
        "latents_mean": [0.0, 0.0],
        "latents_std": [1.0, 1.0],
        "is_residual": False,
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": None,
        "scale_factor_temporal": 1,
        "scale_factor_spatial": 1,
    }
    torch.manual_seed(4)
    reference = diffusers.AutoencoderKLWan(**config).eval()
    actual = AutoencoderKLWan(config).eval()
    actual.load_state_dict(reference.state_dict())
    pixels = torch.randn(1, 3, 8, 8)
    with torch.inference_mode():
        expected_latent = reference.encode(pixels.unsqueeze(2)).latent_dist.mode()[:, :, 0]
        output_latent = actual.encode(pixels)
        expected_image = reference.decode(expected_latent.unsqueeze(2)).sample[:, :, 0]
        output_image = actual.decode(expected_latent)
    assert torch.equal(expected_latent, output_latent)
    assert torch.equal(expected_image, output_image)


class _Inputs(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return _Inputs({name: value.to(device) for name, value in self.items()})


class _TinyProcessor:
    def __call__(self, **_kwargs):
        return _Inputs(
            input_ids=torch.zeros(1, 40, dtype=torch.long),
            attention_mask=torch.ones(1, 40, dtype=torch.long),
            pixel_values=torch.zeros(4, 24),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            mm_token_type_ids=torch.zeros(1, 40, dtype=torch.long),
        )


class _LastLayer(torch.nn.Module):
    def forward(self, x):
        return x


class _TinyText(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=torch.nn.ModuleList([_LastLayer()]))
        )

    @property
    def dtype(self):
        return torch.float32

    def forward(self, input_ids, **_kwargs):
        values = torch.arange(input_ids.shape[1] * 12, dtype=torch.float32).reshape(1, -1, 12)
        return self.model.language_model.layers[-1](values)


class _Distribution:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value

    def sample(self):
        return self.value


class _ReferenceVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            scale_factor_temporal=4,
            scale_factor_spatial=8,
            latents_mean=[0.0] * 4,
            latents_std=[1.0] * 4,
        )

    @property
    def dtype(self):
        return torch.float32

    def encode(self, pixels):
        value = torch.nn.functional.avg_pool2d(pixels[:, :, 0], 8)
        value = torch.cat((value, value.mean(1, keepdim=True)), 1).unsqueeze(2)
        return SimpleNamespace(latent_dist=_Distribution(value))

    def decode(self, latent, return_dict=True):
        value = torch.nn.functional.interpolate(latent[:, :3, 0], scale_factor=8, mode="nearest")
        value = value.unsqueeze(2).clamp(-1, 1)
        return SimpleNamespace(sample=value) if return_dict else (value,)


class _NativeVAE:
    latents_mean = [0.0] * 4
    latents_std = [1.0] * 4

    def encode(self, pixels):
        value = torch.nn.functional.avg_pool2d(pixels, 8)
        return torch.cat((value, value.mean(1, keepdim=True)), 1)

    encode_sample = encode

    def decode(self, latent):
        return torch.nn.functional.interpolate(latent[:, :3], scale_factor=8, mode="nearest").clamp(
            -1, 1
        )


@pytest.mark.parametrize("plus", (False, True), ids=("base", "plus"))
@pytest.mark.parametrize(("seed", "guidance"), ((0, 1.0), (7, 4.0)))
def test_tiny_pipeline_end_to_end_exact(plus, seed, guidance, monkeypatch):
    diffusers = pytest.importorskip("diffusers")
    config = _TINY | ({"plus": True} if plus else {})
    torch.manual_seed(9)
    reference_transformer = (
        diffusers.JoyImageEditPlusTransformer3DModel(**_TINY)
        if plus
        else diffusers.JoyImageEditTransformer3DModel(**_TINY)
    ).eval()
    native_transformer = JoyImageEditTransformer3DModel(config).eval()
    native_transformer.load_state_dict(reference_transformer.state_dict())
    scheduler_config = read_json(
        Path(__file__).parents[1]
        / "fixtures/joyai_image_configs/base/scheduler/scheduler_config.json"
    )
    reference_scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    reference_cls = diffusers.JoyImageEditPlusPipeline if plus else diffusers.JoyImageEditPipeline
    reference = reference_cls(
        scheduler=reference_scheduler,
        vae=_ReferenceVAE(),
        text_encoder=_TinyText(),
        tokenizer=None,
        transformer=reference_transformer,
        processor=_TinyProcessor(),
    )
    monkeypatch.setattr(
        type(reference), "_execution_device", property(lambda _self: torch.device("cpu"))
    )
    if not plus:
        monkeypatch.setattr(
            reference.vae_image_processor, "get_default_height_width", lambda *_args: (32, 32)
        )
    else:
        monkeypatch.setattr(
            reference.image_processor, "get_default_height_width", lambda *_args: (32, 32)
        )
    image = _image().resize((32, 32))
    kwargs = {"images": [image]} if plus else {"image": image}
    expected = reference(
        prompt=PROMPT,
        height=32,
        width=32,
        num_inference_steps=2,
        guidance_scale=guidance,
        generator=torch.Generator().manual_seed(seed),
        output_type="pt",
        **kwargs,
    ).images[0]
    expected = expected.permute(1, 2, 0).mul(255).round().to(torch.uint8)

    backend = JoyAIImageBackend(
        ".", model_id="joyai-image-edit-plus" if plus else "joyai-image-edit"
    )
    backend.components = _Components(
        tokenizer=None,
        text=None,
        transformer=native_transformer,
        vae=_NativeVAE(),
        scheduler=scheduler_config,
    )
    values = torch.arange(40 * 12, dtype=torch.float32).reshape(1, 40, 12)[:, 34:]
    monkeypatch.setattr(
        "imagehub.backends.joyai_image._conditioning",
        lambda *_args, **_kwargs: (values, torch.ones(1, 6, dtype=torch.long)),
    )
    monkeypatch.setattr("imagehub.backends.joyai_image._bucket", lambda h, w: (h, w))
    actual = (
        backend._infer(
            T2IRequest(PROMPT),
            image=image,
            height=32,
            width=32,
            steps=2,
            guidance=guidance,
            seed=seed,
        )
        .images[0]
        .tensor
    )
    assert torch.equal(actual, expected)


MODEL_IDS = ("joyai-image-edit", "joyai-image-edit-plus")
PROMPT = "Turn the square blue."


def _checkpoint_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    paths = (
        [
            directory / name
            for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))
        ]
        if index
        else [next(directory.glob("*.safetensors"))]
    )
    result = {}
    for path in paths:
        with path.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(length))
        result |= {
            name: tuple(value["shape"]) for name, value in header.items() if name != "__metadata__"
        }
    return result


def _image():
    pil = pytest.importorskip("PIL.Image")
    data = torch.arange(64 * 80 * 3, dtype=torch.int64).remainder(256).to(torch.uint8)
    return pil.frombytes("RGB", (80, 64), bytes(data.flatten().tolist()))


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_real_checkpoint_headers(model_id, record):
    root = snapshot_dir(model_id)
    result = {}
    for component, cls in {
        "text_encoder": Qwen3VLForConditionalGeneration,
        "transformer": JoyImageEditTransformer3DModel,
        "vae": AutoencoderKLWan,
    }.items():
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
        shapes = _checkpoint_shapes(root / component)
        prepared = model.prepare_state_dict(
            {name: torch.empty(shape, device="meta") for name, shape in shapes.items()}
        )
        expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
        actual = {name: tuple(value.shape) for name, value in prepared.items()}
        result[component] = {
            "missing": sorted(expected.keys() - actual.keys()),
            "extra": sorted(actual.keys() - expected.keys()),
            "mismatch": sorted(
                name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
            ),
            "tensors": len(shapes),
        }
    record(f"header_{model_id}", **result)
    assert all(
        not value[key] for value in result.values() for key in ("missing", "extra", "mismatch")
    )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_real_tokenizer_and_vision_preprocessing(model_id, record):
    transformers = pytest.importorskip("transformers")
    root = snapshot_dir(model_id)
    processor = transformers.Qwen3VLProcessor.from_pretrained(
        root / "processor", local_files_only=True
    )
    image = _image()
    text = _MULTI_TEMPLATE.format(
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{PROMPT}<|im_end|>\n"
    )
    expected = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    pixels, grid = _vision_pixels(image)
    count = int(grid.prod() // 4)
    expanded = text.replace("<|image_pad|>", "<|image_pad|>" * count)
    ids = torch.tensor([Qwen2Tokenizer(root / "tokenizer").tokenize(expanded)])
    passed = (
        torch.equal(ids, expected.input_ids)
        and torch.equal(grid, expected.image_grid_thw)
        and torch.equal(pixels, expected.pixel_values)
    )
    record(f"tokenizer_vision_{model_id}", passed=passed, tolerance="exact")
    assert passed


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_real_text_and_vision_conditioning(model_id, record, tmp_path):
    transformers = pytest.importorskip("transformers")
    root = snapshot_dir(model_id)
    image = _image()
    processor = transformers.Qwen3VLProcessor.from_pretrained(
        root / "processor", local_files_only=True
    )
    text = _MULTI_TEMPLATE.format(
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{PROMPT}<|im_end|>\n"
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    reference = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    captured = {}

    def capture(_module, _args, output):
        captured["value"] = output[0] if isinstance(output, tuple) else output

    handle = reference.model.language_model.layers[-1].register_forward_hook(capture)
    with torch.inference_mode():
        reference(**inputs)
    handle.remove()
    expected_file = tmp_path / "conditioning.pt"
    torch.save(captured["value"], expected_file)
    del reference
    gc.collect()

    ours = Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", device=torch.device("cpu"), dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(
            inputs.input_ids,
            inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            mm_token_type_ids=inputs.mm_token_type_ids,
        )[-1]
    _check(record, f"conditioning_{model_id}", actual, torch.load(expected_file, weights_only=True))


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_real_transformer_step(model_id, record, tmp_path):
    diffusers = pytest.importorskip("diffusers")
    root = snapshot_dir(model_id)
    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator().manual_seed(4)
    text = torch.randn(1, 7, config["text_dim"], generator=generator)
    timestep = torch.tensor([500.0])
    if model_id.endswith("plus"):
        hidden = torch.randn(1, 8, config["in_channels"], 1, 2, 2, generator=generator)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
        shapes = [[(1, 2, 2), (1, 1, 2)]]
        cls = diffusers.JoyImageEditPlusTransformer3DModel
        kwargs = {"encoder_hidden_states_mask": mask, "shape_list": shapes}
    else:
        hidden = torch.randn(1, 2, config["in_channels"], 1, 4, 4, generator=generator)
        cls = diffusers.JoyImageEditTransformer3DModel
        kwargs = {}
    reference = cls.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(hidden, timestep, text, **kwargs).sample
    path = tmp_path / "transformer.pt"
    torch.save(expected, path)
    del reference, expected
    gc.collect()
    ours = JoyImageEditTransformer3DModel.from_pretrained(
        root / "transformer", device=torch.device("cpu"), dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(
            hidden,
            timestep,
            text,
            kwargs.get("encoder_hidden_states_mask"),
            kwargs.get("shape_list"),
        )
    _check(record, f"transformer_{model_id}", actual, torch.load(path, weights_only=True))


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_real_vae(model_id, record, tmp_path):
    diffusers = pytest.importorskip("diffusers")
    root = snapshot_dir(model_id)
    pixels = torch.randn(1, 3, 64, 80, generator=torch.Generator().manual_seed(5))
    reference = diffusers.AutoencoderKLWan.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        latent = reference.encode(pixels.unsqueeze(2)).latent_dist.mode()[:, :, 0]
        expected = reference.decode(latent.unsqueeze(2)).sample[:, :, 0]
    path = tmp_path / "vae.pt"
    torch.save((latent, expected), path)
    del reference, latent, expected
    gc.collect()
    ours = AutoencoderKLWan.from_pretrained(
        root / "vae", device=torch.device("cpu"), dtype=torch.float32
    )
    latent, expected = torch.load(path, weights_only=True)
    with torch.inference_mode():
        actual_latent = ours.encode(pixels)
        actual = ours.decode(latent)
    _check(record, f"vae_encode_{model_id}", actual_latent, latent)
    _check(record, f"vae_decode_{model_id}", actual, expected)


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("seed", (0, 42))
def test_real_end_to_end(model_id, seed, record):
    diffusers = pytest.importorskip("diffusers")
    root = snapshot_dir(model_id)
    image = _image().resize((512, 512))
    cls = (
        diffusers.JoyImageEditPlusPipeline
        if model_id.endswith("plus")
        else diffusers.JoyImageEditPipeline
    )
    reference = cls.from_pretrained(root, local_files_only=True, torch_dtype=torch.float32)
    torch.manual_seed(seed)
    kwargs = {"images": [image]} if model_id.endswith("plus") else {"image": image}
    expected = reference(
        prompt=PROMPT,
        height=512,
        width=512,
        num_inference_steps=2,
        guidance_scale=4.0,
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    ).images[0]
    expected = torch.frombuffer(bytearray(expected.tobytes()), dtype=torch.uint8).reshape(
        expected.height, expected.width, 3
    )
    del reference
    gc.collect()
    torch.manual_seed(seed)
    actual = (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT, image=image, width=512, height=512, steps=2, guidance=4.0, seed=seed
        )
        .images[0]
        .tensor
    )
    passed = torch.equal(actual, expected)
    name = f"e2e_{model_id}" if seed == 0 else f"seed_e2e_{model_id}"
    record(
        name,
        passed=passed,
        seed=seed,
        steps=2,
        max_pixel_diff=int((actual.int() - expected.int()).abs().max()),
    )
    assert passed
