"""Qwen-Image-Edit real-checkpoint parity on CPU float32."""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, reference_in_subprocess, snapshot_dir

from imagehub import pipeline
from imagehub.backends.qwen_image import (
    _EDIT_DROP_INDEX,
    _EDIT_TEMPLATE,
    _edit_dimensions,
    _pack_original_latents,
    _resize_image,
    _vae_pixels,
    _vision_pixels,
    decode_original_latents,
)
from imagehub.checkpoints import read_json
from imagehub.image import Image
from imagehub.models.autoencoder_kl_qwen_image import AutoencoderKLQwenImage
from imagehub.models.qwen2_5_vl_text import Qwen25VLForConditionalGeneration
from imagehub.models.qwen_image_transformer import QwenImageTransformer2DModel
from imagehub.schedulers.flow_match_euler import FlowMatchEulerDiscreteScheduler, qwen_image_mu
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

CPU = torch.device("cpu")
PROMPT = "make the sky a vivid sunset"
NEGATIVE_PROMPT = "blurry"
REVISIONS = {
    "qwen-image-edit": "ac7f9318f633fc4b5778c59367c8128225f1e3de",
    "qwen-image-edit-2509": "d3968ef930e841f4c73640fb8afa3b306a78167e",
}


@pytest.fixture(params=tuple(REVISIONS), ids=tuple(REVISIONS))
def model_id(request):
    return request.param


def _is_plus(model_id: str) -> bool:
    return model_id == "qwen-image-edit-2509"


def _reference_pipeline_class(model_id: str):
    return (
        diffusers.QwenImageEditPlusPipeline
        if _is_plus(model_id)
        else diffusers.QwenImageEditPipeline
    )


def _reference_text(prompt: str, image_count: int, *, plus: bool) -> str:
    placeholders = "".join(
        (f"Picture {index}: " if plus else "") + "<|vision_start|><|image_pad|><|vision_end|>"
        for index in range(1, image_count + 1)
    )
    return _EDIT_TEMPLATE.format(placeholders + prompt)


def _native_text(prompt: str, grids: list[torch.Tensor], *, plus: bool) -> str:
    placeholders = "".join(
        (f"Picture {index}: " if plus else "")
        + "<|vision_start|>"
        + "<|image_pad|>" * int(grid.prod() // 4)
        + "<|vision_end|>"
        for index, grid in enumerate(grids, 1)
    )
    return _EDIT_TEMPLATE.format(placeholders + prompt)


def _shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    files = [directory / "diffusion_pytorch_model.safetensors"] if index is None else []
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    result = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result.update(
            {
                name: tuple(entry["shape"])
                for name, entry in header.items()
                if name != "__metadata__"
            }
        )
    return result


def _header(expected, actual):
    return {
        "missing": sorted(expected.keys() - actual.keys()),
        "extra": sorted(actual.keys() - expected.keys()),
        "mismatch": sorted(
            name for name in expected.keys() & actual.keys() if expected[name] != actual[name]
        ),
        "tensors": len(actual),
    }


def test_published_tensor_headers_match_models(record, model_id):
    root = snapshot_dir(model_id)
    with torch.device("meta"):
        text = Qwen25VLForConditionalGeneration(read_json(root / "text_encoder/config.json"))
        transformer = QwenImageTransformer2DModel(read_json(root / "transformer/config.json"))
        vae = AutoencoderKLQwenImage(read_json(root / "vae/config.json"))
    result = {
        "text_encoder": _header(
            {name: tuple(value.shape) for name, value in text.state_dict().items()},
            _shapes(root / "text_encoder"),
        ),
        "transformer": _header(
            {name: tuple(value.shape) for name, value in transformer.state_dict().items()},
            _shapes(root / "transformer"),
        ),
        "vae": _header(
            {name: tuple(value.shape) for name, value in vae.state_dict().items()},
            _shapes(root / "vae"),
        ),
    }
    record(f"header_{model_id}", revision=REVISIONS[model_id], **result)
    for entry in result.values():
        assert not entry["missing"] and not entry["extra"] and not entry["mismatch"]


def test_processor_and_text_encoder(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    plus = _is_plus(model_id)
    source = Image(
        torch.arange(56 * 84 * 3, dtype=torch.int64).reshape(56, 84, 3).to(torch.uint8)
    ).to_pil()
    text = _reference_text(PROMPT, 1, plus=plus)
    processor = transformers.Qwen2VLProcessor.from_pretrained(
        root / "processor", local_files_only=True
    )
    inputs = processor(text=[text], images=[source] if plus else source, return_tensors="pt")
    reference = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(**inputs, output_hidden_states=True).hidden_states[-1][
            :, _EDIT_DROP_INDEX:
        ]
    path = tmp_path / "text.pt"
    torch.save(expected, path)
    del reference, expected
    gc.collect()

    config = read_json(root / "processor/preprocessor_config.json")
    pixels, grid = _vision_pixels(source, config)
    native_text = _native_text(PROMPT, [grid], plus=plus)
    ids = torch.tensor([Qwen2Tokenizer(root / "processor").tokenize(native_text)])
    ids_match = torch.equal(ids, inputs["input_ids"])
    pixels_match = torch.equal(pixels, inputs["pixel_values"])
    model = Qwen25VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = model(
            ids,
            torch.ones_like(ids),
            pixel_values=pixels,
            image_grid_thw=grid,
        )[-1][:, _EDIT_DROP_INDEX:]
    expected = torch.load(path, weights_only=True)
    measured = max_diff(actual, expected)
    record(
        f"qwen_image_edit_conditioner_{model_id}",
        ids_match=ids_match,
        pixels_match=pixels_match,
        **measured,
        tolerance=0.0,
    )
    assert ids_match and pixels_match
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_transformer_step(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    generator = torch.Generator("cpu").manual_seed(3)
    hidden = torch.randn(1, 8, 64, generator=generator)
    context = torch.randn(1, 12, 3584, generator=generator)
    kwargs = {
        "hidden_states": hidden,
        "encoder_hidden_states": context,
        "timestep": torch.tensor([0.75]),
        "img_shapes": [[(1, 2, 2), (1, 2, 2)]],
    }
    reference = diffusers.QwenImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = reference(**kwargs, return_dict=False)[0]
    path = tmp_path / "transformer.pt"
    torch.save(expected, path)
    del reference, expected
    gc.collect()
    ours = QwenImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual = ours(**kwargs)
    expected = torch.load(path, weights_only=True)
    measured = max_diff(actual, expected)
    record(f"qwen_image_edit_transformer_{model_id}", **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_vae_encode_decode(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    image = torch.randn(1, 3, 1, 32, 48, generator=torch.Generator("cpu").manual_seed(4))
    reference = diffusers.AutoencoderKLQwenImage.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        encoded = reference.encode(image).latent_dist.mode()
        expected = reference.decode(encoded, return_dict=False)[0]
    path = tmp_path / "vae.pt"
    torch.save((encoded, expected), path)
    del reference, encoded, expected
    gc.collect()
    ours = AutoencoderKLQwenImage.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_encoded = ours.encode(image)
        actual = ours.decode(actual_encoded)
    encoded, expected = torch.load(path, weights_only=True)
    encode_diff, decode_diff = max_diff(actual_encoded, encoded), max_diff(actual, expected)
    record(
        f"qwen_image_edit_vae_{model_id}",
        encode_max_abs=encode_diff["max_abs"],
        decode_max_abs=decode_diff["max_abs"],
        tolerance=0.0,
    )
    torch.testing.assert_close(actual_encoded, encoded, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _reference_image_stages(root: str, model_id: str, out_path: str) -> None:
    import numpy as np
    from diffusers.image_processor import VaeImageProcessor

    source = torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8)
    image = Image(source).to_pil()
    width, height = _edit_dimensions(1024 * 1024, image.width, image.height)
    processor = VaeImageProcessor(vae_scale_factor=16)
    resized = processor.resize(image, height, width)
    pixels = processor.preprocess(resized, height, width).unsqueeze(2)
    vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        f"{root}/vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        encoded = vae.encode(pixels).latent_dist.mode()
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        pipeline_class = _reference_pipeline_class(model_id)
        source_latents = pipeline_class._pack_latents(
            (encoded - mean) / std, 1, 16, encoded.shape[-2], encoded.shape[-1]
        )

        noise = torch.randn((1, 1, 16, 4, 4), generator=torch.Generator("cpu").manual_seed(42))
        packed = pipeline_class._pack_latents(noise, 1, 16, 4, 4)
        unpacked = pipeline_class._unpack_latents(packed, 32, 32, 8)
        decoded = vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
        output = processor.postprocess(decoded, output_type="np")
    torch.save(
        (
            pixels,
            source_latents,
            packed,
            torch.from_numpy((output * 255).round().astype(np.uint8)),
        ),
        out_path,
    )


def test_image_latents_decode_and_np_output(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    expected_pixels, expected_source, expected_noise, expected_output = reference_in_subprocess(
        "reference.test_ref_qwen_image_edit._reference_image_stages",
        {"root": str(root), "model_id": model_id},
        tmp_path / "image-stages.pt",
    )

    source = Image(torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8))
    width, height = _edit_dimensions(1024 * 1024, 96, 64)
    pixels = _vae_pixels(source, width, height)
    vae = AutoencoderKLQwenImage.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        encoded = vae.encode(pixels)
        mean = torch.tensor(vae.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.latents_std).view(1, -1, 1, 1, 1)
        source_latents = _pack_original_latents((encoded - mean) / std)
        noise = torch.randn((1, 1, 16, 4, 4), generator=torch.Generator("cpu").manual_seed(42))
        packed = _pack_original_latents(noise.permute(0, 2, 1, 3, 4))
        output = decode_original_latents(vae, packed, 32, 32)

    pixel_diff = max_diff(pixels, expected_pixels)
    source_diff = max_diff(source_latents, expected_source)
    noise_diff = max_diff(packed, expected_noise)
    output_diff = int((output.int() - expected_output.int()).abs().max())
    record(
        f"qwen_image_edit_image_stages_{model_id}",
        pixels_max_abs=pixel_diff["max_abs"],
        source_latents_max_abs=source_diff["max_abs"],
        seed_42_noise_max_abs=noise_diff["max_abs"],
        output_pixel_max_abs=output_diff,
        tolerance=0.0,
    )
    torch.testing.assert_close(pixels, expected_pixels, atol=0, rtol=0)
    torch.testing.assert_close(source_latents, expected_source, atol=0, rtol=0)
    torch.testing.assert_close(packed, expected_noise, atol=0, rtol=0)
    assert output_diff == 0


def _reference_scheduler_stages(root: str, out_path: str) -> None:
    import numpy as np
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift

    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        f"{root}/scheduler", local_files_only=True
    )
    config = scheduler.config
    mu = calculate_shift(
        4,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(2, device="cpu", sigmas=np.linspace(1.0, 0.5, 2), mu=mu)
    torch.save((scheduler.sigmas, scheduler.timesteps), out_path)


def test_scheduler_stages(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    expected_sigmas, expected_timesteps = reference_in_subprocess(
        "reference.test_ref_qwen_image_edit._reference_scheduler_stages",
        {"root": str(root)},
        tmp_path / "scheduler-stages.pt",
    )
    config = read_json(root / "scheduler/scheduler_config.json")
    scheduler = FlowMatchEulerDiscreteScheduler(config)
    scheduler.set_timesteps(
        2,
        sigmas=torch.linspace(1.0, 0.5, 2, dtype=torch.float64).tolist(),
        mu=qwen_image_mu(32, 32, config),
    )
    sigma_diff = max_diff(scheduler.sigmas, expected_sigmas)
    timestep_diff = max_diff(scheduler.timesteps, expected_timesteps)
    record(
        f"qwen_image_edit_scheduler_{model_id}",
        sigmas_max_abs=sigma_diff["max_abs"],
        timesteps_max_abs=timestep_diff["max_abs"],
        tolerance=0.0,
    )
    torch.testing.assert_close(scheduler.sigmas, expected_sigmas, atol=0, rtol=0)
    torch.testing.assert_close(scheduler.timesteps, expected_timesteps, atol=0, rtol=0)


def _reference_conditioning_stages(root: str, model_id: str, out_path: str) -> None:
    from diffusers.image_processor import VaeImageProcessor

    plus = _is_plus(model_id)
    source = torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8)
    image = Image(source).to_pil()
    area = 384 * 384 if plus else 1024 * 1024
    width, height = _edit_dimensions(area, image.width, image.height)
    image = VaeImageProcessor(vae_scale_factor=16).resize(image, height, width)
    text = _reference_text(PROMPT, 1, plus=plus)
    processor = transformers.Qwen2VLProcessor.from_pretrained(
        f"{root}/processor", local_files_only=True
    )
    inputs = processor(text=[text], images=[image] if plus else image, return_tensors="pt")
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{root}/text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        hidden = model(**inputs, output_hidden_states=True).hidden_states[-1][:, _EDIT_DROP_INDEX:]
    torch.save(
        (
            inputs["input_ids"],
            inputs["pixel_values"],
            inputs["image_grid_thw"],
            hidden,
        ),
        out_path,
    )


def test_real_resolution_conditioning(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    plus = _is_plus(model_id)
    expected_ids, expected_pixels, expected_grid, expected_hidden = reference_in_subprocess(
        "reference.test_ref_qwen_image_edit._reference_conditioning_stages",
        {"root": str(root), "model_id": model_id},
        tmp_path / "conditioning-stages.pt",
    )
    source = Image(torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8))
    area = 384 * 384 if plus else 1024 * 1024
    width, height = _edit_dimensions(area, 96, 64)
    image = _resize_image(source, width, height)
    config = read_json(root / "processor/preprocessor_config.json")
    pixels, grid = _vision_pixels(image, config)
    text = _native_text(PROMPT, [grid], plus=plus)
    ids = torch.tensor([Qwen2Tokenizer(root / "processor").tokenize(text)])
    model = Qwen25VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        hidden = model(ids, torch.ones_like(ids), pixel_values=pixels, image_grid_thw=grid)[-1][
            :, _EDIT_DROP_INDEX:
        ]
    hidden_diff = max_diff(hidden, expected_hidden)
    record(
        f"qwen_image_edit_real_conditioning_{model_id}",
        ids_match=torch.equal(ids, expected_ids),
        pixels_max_abs=max_diff(pixels, expected_pixels)["max_abs"],
        hidden_max_abs=hidden_diff["max_abs"],
        tolerance=0.0,
    )
    assert torch.equal(ids, expected_ids)
    assert torch.equal(pixels, expected_pixels)
    assert torch.equal(grid, expected_grid)
    torch.testing.assert_close(hidden, expected_hidden, atol=0, rtol=0)


def _long_transformer_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator("cpu").manual_seed(5)
    hidden = torch.randn(1, 4060, 64, generator=generator)
    context = torch.randn(1, 1363, 3584, generator=generator)
    return hidden, context, torch.tensor([1.0])


def _reference_long_transformer_first_block(root: str, out_path: str) -> None:
    hidden, context, timestep = _long_transformer_inputs()
    model = diffusers.QwenImageTransformer2DModel.from_pretrained(
        f"{root}/transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        hidden = model.img_in(hidden)
        context = model.txt_in(model.txt_norm(context))
        temb = model.time_text_embed(timestep, hidden)
        rotary = model.pos_embed(
            [[(1, 2, 2), (1, 52, 78)]],
            max_txt_seq_len=context.shape[1],
            device=hidden.device,
        )
        context, hidden = model.transformer_blocks[0](hidden, context, None, temb, rotary)
    torch.save((context, hidden), out_path)


def test_long_transformer_first_block(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    expected_context, expected_hidden = reference_in_subprocess(
        "reference.test_ref_qwen_image_edit._reference_long_transformer_first_block",
        {"root": str(root)},
        tmp_path / "long-transformer-first-block.pt",
    )
    hidden, context, timestep = _long_transformer_inputs()
    model = QwenImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        hidden = model.img_in(hidden)
        context = model.txt_in(model.txt_norm(context))
        temb = model.time_text_embed(timestep, hidden)
        rotary = model.pos_embed([[(1, 2, 2), (1, 52, 78)]], context.shape[1], hidden.device)
        context, hidden = model.transformer_blocks[0](hidden, context, None, temb, rotary)
    context_diff = max_diff(context, expected_context)
    hidden_diff = max_diff(hidden, expected_hidden)
    record(
        f"qwen_image_edit_long_transformer_first_block_{model_id}",
        context_max_abs=context_diff["max_abs"],
        hidden_max_abs=hidden_diff["max_abs"],
        tolerance=0.0,
    )
    torch.testing.assert_close(context, expected_context, atol=0, rtol=0)
    torch.testing.assert_close(hidden, expected_hidden, atol=0, rtol=0)


def _load_reference_pipeline(root: str, model_id: str):
    """Build the public diffusion pipeline after prompt embeddings have been computed."""
    transformer = diffusers.QwenImageTransformer2DModel.from_pretrained(
        f"{root}/transformer", local_files_only=True, torch_dtype=torch.float32
    )
    vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
        f"{root}/vae", local_files_only=True, torch_dtype=torch.float32
    )
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        f"{root}/scheduler", local_files_only=True
    )
    return _reference_pipeline_class(model_id)(
        scheduler=scheduler,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        processor=None,
        transformer=transformer,
    )


def _reference_prompt_embeddings(
    root: str, images: list, prompts: list[str], model_id: str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    plus = _is_plus(model_id)
    processor = transformers.Qwen2VLProcessor.from_pretrained(
        f"{root}/processor", local_files_only=True
    )
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{root}/text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    area = 384 * 384 if plus else 1024 * 1024
    images = [
        _resize_image(image, *_edit_dimensions(area, image.width, image.height)) for image in images
    ]
    result = []
    with torch.inference_mode():
        for prompt in prompts:
            text = _reference_text(prompt, len(images), plus=plus)
            inputs = processor(
                text=[text], images=images if plus else images[0], return_tensors="pt"
            )
            hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
            result.append(
                (hidden[:, _EDIT_DROP_INDEX:], inputs["attention_mask"][:, _EDIT_DROP_INDEX:])
            )
    return result


def _reference_e2e(root: str, seed: int, model_id: str, out_path: str) -> None:
    import numpy as np

    plus = _is_plus(model_id)
    source = torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8)
    images = [Image(source).to_pil()]
    if plus:
        images.append(Image(source.flip(1)).to_pil())
    (prompt_embeds, prompt_mask), (negative_prompt_embeds, negative_mask) = (
        _reference_prompt_embeddings(root, images, [PROMPT, NEGATIVE_PROMPT], model_id)
    )
    gc.collect()
    pipe = _load_reference_pipeline(root, model_id)
    pipe.set_progress_bar_config(disable=True)
    with torch.inference_mode():
        result = pipe(
            image=images if plus else images[0],
            prompt=None,
            negative_prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_mask,
            true_cfg_scale=4.0,
            num_inference_steps=2,
            width=32,
            height=32,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="np",
        ).images
    torch.save(torch.from_numpy((result * 255).round().astype(np.uint8)), out_path)


def _native_e2e(root, seed, model_id):
    source = Image(torch.arange(64 * 96 * 3, dtype=torch.int64).reshape(64, 96, 3).to(torch.uint8))
    images = [source]
    if _is_plus(model_id):
        images.append(Image(source.tensor.flip(1)))
    return (
        pipeline("t2i", backend=model_id, model=root)(
            PROMPT,
            image=images if _is_plus(model_id) else source,
            negative_prompt=NEGATIVE_PROMPT,
            true_cfg=4.0,
            steps=2,
            width=32,
            height=32,
            seed=seed,
        )
        .images[0]
        .tensor
    )


def test_end_to_end(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    expected = reference_in_subprocess(
        "reference.test_ref_qwen_image_edit._reference_e2e",
        {"root": str(root), "seed": 0, "model_id": model_id},
        tmp_path / "reference-0.pt",
    )[0]
    actual = _native_e2e(root, 0, model_id)
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    passed = pixel_diff == 0
    record(f"e2e_{model_id}", pixel_max_abs=pixel_diff, passed=passed, tolerance=0)
    assert passed


def test_seed_end_to_end(record, tmp_path, model_id):
    root = snapshot_dir(model_id)
    expected = reference_in_subprocess(
        "reference.test_ref_qwen_image_edit._reference_e2e",
        {"root": str(root), "seed": 42, "model_id": model_id},
        tmp_path / "reference-42.pt",
    )[0]
    actual = _native_e2e(root, 42, model_id)
    pixel_diff = int((actual.int() - expected.int()).abs().max())
    passed = pixel_diff == 0
    record(f"seed_e2e_{model_id}", pixel_max_abs=pixel_diff, passed=passed, tolerance=0)
    assert passed
