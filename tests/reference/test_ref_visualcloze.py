"""VisualCloze preprocessing, prompt and pipeline parity with diffusers."""

import gc
import json
import struct

import pytest
import torch
from helpers import snapshot_dir, write_tiny_flux1

from imagehub import pipeline
from imagehub.backends import visualcloze as visualcloze_backend
from imagehub.backends.flux import FluxComponents, encode_prompt
from imagehub.backends.visualcloze import _layout_prompt, _prompt, preprocess_image
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModel
from imagehub.models.flux_transformer import FluxTransformer2DModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.tokenization import CLIPTokenizer
from imagehub.tokenization_t5 import T5Tokenizer

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
PILImage = pytest.importorskip("PIL.Image")


TINY = {
    "patch_size": 1,
    "in_channels": 384,
    "out_channels": 64,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 32,
    "pooled_projection_dim": 32,
    "guidance_embeds": True,
    "axes_dims_rope": [2, 2, 4],
}
CPU = torch.device("cpu")
PROMPT = "A small red cube beside a blue sphere."


def make(width: int, height: int, seed: int):
    data = torch.randint(
        0,
        256,
        (width * height * 3,),
        generator=torch.Generator().manual_seed(seed),
        dtype=torch.uint8,
    )
    return PILImage.frombytes("RGB", (width, height), bytes(data.tolist()))


def _tiny_root(tmp_path, name: str):
    return write_tiny_flux1(
        tmp_path / name,
        in_channels=384,
        out_channels=64,
        latent_channels=16,
        scheduler_config={"base_image_seq_len": 256, "max_image_seq_len": 4096},
    )


def _reference_pipeline(root, combined: bool = False):
    from diffusers import AutoencoderKL as ReferenceVAE
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers import (
        FluxTransformer2DModel as ReferenceTransformer,
    )
    from diffusers.pipelines.visualcloze.pipeline_visualcloze_combined import VisualClozePipeline
    from diffusers.pipelines.visualcloze.pipeline_visualcloze_generation import (
        VisualClozeGenerationPipeline,
    )
    from transformers import CLIPTextModel as ReferenceCLIP
    from transformers import CLIPTokenizer, T5TokenizerFast
    from transformers import T5EncoderModel as ReferenceT5

    cls = VisualClozePipeline if combined else VisualClozeGenerationPipeline
    result = cls(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler"),
        vae=ReferenceVAE.from_pretrained(root / "vae", torch_dtype=torch.float32),
        text_encoder=ReferenceCLIP.from_pretrained(
            root / "text_encoder", torch_dtype=torch.float32
        ),
        tokenizer=CLIPTokenizer.from_pretrained(root / "tokenizer"),
        text_encoder_2=ReferenceT5.from_pretrained(
            root / "text_encoder_2", torch_dtype=torch.float32
        ),
        tokenizer_2=T5TokenizerFast.from_pretrained(root / "tokenizer_2"),
        transformer=ReferenceTransformer.from_pretrained(
            root / "transformer", torch_dtype=torch.float32
        ),
    )
    result.set_progress_bar_config(disable=True)
    generation = result.generation_pipe if combined else result
    generation.image_processor.get_layout_prompt = lambda size: _layout_prompt(*size)
    return result


def _pixels(image):
    return torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(
        image.height, image.width, 3
    )


def _native_generation(root, monkeypatch, grid, content_prompt):
    captured = {}
    original = visualcloze_backend._decode_generation

    def capture(*args, **kwargs):
        captured["latents"] = args[1].clone()
        return original(*args, **kwargs)

    monkeypatch.setattr(visualcloze_backend, "_decode_generation", capture)
    result = pipeline("t2i", backend="visualcloze-384", model=root)(
        "task",
        image=grid,
        content_prompt=content_prompt,
        steps=2,
        guidance=30,
        max_sequence_length=8,
        seed=0,
        upsampling_strength=0,
    )
    return captured["latents"], torch.stack([image.tensor for image in result.images])


def _support_all_target_last_row(reference):
    """Backport the processor's all-target-row handling for installed reference releases."""
    original = reference.image_processor.preprocess_image

    def preprocess(grid, vae_scale_factor):
        if all(image is None for image in grid[-1]):
            replacement = [list(row) for row in grid]
            source = next(image for row in grid[:-1] for image in row if image is not None)
            replacement[-1][0] = PILImage.new("RGB", source.size)
            images, sizes, _ = original(replacement, vae_scale_factor)
            return images, sizes, [1] * len(grid[-1])
        return original(grid, vae_scale_factor)

    reference.image_processor.preprocess_image = preprocess


def test_transformer_with_visualcloze_channels_matches_diffusers_exactly(record):
    torch.manual_seed(0)
    reference = diffusers.FluxTransformer2DModel(**TINY).eval()
    native = FluxTransformer2DModel(TINY).eval()
    native.load_state_dict(reference.state_dict())
    hidden = torch.randn(1, 7, 384)
    context = torch.randn(1, 8, 32)
    pooled = torch.randn(1, 32)
    timestep = torch.tensor([0.5])
    ids = torch.zeros(7, 3)
    text_ids = torch.zeros(8, 3)
    guidance = torch.tensor([30.0])
    with torch.inference_mode():
        expected = reference(
            hidden, context, pooled, timestep, ids, text_ids, guidance, return_dict=False
        )[0]
        actual = native(hidden, context, pooled, timestep, ids, text_ids, guidance)
    assert torch.equal(actual, expected)
    record("visualcloze_transformer_random", exact=True, tolerance="exact")


def test_processor_preprocessing_and_official_layout_prompt(record):
    from diffusers.pipelines.visualcloze.visualcloze_utils import VisualClozeProcessor

    processor = VisualClozeProcessor(vae_scale_factor=16, resolution=384)
    grid = [[make(73, 41, 0), make(101, 77, 1)], [make(115, 69, 2), None]]
    expected, expected_sizes, expected_targets = processor.preprocess_image(grid, 8)
    actual, sizes, targets = preprocess_image(grid, resolution=384, vae_scale=8)
    assert sizes == expected_sizes and targets == expected_targets
    assert all(
        torch.equal(a, b)
        for row_a, row_b in zip(actual, expected, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )
    assert isinstance(processor.get_layout_prompt((2, 3)), tuple)
    assert (
        _layout_prompt(2, 3)
        == "A grid layout with 2 rows and 3 columns, displaying 6 images arranged side by side."
    )
    assert (
        _prompt("task", "target", 2, 3)
        == "A grid layout with 2 rows and 3 columns, displaying 6 images arranged side by side. "
        "task The last image of the last row depicts: target"
    )
    record("visualcloze_preprocess_and_prompt", exact=True, tolerance="exact")


def _checkpoint_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is not None:
        files = [
            directory / name
            for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))
        ]
    else:
        files = [next(directory.glob("*.safetensors"))]
    result = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result |= {
            name: tuple(value["shape"]) for name, value in header.items() if name != "__metadata__"
        }
    return result


def test_real_checkpoint_headers(record):
    root = snapshot_dir("visualcloze-384")
    result = {}
    for component, cls in {
        "text_encoder": CLIPTextModel,
        "text_encoder_2": T5EncoderModel,
        "transformer": FluxTransformer2DModel,
        "vae": AutoencoderKL,
    }.items():
        shapes = _checkpoint_shapes(root / component)
        with torch.device("meta"):
            model = cls(read_json(root / component / "config.json"))
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
    record("header_visualcloze-384", **result)
    assert all(
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tiny_generation_matches_patched_diffusers_exactly(tmp_path, record):
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        FluxTransformer2DModel,
    )
    from diffusers.pipelines.visualcloze.pipeline_visualcloze_generation import (
        VisualClozeGenerationPipeline,
    )
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    root = write_tiny_flux1(
        tmp_path / "visualcloze",
        in_channels=384,
        out_channels=64,
        latent_channels=16,
        scheduler_config={"base_image_seq_len": 256, "max_image_seq_len": 4096},
    )
    reference = VisualClozeGenerationPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler"),
        vae=AutoencoderKL.from_pretrained(root / "vae"),
        text_encoder=CLIPTextModel.from_pretrained(root / "text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(root / "tokenizer"),
        text_encoder_2=T5EncoderModel.from_pretrained(root / "text_encoder_2"),
        tokenizer_2=T5TokenizerFast.from_pretrained(root / "tokenizer_2"),
        transformer=FluxTransformer2DModel.from_pretrained(root / "transformer"),
    )
    reference.set_progress_bar_config(disable=True)
    reference.image_processor.get_layout_prompt = lambda size: _layout_prompt(*size)
    grid = [[make(32, 32, 0), make(32, 32, 1)], [make(32, 32, 2), None]]
    expected = reference(
        "task",
        None,
        grid,
        num_inference_steps=2,
        guidance_scale=30,
        max_sequence_length=8,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0][0]
    actual = (
        pipeline("t2i", backend="visualcloze-384", model=root)(
            "task",
            image=grid,
            steps=2,
            guidance=30,
            max_sequence_length=8,
            seed=0,
            upsampling_strength=0,
        )
        .images[0]
        .tensor
    )
    expected = torch.frombuffer(bytearray(expected.tobytes()), dtype=torch.uint8).reshape(
        expected.height, expected.width, 3
    )
    assert torch.equal(actual, expected)
    record("visualcloze_tiny_generation", exact=True, tolerance="exact")


@pytest.mark.parametrize("content_prompt", [None, "a red circle"])
def test_tiny_prompt_embeddings_match_patched_diffusers_exactly(tmp_path, content_prompt):
    root = _tiny_root(tmp_path, f"prompt-{content_prompt}")
    reference = _reference_pipeline(root)
    components = FluxComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        expected = reference.encode_prompt(
            _layout_prompt(2, 2),
            "task",
            content_prompt,
            device=torch.device("cpu"),
            max_sequence_length=8,
        )
        actual = encode_prompt(
            components,
            _prompt("task", content_prompt, 2, 2),
            torch.device("cpu"),
            1,
            8,
        )
    assert all(
        torch.equal(value, expected_value)
        for value, expected_value in zip(actual, expected, strict=True)
    )


def _generation_cases():
    return [
        ("one_target", [[make(32, 32, 0), make(32, 32, 1)], [make(32, 32, 2), None]], None),
        (
            "several_square_targets",
            [[make(32, 32, 0), make(32, 32, 1), make(32, 32, 2)], [make(32, 32, 3), None, None]],
            None,
        ),
        (
            "two_example_rows",
            [
                [make(32, 32, 0), make(32, 32, 1)],
                [make(32, 32, 2), make(32, 32, 3)],
                [make(32, 32, 4), None],
            ],
            None,
        ),
        (
            "non_square_target",
            [[make(48, 32, 0), make(48, 32, 1)], [make(48, 32, 2), None]],
            None,
        ),
        (
            "content_prompt",
            [[make(32, 32, 0), make(32, 32, 1)], [make(32, 32, 2), None]],
            "a red circle",
        ),
        ("all_targets", [[make(32, 32, 0), make(32, 32, 1)], [None, None]], None),
    ]


@pytest.mark.parametrize(("name", "grid", "content_prompt"), _generation_cases())
def test_tiny_generation_latents_and_pixels_match_patched_diffusers_exactly(
    tmp_path, monkeypatch, name, grid, content_prompt
):
    root = _tiny_root(tmp_path, name)
    reference = _reference_pipeline(root)
    if name == "all_targets":
        _support_all_target_last_row(reference)
    kwargs = {
        "task_prompt": "task",
        "content_prompt": content_prompt,
        "image": grid,
        "num_inference_steps": 2,
        "guidance_scale": 30,
        "max_sequence_length": 8,
    }
    expected_latents = reference(
        **kwargs, generator=torch.Generator("cpu").manual_seed(0), output_type="latent"
    ).images
    expected_pixels = torch.stack(
        [
            _pixels(image)
            for image in reference(
                **kwargs, generator=torch.Generator("cpu").manual_seed(0)
            ).images[0]
        ]
    )
    actual_latents, actual_pixels = _native_generation(root, monkeypatch, grid, content_prompt)
    assert torch.equal(actual_latents, expected_latents)
    assert torch.equal(actual_pixels, expected_pixels)


@pytest.mark.parametrize(
    ("strength", "height", "width", "content_prompt", "several_targets"),
    [
        (0.3, 32, 32, None, False),
        (1.0, 48, 32, None, False),
        (0.0, 48, 32, None, False),
        (0.3, 48, 32, "a red circle", False),
        (1.0, 48, 32, None, True),
    ],
)
def test_tiny_two_stage_matches_patched_diffusers_exactly(
    tmp_path, record, strength, height, width, content_prompt, several_targets
):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
    from diffusers.pipelines.visualcloze.pipeline_visualcloze_combined import VisualClozePipeline
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    root = write_tiny_flux1(
        tmp_path / f"visualcloze-{strength}",
        in_channels=384,
        out_channels=64,
        latent_channels=16,
        scheduler_config={"base_image_seq_len": 256, "max_image_seq_len": 4096},
    )
    reference = VisualClozePipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler"),
        vae=AutoencoderKL.from_pretrained(root / "vae"),
        text_encoder=CLIPTextModel.from_pretrained(root / "text_encoder"),
        tokenizer=CLIPTokenizer.from_pretrained(root / "tokenizer"),
        text_encoder_2=T5EncoderModel.from_pretrained(root / "text_encoder_2"),
        tokenizer_2=T5TokenizerFast.from_pretrained(root / "tokenizer_2"),
        transformer=FluxTransformer2DModel.from_pretrained(root / "transformer"),
    )
    reference.set_progress_bar_config(disable=True)
    reference.generation_pipe.image_processor.get_layout_prompt = lambda size: _layout_prompt(*size)
    grid = (
        [[make(32, 32, 0), make(32, 32, 1), make(32, 32, 2)], [make(32, 32, 3), None, None]]
        if several_targets
        else [[make(32, 32, 0), make(32, 32, 1)], [make(32, 32, 2), None]]
    )
    expected = reference(
        "task",
        content_prompt,
        grid,
        upsampling_height=height,
        upsampling_width=width,
        upsampling_strength=strength,
        num_inference_steps=2,
        guidance_scale=30,
        max_sequence_length=8,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0][0]
    actual = (
        pipeline("t2i", backend="visualcloze-384", model=root)(
            "task",
            image=grid,
            content_prompt=content_prompt,
            steps=2,
            guidance=30,
            max_sequence_length=8,
            seed=0,
            upsampling_strength=strength,
            upsampling_height=height,
            upsampling_width=width,
        )
        .images[0]
        .tensor
    )
    expected = torch.frombuffer(bytearray(expected.tobytes()), dtype=torch.uint8).reshape(
        expected.height, expected.width, 3
    )
    assert torch.equal(actual, expected)
    record(
        f"visualcloze_tiny_two_stage_{strength}_{height}x{width}_{content_prompt}_{several_targets}",
        exact=True,
        tolerance="exact",
    )


def _check_exact(record, name, actual, expected):
    exact = torch.equal(actual, expected)
    record(name, exact=exact, tolerance="exact")
    assert exact


def test_real_components_cpu_float32(record):
    from diffusers import AutoencoderKL as ReferenceVAE
    from diffusers import FluxTransformer2DModel as ReferenceTransformer
    from transformers import CLIPTextModel as ReferenceCLIP
    from transformers import CLIPTokenizer as ReferenceCLIPTokenizer
    from transformers import T5EncoderModel as ReferenceT5
    from transformers import T5TokenizerFast

    root = snapshot_dir("visualcloze-384")
    reference_tokenizer = ReferenceCLIPTokenizer.from_pretrained(
        root / "tokenizer", local_files_only=True
    )
    native_tokenizer = CLIPTokenizer(root / "tokenizer")
    reference_ids = reference_tokenizer(
        PROMPT, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    ).input_ids
    native_ids = torch.tensor([native_tokenizer(PROMPT)])
    _check_exact(record, "visualcloze_real_clip_tokens", native_ids, reference_ids)
    reference_clip = ReferenceCLIP.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_clip = reference_clip(reference_ids).pooler_output
    del reference_clip
    native_clip = CLIPTextModel.from_pretrained(
        root / "text_encoder", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        hidden = native_clip(native_ids)
        actual_clip = hidden[torch.arange(hidden.shape[0]), native_ids.int().argmax(dim=-1)]
    _check_exact(record, "visualcloze_real_clip", actual_clip, expected_clip)
    del native_clip, expected_clip, actual_clip
    gc.collect()

    reference_tokenizer_2 = T5TokenizerFast.from_pretrained(
        root / "tokenizer_2", local_files_only=True
    )
    native_tokenizer_2 = T5Tokenizer(root / "tokenizer_2")
    reference_ids_2 = reference_tokenizer_2(
        [PROMPT], padding="max_length", max_length=32, truncation=True, return_tensors="pt"
    ).input_ids
    native_ids_2, _ = native_tokenizer_2([PROMPT], 32)
    _check_exact(record, "visualcloze_real_t5_tokens", native_ids_2, reference_ids_2)
    reference_t5 = ReferenceT5.from_pretrained(
        root / "text_encoder_2", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_t5 = reference_t5(reference_ids_2).last_hidden_state
    del reference_t5
    native_t5 = T5EncoderModel.from_pretrained(
        root / "text_encoder_2", device=CPU, dtype=torch.float32
    )
    with torch.inference_mode():
        actual_t5 = native_t5(native_ids_2)[0]
    _check_exact(record, "visualcloze_real_t5", actual_t5, expected_t5)
    del native_t5, expected_t5, actual_t5
    gc.collect()

    config = read_json(root / "transformer" / "config.json")
    generator = torch.Generator("cpu").manual_seed(0)
    sequence = 4 * (384 // 16) ** 2
    hidden = torch.randn(1, sequence, config["in_channels"], generator=generator)
    context = torch.randn(1, 32, config["joint_attention_dim"], generator=generator)
    pooled = torch.randn(1, config["pooled_projection_dim"], generator=generator)
    image_ids = torch.zeros(sequence, 3)
    image_ids[:, 1:] = torch.cartesian_prod(torch.arange(24), torch.arange(48)).repeat(2, 1)
    text_ids = torch.zeros(32, 3)
    inputs = (hidden, context, pooled)
    ids = (image_ids, text_ids, torch.tensor([30.0]))
    timesteps = (0.25, 0.5, 0.75)
    reference_transformer = ReferenceTransformer.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_outputs = [
            reference_transformer(*inputs, torch.tensor([timestep]), *ids).sample
            for timestep in timesteps
        ]
    del reference_transformer
    gc.collect()
    native_transformer = FluxTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    for timestep, expected in zip(timesteps, expected_outputs, strict=True):
        with torch.inference_mode():
            actual = native_transformer(*inputs, torch.tensor([timestep]), *ids)
        _check_exact(record, f"visualcloze_real_transformer_{timestep}", actual, expected)
    del native_transformer
    gc.collect()

    image = torch.rand(1, 3, 384, 384, generator=generator) * 2 - 1
    latents = torch.randn(1, 16, 48, 48, generator=generator)
    reference_vae = ReferenceVAE.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected_encode = reference_vae.encode(image).latent_dist.mode()
        expected_decode = reference_vae.decode(latents).sample
    del reference_vae
    native_vae = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual_encode = native_vae.encode(image)
        actual_decode = native_vae.decode(latents)
    _check_exact(record, "visualcloze_real_vae_encode", actual_encode, expected_encode)
    _check_exact(record, "visualcloze_real_vae_decode", actual_decode, expected_decode)


def test_real_end_to_end_both_stages(record, monkeypatch):
    root = snapshot_dir("visualcloze-384")
    grid = [[make(32, 32, 0), make(32, 32, 1)], [make(32, 32, 2), None]]
    reference = _reference_pipeline(root, combined=True)
    seen = {}
    # Held here so both pipelines can be released before the native model loads.
    held = {"generation": reference.generation_pipe, "upsampling": reference.upsampling_pipe}

    def spy_upsampling(**options):
        seen["generated"] = list(options["image"])
        return held["upsampling"](**options)

    def keep_generation_latents(pipe, step, timestep, tensors):
        if pipe is held["generation"] and step == 7:
            seen["latents"] = tensors["latents"].clone()
        return {}

    reference.upsampling_pipe = spy_upsampling
    expected_upsampling = _pixels(
        reference(
            task_prompt=PROMPT,
            image=grid,
            num_inference_steps=8,
            guidance_scale=30,
            max_sequence_length=512,
            upsampling_strength=0.3,
            upsampling_height=512,
            upsampling_width=512,
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=keep_generation_latents,
        ).images[0][0]
    )
    expected_generation = _pixels(seen["generated"][0])
    expected_latents = seen["latents"]
    held.clear()
    del reference
    gc.collect()

    original = visualcloze_backend._decode_generation
    captured = {}

    def capture(*args, **options):
        captured["latents"] = args[1].clone()
        captured["generated"] = original(*args, **options)
        return captured["generated"]

    monkeypatch.setattr(visualcloze_backend, "_decode_generation", capture)
    upsampling = pipeline("t2i", backend="visualcloze-384", model=root)(
        PROMPT,
        image=grid,
        steps=8,
        guidance=30,
        max_sequence_length=512,
        seed=0,
        upsampling_strength=0.3,
        upsampling_height=512,
        upsampling_width=512,
    )
    _check_exact(
        record, "visualcloze_real_generation_latents", captured["latents"], expected_latents
    )
    _check_exact(
        record,
        "visualcloze_real_generation_pixels",
        _pixels(captured["generated"][0]),
        expected_generation,
    )
    _check_exact(
        record,
        "visualcloze_real_upsampling_pixels",
        upsampling.images[0].tensor,
        expected_upsampling,
    )
    record(
        "e2e_visualcloze-384",
        max_pixel_diff=0,
        passed=True,
        seed=0,
        steps=8,
        guidance=30,
        upsampling_strength=0.3,
        upsampling_size=[512, 512],
    )


def test_real_seed_without_upsampling_matches_reference(record):
    root = snapshot_dir("visualcloze-384")
    grid = [[make(32, 32, 0), make(32, 32, 1)], [make(32, 32, 2), None]]
    reference = _reference_pipeline(root, combined=True)
    expected = _pixels(
        reference(
            PROMPT,
            None,
            grid,
            num_inference_steps=4,
            guidance_scale=30,
            max_sequence_length=512,
            upsampling_strength=0,
            generator=torch.Generator("cpu").manual_seed(7),
        ).images[0][0]
    )
    del reference
    gc.collect()
    actual = (
        pipeline("t2i", backend="visualcloze-384", model=root)(
            PROMPT,
            image=grid,
            steps=4,
            guidance=30,
            max_sequence_length=512,
            seed=7,
            upsampling_strength=0,
        )
        .images[0]
        .tensor
    )
    _check_exact(record, "visualcloze_real_seed", actual, expected)
    record("seed_e2e_visualcloze-384", max_pixel_diff=0, passed=True, seed=7, steps=4)
