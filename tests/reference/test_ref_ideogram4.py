"""Ideogram 4 fp8 checkpoint parity on CPU float32."""

import json
import multiprocessing
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends.ideogram4 import Ideogram4Components
from imagehub.models.autoencoder_ideogram4 import AutoencoderIdeogram4
from imagehub.models.fp8 import FP8_SCALE_SUFFIX, FP8_WEIGHT_DTYPE
from imagehub.models.ideogram4_text import build_text_inputs, encode_text, tokenize_prompt
from imagehub.schedulers.ideogram4 import PRESETS, get_schedule_for_resolution, make_step_intervals
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
ideogram4 = pytest.importorskip("ideogram4")
numpy = pytest.importorskip("numpy")
reference_pipeline = pytest.importorskip("ideogram4.pipeline_ideogram4")
reference_scheduler = pytest.importorskip("ideogram4.scheduler")
transformers = pytest.importorskip("transformers")

MODEL_ID = "ideogram4-fp8"
PROMPT = (
    '{"high_level_description":"An astronaut riding a horse",'
    '"style_description":{"aesthetics":"cinematic","lighting":"daylight",'
    '"photo":"editorial","medium":"digital","color_palette":["#88AACC"]},'
    '"compositional_deconstruction":{"background":"mountains","elements":[]}}'
)


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _headers(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    files = (
        [
            directory / name
            for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))
        ]
        if index is not None
        else list(directory.glob("*.safetensors"))
    )
    result = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        result |= {name: entry for name, entry in header.items() if name != "__metadata__"}
    return result


def _load_reference(root):
    def local_file(repo_id, filename, **kwargs):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    reference_pipeline.hf_hub_download = local_file
    config = reference_pipeline.Ideogram4PipelineConfig(weights_repo=str(root))
    return reference_pipeline.Ideogram4Pipeline.from_pretrained(
        config=config, device="cpu", dtype=torch.float32
    )


def _in_subprocess(function, *args):
    """Run a reference computation in a fresh process.

    The official pipeline peaks near 52 GB while loading and the native one near 40 GB; memory
    freed in one process is not returned to the system, so the reference runs in its own process.
    """
    context = multiprocessing.get_context("spawn")
    # Same intra-op thread count as this process (set by the reference conftest): the CPU
    # kernels split their sums by thread, so a different count changes the results.
    with context.Pool(
        1, initializer=torch.set_num_threads, initargs=(torch.get_num_threads(),)
    ) as pool:
        return pool.apply(function, args)


def _reference_text(root, inputs, output):
    reference = _load_reference(root)
    tokens = reference._tokenize(PROMPT)
    features = reference._encode_text(
        inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
    )
    torch.save({"tokens": tokens, "features": features}, output)


def _reference_step(root, inputs, conditional, unconditional, output):
    reference = _load_reference(root)
    features = reference._encode_text(
        inputs["token_ids"], inputs["text_position_ids"], inputs["indicator"]
    )
    conditional = dict(conditional, llm_features=features)
    image = slice(int(inputs["max_text_tokens"]), None)
    unconditional = dict(unconditional, llm_features=torch.zeros_like(features[:, image]))
    with torch.inference_mode():
        torch.save(
            {
                "features": features,
                "conditional": reference.conditional_transformer(**conditional),
                "unconditional": reference.unconditional_transformer(**unconditional),
            },
            output,
        )


def _reference_image(root, preset_name, seed, output):
    reference = _load_reference(root)
    preset = PRESETS[preset_name]
    image = reference(
        PROMPT,
        height=1024,
        width=1024,
        num_steps=preset.num_steps,
        guidance_schedule=preset.guidance_schedule,
        mu=preset.mu,
        std=preset.std,
        seed=seed,
    )[0]
    torch.save(torch.from_numpy(numpy.array(image)), output)


def _local_reference(root, monkeypatch):
    def local_file(repo_id, filename, **kwargs):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)

    monkeypatch.setattr(reference_pipeline, "hf_hub_download", local_file)
    config = reference_pipeline.Ideogram4PipelineConfig(weights_repo=str(root))
    return reference_pipeline.Ideogram4Pipeline.from_pretrained(
        config=config, device="cpu", dtype=torch.float32
    )


def test_ideogram4_headers_include_all_components_and_fp8_scales(record):
    root = snapshot_dir(MODEL_ID)
    result = {}
    for component in ("text_encoder", "transformer", "unconditional_transformer", "vae"):
        header = _headers(root / component)
        fp8 = [name for name, entry in header.items() if entry["dtype"] == "F8_E4M3"]
        scales = [name for name in header if name.endswith(FP8_SCALE_SUFFIX)]
        result[component] = {"tensors": len(header), "fp8": len(fp8), "scales": len(scales)}
        if component != "vae":
            assert fp8 and len(fp8) == len(scales)
    record("header_ideogram4-fp8", **result)


def test_ideogram4_tokens_and_text_features(record, tmp_path):
    root = snapshot_dir(MODEL_ID)
    tokenizer = Qwen2Tokenizer(root / "tokenizer")
    inputs = build_text_inputs(tokenizer, [PROMPT], 1024, 1024)
    _in_subprocess(_reference_text, root, inputs, tmp_path / "text.pt")

    saved = torch.load(tmp_path / "text.pt", weights_only=False)
    actual_tokens = tokenize_prompt(tokenizer, PROMPT)
    assert (
        torch.equal(actual_tokens[0], saved["tokens"][0]) and actual_tokens[1] == saved["tokens"][1]
    )
    native = Ideogram4Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    with torch.inference_mode():
        actual = encode_text(
            native.text_encoder,
            inputs["token_ids"],
            inputs["text_position_ids"],
            inputs["indicator"],
        )
    _check(record, "ideogram4_text_features", actual, saved["features"])


def test_ideogram4_transformers_single_step_1024(record, tmp_path):
    # Save-then-compare, as above; both sides get the reference's text features.
    root = snapshot_dir(MODEL_ID)
    tokenizer = Qwen2Tokenizer(root / "tokenizer")
    inputs = build_text_inputs(tokenizer, [PROMPT], 1024, 1024)
    generator = torch.Generator("cpu").manual_seed(0)
    noise = torch.randn(1, inputs["position_ids"].shape[1], 128, generator=generator)
    timestep = torch.tensor([0.75])
    image = slice(int(inputs["max_text_tokens"]), None)

    def arguments(features):
        conditional = dict(
            llm_features=features,
            x=noise,
            t=timestep,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        unconditional = dict(
            llm_features=torch.zeros_like(features[:, image]),
            x=noise[:, image],
            t=timestep,
            position_ids=inputs["position_ids"][:, image],
            segment_ids=inputs["segment_ids"][:, image],
            indicator=inputs["indicator"][:, image],
        )
        return conditional, unconditional

    conditional, unconditional = arguments(torch.empty(1, 0))
    conditional.pop("llm_features")
    unconditional.pop("llm_features")
    _in_subprocess(_reference_step, root, inputs, conditional, unconditional, tmp_path / "step.pt")

    saved = torch.load(tmp_path / "step.pt", weights_only=True)
    native = Ideogram4Components.load(root, device=torch.device("cpu"), dtype=torch.float32)
    conditional, unconditional = arguments(saved["features"])
    with torch.inference_mode():
        actual_conditional = native.conditional_transformer(**conditional)
        actual_unconditional = native.unconditional_transformer(**unconditional)
    _check(record, "ideogram4_transformer_conditional", actual_conditional, saved["conditional"])
    _check(
        record, "ideogram4_transformer_unconditional", actual_unconditional, saved["unconditional"]
    )


def test_ideogram4_autoencoder_decode_encode(record, monkeypatch):
    root = snapshot_dir(MODEL_ID)
    reference = _local_reference(root, monkeypatch)
    native = AutoencoderIdeogram4.from_pretrained(
        root / "vae", device=torch.device("cpu"), dtype=torch.float32
    )
    generator = torch.Generator("cpu").manual_seed(0)
    latents = torch.randn(1, 32, 16, 16, generator=generator)
    image = torch.randn(1, 3, 128, 128, generator=generator)
    _check(
        record,
        "ideogram4_vae_decode",
        native.decode(latents),
        reference.autoencoder.decoder(latents),
    )
    _check(
        record,
        "ideogram4_vae_encode",
        native.encode(image),
        reference.autoencoder.encoder(image).chunk(2, 1)[0],
    )


def test_ideogram4_sampler_intervals_exact(record):
    preset = PRESETS["V4_TURBO_12"]
    intervals = make_step_intervals(preset.num_steps)
    reference_intervals = reference_scheduler.make_step_intervals(preset.num_steps)
    schedule = get_schedule_for_resolution((1024, 1024), known_mean=preset.mu, std=preset.std)
    exact = torch.equal(intervals, reference_intervals)
    record("ideogram4_sampler_intervals", exact=exact, first=float(schedule(intervals[-1])))
    assert exact


@pytest.mark.parametrize("seed", [0, 42])
def test_ideogram4_end_to_end_1024(record, tmp_path, seed):
    root = snapshot_dir(MODEL_ID)
    _in_subprocess(_reference_image, root, "V4_TURBO_12", seed, tmp_path / f"expected-{seed}.pt")
    actual = (
        pipeline("t2i", backend="ideogram4", model=root)(PROMPT, preset="V4_TURBO_12", seed=seed)
        .images[0]
        .tensor
    )
    expected_tensor = torch.load(tmp_path / f"expected-{seed}.pt", weights_only=True)
    name = "e2e_ideogram4-fp8" if seed == 0 else "seed_e2e_ideogram4-fp8"
    _check(record, name, actual, expected_tensor)
    assert FP8_WEIGHT_DTYPE == torch.float8_e4m3fn
