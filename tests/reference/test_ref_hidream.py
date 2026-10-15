"""HiDream-I1 Fast parity on CPU float32 against the real checkpoint.

The snapshot is a ``hidream-i1-fast`` download: the published repository plus the Llama 3.1
encoder and tokenizer under ``text_encoder_4``. The float32 transformer needs about 68 GB, so every
reference result is computed, saved to disk and freed before the library's side is loaded. Both
the text encoders and the transformer are never held next to each other, and the reference
pipeline is run in two stages with its own ``encode_prompt`` output fed to its own ``__call__``.
"""

import gc
import json
import struct

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub import pipeline
from imagehub.backends import hidream as hidream_backend
from imagehub.backends.hidream import HiDreamComponents, decode_latents, encode_prompts
from imagehub.catalog import model_info
from imagehub.checkpoints import read_json
from imagehub.models.autoencoder_kl import AutoencoderKL
from imagehub.models.clip_text import CLIPTextModelWithProjection
from imagehub.models.hidream_transformer import HiDreamImageTransformer2DModel
from imagehub.models.llama_text import LlamaTextModel
from imagehub.models.t5 import T5EncoderModel
from imagehub.schedulers.flow_match_lcm import FlowMatchLCMScheduler
from imagehub.schedulers.unipc import UniPCMultistepScheduler

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("numpy")

CPU = torch.device("cpu")
MODEL_IDS = ("hidream-i1-fast", "hidream-i1-dev", "hidream-i1-full")
PROMPT = 'A cat holding a sign that says "Hi-Dreams.ai".'
SEED_PROMPT = "a photograph of an astronaut riding a horse on the moon, 35mm film"
MAX_LENGTH = 128
REDUCED_STEPS = 4
TOKENIZER_PROMPTS = [
    "",
    PROMPT,
    SEED_PROMPT,
    "naïve café 日本語のテキスト 🐱 12345678 it's WE'LL",
    "line1\nline2\r\n\r\n  two  spaces <|begin_of_text|> <|eot_id|> </s> <pad>",
    "a long prompt " * 80,
]
_EMBEDDINGS: dict[str, object] = {}


def _check(record, name, actual, expected):
    measured = max_diff(actual, expected)
    record(name, **measured, tolerance=0.0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _check_pixels(record, name, actual, expected, **extra):
    passed = torch.equal(actual, expected)
    record(
        name,
        passed=passed,
        max_pixel_diff=int((actual.to(torch.int16) - expected.to(torch.int16)).abs().max()),
        **extra,
    )
    assert passed


def _checkpoint_shapes(directory):
    index = next(directory.glob("*.safetensors.index.json"), None)
    if index is not None:
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = [directory / name for name in sorted(set(weight_map.values()))]
    else:
        files = [next(directory.glob("*.safetensors"))]
    shapes = {}
    for path in files:
        with path.open("rb") as source:
            (length,) = struct.unpack("<Q", source.read(8))
            header = json.loads(source.read(length))
        shapes |= {
            name: tuple(entry["shape"]) for name, entry in header.items() if name != "__metadata__"
        }
    return shapes


def _reference_llama(root):
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "text_encoder_4")
    encoder = transformers.LlamaForCausalLM.from_pretrained(
        root / "text_encoder_4",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.float32,
    )
    return tokenizer, encoder


def _reference_embeddings(root, prompt, tmp_path_factory):
    """The reference pipeline's ``encode_prompt`` output for ``prompt`` (saved to disk)."""
    if prompt not in _EMBEDDINGS:
        tokenizer, encoder = _reference_llama(root)
        text_pipe = diffusers.HiDreamImagePipeline.from_pretrained(
            root,
            transformer=None,
            vae=None,
            tokenizer_4=tokenizer,
            text_encoder_4=encoder,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        with torch.inference_mode():
            t5, _, llama, _, pooled, _ = text_pipe.encode_prompt(
                prompt, do_classifier_free_guidance=False, max_sequence_length=MAX_LENGTH
            )
        path = tmp_path_factory.mktemp("embeddings") / "embeddings.pt"
        torch.save({"t5": t5, "llama": llama, "pooled": pooled}, path)
        del text_pipe, encoder, tokenizer, t5, llama, pooled
        gc.collect()
        _EMBEDDINGS[prompt] = path
    return torch.load(_EMBEDDINGS[prompt], weights_only=True)


@pytest.fixture(scope="module", params=MODEL_IDS)
def model_id(request):
    return request.param


@pytest.fixture(scope="module")
def root(model_id):
    return snapshot_dir(model_id)


def test_header(root, model_id, record):
    result = {}
    for component, cls in {
        "text_encoder": CLIPTextModelWithProjection,
        "text_encoder_2": CLIPTextModelWithProjection,
        "text_encoder_3": T5EncoderModel,
        "text_encoder_4": LlamaTextModel,
        "transformer": HiDreamImageTransformer2DModel,
        "vae": AutoencoderKL,
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
        not entry[key] for entry in result.values() for key in ("missing", "extra", "mismatch")
    )


def test_tokenizers(root, model_id, record):
    components = HiDreamComponents.load(root, device=CPU, dtype=torch.float32)
    llama_reference = transformers.AutoTokenizer.from_pretrained(root / "text_encoder_4")
    llama_reference.pad_token = llama_reference.eos_token
    references = {
        "tokenizer": transformers.CLIPTokenizer.from_pretrained(root / "tokenizer"),
        "tokenizer_2": transformers.CLIPTokenizer.from_pretrained(root / "tokenizer_2"),
        "tokenizer_3": transformers.T5Tokenizer.from_pretrained(root / "tokenizer_3"),
        "tokenizer_4": llama_reference,
    }
    for name, reference in references.items():
        ours = getattr(components, name)
        limit = min(
            MAX_LENGTH, 218 if name in ("tokenizer", "tokenizer_2") else reference.model_max_length
        )
        for prompt in TOKENIZER_PROMPTS:
            expected = reference(
                [prompt],
                padding="max_length",
                max_length=limit,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            if name in ("tokenizer", "tokenizer_2"):
                ours.max_length = limit
                ids = torch.tensor([ours(prompt)])
            else:
                ids, mask = ours([prompt], limit)
                assert torch.equal(mask, expected.attention_mask.bool()), (name, prompt)
            assert torch.equal(ids, expected.input_ids), (name, prompt)
    record(f"tokenizers_{model_id}", prompts=len(TOKENIZER_PROMPTS), identical=True)


def test_text_encoders(root, record, tmp_path_factory):
    expected = _reference_embeddings(root, PROMPT, tmp_path_factory)
    components = HiDreamComponents.load(root, device=CPU, dtype=torch.float32)
    pooled, t5, llama = encode_prompts(components, [PROMPT], MAX_LENGTH)
    _check(record, "hidream_pooled_clip", pooled, expected["pooled"])
    _check(record, "hidream_t5", t5, expected["t5"])
    _check(record, "hidream_llama_layers", llama, expected["llama"])


def test_transformer(root, record, tmp_path_factory, tmp_path):
    expected_embeddings = _reference_embeddings(root, PROMPT, tmp_path_factory)
    t5, llama, pooled = (expected_embeddings[key] for key in ("t5", "llama", "pooled"))
    channels = read_json(root / "transformer" / "config.json")["in_channels"]
    noise = torch.randn(1, channels, 128, 128, generator=torch.Generator("cpu").manual_seed(0))
    timesteps = [torch.tensor([1000.0]), torch.tensor([562.5])]

    reference = diffusers.HiDreamImageTransformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    with torch.inference_mode():
        expected = [
            reference(
                hidden_states=noise,
                timesteps=timestep,
                encoder_hidden_states_t5=t5,
                encoder_hidden_states_llama3=llama,
                pooled_embeds=pooled,
                return_dict=False,
            )[0]
            for timestep in timesteps
        ]
    output = tmp_path / "transformer.pt"
    torch.save(expected, output)
    del reference, expected
    gc.collect()

    native = HiDreamImageTransformer2DModel.from_pretrained(
        root / "transformer", device=CPU, dtype=torch.float32
    )
    expected = torch.load(output, weights_only=True)
    with torch.inference_mode():
        for index, (timestep, reference_output) in enumerate(zip(timesteps, expected, strict=True)):
            actual = native(noise, timestep, t5, llama, pooled)
            _check(record, f"hidream_transformer_t{index}", actual, reference_output)
    del native
    gc.collect()


def test_vae_decode(root, record):
    reference = diffusers.AutoencoderKL.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    latents = torch.randn(
        1,
        reference.config.latent_channels,
        128,
        128,
        generator=torch.Generator("cpu").manual_seed(3),
    )
    scaled = latents / reference.config.scaling_factor + reference.config.shift_factor
    with torch.inference_mode():
        expected = reference.decode(scaled, return_dict=False)[0]
    del reference
    gc.collect()
    native = AutoencoderKL.from_pretrained(root / "vae", device=CPU, dtype=torch.float32)
    with torch.inference_mode():
        actual = native.decode(latents / native.scaling_factor + (native.shift_factor or 0.0))
    _check(record, "hidream_vae_decode", actual, expected)
    pixels = decode_latents(native, latents)
    expected_pixels = ((expected / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1) * 255).round()
    assert torch.equal(pixels, expected_pixels.to(torch.uint8))


def test_scheduler(root, model_id, record):
    config = read_json(root / "scheduler" / "scheduler_config.json")
    if config["_class_name"] == "FlowMatchLCMScheduler":
        reference, ours = (
            diffusers.FlowMatchLCMScheduler.from_config(config),
            FlowMatchLCMScheduler(config),
        )
        reference.set_timesteps(16, mu=1.15)
        ours.set_timesteps(16, mu=1.15)
    else:
        reference, ours = (
            diffusers.UniPCMultistepScheduler.from_config(config),
            UniPCMultistepScheduler(config),
        )
        reference.set_timesteps(16)
        ours.set_timesteps(16)
    _check(record, "hidream_scheduler_timesteps", ours.timesteps, reference.timesteps)
    _check(record, "hidream_scheduler_sigmas", ours.sigmas, reference.sigmas)


def _reference_run(root, prompt, tmp_path_factory, *, guidance, **kwargs):
    """Run the reference pipeline from its own prompt embeddings; return latents and pixels."""
    embeddings = _reference_embeddings(root, prompt, tmp_path_factory)
    negative = _reference_embeddings(root, "", tmp_path_factory) if guidance > 1 else None
    reference = diffusers.HiDreamImagePipeline.from_pretrained(
        root,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        text_encoder_4=None,
        tokenizer_4=None,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    reference.set_progress_bar_config(disable=True)
    captured = {}

    def capture(_pipe, _index, _timestep, callback_kwargs):
        captured["latents"] = callback_kwargs["latents"].clone()
        return callback_kwargs

    seed = kwargs.pop("seed")
    # The scheduler draws its step noise from the global random state.
    torch.manual_seed(seed)
    images = reference(
        prompt_embeds_t5=embeddings["t5"],
        prompt_embeds_llama3=embeddings["llama"],
        pooled_prompt_embeds=embeddings["pooled"],
        negative_prompt_embeds_t5=None if negative is None else negative["t5"],
        negative_prompt_embeds_llama3=None if negative is None else negative["llama"],
        negative_pooled_prompt_embeds=None if negative is None else negative["pooled"],
        guidance_scale=guidance,
        generator=torch.Generator("cpu").manual_seed(seed),
        callback_on_step_end=capture,
        callback_on_step_end_tensor_inputs=["latents"],
        output_type="pt",
        **kwargs,
    ).images
    pixels = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1)
    del reference
    gc.collect()
    return {"latents": captured["latents"], "pixels": pixels}


def _our_run(root, model_id, monkeypatch, prompt, **options):
    captured = {}
    original = hidream_backend.denoise

    def capture(*args, **kwargs):
        captured["latents"] = original(*args, **kwargs)
        return captured["latents"]

    monkeypatch.setattr(hidream_backend, "denoise", capture)
    state = torch.get_rng_state()
    result = pipeline("t2i", backend=model_id, model=root)(prompt, **options)
    assert torch.equal(torch.get_rng_state(), state), "global random state was touched"
    monkeypatch.setattr(hidream_backend, "denoise", original)
    gc.collect()
    return {
        "latents": captured["latents"],
        "pixels": torch.stack([i.tensor for i in result.images]),
    }


def test_end_to_end(root, model_id, record, tmp_path, tmp_path_factory, monkeypatch):
    guidance = model_info(model_id).config["defaults"]["guidance"]
    expected = _reference_run(
        root, PROMPT, tmp_path_factory, guidance=guidance, seed=0, num_inference_steps=REDUCED_STEPS
    )
    output = tmp_path / "end-to-end.pt"
    torch.save(expected, output)
    del expected
    gc.collect()

    actual = _our_run(root, model_id, monkeypatch, PROMPT, steps=REDUCED_STEPS, seed=0)
    expected = torch.load(output, weights_only=True)
    _check(record, "hidream_e2e_latents", actual["latents"], expected["latents"])
    _check_pixels(
        record,
        f"e2e_{model_id.replace('-', '_')}",
        actual["pixels"],
        expected["pixels"],
        steps=REDUCED_STEPS,
        seed=0,
    )


def test_seed_end_to_end(root, model_id, record, tmp_path, tmp_path_factory, monkeypatch):
    guidance = model_info(model_id).config["defaults"]["guidance"]
    # A non-square request: the pipeline rescales it to about one megapixel (1248 x 832 here).
    size = dict(height=1024, width=1536)
    expected = _reference_run(
        root,
        SEED_PROMPT,
        tmp_path_factory,
        guidance=guidance,
        seed=42,
        num_inference_steps=REDUCED_STEPS,
        **size,
    )["pixels"]
    output = tmp_path / "seed.pt"
    torch.save(expected, output)
    del expected
    gc.collect()

    actual = _our_run(
        root, model_id, monkeypatch, SEED_PROMPT, steps=REDUCED_STEPS, seed=42, **size
    )["pixels"]
    expected = torch.load(output, weights_only=True)
    assert tuple(actual.shape[1:3]) == (832, 1248)
    _check_pixels(
        record,
        f"seed_e2e_{model_id.replace('-', '_')}",
        actual,
        expected,
        steps=REDUCED_STEPS,
        seed=42,
    )
