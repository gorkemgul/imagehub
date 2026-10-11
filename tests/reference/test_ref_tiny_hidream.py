"""HiDream-I1 end to end on a tiny random snapshot in the published layout, against the pipeline."""

import pytest
import torch
from helpers import write_tiny_hidream

from imagehub import pipeline

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("numpy")

CASES = {
    "defaults": dict(prompt="a cat", steps=16, guidance=0.0, hw=(1024, 1024), seed=0, msl=128),
    "cfg-negative": dict(
        prompt="a b c", steps=3, guidance=3.5, hw=(512, 512), seed=1, msl=16, negative_prompt="ab"
    ),
    "non-square": dict(prompt="ab abc", steps=3, guidance=0.0, hw=(128, 192), seed=7, msl=24),
    "non-square-cfg-two-images": dict(
        prompt="cat hi",
        steps=2,
        guidance=2.0,
        hw=(192, 128),
        seed=11,
        msl=12,
        negative_prompt="bc",
        num_images=2,
    ),
}


def _load_reference(root):
    tokenizer = transformers.AutoTokenizer.from_pretrained(root / "text_encoder_4")
    encoder = transformers.LlamaForCausalLM.from_pretrained(
        root / "text_encoder_4",
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.float32,
    )
    ref = diffusers.HiDreamImagePipeline.from_pretrained(
        root,
        tokenizer_4=tokenizer,
        text_encoder_4=encoder,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    ref.set_progress_bar_config(disable=True)
    return ref


@pytest.fixture(scope="module")
def references(tmp_path_factory):
    result = {}
    for seed in (0, 1):
        root = write_tiny_hidream(tmp_path_factory.mktemp(f"hidream{seed}"), seed=seed)
        result[seed] = (root, _load_reference(root))
    return result


def _reference_call(ref, case, **extra):
    height, width = case["hw"]
    # The pipeline draws the step noise from the global random state.
    torch.manual_seed(case["seed"])
    return ref(
        case["prompt"],
        negative_prompt=case.get("negative_prompt"),
        num_inference_steps=case["steps"],
        guidance_scale=case["guidance"],
        height=height,
        width=width,
        num_images_per_prompt=case.get("num_images", 1),
        max_sequence_length=case["msl"],
        generator=torch.Generator("cpu").manual_seed(case["seed"]),
        **extra,
    )


@pytest.mark.parametrize("weights_seed", [0, 1])
@pytest.mark.parametrize("name", sorted(CASES))
def test_tiny_snapshot_matches_hidream_pipeline(references, name, weights_seed, record):
    case = CASES[name]
    root, ref = references[weights_seed]
    expected = _reference_call(ref, case, output_type="np").images
    expected = torch.from_numpy((expected * 255).round().astype("uint8"))
    height, width = case["hw"]
    state = torch.get_rng_state()
    result = pipeline("t2i", backend="hidream-i1-fast", model=root)(
        case["prompt"],
        negative_prompt=case.get("negative_prompt"),
        steps=case["steps"],
        guidance=case["guidance"],
        height=height,
        width=width,
        num_images=case.get("num_images", 1),
        max_sequence_length=case["msl"],
        seed=case["seed"],
    )
    assert torch.equal(torch.get_rng_state(), state), "global random state was touched"
    actual = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual, expected)
    record(f"hidream_tiny_{name}_{weights_seed}", max_abs_diff=0.0, tolerance="exact")


def test_tiny_snapshot_latents_match_hidream_pipeline(references):
    from imagehub.backends.hidream import HiDreamBackend, denoise, effective_size

    case = CASES["non-square-cfg-two-images"]
    root, ref = references[0]
    expected = _reference_call(ref, case, output_type="latent").images
    backend = HiDreamBackend(root).load()
    try:
        components = backend.components
        assert components is not None
        height, width = case["hw"]
        width, height = effective_size(width, height, 2)
        noise = torch.randn(
            (case["num_images"], 4, height // 2, width // 2),
            generator=torch.Generator("cpu").manual_seed(case["seed"]),
        )
        actual = denoise(
            components,
            prompt=case["prompt"],
            negative_prompt=case["negative_prompt"],
            steps=case["steps"],
            guidance=case["guidance"],
            latents=noise,
            max_sequence_length=case["msl"],
            noise_generator=torch.Generator("cpu").manual_seed(case["seed"]),
        )
    finally:
        backend.unload()
    assert torch.equal(actual, expected)


def test_tiny_snapshot_text_embeddings_match_hidream_pipeline(references):
    from imagehub.backends.hidream import HiDreamBackend, encode_prompts

    root, ref = references[1]
    prompt, negative = "a cat hi", "bc a"
    with torch.no_grad():
        t5, t5_negative, llama, llama_negative, pooled, pooled_negative = ref.encode_prompt(
            prompt,
            negative_prompt=negative,
            do_classifier_free_guidance=True,
            max_sequence_length=20,
        )
    backend = HiDreamBackend(root).load()
    try:
        assert backend.components is not None
        pooled_ours, t5_ours, llama_ours = encode_prompts(
            backend.components, [prompt, negative], 20
        )
    finally:
        backend.unload()
    assert torch.equal(pooled_ours[:1], pooled)
    assert torch.equal(t5_ours[:1], t5)
    assert torch.equal(llama_ours[:, :1], llama)
    assert torch.equal(pooled_ours[1:], pooled_negative)
    assert torch.equal(t5_ours[1:], t5_negative)
    assert torch.equal(llama_ours[:, 1:], llama_negative)
