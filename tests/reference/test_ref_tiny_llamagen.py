"""Tiny-snapshot LlamaGen end-to-end parity against the official sampling driver."""

import json
import shutil

import pytest
import torch
from helpers import TINY_LLAMAGEN_GPT, TINY_LLAMAGEN_VQ, write_tiny_llamagen
from llamagen_driver import left_pad, reference_captions, reference_sample

from imagehub import pipeline
from imagehub.backends.llamagen import LlamaGenComponents, encode_prompts, generate, to_uint8

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
sentencepiece = pytest.importorskip("sentencepiece")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
np = pytest.importorskip("numpy")

PROMPT = "A cat on a mat, an astronaut riding a horse"
IMAGE_SIZE = 64
LATENT = IMAGE_SIZE // 16


def _write_spiece(directory):
    """Replace the fixture tokenizer with a trained SentencePiece model (both loaders read it)."""
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (directory / name).unlink()
    corpus = directory / "corpus.txt"
    corpus.write_text(
        "a cat on a mat\na blurry photograph\nan astronaut riding a horse\n", encoding="utf-8"
    )
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(directory / "spiece"),
        vocab_size=64,
        model_type="unigram",
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    corpus.unlink()
    (directory / "spiece.vocab").unlink()
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "add_prefix_space": True,
                "eos_token": "</s>",
                "extra_ids": 100,
                "legacy": True,
                "model_max_length": 300,
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            }
        )
    )
    (directory / "special_tokens_map.json").write_text(
        json.dumps({"eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"})
    )


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, llamagen_reference):
    gpt, generate_module, vq = llamagen_reference
    root = write_tiny_llamagen(
        tmp_path_factory.mktemp("llamagen") / "snapshot", image_size=IMAGE_SIZE
    )
    _write_spiece(root / "text_encoder")
    reference_gpt = gpt.Transformer(
        gpt.ModelArgs(**TINY_LLAMAGEN_GPT, block_size=LATENT**2, model_type="t2i")
    )
    reference_gpt.load_state_dict(
        torch.load(root / f"t2i_XL_stage1_{IMAGE_SIZE}.pt", weights_only=True)["model"]
    )
    reference_vq = vq.VQ_16(**TINY_LLAMAGEN_VQ)
    incompatible = reference_vq.load_state_dict(
        torch.load(root / "vq_ds16_t2i.pt", weights_only=True)["model"], strict=False
    )
    assert not incompatible.unexpected_keys
    assert all(
        name.startswith(("encoder.", "quant_conv.", "quantize.codebook_used"))
        for name in incompatible.missing_keys
    )
    components = LlamaGenComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    return root, generate_module, reference_gpt.eval(), reference_vq.eval(), components


def _reference(tiny, tmp_path, prompt=PROMPT, **options):
    root, generate_module, reference_gpt, reference_vq, _ = tiny
    _, embeddings, masks = reference_captions(root / "text_encoder", [prompt])
    c_indices, c_masks = left_pad(embeddings, masks)
    return reference_sample(
        generate_module,
        reference_gpt,
        reference_vq,
        c_indices,
        c_masks,
        latent_size=LATENT,
        embed_dim=TINY_LLAMAGEN_VQ["codebook_embed_dim"],
        output=tmp_path / "sample.png",
        **options,
    )


def _check(tiny, tmp_path, record, name, *, seed=0, prompt=PROMPT, **options):
    root, _, _, _, components = tiny
    expected_tokens, expected_pixels = _reference(tiny, tmp_path, prompt, seed=seed, **options)
    embeddings, masks = encode_prompts(components, [prompt], torch.device("cpu"))
    defaults = {"cfg_scale": 7.5, "top_k": 1000}
    tokens = generate(
        components.gpt,
        embeddings,
        LATENT**2,
        masks,
        generator=torch.Generator("cpu").manual_seed(seed),
        **(defaults | options),
    )
    assert torch.equal(tokens, expected_tokens)
    assert len(set(tokens.flatten().tolist())) > 1

    result = pipeline("t2i", backend="llamagen", model=root)(prompt, seed=seed, **options)
    pixels = result.images[0].tensor
    assert torch.equal(pixels, expected_pixels)
    assert expected_pixels.float().std() > 1.0
    record(name, tokens_equal=True, pixels_equal=True, seed=seed, **options)


def test_defaults(tiny, tmp_path, record):
    _check(tiny, tmp_path, record, "llamagen_tiny_default")


def test_pipeline_defaults_match_the_script_defaults(tiny, tmp_path):
    root, *_ = tiny
    _, expected = _reference(tiny, tmp_path, seed=3)
    result = pipeline("t2i", backend="llamagen", model=root)(PROMPT, seed=3)
    assert torch.equal(result.images[0].tensor, expected)
    assert result.metadata["cfg_scale"] == 7.5 and result.metadata["top_k"] == 1000


def test_guidance_one_disables_cfg(tiny, tmp_path, record):
    _check(tiny, tmp_path, record, "llamagen_tiny_cfg_1", cfg_scale=1.0)


def test_top_k_and_temperature(tiny, tmp_path, record):
    _check(tiny, tmp_path, record, "llamagen_tiny_top_k", top_k=5, temperature=0.7, cfg_scale=3.0)


def test_top_p(tiny, tmp_path, record):
    _check(tiny, tmp_path, record, "llamagen_tiny_top_p", top_p=0.6, top_k=0, temperature=1.3)


def test_top_k_and_top_p_together(tiny, tmp_path, record):
    _check(tiny, tmp_path, record, "llamagen_tiny_top_k_top_p", top_k=20, top_p=0.8)


def test_cfg_interval(tiny, tmp_path, record):
    _check(tiny, tmp_path, record, "llamagen_tiny_cfg_interval", cfg_interval=4, cfg_scale=5.0)


@pytest.mark.parametrize("seed", [0, 123456789])
def test_seeds(tiny, tmp_path, record, seed):
    _check(tiny, tmp_path, record, f"llamagen_tiny_seed_{seed}", seed=seed)


def test_to_uint8_matches_save_image(tiny, tmp_path):
    utils = pytest.importorskip("torchvision.utils")
    pil_image = pytest.importorskip("PIL.Image")
    generator = torch.Generator().manual_seed(0)
    samples = torch.randn(1, 3, 64, 64, generator=generator) * 1.2
    utils.save_image(samples, tmp_path / "x.png", nrow=4, normalize=True, value_range=(-1, 1))
    expected = torch.from_numpy(np.array(pil_image.open(tmp_path / "x.png").convert("RGB")))
    assert torch.equal(to_uint8(samples)[0], expected)
    shutil.rmtree(tmp_path / "x.png", ignore_errors=True)
