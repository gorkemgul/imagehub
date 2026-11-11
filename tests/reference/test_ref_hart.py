"""Exact HART parity against the pinned official implementation and checkpoint."""

import gc
import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from helpers import snapshot_dir
from reference_sources import reference_source

from imagehub import pipeline
from imagehub.backends.hart import SYSTEM_PROMPT, HARTComponents, _diffusion_sample, encode_prompt
from imagehub.checkpoints import load_weights
from imagehub.models.hart import HARTBlock, HARTDecoder, HARTDiffusionMLP, HARTQuantizer
from imagehub.models.qwen2 import Qwen2Model
from imagehub.tokenization_qwen import Qwen2Tokenizer

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")

REFERENCE_ROOT = reference_source("hart")
PROMPT = "A blue Porsche 356 parked in front of a yellow brick wall."


@pytest.fixture(scope="module")
def hart_reference():
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"HART reference checkout not found at {REFERENCE_ROOT}")
    kernels = types.ModuleType("hart_backend.fused_kernels")

    def rms_norm(out, value, weight, epsilon, use_quant):
        assert not use_quant
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(-1, keepdim=True) + epsilon
        )
        out.copy_(weight * normalized.to(value.dtype))

    kernels.rms_norm = rms_norm
    backend = types.ModuleType("hart_backend")
    backend.fused_kernels = kernels
    sys.modules["hart_backend"] = backend
    sys.modules["hart_backend.fused_kernels"] = kernels
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        basic = importlib.import_module("hart.modules.networks.basic_hart")
        vae = importlib.import_module("hart.modules.networks.basic_vae")
        diffusion = importlib.import_module("hart.modules.diffusion.diffloss")
        quantizer = importlib.import_module(
            "hart.modules.models.autoencoder.quantize.var_quantize_multiple_res"
        )
        yield basic, vae, diffusion, quantizer
    finally:
        sys.path.remove(str(REFERENCE_ROOT))
        for name in list(sys.modules):
            if name == "hart_backend" or name.startswith(("hart_backend.", "hart.")):
                del sys.modules[name]


def test_random_qwen2_text_encoder_is_exact(record):
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "use_sliding_window": False,
        "max_position_embeddings": 128,
    }
    torch.manual_seed(0)
    reference = transformers.Qwen2Model(transformers.Qwen2Config(**config)).eval()
    ours = Qwen2Model(config).eval()
    ours.load_checkpoint(reference.state_dict())
    ids = torch.randint(0, 64, (2, 7))
    mask = torch.tensor([[1] * 7, [1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        expected = reference(ids, attention_mask=mask).last_hidden_state
        actual = ours(ids, mask.bool())
    passed = torch.equal(actual, expected)
    record("hart_qwen2_random", passed=passed, tolerance="exact")
    assert passed


def test_random_hart_block_is_exact(hart_reference, record):
    basic, _, _, _ = hart_reference
    config = {
        "embed_dim": 32,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "norm_eps": 1e-6,
        "patch_nums": (1, 2, 4),
        "context_token": 4,
    }

    def norm(size, **options):
        return basic.LlamaRMSNormFused(size, eps=config["norm_eps"], **options)

    torch.manual_seed(1)
    reference = basic.AdaLNSelfAttn(
        cond_dim=32,
        shared_aln=False,
        block_idx=0,
        embed_dim=32,
        norm_layer=norm,
        num_heads=4,
        mlp_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        last_drop_p=0.0,
        attn_l2_norm=False,
        flash_if_available=False,
        fused_if_available=False,
        mlp_type="llama",
        attn_type="llama",
        max_position_embeddings=32,
        patch_nums=(1, 2, 4),
        context_token=4,
        disable_aln=True,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
    ).eval()
    ours = HARTBlock(config, 0).eval()
    ours.load_state_dict(reference.state_dict())
    value = torch.randn(2, 4, 32)
    positions = torch.arange(4).repeat(2, 1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    reference.attn.kv_caching(True)
    ours.attn.kv_caching(True)
    with torch.no_grad():
        expected = reference(
            value,
            None,
            None,
            si=0,
            context_position_ids=positions,
            context_mask=mask,
        )
        actual = ours(value, stage=0, context_position_ids=positions)
        visual = torch.randn(2, 4, 32)
        expected_visual = reference(
            visual,
            None,
            None,
            si=1,
            context_position_ids=positions,
            context_mask=mask,
        )
        actual_visual = ours(visual, stage=1, context_position_ids=positions)
    passed = torch.equal(actual, expected) and torch.equal(actual_visual, expected_visual)
    record("hart_block_random", passed=passed, stages=2, tolerance="exact")
    assert passed


def test_random_diffusion_head_is_exact(hart_reference, record):
    _, _, diffusion, _ = hart_reference
    torch.manual_seed(2)
    reference = diffusion.SimpleMLPAdaLN(
        in_channels=8,
        model_channels=32,
        out_channels=16,
        z_channels=32,
        num_res_blocks=2,
    ).eval()
    ours = HARTDiffusionMLP(8, 32, 32, 2).eval()
    ours.load_state_dict(reference.state_dict())
    value = torch.randn(3, 8)
    timestep = torch.tensor([0, 100, 999])
    condition = torch.randn(3, 32)
    with torch.no_grad():
        expected = reference(value, timestep, condition)
        actual = ours(value, timestep, condition)
        reference_loss = diffusion.DiffLoss(8, 32, 2, 32, "8").eval()
        reference_loss.net.load_state_dict(ours.state_dict())
        sample_condition = torch.randn(4, 32)
        torch.manual_seed(5)
        with _cpu_cuda_shim():
            expected_sample = reference_loss.sample(sample_condition, cfg=4.5 * 12 / 13)
        holder = types.SimpleNamespace(
            config={"latent_channels": 8},
            diffloss=types.SimpleNamespace(net=ours),
        )
        actual_sample = _diffusion_sample(
            cast(Any, holder),
            sample_condition,
            4.5 * 12 / 13,
            torch.Generator().manual_seed(5),
        )
    passed = torch.equal(actual, expected) and torch.equal(actual_sample, expected_sample)
    record("hart_diffusion_mlp_random", passed=passed, tolerance="exact")
    assert passed


def test_random_autoencoder_decoder_is_exact(hart_reference, record):
    _, vae, _, _ = hart_reference
    config = {"ch": 32, "ch_mult": [1, 1, 2], "z_channels": 8, "dropout": 0.0}
    torch.manual_seed(3)
    reference = vae.Decoder(
        ch=32,
        ch_mult=(1, 1, 2),
        z_channels=8,
        dropout=0.0,
        num_res_blocks=2,
        in_channels=3,
        using_sa=True,
        using_mid_sa=True,
    ).eval()
    ours = HARTDecoder(config).eval()
    ours.load_state_dict(reference.state_dict())
    value = torch.randn(1, 8, 4, 4)
    with torch.no_grad():
        expected = reference(value)
        actual = ours(value)
    passed = torch.equal(actual, expected)
    record("hart_decoder_random", passed=passed, tolerance="exact")
    assert passed


def test_random_multiscale_quantizer_is_exact(hart_reference, record):
    _, _, _, quantizer = hart_reference
    reference = quantizer.VectorQuantizer2(
        vocab_size=32,
        Cvae=8,
        using_znorm=False,
        v_patch_nums=(1, 2, 4),
        quant_resi=0.5,
        share_quant_resi=4,
    ).eval()
    ours = HARTQuantizer(32, 8).eval()
    state = reference.state_dict()
    state.pop("ema_vocab_hit_SV")
    ours.load_state_dict(state, strict=False)
    generator = torch.Generator().manual_seed(4)
    reference_hat = torch.zeros(2, 8, 4, 4)
    actual_hat = reference_hat.clone()
    passed = True
    for stage, size in enumerate((1, 2)):
        tokens = torch.randn(2, 8, size, size, generator=generator)
        expected_hat, expected_next = reference.get_next_autoregressive_input(
            stage, 3, reference_hat, tokens, patch_nums=(1, 2, 4)
        )
        actual_hat, actual_next = ours.next_input(stage, (1, 2, 4), actual_hat, tokens)
        passed = passed and torch.equal(actual_hat, expected_hat)
        passed = passed and torch.equal(actual_next, expected_next)
    record("hart_quantizer_random", passed=passed, stages=2, tolerance="exact")
    assert passed


def test_real_checkpoint_headers_and_tokenizer(record, hart_reference):
    root = snapshot_dir("hart")
    llm = torch.load(root / "llm/ema_model.bin", map_location="cpu", weights_only=True, mmap=True)
    tokenizer_state = torch.load(
        root / "tokenizer/pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    components = HARTComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    assert set(components.transformer.state_dict()) == set(llm)
    prepared = components.autoencoder.prepare_state_dict(tokenizer_state)
    assert set(components.autoencoder.state_dict()) == set(prepared)
    ours = Qwen2Tokenizer(root / "text_encoder")
    reference = transformers.AutoTokenizer.from_pretrained(
        root / "text_encoder", local_files_only=True
    )
    cases = [PROMPT, "Café — 東京", "  spaced   text  ", "🐱"]
    matches = []
    for text in cases:
        ids, mask = ours(text, 300)
        expected = reference(
            text,
            max_length=300,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        matches.append(
            torch.equal(ids, expected.input_ids)
            and torch.equal(mask, expected.attention_mask.bool())
        )
    record(
        "header_hart",
        transformer_tensors=len(llm),
        autoencoder_inference_tensors=len(prepared),
        missing=[],
        extra=[],
        mismatch=[],
    )
    record("hart_tokenizer", cases=len(cases), passed=all(matches))
    assert all(matches)

    reference_model = (
        transformers.AutoModel.from_pretrained(root / "text_encoder", local_files_only=True)
        .float()
        .eval()
    )
    reference_encode = importlib.import_module("hart.utils").encode_prompts
    with torch.no_grad():
        _, expected_mask, expected_positions, expected_context = reference_encode(
            [PROMPT], reference_model, reference, 300, SYSTEM_PROMPT, True
        )
        actual_context, actual_mask, actual_positions = encode_prompt(
            components, PROMPT, torch.device("cpu"), True
        )
    text_passed = (
        torch.equal(actual_context, expected_context.float())
        and torch.equal(actual_mask, expected_mask)
        and torch.equal(actual_positions, expected_positions)
    )
    record("hart_text_encoder", passed=text_passed, tolerance="exact")
    assert text_passed


@contextmanager
def _cpu_cuda_shim():
    tensor_cuda, module_cuda = torch.Tensor.cuda, torch.nn.Module.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    try:
        yield
    finally:
        torch.Tensor.cuda, torch.nn.Module.cuda = tensor_cuda, module_cuda


def _official_image(root: Path, seed: int) -> torch.Tensor:
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        importlib.import_module("hart.modules.models.transformer")
        autoencoder = importlib.import_module(
            "hart.modules.models.autoencoder.hart_autoencoder_with_disc"
        )
        # Transformers 5 expects bookkeeping absent from the pinned HART wrapper.
        if not hasattr(autoencoder.HARTAutoEncoderWithDisc, "all_tied_weights_keys"):
            autoencoder.HARTAutoEncoderWithDisc.all_tied_weights_keys = {}
        encode = importlib.import_module("hart.utils").encode_prompts
        constants = importlib.import_module("hart.utils.constants")
        with _cpu_cuda_shim():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                root / "text_encoder", local_files_only=True
            )
            text_model = (
                transformers.AutoModel.from_pretrained(root / "text_encoder", local_files_only=True)
                .float()
                .eval()
            )
            model_config = transformers.AutoConfig.from_pretrained(
                root / "llm", local_files_only=True
            )
            model = transformers.AutoModel.from_config(model_config)
            model.load_state_dict(load_weights(root / "llm/ema_model.bin"))
            model.eval()
            _, mask, positions, context = encode(
                [PROMPT], text_model, tokenizer, 300, constants.llm_system_prompt, True
            )
            torch.manual_seed(seed)
            with torch.no_grad():
                image = model.autoregressive_infer_cfg(
                    B=1,
                    label_B=context,
                    cfg=4.5,
                    g_seed=seed,
                    more_smooth=False,
                    context_position_ids=positions,
                    context_mask=mask,
                )
            return image.mul(255).permute(0, 2, 3, 1).to(torch.uint8)
    finally:
        sys.path.remove(str(REFERENCE_ROOT))


@pytest.mark.parametrize(("seed", "record_name"), [(0, "e2e_hart"), (42, "seed_e2e_hart")])
def test_real_end_to_end(seed, record_name, record, hart_reference):
    root = snapshot_dir("hart")
    expected = _official_image(root, seed)
    gc.collect()
    with pipeline("t2i", backend="hart", model=root, device="cpu") as native:
        result = native(PROMPT, seed=seed)
    actual = result.images[0].tensor.unsqueeze(0)
    passed = torch.equal(actual, expected)
    record(record_name, seed=seed, pixels_equal=passed, passed=passed)
    assert passed
