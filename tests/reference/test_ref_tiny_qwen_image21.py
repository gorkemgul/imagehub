import pytest
import torch
from helpers import write_tiny_qwen_image21

from imagehub import pipeline
from imagehub.backends.qwen_image import QwenImageComponents, denoise

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "QwenImage21Pipeline"):
    pytest.skip(
        "needs the diffusers commit pinned for Qwen-Image 2.1 (.venv-ref-qwen-image-2.1)",
        allow_module_level=True,
    )
transformers = pytest.importorskip("transformers")


class _TextOnlyProcessor:
    def __init__(self, root):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(root, local_files_only=True)

    def apply_chat_template(self, messages, *, tokenize, return_dict):
        del tokenize, return_dict
        text = "".join(content["text"] for content in messages[0]["content"])
        rendered = f"<|im_start|>system\n{text}<|im_end|>\n"
        return [self.tokenizer.encode(rendered)]

    def __call__(self, *, text, padding, padding_side, return_tensors):
        self.tokenizer.padding_side = padding_side
        return self.tokenizer(
            text, padding=padding, return_tensors=return_tensors, add_special_tokens=True
        )


def _reference_pipeline(root):
    text_encoder = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        root / "text_encoder", local_files_only=True, torch_dtype=torch.float32
    )
    transformer = diffusers.QwenImage21Transformer2DModel.from_pretrained(
        root / "transformer", local_files_only=True, torch_dtype=torch.float32
    )
    vae = diffusers.AutoencoderKLQwenImage21.from_pretrained(
        root / "vae", local_files_only=True, torch_dtype=torch.float32
    )
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        root / "scheduler", local_files_only=True
    )
    reference = diffusers.QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=_TextOnlyProcessor(root / "processor"),
        transformer=transformer,
    )
    reference.set_progress_bar_config(disable=True)
    return reference


@pytest.mark.parametrize(
    "true_cfg,negative_prompt", [(1.0, None), (4.0, "blurry")], ids=["unguided", "guided"]
)
def test_tiny_qwen_image21_generation_matches_diffusers_exactly(
    tmp_path, true_cfg, negative_prompt
):
    root = write_tiny_qwen_image21(tmp_path / "tiny")
    reference = _reference_pipeline(root)
    raw = torch.randn(1, 1, 64, 2, 2, generator=torch.Generator("cpu").manual_seed(42))
    noise = raw.view(1, 64, 4).transpose(1, 2)
    expected_latents = reference(
        "a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg,
        num_inference_steps=2,
        height=32,
        width=32,
        latents=noise.clone(),
        output_type="latent",
        use_kv_cache=True,
    ).images
    expected_pixels = reference(
        "a cat",
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg,
        num_inference_steps=2,
        height=32,
        width=32,
        latents=noise.clone(),
        output_type="np",
        use_kv_cache=True,
    ).images
    expected_pixels = torch.from_numpy((expected_pixels * 255).round().astype("uint8"))

    components = QwenImageComponents.load(root, device=torch.device("cpu"), dtype=torch.float32)
    actual_latents = denoise(
        components,
        prompt="a cat",
        negative_prompt=negative_prompt,
        steps=2,
        true_cfg=true_cfg,
        width=32,
        height=32,
        latents=noise.clone(),
    )
    torch.testing.assert_close(actual_latents, expected_latents, rtol=0, atol=0)
    result = pipeline("t2i", backend="qwen-image", model=root)(
        "a cat",
        negative_prompt=negative_prompt,
        true_cfg=true_cfg,
        steps=2,
        width=32,
        height=32,
        latents=noise,
    )
    actual_pixels = torch.stack([image.tensor for image in result.images])
    assert torch.equal(actual_pixels, expected_pixels)
