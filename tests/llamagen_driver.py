"""A CPU reproduction of the official ``sample_t2i.py`` driver for parity tests.

It follows the script (caption embeddings from ``T5Embedder`` without text preprocessing, left
padding by flipping the masks, ``generate``, ``vq_model.decode_code`` and ``save_image``) without
the CUDA-only settings. The random seed is set right before sampling.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def reference_captions(text_dir: Path, prompts: list[str], max_length: int = 120):
    """Caption embeddings and masks as ``T5Embedder.get_text_embeddings`` returns them."""
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(text_dir)
    model = transformers.T5EncoderModel.from_pretrained(
        text_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    texts = [prompt.lower().strip() for prompt in prompts]
    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        )["last_hidden_state"].detach()
    return encoded["input_ids"], embeddings, encoded["attention_mask"]


def left_pad(caption_embs: torch.Tensor, emb_masks: torch.Tensor):
    """The driver's naive left padding; returns ``(c_indices, c_emb_masks)``."""
    new_emb_masks = torch.flip(emb_masks, dims=[-1])
    new_caption_embs = []
    for caption_emb, emb_mask in zip(caption_embs, emb_masks, strict=True):
        valid_num = int(emb_mask.sum().item())
        new_caption_embs.append(torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]]))
    new_caption_embs = torch.stack(new_caption_embs)
    return new_caption_embs * new_emb_masks[:, :, None], new_emb_masks


def reference_sample(
    generate_module,
    gpt_model,
    vq_model,
    c_indices,
    c_emb_masks,
    *,
    seed: int,
    latent_size: int,
    embed_dim: int,
    output: Path,
    cfg_scale: float = 7.5,
    temperature: float = 1.0,
    top_k: int = 1000,
    top_p: float = 1.0,
    cfg_interval: int = -1,
):
    """Token ids and the saved image of the first caption as uint8 ``(height, width, 3)``."""
    torchvision_utils = pytest.importorskip("torchvision.utils")
    pil_image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")
    torch.manual_seed(seed)
    index_sample = generate_module.generate(
        gpt_model,
        c_indices,
        latent_size**2,
        c_emb_masks,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        sample_logits=True,
    )
    qzshape = [len(c_indices), embed_dim, latent_size, latent_size]
    with torch.no_grad():
        samples = vq_model.decode_code(index_sample, qzshape)
    torchvision_utils.save_image(samples, output, nrow=4, normalize=True, value_range=(-1, 1))
    pixels = torch.from_numpy(np.array(pil_image.open(output).convert("RGB")))
    return index_sample, pixels
