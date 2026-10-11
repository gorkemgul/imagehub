"""Published HunyuanDiT transformer checkpoint parity on CPU float32."""

import gc

import pytest
import torch
from helpers import max_diff, snapshot_dir

from imagehub.models.hunyuan_dit_transformer import (
    HunyuanDiT2DModel,
    get_2d_rotary_pos_embed,
    get_resize_crop_region_for_grid,
)

pytestmark = pytest.mark.reference
diffusers = pytest.importorskip("diffusers")


@pytest.mark.parametrize(
    "model_id", ["hunyuan-dit-v1.1-distilled", "hunyuan-dit-v1.2", "hunyuan-dit-v1.2-distilled"]
)
def test_hunyuan_dit_transformer_checkpoint_matches_reference(record, model_id):
    root = snapshot_dir(model_id) / "transformer"
    reference = diffusers.HunyuanDiT2DModel.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.float32
    ).eval()
    model = HunyuanDiT2DModel.from_pretrained(root, dtype=torch.float32)

    generator = torch.Generator("cpu").manual_seed(0)
    hidden_states = torch.randn(2, 4, 128, 128, generator=generator)
    encoder_hidden_states = torch.randn(2, 77, 1024, generator=generator)
    encoder_hidden_states_t5 = torch.randn(2, 256, 2048, generator=generator)
    text_embedding_mask = torch.ones(2, 77, dtype=torch.long)
    text_embedding_mask[1, 20:] = 0
    text_embedding_mask_t5 = torch.ones(2, 256, dtype=torch.long)
    text_embedding_mask_t5[0, 200:] = 0
    text_embedding_mask_t5[1, 80:] = 0
    image_meta_size = torch.tensor([[1024, 1024, 1024, 1024, 0, 0]] * 2, dtype=torch.float32)
    style = torch.zeros(2, dtype=torch.long)
    timestep = torch.full((2,), 500.0)
    grid_size = (64, 64)
    crops = get_resize_crop_region_for_grid(grid_size, 32)
    image_rotary_emb = get_2d_rotary_pos_embed(88, crops, grid_size)

    kwargs = {
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "text_embedding_mask": text_embedding_mask,
        "encoder_hidden_states_t5": encoder_hidden_states_t5,
        "text_embedding_mask_t5": text_embedding_mask_t5,
        "image_meta_size": image_meta_size,
        "style": style,
        "image_rotary_emb": image_rotary_emb,
    }
    with torch.inference_mode():
        expected = reference(hidden_states, **kwargs).sample
        actual = model(hidden_states, **kwargs)
    measured = max_diff(actual, expected)
    record(
        "hunyuan_dit_transformer"
        if model_id == "hunyuan-dit-v1.1-distilled"
        else f"hunyuan_dit_transformer_{model_id}",
        **measured,
        batch=2,
        height=1024,
        width=1024,
        timestep=500,
        tolerance=0.0,
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    del model, reference
    gc.collect()
