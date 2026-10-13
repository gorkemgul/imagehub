import pytest
import torch

from imagehub.backends.ldm import LDMBackend, LDMUNet
from imagehub.models.ldm_bert import LDMBertModel
from imagehub.types import T2IRequest


def test_ldm_bert_has_published_weight_layout():
    config = {
        "vocab_size": 17,
        "max_position_embeddings": 7,
        "d_model": 8,
        "encoder_layers": 2,
        "encoder_ffn_dim": 16,
        "encoder_attention_heads": 2,
        "head_dim": 4,
    }
    model = LDMBertModel(config)
    assert "model.layers.0.self_attn.q_proj.weight" in model.state_dict()
    assert tuple(model(torch.tensor([[1, 2, 3]])).shape) == (1, 3, 8)


def test_ldm_unet_supplies_legacy_context_dimension():
    with torch.device("meta"):
        unet = LDMUNet(
            {
                "block_out_channels": [32, 32],
                "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
                "norm_num_groups": 32,
            }
        )
    assert unet.config["cross_attention_dim"] == 1280


def test_ldm_validates_options_before_loading(tmp_path):
    backend = LDMBackend(tmp_path)
    request = T2IRequest(prompt="cat")
    with pytest.raises(TypeError, match="Unknown LDM options"):
        backend.validate_options(request, {"unknown": 1})
    with pytest.raises(ValueError, match="multiples of 8"):
        backend.validate_options(request, {"width": 257})
    with pytest.raises(ValueError, match="eta"):
        backend.validate_options(request, {"eta": 0.1})
