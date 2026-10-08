import pytest
import torch

from imagehub.models.clip_vision import CLIPVisionModelWithProjection
from imagehub.models.prior_transformer import PriorTransformer
from imagehub.schedulers.unclip import UnCLIPScheduler

pytestmark = pytest.mark.reference
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


VISION_TINY = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 16,
    "patch_size": 4,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-5,
    "attention_dropout": 0.0,
    "projection_dim": 24,
}
VISION_MEDIUM = VISION_TINY | {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "image_size": 32,
    "patch_size": 4,
    "projection_dim": 96,
}


@pytest.mark.parametrize("config", [VISION_TINY, VISION_MEDIUM], ids=["tiny", "medium"])
def test_clip_vision_random_weights(config):
    torch.manual_seed(0)
    ref_config = transformers.CLIPVisionConfig(**config)
    ref = transformers.CLIPVisionModelWithProjection(ref_config).eval()
    assert ref.config._attn_implementation == "sdpa"
    ours = CLIPVisionModelWithProjection(config).eval()
    ours.load_checkpoint(ref.state_dict())
    pixels = torch.randn(2, 3, config["image_size"], config["image_size"])
    with torch.inference_mode():
        actual_embeds, actual_last = ours(pixels)
        expected = ref(pixel_values=pixels)
    exact(actual_embeds, expected.image_embeds)
    exact(actual_last, expected.last_hidden_state)


PRIOR_TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 8,
    "num_layers": 2,
    "embedding_dim": 16,
    "num_embeddings": 7,
    "additional_embeddings": 4,
    "time_embed_dim": 16,
    "embedding_proj_dim": 16,
    "clip_embed_dim": 12,
}
PRIOR_MEDIUM = PRIOR_TINY | {
    "num_attention_heads": 8,
    "attention_head_dim": 16,
    "num_layers": 4,
    "embedding_dim": 96,
    "time_embed_dim": 128,
    "embedding_proj_dim": 80,
    "clip_embed_dim": 64,
}


@pytest.mark.parametrize("config", [PRIOR_TINY, PRIOR_MEDIUM], ids=["tiny", "medium"])
def test_prior_transformer_random_weights(config):
    torch.manual_seed(0)
    ref = diffusers.PriorTransformer(**config).eval()
    ours = PriorTransformer(config).eval()
    ours.load_checkpoint(ref.state_dict())
    hidden_states = torch.randn(2, config["embedding_dim"])
    proj_embedding = torch.randn(2, config["embedding_proj_dim"])
    encoder_hidden_states = torch.randn(2, config["num_embeddings"], config["embedding_dim"])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]])
    with torch.inference_mode():
        actual = ours(
            hidden_states,
            271,
            proj_embedding,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        expected = ref(
            hidden_states,
            271,
            proj_embedding,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        ).predicted_image_embedding
        actual_processed = ours.post_process_latents(actual)
        expected_processed = ref.post_process_latents(expected)
    exact(actual, expected)
    exact(actual_processed, expected_processed)


def test_unclip_scheduler_25_step_trajectory():
    config = {
        "num_train_timesteps": 1000,
        "variance_type": "fixed_small_log",
        "clip_sample": True,
        "clip_sample_range": 10.0,
        "prediction_type": "sample",
        "beta_schedule": "squaredcos_cap_v2",
    }
    ref = diffusers.UnCLIPScheduler(**config)
    ours = UnCLIPScheduler(config)
    ref.set_timesteps(25)
    ours.set_timesteps(25)
    exact(ours.timesteps, ref.timesteps)
    torch.manual_seed(7)
    expected = torch.randn(2, 64)
    actual = expected.clone()
    outputs = [torch.randn_like(actual) for _ in range(25)]
    ref_generator = torch.Generator(device="cpu").manual_seed(11)
    our_generator = torch.Generator(device="cpu").manual_seed(11)
    for index, timestep in enumerate(ref.timesteps):
        previous = ref.timesteps[index + 1] if index + 1 < len(ref.timesteps) else None
        expected = ref.step(
            outputs[index],
            int(timestep),
            expected,
            prev_timestep=None if previous is None else int(previous),
            generator=ref_generator,
        ).prev_sample
        actual = ours.step(
            outputs[index],
            int(timestep),
            actual,
            prev_timestep=None if previous is None else int(previous),
            generator=our_generator,
        )
        exact(actual, expected)
