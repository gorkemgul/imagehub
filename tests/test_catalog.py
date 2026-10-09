import re

import pytest

from imagehub.catalog import MODEL_CATALOG, ModelSpec, SubRepo, available_models, model_info
from imagehub.types import Task


def test_catalog_contents():
    assert available_models() == {
        "t2i": (
            "amused-256",
            "amused-512",
            "auraflow-v0.3",
            "cogview4-6b",
            "dmd2-sdxl-1step",
            "dmd2-sdxl-4step",
            "flux1-dev",
            "flux1-kontext-dev",
            "flux1-krea-dev",
            "flux1-schnell",
            "flux2-klein-4b",
            "flux2-klein-9b",
            "hidream-i1-fast",
            "hunyuan-dit-v1.1-distilled",
            "hyper-sdxl-1step",
            "ideogram4-fp8",
            "janus-pro-1b",
            "janus-pro-7b",
            "kandinsky-2.2",
            "kolors",
            "krea2-turbo",
            "lcm-dreamshaper-v7",
            "longcat-image",
            "lumina-image-2.0",
            "pixart-alpha-xl-1024",
            "pixart-sigma-xl-1024",
            "playground-v2.5",
            "qwen-image-2.1",
            "sana-1600m-1024",
            "sana-600m-1024",
            "sd-turbo",
            "sd-v1-1",
            "sd-v1-2",
            "sd-v1-3",
            "sd-v1-4",
            "sd-v1-5",
            "sd-v2-1",
            "sd3-medium",
            "sd3.5-large",
            "sd3.5-large-turbo",
            "sd3.5-medium",
            "sdxl-base-1.0",
            "sdxl-lightning-2step",
            "sdxl-lightning-4step",
            "sdxl-lightning-8step",
            "sdxl-turbo",
            "ssd-1b",
            "stable-cascade",
            "stable-cascade-lite",
            "z-image",
            "z-image-turbo",
        )
    }
    for spec in (
        value
        for value in MODEL_CATALOG.values()
        if value.family == "stable-diffusion" and value.catalog_id == "stable-diffusion-1"
    ):
        assert spec.task is Task.T2I
        assert spec.family == "stable-diffusion"
        assert spec.catalog_id == "stable-diffusion-1"
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
        assert spec.source == f"https://huggingface.co/{spec.repo_id}"
        assert spec.license == "CreativeML-OpenRAIL-M"
        assert len(set(spec.files)) == len(spec.files) == 12
        assert not any(name.endswith((".py", ".ckpt")) or "fp16" in name for name in spec.files)
        assert not any(
            name.startswith(("safety_checker/", "feature_extractor/")) for name in spec.files
        )

    sd21 = model_info("sd-v2-1")
    assert sd21.task is Task.T2I
    assert sd21.family == "stable-diffusion"
    assert sd21.repo_id == "sd2-community/stable-diffusion-2-1"
    assert sd21.source == "https://huggingface.co/sd2-community/stable-diffusion-2-1"
    assert sd21.revision == "bb2154823665391b4fb29b0b9cf82a198964ee05"
    assert sd21.provenance == "community-mirror"
    assert sd21.catalog_id == "stable-diffusion-2"
    assert sd21.format == "diffusers"
    assert sd21.license == "CreativeML-OpenRAIL++-M"
    assert sd21.config == {"defaults": {"steps": 50, "guidance": 7.5}}
    assert sd21.files == model_info("sd-turbo").files

    sd_turbo = model_info("sd-turbo")
    assert sd_turbo.task is Task.T2I
    assert sd_turbo.family == "stable-diffusion"
    assert sd_turbo.repo_id == "stabilityai/sd-turbo"
    assert sd_turbo.revision == "b261bac6fd2cf515557d5d0707481eafa0485ec2"
    assert sd_turbo.provenance == "author"
    assert sd_turbo.catalog_id == "sdxl-turbo"
    assert sd_turbo.format == "diffusers"
    assert sd_turbo.license == "sai-nc-community"
    assert sd_turbo.config == {"defaults": {"steps": 1, "guidance": 0.0}}
    assert sd_turbo.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    lcm = model_info("lcm-dreamshaper-v7")
    assert lcm.task is Task.T2I
    assert lcm.family == "stable-diffusion"
    assert lcm.repo_id == "SimianLuo/LCM_Dreamshaper_v7"
    assert lcm.revision == "a85df6a8bd976cdd08b4fd8f3b73f229c9e54df5"
    assert lcm.provenance == "author"
    assert lcm.catalog_id == "lcm"
    assert lcm.format == "diffusers"
    assert lcm.license == "MIT"
    assert lcm.config == {"defaults": {"steps": 4, "guidance": 8.0}}
    assert len(set(lcm.files)) == len(lcm.files) == 12
    assert not any(
        "fp16" in name or name.endswith((".onnx", ".ckpt", ".bin")) for name in lcm.files
    )
    assert not any(name.startswith(("safety_checker/", "feature_extractor/")) for name in lcm.files)

    sdxl = model_info("sdxl-base-1.0")
    assert sdxl.task is Task.T2I
    assert sdxl.family == "stable-diffusion-xl"
    assert sdxl.repo_id == "stabilityai/stable-diffusion-xl-base-1.0"
    assert sdxl.revision == "462165984030d82259a11f4367a4eed129e94a7b"
    assert sdxl.provenance == "author"
    assert sdxl.catalog_id == "sdxl"
    assert sdxl.format == "diffusers"
    assert sdxl.license == "CreativeML-OpenRAIL++-M"
    assert len(set(sdxl.files)) == len(sdxl.files) == 18

    for steps in (2, 4, 8):
        lightning = model_info(f"sdxl-lightning-{steps}step")
        assert lightning.task is Task.T2I
        assert lightning.family == "stable-diffusion-xl"
        assert lightning.repo_id == "ByteDance/SDXL-Lightning"
        assert lightning.revision == "c9a24f48e1c025556787b0c58dd67a091ece2e44"
        assert lightning.provenance == "author"
        assert lightning.catalog_id == "sdxl-lightning"
        assert lightning.format == "diffusers"
        assert lightning.license == "CreativeML-OpenRAIL++-M"
        assert lightning.base == "sdxl-base-1.0"
        assert lightning.files == (f"sdxl_lightning_{steps}step_unet.safetensors",)
        assert lightning.overlay == {
            "unet/diffusion_pytorch_model.safetensors": (
                f"sdxl_lightning_{steps}step_unet.safetensors"
            )
        }
        assert lightning.config == {
            "defaults": {"steps": steps, "guidance": 0.0},
            "scheduler": {"timestep_spacing": "trailing"},
        }

    hyper = model_info("hyper-sdxl-1step")
    assert hyper.task is Task.T2I
    assert hyper.family == "stable-diffusion-xl"
    assert hyper.source == "https://huggingface.co/ByteDance/Hyper-SD"
    assert hyper.repo_id == "ByteDance/Hyper-SD"
    assert hyper.revision == "bc08d970a87c74c71209491d64e3525845698863"
    assert hyper.provenance == "author"
    assert hyper.catalog_id == "hyper-sd"
    assert hyper.format == "diffusers"
    assert hyper.license == "Unspecified"
    assert hyper.base == "sdxl-base-1.0"
    assert hyper.files == ("Hyper-SDXL-1step-Unet.safetensors",)
    assert hyper.overlay == {
        "unet/diffusion_pytorch_model.safetensors": "Hyper-SDXL-1step-Unet.safetensors"
    }
    assert hyper.config == {
        "defaults": {"steps": 1, "guidance": 0.0, "timesteps": [800]},
        "scheduler": {"_class_name": "LCMScheduler"},
    }

    for steps, timesteps in ((4, [999, 749, 499, 249]), (1, [399])):
        dmd2 = model_info(f"dmd2-sdxl-{steps}step")
        source = f"dmd2_sdxl_{steps}step_unet.bin"
        assert dmd2.task is Task.T2I
        assert dmd2.family == "stable-diffusion-xl"
        assert dmd2.source == "https://huggingface.co/tianweiy/DMD2"
        assert dmd2.repo_id == "tianweiy/DMD2"
        assert dmd2.revision == "be22767697a1f3ca656b73c776e15fa335c86c6c"
        assert dmd2.provenance == "author"
        assert dmd2.catalog_id == "dmd2"
        assert dmd2.format == "diffusers"
        assert dmd2.license == "CC-BY-NC-4.0"
        assert dmd2.base == "sdxl-base-1.0"
        assert dmd2.base_exclude == ("unet/diffusion_pytorch_model.safetensors",)
        assert dmd2.files == (source,)
        assert dmd2.overlay == {"unet/diffusion_pytorch_model.bin": source}
        assert dmd2.config == {
            "defaults": {"steps": steps, "guidance": 0.0, "timesteps": timesteps},
            "scheduler": {"_class_name": "LCMScheduler"},
        }

    turbo = model_info("sdxl-turbo")
    assert turbo.task is Task.T2I
    assert turbo.family == "stable-diffusion-xl"
    assert turbo.repo_id == "stabilityai/sdxl-turbo"
    assert turbo.revision == "71153311d3dbb46851df1931d3ca6e939de83304"
    assert turbo.provenance == "author"
    assert turbo.catalog_id == "sdxl-turbo"
    assert turbo.format == "diffusers"
    assert turbo.license == "sai-nc-community"
    assert turbo.config == {"defaults": {"steps": 1, "guidance": 0.0}}
    assert len(set(turbo.files)) == len(turbo.files) == 18
    assert not any("fp16" in name or name.endswith((".onnx", ".ckpt")) for name in turbo.files)

    z_image = model_info("z-image-turbo")
    assert z_image.task is Task.T2I
    assert z_image.family == "z-image"
    assert z_image.repo_id == "Tongyi-MAI/Z-Image-Turbo"
    assert z_image.revision == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert z_image.provenance == "author"
    assert z_image.catalog_id == "z-image"
    assert z_image.format == "diffusers"
    assert z_image.license == "Apache-2.0"
    assert len(set(z_image.files)) == len(z_image.files) == 19
    assert not any("fp16" in name or name.endswith((".onnx", ".ckpt")) for name in z_image.files)

    z_image_base = model_info("z-image")
    assert z_image_base.task is Task.T2I
    assert z_image_base.family == "z-image"
    assert z_image_base.repo_id == "Tongyi-MAI/Z-Image"
    assert z_image_base.revision == "04cc4abb7c5069926f75c9bfde9ef43d49423021"
    assert z_image_base.provenance == "author"
    assert z_image_base.catalog_id == "z-image"
    assert z_image_base.format == "diffusers"
    assert z_image_base.license == "Apache-2.0"
    assert z_image_base.config == {"defaults": {"steps": 50, "guidance": 4.0}}
    assert len(set(z_image_base.files)) == len(z_image_base.files) == 18
    assert not any(
        "fp16" in name or name.endswith((".onnx", ".ckpt")) for name in z_image_base.files
    )

    qwen_image = model_info("qwen-image-2.1")
    assert qwen_image.family == "qwen-image"
    assert qwen_image.catalog_id == "qwen-image-2"
    assert qwen_image.license == "Qwen Research License"
    assert len(set(qwen_image.files)) == len(qwen_image.files) == 25

    janus = model_info("janus-pro-1b")
    assert janus.task is Task.T2I
    assert janus.family == "janus"
    assert janus.repo_id == "deepseek-community/Janus-Pro-1B"
    assert janus.revision == "1655280bb75959cc1cb85529a2a8b26e7016072e"
    assert janus.provenance == "community-conversion"
    assert janus.catalog_id == "janus-pro"
    assert janus.format == "transformers"
    assert janus.license == "DeepSeek Model License"
    assert janus.files == (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
    )

    janus_7b = model_info("janus-pro-7b")
    assert janus_7b.task is Task.T2I
    assert janus_7b.family == "janus"
    assert janus_7b.repo_id == "deepseek-community/Janus-Pro-7B"
    assert janus_7b.revision == "8f23139ed8d37346501091b3690291cb7b16f9fc"
    assert janus_7b.provenance == "community-conversion"
    assert janus_7b.catalog_id == "janus-pro"
    assert janus_7b.format == "transformers"
    assert janus_7b.license == "DeepSeek Model License"
    assert janus_7b.files == (
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )

    ssd1b = model_info("ssd-1b")
    assert ssd1b.task is Task.T2I
    assert ssd1b.family == "stable-diffusion-xl"
    assert ssd1b.repo_id == "segmind/SSD-1B"
    assert ssd1b.revision == "60987f37e94cd59c36b1cba832b9f97b57395a10"
    assert ssd1b.provenance == "author"
    assert ssd1b.catalog_id == "ssd-1b"
    assert ssd1b.format == "diffusers"
    assert ssd1b.license == "Apache-2.0"
    assert len(set(ssd1b.files)) == len(ssd1b.files) == 18

    playground = model_info("playground-v2.5")
    assert playground.task is Task.T2I
    assert playground.family == "stable-diffusion-xl"
    assert playground.repo_id == "playgroundai/playground-v2.5-1024px-aesthetic"
    assert playground.revision == "1e032f13f2fe6db2dc49947dbdbd196e753de573"
    assert playground.provenance == "author"
    assert playground.catalog_id == "playground-v2"
    assert playground.format == "diffusers"
    assert playground.license == "playground-v2dot5-community"
    assert playground.config == {"defaults": {"guidance": 3.0}}
    assert len(set(playground.files)) == len(playground.files) == 18
    assert not any("fp16" in name or name.endswith((".onnx", ".ckpt")) for name in playground.files)

    for model_id in ("sd3-medium", "sd3.5-medium", "sd3.5-large", "sd3.5-large-turbo"):
        spec = model_info(model_id)
        assert spec.task is Task.T2I
        assert spec.family == "stable-diffusion-3"
        assert spec.catalog_id == "stable-diffusion-3"
        assert spec.provenance == "author"
        assert spec.format == "diffusers"
        assert spec.license == "Stability AI Community License"
        assert spec.gated
        assert not any("fp16" in name or "single_file" in name for name in spec.files)

    sd3_medium = model_info("sd3-medium")
    sd35_large = model_info("sd3.5-large")
    sd35_large_turbo = model_info("sd3.5-large-turbo")
    assert sd3_medium.repo_id == "stabilityai/stable-diffusion-3-medium-diffusers"
    assert sd3_medium.revision == "ea42f8cef0f178587cf766dc8129abd379c90671"
    assert sd3_medium.config == {"defaults": {"steps": 28, "guidance": 7.0}}
    assert sd35_large_turbo.repo_id == "stabilityai/stable-diffusion-3.5-large-turbo"
    assert sd35_large_turbo.revision == "ec07796fc06b096cc56de9762974a28f4c632eda"
    assert sd35_large_turbo.config == {"defaults": {"steps": 4, "guidance": 0.0}}
    assert sd35_large_turbo.files == sd35_large.files


def test_provenance_and_formats():
    assert model_info("sd-v1-5").provenance == "community-mirror"
    assert model_info("sd-v1-5").revision == "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    for version in (1, 2, 3, 4):
        assert model_info(f"sd-v1-{version}").provenance == "author"
    assert "unet/diffusion_pytorch_model.bin" in model_info("sd-v1-1").files
    assert "unet/diffusion_pytorch_model.safetensors" in model_info("sd-v1-4").files


def test_flux2_klein_9b_catalog_entry():
    spec = model_info("flux2-klein-9b")
    assert spec.task is Task.T2I
    assert spec.family == "flux2"
    assert spec.repo_id == "black-forest-labs/FLUX.2-klein-9B"
    assert spec.revision == "92196c8e11f7b6cf2b7493e037d8c5345c559216"
    assert spec.provenance == "author"
    assert spec.catalog_id == "flux-2"
    assert spec.format == "diffusers"
    assert spec.license == "FLUX Non-Commercial License"
    assert spec.gated
    assert spec.config == {"defaults": {"steps": 4, "guidance": 1.0}}
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/added_tokens.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_hunyuan_dit_catalog_entry():
    spec = model_info("hunyuan-dit-v1.1-distilled")
    assert spec.task is Task.T2I
    assert spec.family == "hunyuan-dit"
    assert spec.repo_id == "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
    assert spec.revision == "527cf2ecce7c04021975938f8b0e44e35d2b1ed9"
    assert spec.provenance == "author"
    assert spec.catalog_id == "hunyuan-dit"
    assert spec.format == "diffusers"
    assert spec.license == "Tencent Hunyuan Community License"
    assert not spec.gated
    assert spec.config == {"defaults": {"steps": 25, "guidance": 5.0}}
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.safetensors.index.json",
        "text_encoder_2/model-00001-of-00002.safetensors",
        "text_encoder_2/model-00002-of-00002.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.txt",
        "tokenizer_2/special_tokens_map.json",
        "tokenizer_2/spiece.model",
        "tokenizer_2/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_pixart_catalog_entries():
    common = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )
    expected = {
        "pixart-sigma-xl-1024": (
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
            "e102b3591cc82e97071b8b4cb90d834d0c487207",
            "pixart-sigma",
            300,
            common,
        ),
        "pixart-alpha-xl-1024": (
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            "b89adadeccd9ead2adcb9fa2825d3fabec48d404",
            "pixart-alpha",
            120,
            common[:6] + ("tokenizer/added_tokens.json",) + common[6:],
        ),
    }
    for model_id, (repo, revision, catalog_id, max_length, files) in expected.items():
        spec = model_info(model_id)
        assert spec.task is Task.T2I
        assert spec.family == "pixart"
        assert spec.repo_id == repo
        assert spec.revision == revision
        assert spec.provenance == "author"
        assert spec.catalog_id == catalog_id
        assert spec.format == "diffusers"
        assert spec.license == "CreativeML-OpenRAIL++-M"
        assert not spec.gated
        assert spec.config == {
            "defaults": {"steps": 20, "guidance": 4.5, "max_sequence_length": max_length}
        }
        assert spec.files == files


def test_cogview4_catalog_entry():
    spec = model_info("cogview4-6b")
    assert spec.task is Task.T2I
    assert spec.family == "cogview4"
    assert spec.source == "https://huggingface.co/zai-org/CogView4-6B"
    assert spec.repo_id == "zai-org/CogView4-6B"
    assert spec.revision == "63a52b7f6dace7033380cd6da14d0915eab3e6b5"
    assert spec.provenance == "author"
    assert spec.catalog_id == "cogview4"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {"defaults": {"steps": 50, "guidance": 5.0}}
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_ideogram4_catalog_entry():
    spec = model_info("ideogram4-fp8")
    assert spec.task is Task.T2I
    assert spec.family == "ideogram4"
    assert spec.source == "https://huggingface.co/ideogram-ai/ideogram-4-fp8"
    assert spec.repo_id == "ideogram-ai/ideogram-4-fp8"
    assert spec.revision == "ee79a7237b519f1402ceacf952f30c8a31ec5073"
    assert spec.provenance == "author"
    assert spec.catalog_id == "ideogram-4"
    assert spec.format == "diffusers"
    assert spec.license == "Ideogram 4 Non-Commercial License"
    assert spec.gated
    assert spec.config == {"defaults": {"preset": "V4_QUALITY_48"}}
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/chat_template.jinja",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "unconditional_transformer/config.json",
        "unconditional_transformer/diffusion_pytorch_model.safetensors",
        "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_kandinsky22_catalog_entry():
    spec = model_info("kandinsky-2.2")
    assert spec.task is Task.T2I
    assert spec.family == "kandinsky22"
    assert spec.source == "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"
    assert spec.repo_id == "kandinsky-community/kandinsky-2-2-decoder"
    assert spec.revision == "9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3"
    assert spec.files == (
        "model_index.json",
        "movq/config.json",
        "movq/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    )
    assert spec.subfolders == {
        "prior": SubRepo(
            repo_id="kandinsky-community/kandinsky-2-2-prior",
            revision="9fc51ad5732afc5d031724219d22e6c42179c5a8",
            files=(
                "model_index.json",
                "image_encoder/config.json",
                "image_encoder/model.safetensors",
                "image_processor/preprocessor_config.json",
                "prior/config.json",
                "prior/diffusion_pytorch_model.safetensors",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
            ),
        )
    }
    assert spec.provenance == "author"
    assert spec.catalog_id == "kandinsky-2"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {
            "steps": 100,
            "guidance": 4.0,
            "prior_steps": 25,
            "prior_guidance": 4.0,
        }
    }


def test_auraflow_catalog_entry():
    spec = model_info("auraflow-v0.3")
    assert spec.task is Task.T2I
    assert spec.family == "auraflow"
    assert spec.source == "https://huggingface.co/fal/AuraFlow-v0.3"
    assert spec.repo_id == "fal/AuraFlow-v0.3"
    assert spec.revision == "2cd8588f04c886002be4571697d84654a50e3af3"
    assert spec.provenance == "author"
    assert spec.catalog_id == "auraflow"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 50, "guidance": 3.5, "width": 1024, "height": 1024}
    }
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/added_tokens.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


@pytest.mark.parametrize(
    ("model_id", "repo", "revision", "transformer"),
    [
        (
            "sana-1600m-1024",
            "Efficient-Large-Model/Sana_1600M_1024px_diffusers",
            "ac0da2ff55fbe434795be0dce883042e4d49e2fc",
            (
                "transformer/diffusion_pytorch_model.safetensors.index.json",
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            ),
        ),
        (
            "sana-600m-1024",
            "Efficient-Large-Model/Sana_600M_1024px_diffusers",
            "28f3af7689de15f3883d5863059a2fca0aa9b829",
            ("transformer/diffusion_pytorch_model.safetensors",),
        ),
    ],
)
def test_sana_catalog_entries(model_id, repo, revision, transformer):
    spec = model_info(model_id)
    assert spec.task is Task.T2I
    assert spec.family == "sana"
    assert spec.source == f"https://huggingface.co/{repo}"
    assert spec.repo_id == repo
    assert spec.revision == revision
    assert spec.provenance == "author"
    assert spec.catalog_id == "sana"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 20, "guidance": 4.5, "width": 1024, "height": 1024}
    }
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        *transformer,
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_kolors_catalog_entry():
    spec = model_info("kolors")
    assert spec.task is Task.T2I
    assert spec.family == "kolors"
    assert spec.source == "https://huggingface.co/Kwai-Kolors/Kolors-diffusers"
    assert spec.repo_id == "Kwai-Kolors/Kolors-diffusers"
    assert spec.revision == "7e091c75199e910a26cd1b51ed52c28de5db3711"
    assert spec.provenance == "author"
    assert spec.catalog_id == "kolors"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 50, "guidance": 5.0, "width": 1024, "height": 1024}
    }
    # Only float16 weights are published, so the variant file names are listed.
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.fp16.json",
        "text_encoder/model.fp16-00001-of-00003.safetensors",
        "text_encoder/model.fp16-00002-of-00003.safetensors",
        "text_encoder/model.fp16-00003-of-00003.safetensors",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.txt",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    )


def test_hidream_catalog_entry():
    spec = model_info("hidream-i1-fast")
    assert spec.task is Task.T2I
    assert spec.family == "hidream"
    assert spec.source == "https://huggingface.co/HiDream-ai/HiDream-I1-Fast"
    assert spec.repo_id == "HiDream-ai/HiDream-I1-Fast"
    assert spec.revision == "4856a5d8cd6fbd194780ed9f289bdf696d3afc10"
    assert spec.provenance == "author"
    assert spec.catalog_id == "hidream-i1"
    assert spec.format == "diffusers"
    assert spec.license == "MIT"
    assert spec.gated
    assert spec.config == {
        "defaults": {"steps": 16, "guidance": 0.0, "width": 1024, "height": 1024}
    }
    assert len(spec.files) == len(set(spec.files)) == 33
    assert spec.files[:3] == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
    )
    shards = tuple(
        f"transformer/diffusion_pytorch_model-0000{index}-of-00007.safetensors"
        for index in range(1, 8)
    )
    assert [name for name in spec.files if name.startswith("transformer/")] == [
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        *shards,
    ]
    assert spec.subfolders == {
        "text_encoder_4": SubRepo(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            revision="0e9e39f249a16976918f6564b8830bc894c89659",
            files=(
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "model-00001-of-00004.safetensors",
                "model-00002-of-00004.safetensors",
                "model-00003-of-00004.safetensors",
                "model-00004-of-00004.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        )
    }


@pytest.mark.parametrize(
    ("model_id", "decoder", "prior"),
    [
        ("stable-cascade", "decoder", "prior"),
        ("stable-cascade-lite", "decoder_lite", "prior_lite"),
    ],
)
def test_stable_cascade_catalog_entries(model_id, decoder, prior):
    spec = model_info(model_id)
    assert spec.task is Task.T2I
    assert spec.family == "stable_cascade"
    assert spec.source == "https://huggingface.co/stabilityai/stable-cascade"
    assert spec.repo_id == "stabilityai/stable-cascade"
    assert spec.revision == "a89f66d459ae653e3b4d4f992a7c3789d0dc4d16"
    assert spec.files == (
        "model_index.json",
        f"{decoder}/config.json",
        f"{decoder}/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "vqgan/config.json",
        "vqgan/diffusion_pytorch_model.safetensors",
    )
    assert spec.subfolders == {
        "prior": SubRepo(
            repo_id="stabilityai/stable-cascade-prior",
            revision="7ca32c21c3b4d4e35bbb94fcfedfb4fa2259bd91",
            files=(
                "model_index.json",
                f"{prior}/config.json",
                f"{prior}/diffusion_pytorch_model.safetensors",
                "scheduler/scheduler_config.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
            ),
        )
    }
    assert spec.provenance == "author"
    assert spec.catalog_id == "wuerstchen"
    assert spec.format == "diffusers"
    assert spec.license == "stable-cascade-nc-community"
    assert not spec.gated
    assert spec.config == {
        "prior_component": prior,
        "decoder_component": decoder,
        "defaults": {"steps": 12, "guidance": 0.0, "prior_steps": 60, "prior_guidance": 4.0},
    }


def test_subrepo_validation_and_immutability():
    base = model_info("sd-v1-5")
    subrepo = SubRepo("owner/repo", "a" * 40, ("config.json",))
    spec = ModelSpec(**{**base.__dict__, "id": "subrepo-test", "subfolders": {"prior": subrepo}})
    assert spec.subfolders == {"prior": subrepo}
    with pytest.raises(TypeError):
        spec.subfolders["other"] = subrepo
    for revision in ("main", "A" * 40):
        with pytest.raises(ValueError, match="revision"):
            SubRepo("owner/repo", revision, ("config.json",))
    for folder in ("", "/prior", "../prior", "prior/../bad"):
        with pytest.raises(ValueError, match="subfolder"):
            ModelSpec(**{**base.__dict__, "id": f"bad-{folder}", "subfolders": {folder: subrepo}})
    for name in ("", "/config.json", "../config.json"):
        with pytest.raises(ValueError, match="file"):
            SubRepo("owner/repo", "a" * 40, (name,))


def test_unknown_model_lists_available():
    with pytest.raises(ValueError, match="sd-v1-4, sd-v1-5"):
        model_info("sd-v15")


def test_spec_validation():
    base = model_info("sd-v1-4")
    with pytest.raises(ValueError, match="revision"):
        ModelSpec(**{**base.__dict__, "revision": "main"})
    with pytest.raises(ValueError, match="provenance"):
        ModelSpec(**{**base.__dict__, "provenance": "unknown"})
    with pytest.raises(ValueError, match="base"):
        ModelSpec(**{**base.__dict__, "id": "bad-base", "base": "missing"})
    with pytest.raises(ValueError, match="base_exclude"):
        ModelSpec(**{**base.__dict__, "id": "excluded-without-base", "base_exclude": ("x",)})
    with pytest.raises(ValueError, match="overlay source"):
        ModelSpec(
            **{
                **base.__dict__,
                "id": "bad-source",
                "base": "sdxl-base-1.0",
                "overlay": {"unet/model.safetensors": "missing.safetensors"},
            }
        )
    for overlay in (
        {"../model.safetensors": base.files[0]},
        {"unet/model.safetensors": "../model.safetensors"},
    ):
        with pytest.raises(ValueError, match="overlay"):
            ModelSpec(
                **{
                    **base.__dict__,
                    "id": "bad-path",
                    "base": "sdxl-base-1.0",
                    "files": (base.files[0], "../model.safetensors"),
                    "overlay": overlay,
                }
            )
    for base_exclude in ("../model.safetensors", "/model.safetensors"):
        with pytest.raises(ValueError, match="base_exclude"):
            ModelSpec(
                **{
                    **base.__dict__,
                    "id": "bad-exclude",
                    "base": "sdxl-base-1.0",
                    "base_exclude": (base_exclude,),
                }
            )


def test_image_input_field():
    base = MODEL_CATALOG["sdxl-base-1.0"]
    with_image = {spec.id: spec.image_input for spec in MODEL_CATALOG.values() if spec.image_input}
    assert with_image == {"flux1-kontext-dev": "optional"}
    for value in (None, "optional", "required"):
        assert ModelSpec(**{**base.__dict__, "id": "x", "image_input": value}).image_input == value
    with pytest.raises(ValueError, match="image_input"):
        ModelSpec(**{**base.__dict__, "id": "x", "image_input": "sometimes"})


def test_lumina_image_2_catalog_entry():
    spec = model_info("lumina-image-2.0")
    assert spec.task is Task.T2I
    assert spec.family == "lumina2"
    assert spec.source == "https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0"
    assert spec.repo_id == "Alpha-VLLM/Lumina-Image-2.0"
    assert spec.revision == "53504abd8178b30685b6c4c7a4cd181ff78b73e9"
    assert spec.provenance == "author"
    assert spec.catalog_id == "lumina-image-2"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 30, "guidance": 4.0, "width": 1024, "height": 1024}
    }
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00001-of-00003.safetensors",
        "text_encoder/model-00002-of-00003.safetensors",
        "text_encoder/model-00003-of-00003.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


@pytest.mark.parametrize(
    ("model_id", "revision", "size"),
    [
        ("amused-256", "09b6259bf96dbe6d70a852b70812420fe02df55e", 256),
        ("amused-512", "6f3f309ba188a08e347bff9691a0ac791817ca97", 512),
    ],
)
def test_amused_catalog_entries(model_id, revision, size):
    spec = model_info(model_id)
    assert spec.task is Task.T2I
    assert spec.family == "amused"
    assert spec.source == f"https://huggingface.co/amused/{model_id}"
    assert spec.repo_id == f"amused/{model_id}"
    assert spec.revision == revision
    assert spec.provenance == "author"
    assert spec.catalog_id == "amused"
    assert spec.format == "diffusers"
    assert spec.license == "OpenRAIL++"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 12, "guidance": 10.0, "width": size, "height": size}
    }
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vqvae/config.json",
        "vqvae/diffusion_pytorch_model.safetensors",
    )
