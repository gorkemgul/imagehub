import re

import pytest

from imagehub.catalog import MODEL_CATALOG, ModelSpec, SubRepo, available_models, model_info
from imagehub.types import Task


def test_catalog_contents():
    assert available_models() == {
        "t2i": (
            "amused-256",
            "amused-512",
            "animagine-xl-4.0",
            "auraflow-v0.3",
            "bk-sdm-base",
            "bk-sdm-small",
            "bk-sdm-tiny",
            "bk-sdm-v2-base",
            "bk-sdm-v2-small",
            "bk-sdm-v2-tiny",
            "chroma1-base",
            "chroma1-flash",
            "chroma1-hd",
            "cogview3-plus-3b",
            "cogview4-6b",
            "cosmos-predict2-2b-t2i",
            "deepfloyd-if-i-xl",
            "dmd2-sdxl-1step",
            "dmd2-sdxl-4step",
            "dreamlite-base",
            "dreamlite-mobile",
            "dreamshaper-8",
            "dreamshaper-xl-1.0",
            "emu3-gen",
            "ernie-image",
            "flux-1-lite-8b",
            "flux1-dev",
            "flux1-kontext-dev",
            "flux1-krea-dev",
            "flux1-schnell",
            "flux2-klein-4b",
            "flux2-klein-9b",
            "glm-image",
            "hart",
            "hidream-i1-dev",
            "hidream-i1-fast",
            "hidream-i1-full",
            "hidream-o1-image",
            "hunyuan-dit-v1.1-distilled",
            "hunyuan-dit-v1.2",
            "hunyuan-dit-v1.2-distilled",
            "hyper-sdxl-1step",
            "ideogram4-fp8",
            "janus-pro-1b",
            "janus-pro-7b",
            "joyai-image-edit",
            "joyai-image-edit-plus",
            "juggernaut-xl-v9",
            "kandinsky-2-1",
            "kandinsky-2.2",
            "kandinsky-3",
            "kandinsky-5-t2i-lite",
            "koala-1b",
            "koala-700m",
            "koala-lightning-1b",
            "koala-lightning-700m",
            "kolors",
            "krea2-turbo",
            "lcm-dreamshaper-v7",
            "ldm-text2im-large-256",
            "llamagen-t2i-xl-256",
            "llamagen-t2i-xl-512",
            "longcat-image",
            "lumina-dimoo",
            "lumina-image-2.0",
            "lumina-next",
            "meissonic",
            "omnigen-v1",
            "ovis-image-7b",
            "pixart-alpha-xl-1024",
            "pixart-alpha-xl-512",
            "pixart-lcm-xl-1024",
            "pixart-sigma-xl-1024",
            "pixart-sigma-xl-256",
            "pixart-sigma-xl-2k",
            "pixart-sigma-xl-512",
            "pixeldit",
            "playground-v2",
            "playground-v2.5",
            "prx-1024-t2i-beta",
            "prx-512-t2i-sft",
            "qwen-image",
            "qwen-image-2.1",
            "sana-1.5-4.8b-1024",
            "sana-1600m-1024",
            "sana-600m-1024",
            "sana-sprint-1.6b-1024",
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
            "sdxl-refiner-1.0",
            "sdxl-turbo",
            "sdxs-512-0.9",
            "segmind-small-sd",
            "segmind-tiny-sd",
            "segmind-vega",
            "show-o-512",
            "ssd-1b",
            "stable-cascade",
            "stable-cascade-lite",
            "wuerstchen-v2",
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

    dreamshaper = model_info("dreamshaper-8")
    assert dreamshaper.repo_id == "Lykon/dreamshaper-8"
    assert dreamshaper.revision == "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917"
    assert dreamshaper.provenance == "author"
    assert dreamshaper.catalog_id == "stable-diffusion-1"
    assert dreamshaper.config == {"defaults": {"steps": 50, "guidance": 7.5}}
    assert dreamshaper.files == model_info("sd-v1-4").files

    bk_sdm = [model_info(f"bk-sdm-{variant}") for variant in ("base", "small", "tiny")]
    assert [spec.revision for spec in bk_sdm] == [
        "0c03b7a0369b49f97d1acf7256d4ee55ced9b2e0",
        "572238db7ed3a10858900803f3fc8cca53e893e0",
        "0364108e53b7f7f4d2585e817a0b7a83dc261cfa",
    ]
    for spec in bk_sdm:
        assert spec.task is Task.T2I
        assert spec.family == "stable-diffusion"
        assert spec.catalog_id == "bk-sdm"
        assert spec.provenance == "author"
        assert spec.license == "CreativeML-OpenRAIL-M"
        assert spec.files == model_info("sd-v1-4").files

    bk_sdm_v2 = [model_info(f"bk-sdm-v2-{variant}") for variant in ("base", "small", "tiny")]
    assert [spec.revision for spec in bk_sdm_v2] == [
        "21915be4c3a83f3fa45f93559750df44943509b1",
        "ad82c51a3b568e855a7f79c5b893ea2f9ebb197f",
        "68277af553777858cd47e133f92e4db47321bc74",
    ]
    for spec in bk_sdm_v2:
        assert spec.task is Task.T2I
        assert spec.family == "stable-diffusion"
        assert spec.catalog_id == "bk-sdm"
        assert spec.provenance == "author"
        assert spec.license == "CreativeML-OpenRAIL-M"
        assert spec.config == {"defaults": {"steps": 50, "guidance": 7.5}}
        assert spec.files == model_info("sd-v2-1").files

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

    qwen_image = model_info("qwen-image")
    assert qwen_image.family == "qwen-image"
    assert qwen_image.catalog_id == "qwen-image-2"
    assert qwen_image.license == "Apache-2.0"
    assert model_info("qwen-image-2.1").license == "Qwen Research License"
    assert len(set(qwen_image.files)) == len(qwen_image.files) == 29

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

    vega = model_info("segmind-vega")
    assert vega.task is Task.T2I
    assert vega.family == "stable-diffusion-xl"
    assert vega.repo_id == "segmind/Segmind-Vega"
    assert vega.revision == "7714c4363e5856ff974a4f4b068e8691f26d0b40"
    assert vega.provenance == "author"
    assert vega.catalog_id == "segmind-vega"
    assert vega.format == "diffusers"
    assert vega.license == "Apache-2.0"
    assert vega.reference["diffusers"]["git_commit"] == ("d035dcd7cc7c88e0a154609b62887d50bba9fdc2")
    assert vega.reference["segmind-ssd-1b"]["git_commit"] == (
        "2bfc35f41f21531c61e3d3470bca907fd839d035"
    )
    assert len(set(vega.files)) == len(vega.files) == 18
    assert not any("fp16" in name or name == "segmind-vega.safetensors" for name in vega.files)

    for model_id, repo_id, revision in (
        (
            "koala-700m",
            "etri-vilab/koala-700m",
            "5e73a318a1ee1b97cc31fd50f09ac13e018c0b63",
        ),
        (
            "koala-1b",
            "etri-vilab/koala-1b",
            "8ef26f38fe4d141c35fda24966bd48777fc3c8ad",
        ),
    ):
        koala = model_info(model_id)
        assert koala.task is Task.T2I
        assert koala.family == "stable-diffusion-xl"
        assert koala.repo_id == repo_id
        assert koala.revision == revision
        assert koala.provenance == "author"
        assert koala.catalog_id == "koala"
        assert koala.format == "diffusers"
        assert koala.license == "Unspecified"
        assert koala.reference["diffusers"]["git_commit"] == (
            "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"
        )
        assert koala.reference["sdxl-koala"]["git_commit"] == (
            "df9dac7514cc2bd6882a5161401cd9740349c11b"
        )
        assert len(set(koala.files)) == len(koala.files) == 18
        assert not any("fp16" in name or name.endswith((".bin", ".onnx")) for name in koala.files)

    for model_id, repo_id, revision in (
        (
            "koala-lightning-700m",
            "etri-vilab/koala-lightning-700m",
            "358e0e3576adf2ebc563b64cfcf8a088a4a6407a",
        ),
        (
            "koala-lightning-1b",
            "etri-vilab/koala-lightning-1b",
            "b4114f7577bb08e08a6b9792eb3a3a8b34c95adb",
        ),
    ):
        koala = model_info(model_id)
        assert koala.task is Task.T2I
        assert koala.family == "stable-diffusion-xl"
        assert koala.repo_id == repo_id
        assert koala.revision == revision
        assert koala.provenance == "author"
        assert koala.catalog_id == "koala"
        assert koala.format == "diffusers"
        assert koala.license == "Unspecified"
        assert koala.config == {
            "defaults": {"steps": 10, "guidance": 3.5},
            "scheduler": {"timestep_spacing": "trailing"},
        }
        assert len(set(koala.files)) == len(koala.files) == 18
        assert not any("fp16" in name or name.endswith((".bin", ".onnx")) for name in koala.files)

    for model_id, repo_id, revision, precision in (
        (
            "segmind-small-sd",
            "segmind/small-sd",
            "f8b087e68071a29bbe85a4b07e6e1e8998d59aa7",
            "float32",
        ),
        (
            "segmind-tiny-sd",
            "segmind/tiny-sd",
            "cad0bd7495fa6c4bcca01b19a723dc91627fe84f",
            "float16",
        ),
    ):
        spec = model_info(model_id)
        assert spec.family == "stable-diffusion"
        assert spec.repo_id == repo_id
        assert spec.revision == revision
        assert spec.provenance == "author"
        assert spec.catalog_id == "bk-sdm"
        assert spec.license == "CreativeML-OpenRAIL-M"
        assert spec.config == {"precision": precision}
        assert len(spec.files) == 12
        assert spec.files[-3:] == (
            "text_encoder/pytorch_model.bin",
            "unet/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.bin",
        )

    playground_v2 = model_info("playground-v2")
    assert playground_v2.task is Task.T2I
    assert playground_v2.family == "stable-diffusion-xl"
    assert playground_v2.repo_id == "playgroundai/playground-v2-1024px-aesthetic"
    assert playground_v2.revision == "737a24ebe3ae4e5421110879168be188853d86c8"
    assert playground_v2.provenance == "author"
    assert playground_v2.catalog_id == "playground-v2"
    assert playground_v2.format == "diffusers"
    assert playground_v2.license == "playground-v2-community"
    assert playground_v2.config == {"defaults": {"guidance": 3.0}}
    assert len(set(playground_v2.files)) == len(playground_v2.files) == 18
    assert not any(
        "fp16" in n or n.endswith((".onnx", ".ckpt", ".bin")) for n in playground_v2.files
    )

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

    fine_tunes = {
        model_id: model_info(model_id)
        for model_id in ("juggernaut-xl-v9", "animagine-xl-4.0", "dreamshaper-xl-1.0")
    }
    assert [fine_tunes[model_id].revision for model_id in fine_tunes] == [
        "cf419233522daa0b9ea36c3aff98fa2cab1fb0fb",
        "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96",
        "41e6644752a8c9aa63930e6043c4fd83c7708420",
    ]
    for spec in fine_tunes.values():
        assert spec.family == "stable-diffusion-xl"
        assert spec.catalog_id == "sdxl"
        assert spec.provenance == "author"
        assert spec.format == "diffusers"
        assert spec.reference == {
            "diffusers": {"version": "0.40.0"},
            "transformers": {"version": "5.18.0"},
        }
        assert len(set(spec.files)) == len(spec.files) == 18
        assert not any(name.endswith((".bin", ".ckpt")) for name in spec.files)
    assert fine_tunes["juggernaut-xl-v9"].repo_id == "RunDiffusion/Juggernaut-XL-v9"
    assert fine_tunes["juggernaut-xl-v9"].license == "CreativeML-OpenRAIL-M"
    assert fine_tunes["juggernaut-xl-v9"].config == {"precision": "float16"}
    assert sum("fp16" in name for name in fine_tunes["juggernaut-xl-v9"].files) == 4
    assert fine_tunes["animagine-xl-4.0"].repo_id == "cagliostrolab/animagine-xl-4.0"
    assert fine_tunes["animagine-xl-4.0"].license == "CreativeML-OpenRAIL++-M"
    assert fine_tunes["animagine-xl-4.0"].config == {"precision": "float16"}
    assert not any("fp16" in name for name in fine_tunes["animagine-xl-4.0"].files)
    assert fine_tunes["dreamshaper-xl-1.0"].repo_id == "Lykon/dreamshaper-xl-1-0"
    assert fine_tunes["dreamshaper-xl-1.0"].license == "CreativeML-OpenRAIL++-M"
    assert fine_tunes["dreamshaper-xl-1.0"].config == {"precision": "float32"}

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


@pytest.mark.parametrize(
    ("model_id", "repo", "revision", "steps"),
    [
        (
            "hunyuan-dit-v1.2",
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
            "5e96094e0ad19e7f475de8711f03634ca0ccc40c",
            50,
        ),
        (
            "hunyuan-dit-v1.2-distilled",
            "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
            "ba991d1546d8c50936c4c16398ed0a87b9b99fb1",
            25,
        ),
    ],
)
def test_hunyuan_dit_v12_catalog_entries(model_id, repo, revision, steps):
    spec = model_info(model_id)
    assert spec.task is Task.T2I
    assert spec.family == "hunyuan-dit"
    assert (spec.repo_id, spec.revision) == (repo, revision)
    assert spec.provenance == "author"
    assert spec.catalog_id == "hunyuan-dit"
    assert spec.format == "diffusers"
    assert spec.license == "Tencent Hunyuan Community License"
    assert not spec.gated
    assert spec.config == {"defaults": {"steps": steps, "guidance": 5.0}}
    assert spec.files == model_info("hunyuan-dit-v1.1-distilled").files


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
    alpha_512 = (
        common[:4]
        + tuple(f"text_encoder/model-0000{index}-of-00004.safetensors" for index in range(1, 5))
        + ("tokenizer/added_tokens.json",)
        + common[6:]
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
        "pixart-alpha-xl-512": (
            "PixArt-alpha/PixArt-XL-2-512x512",
            "50f702106901db6d0f8b67eb88e814c56ded2692",
            "pixart-alpha",
            120,
            alpha_512,
        ),
        "pixart-lcm-xl-1024": (
            "PixArt-alpha/PixArt-LCM-XL-2-1024-MS",
            "e153060b10990994d2ed31304b3224da47fbd740",
            "pixart-delta",
            120,
            alpha_512,
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
        if model_id == "pixart-lcm-xl-1024":
            assert spec.config == {
                "defaults": {"steps": 4, "guidance": 0.0, "max_sequence_length": 120}
            }
            assert spec.files == files
            assert spec.base is None
            continue
        size = {"pixart-alpha-xl-512": {"width": 512, "height": 512}}.get(model_id, {})
        assert spec.config == {
            "defaults": {"steps": 20, "guidance": 4.5, "max_sequence_length": max_length} | size
        }
        assert spec.files == files
        assert spec.base is None


def test_pixart_sigma_transformer_only_catalog_entries():
    transformer = ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors")
    expected = {
        "pixart-sigma-xl-256": (
            "PixArt-alpha/PixArt-Sigma-XL-2-256x256",
            "495cb1bafac48d71ca96761409c5fd7e48354df2",
            256,
        ),
        "pixart-sigma-xl-512": (
            "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
            "76fb7eb5a9314bc1e4e479d2f13447517fca9be4",
            512,
        ),
        "pixart-sigma-xl-2k": (
            "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
            "cd274f4c8b01de0dcc87b65887dd357d473bf516",
            2048,
        ),
    }
    base = model_info("pixart-sigma-xl-1024")
    for model_id, (repo, revision, size) in expected.items():
        spec = model_info(model_id)
        assert spec.task is Task.T2I
        assert spec.family == "pixart"
        assert (spec.repo_id, spec.revision) == (repo, revision)
        assert spec.provenance == "author"
        assert spec.catalog_id == "pixart-sigma"
        assert spec.license == "CreativeML-OpenRAIL++-M"
        assert not spec.gated
        assert spec.base == base.id
        assert spec.files == transformer
        assert spec.overlay == {name: name for name in transformer}
        assert not spec.base_exclude
        assert spec.config == {
            "defaults": {
                "steps": 20,
                "guidance": 4.5,
                "max_sequence_length": 300,
                "width": size,
                "height": size,
            }
        }


def test_pixeldit_catalog_entry():
    spec = model_info("pixeldit")
    assert spec.task is Task.T2I
    assert spec.family == spec.catalog_id == "pixeldit"
    assert spec.repo_id == "nvidia/PixelDiT-1300M-1024px"
    assert spec.revision == "7c63b99a7a399918a1d6478b095698a65f664847"
    assert spec.provenance == "author"
    assert spec.format == "pytorch"
    assert "NSCLv1" in spec.license
    assert spec.files == ("LICENSE", "README.md", "config.json", "pixeldit_t2i_v1.pth")
    assert spec.subfolders["text_encoder"].revision == ("569d9809d0c8b6722d4d31b5a77a2ec7a400650a")


def test_cosmos2_catalog_entry():
    spec = model_info("cosmos-predict2-2b-t2i")
    assert spec.family == "cosmos2"
    assert spec.catalog_id == "cosmos-predict2-2b-t2i"
    assert spec.revision == "acdb5fde992a73ef0355f287977d002cbfd127e0"
    assert spec.provenance == "author" and spec.gated
    assert spec.license == "NVIDIA Open Model License"
    assert len(spec.files) == 15
    assert not any(name in spec.files for name in ("model.pt", "tokenizer/tokenizer.pth"))


def test_cogview3_plus_catalog_entry():
    spec = model_info("cogview3-plus-3b")
    assert spec.task is Task.T2I
    assert spec.family == "cogview3"
    assert spec.source == "https://huggingface.co/zai-org/CogView3-Plus-3B"
    assert spec.repo_id == "zai-org/CogView3-Plus-3B"
    assert spec.revision == "5d70e40732ac0efac98524c51a7fa9c82707f1e5"
    assert spec.provenance == "author"
    assert spec.catalog_id == "cogview3-plus"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {"defaults": {"steps": 50, "guidance": 5.0}}
    assert len(spec.files) == 18
    assert "tokenizer/spiece.model" in spec.files
    assert not any(name.endswith(".py") for name in spec.files)


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


def test_hidream_o1_catalog_entry():
    spec = model_info("hidream-o1-image")
    assert spec.task is Task.T2I
    assert spec.family == "hidream-o1"
    assert spec.repo_id == "HiDream-ai/HiDream-O1-Image"
    assert spec.revision == "0b0901d99f200389e138c61946af1185f5f49a13"
    assert spec.provenance == "author"
    assert spec.catalog_id == "hidream-o1-image"
    assert spec.format == "transformers"
    assert spec.license == "MIT"
    assert spec.image_input is None
    assert spec.config == {"defaults": {"steps": 50, "guidance": 5.0, "shift": 3.0}}
    assert spec.files[:4] == (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    assert spec.files[-1] == "model-00008-of-00008.safetensors"


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


def test_kandinsky21_catalog_entry():
    spec = model_info("kandinsky-2-1")
    assert spec.task is Task.T2I
    assert spec.family == "kandinsky21"
    assert spec.repo_id == "kandinsky-community/kandinsky-2-1"
    assert spec.revision == "04b366659663bf60deacb96fd3e38f1cb39a3be4"
    assert spec.files == (
        "ddpm_scheduler/scheduler_config.json",
        "model_index.json",
        "movq/config.json",
        "movq/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/sentencepiece.bpe.model",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    )
    assert spec.subfolders == {
        "prior": SubRepo(
            repo_id="kandinsky-community/kandinsky-2-1-prior",
            revision="b726af62d69111bbf62b5d71bb9d90d9867fbce4",
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


def test_chroma_catalog_entry():
    spec = model_info("chroma1-hd")
    assert spec.task is Task.T2I
    assert spec.family == "chroma"
    assert spec.source == "https://huggingface.co/lodestones/Chroma1-HD"
    assert spec.repo_id == "lodestones/Chroma1-HD"
    assert spec.revision == "0e0c60ece1e82b17cb7f77342d765ba5024c40c0"
    assert spec.provenance == "author"
    assert spec.catalog_id == "chroma1-hd"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 35, "guidance": 5.0, "width": 1024, "height": 1024}
    }
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


@pytest.mark.parametrize(
    ("model_id", "repo", "revision", "defaults"),
    (
        (
            "chroma1-base",
            "lodestones/Chroma1-Base",
            "50ebe125b67794f4ed8e4985d26a4414ca62aed3",
            {"steps": 40, "guidance": 3.0, "width": 1024, "height": 1024},
        ),
        (
            "chroma1-flash",
            "lodestones/Chroma1-Flash",
            "093c4f63b60507234aa53a073f1f9f565b3f4336",
            {"steps": 4, "guidance": 1.0, "width": 1024, "height": 1024},
        ),
    ),
)
def test_chroma_variant_catalog_entries(model_id, repo, revision, defaults):
    spec = model_info(model_id)
    assert spec.task is Task.T2I
    assert spec.family == "chroma"
    assert spec.source == f"https://huggingface.co/{repo}"
    assert spec.repo_id == repo
    assert spec.revision == revision
    assert spec.provenance == "author"
    assert spec.catalog_id == model_id
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {"defaults": defaults}
    assert spec.files == MODEL_CATALOG["chroma1-hd"].files


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


@pytest.mark.parametrize(
    ("model_id", "repo", "revision", "transformer", "defaults"),
    [
        (
            "sana-sprint-1.6b-1024",
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            "19683c58b7ea290e55cedd8950ae1d86ada7ef96",
            ("transformer/diffusion_pytorch_model.safetensors",),
            {"steps": 2, "guidance": 4.5, "width": 1024, "height": 1024},
        ),
        (
            "sana-1.5-4.8b-1024",
            "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
            "9468102c3cebb657f8c4b5f1e5a71e989a15f10d",
            (
                "transformer/diffusion_pytorch_model.safetensors.index.json",
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            ),
            {"steps": 20, "guidance": 4.5, "width": 1024, "height": 1024},
        ),
    ],
)
def test_sana_new_catalog_entries(model_id, repo, revision, transformer, defaults):
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
    assert spec.config == {"defaults": defaults}
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
    ("model_id", "repo_id", "revision", "defaults"),
    [
        (
            "hidream-i1-dev",
            "HiDream-ai/HiDream-I1-Dev",
            "0fad2ea0ccf9a80ddf019ea777eedb27c1ccb232",
            {"steps": 30, "guidance": 0.0, "width": 1024, "height": 1024},
        ),
        (
            "hidream-i1-full",
            "HiDream-ai/HiDream-I1-Full",
            "8ccbbfb270ccdae26d6bb0081df67dc81e4033bf",
            {"steps": 50, "guidance": 5.0, "width": 1024, "height": 1024},
        ),
    ],
)
def test_hidream_variant_catalog_entries(model_id, repo_id, revision, defaults):
    spec = model_info(model_id)
    assert (spec.family, spec.repo_id, spec.revision, spec.config["defaults"]) == (
        "hidream",
        repo_id,
        revision,
        defaults,
    )
    assert spec.files == model_info("hidream-i1-fast").files
    assert spec.subfolders == model_info("hidream-i1-fast").subfolders


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
    assert with_image == {
        "flux1-kontext-dev": "optional",
        "joyai-image-edit": "optional",
        "joyai-image-edit-plus": "required",
        "sdxl-refiner-1.0": "required",
        "omnigen-v1": "optional",
    }
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


def test_meissonic_catalog_entry():
    spec = model_info("meissonic")
    assert spec.task is Task.T2I
    assert spec.family == spec.catalog_id == "meissonic"
    assert spec.source == "https://huggingface.co/MeissonFlow/Meissonic"
    assert spec.repo_id == "MeissonFlow/Meissonic"
    assert spec.revision == "7ee5068bcebaaf6165e5d77c2969aa3b38e17b38"
    assert spec.provenance == "author"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 48, "guidance": 9.0, "width": 1024, "height": 1024}
    }
    assert spec.reference["meissonic"] == {
        "repository": "viiika/Meissonic",
        "git_commit": "a9de06ca8c02c467aec3da275605fc59380299e2",
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


@pytest.mark.parametrize(
    ("model_id", "gpt_file", "size"),
    [
        ("llamagen-t2i-xl-256", "t2i_XL_stage1_256.pt", 256),
        ("llamagen-t2i-xl-512", "t2i_XL_stage2_512.pt", 512),
    ],
)
def test_llamagen_catalog_entries(model_id, gpt_file, size):
    spec = model_info(model_id)
    assert spec.task is Task.T2I
    assert spec.family == "llamagen"
    assert spec.source == "https://huggingface.co/peizesun/llamagen_t2i"
    assert spec.repo_id == "peizesun/llamagen_t2i"
    assert spec.revision == "276f5c5a3d915b922899a03f1912605531574747"
    assert spec.provenance == "author"
    assert spec.catalog_id == "llamagen"
    assert spec.format == "pytorch"
    assert spec.license == "MIT"
    assert not spec.gated
    assert spec.config == {
        "gpt_checkpoint": gpt_file,
        "defaults": {
            "cfg_scale": 7.5,
            "temperature": 1.0,
            "top_k": 1000,
            "top_p": 1.0,
            "cfg_interval": -1,
            "width": size,
            "height": size,
        },
    }
    assert spec.files == ("vq_ds16_t2i.pt", gpt_file)
    assert spec.subfolders == {
        "text_encoder": SubRepo(
            repo_id="google/flan-t5-xl",
            revision="7d6315df2c2fb742f0f5b556879d730926ca9001",
            files=(
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ),
        )
    }


def test_wuerstchen_catalog_entry():
    spec = model_info("wuerstchen-v2")
    assert spec.task is Task.T2I
    assert spec.family == "wuerstchen"
    assert spec.source == "https://huggingface.co/warp-ai/wuerstchen"
    assert spec.repo_id == "warp-ai/wuerstchen"
    assert spec.revision == "c3da41406ddd4d9c48c49aa93981a82354351b83"
    assert spec.files == (
        "model_index.json",
        "decoder/config.json",
        "decoder/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "vqgan/config.json",
        "vqgan/diffusion_pytorch_model.safetensors",
    )
    assert spec.subfolders == {
        "prior": SubRepo(
            repo_id="warp-ai/wuerstchen-prior",
            revision="c41be6339e1b93f95ece99825327cdf9f4abc436",
            files=(
                "model_index.json",
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
    assert spec.catalog_id == "wuerstchen"
    assert spec.format == "diffusers"
    assert spec.license == "mit"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 12, "guidance": 0.0, "prior_steps": 60, "prior_guidance": 4.0}
    }
    assert not any(name.endswith(".bin") for name in spec.files + spec.subfolders["prior"].files)


def test_glm_image_catalog_entry():
    spec = model_info("glm-image")
    assert spec.task is Task.T2I
    assert spec.family == "glm-image"
    assert spec.repo_id == "zai-org/GLM-Image"
    assert spec.revision == "2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4"
    assert spec.catalog_id == "glm-image"
    assert spec.license == "MIT"
    assert not spec.gated
    assert spec.provenance == "author"
    assert spec.image_input is None
    assert spec.config == {"defaults": {"steps": 50, "guidance": 1.5}}
    assert len(spec.files) == 25
    assert not any(name.endswith(".py") for name in spec.files)


def test_lumina_next_catalog_entry():
    spec = model_info("lumina-next")
    assert spec.task is Task.T2I
    assert spec.family == "lumina-next"
    assert spec.source == "https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers"
    assert spec.repo_id == "Alpha-VLLM/Lumina-Next-SFT-diffusers"
    assert spec.revision == "0ee5ec90043acf5cb41fe96274af36eb7fad8d95"
    assert spec.provenance == "author"
    assert spec.catalog_id == "lumina-next"
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
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_lumina_dimoo_catalog_entry():
    spec = model_info("lumina-dimoo")
    assert spec.task is Task.T2I
    assert spec.family == spec.catalog_id == "lumina-dimoo"
    assert spec.repo_id == "Alpha-VLLM/Lumina-DiMOO"
    assert spec.revision == "f014549e17679e41b119959754cdd9320af9bdc8"
    assert spec.provenance == "author"
    assert spec.format == "safetensors"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.reference["lumina-dimoo"]["git_commit"] == (
        "81344e60545ae8d70f11039d3450717d974f831b"
    )
    assert spec.config == {
        "defaults": {"steps": 64, "guidance": 4.0, "width": 1024, "height": 1024}
    }
    assert len(spec.files) == len(set(spec.files)) == 14


def test_ernie_image_catalog_entry():
    spec = model_info("ernie-image")
    assert spec.family == "ernie-image"
    assert spec.task is Task.T2I
    assert spec.repo_id == "baidu/ERNIE-Image"
    assert spec.revision == "5346b31d68c9c23758ba56ef8be5e9dc174c7f99"
    assert spec.catalog_id == "ernie-image"
    assert spec.provenance == "author"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert len(spec.files) == len(set(spec.files)) == 12
    # The optional prompt enhancer (pe/, pe_tokenizer/) is not part of the supported snapshot.
    assert not any(name.startswith(("pe/", "pe_tokenizer/")) for name in spec.files)
    assert dict(spec.config["defaults"]) == {"steps": 50, "guidance": 4.0}


def test_omnigen_catalog_entry():
    spec = model_info("omnigen-v1")
    assert spec.task is Task.T2I
    assert spec.family == "omnigen"
    assert spec.source == "https://huggingface.co/Shitao/OmniGen-v1-diffusers"
    assert spec.repo_id == "Shitao/OmniGen-v1-diffusers"
    assert spec.revision == "016e2f61d12a98303f6bbdf122687694d7984268"
    assert spec.provenance == "author"
    assert spec.catalog_id == "omnigen"
    assert spec.format == "diffusers"
    assert spec.license == "MIT"
    assert not spec.gated
    assert spec.image_input == "optional"
    assert spec.config == {
        "defaults": {
            "steps": 50,
            "guidance": 2.5,
            "image_guidance": 1.6,
            "width": 1024,
            "height": 1024,
        }
    }
    assert spec.files == (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    )


def test_kandinsky3_catalog_entry():
    spec = model_info("kandinsky-3")
    assert spec.task is Task.T2I
    assert spec.family == "kandinsky3"
    assert spec.source == "https://huggingface.co/kandinsky-community/kandinsky-3"
    assert spec.repo_id == "kandinsky-community/kandinsky-3"
    assert spec.revision == "bf79e6c219da8a94abb50235fdc4567eb8fb4632"
    assert spec.provenance == "author"
    assert spec.catalog_id == "kandinsky-3"
    assert spec.format == "diffusers"
    assert spec.license == "Apache-2.0"
    assert not spec.gated
    assert spec.config == {
        "defaults": {"steps": 25, "guidance": 3.0, "width": 1024, "height": 1024}
    }
    # Only float16 weights are published, so the variant file names are listed.
    assert len(set(spec.files)) == len(spec.files) == 17
    assert "movq/diffusion_pytorch_model.fp16.safetensors" in spec.files
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in spec.files
    assert [name for name in spec.files if name.startswith("text_encoder/model.fp16")] == [
        f"text_encoder/model.fp16-0000{index}-of-00005.safetensors" for index in range(1, 6)
    ]
    assert "tokenizer/spiece.model" in spec.files
    assert not any(name.startswith("assets/") for name in spec.files)
    assert not any(name.endswith(("model.safetensors", ".py")) for name in spec.files)
