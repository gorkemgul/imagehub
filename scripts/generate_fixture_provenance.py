"""Generate the per-file provenance manifest for redistributable test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

# Each entry is source repository, immutable revision, and governing license.
SOURCES = {
    "amused_configs": (
        "amused/amused-256 and amused/amused-512",
        "09b6259bf96dbe6d70a852b70812420fe02df55e / 6f3f309ba188a08e347bff9691a0ac791817ca97",
        "OpenRAIL++",
    ),
    "animagine_xl_4_configs": (
        "cagliostrolab/animagine-xl-4.0",
        "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96",
        "CreativeML-OpenRAIL++-M",
    ),
    "animagine_xl_4_tensor_shapes.json": (
        "cagliostrolab/animagine-xl-4.0",
        "2b7c1b397761bf5bd3cc42e5b39ec99314a75a96",
        "CreativeML-OpenRAIL++-M",
    ),
    "auraflow_configs": (
        "fal/AuraFlow-v0.3",
        "2cd8588f04c886002be4571697d84654a50e3af3",
        "Apache-2.0",
    ),
    "bagel_7b_mot_configs": (
        "ByteDance-Seed/BAGEL-7B-MoT",
        "5019f57d168e5816e8f3f701b17cc816bb7cf24b",
        "Apache-2.0",
    ),
    "bk_sdm_configs": (
        "https://github.com/Nota-NetsPresso/BK-SDM",
        "e13670c8a7e2b896f508cbece59335930703be6c",
        "CreativeML-OpenRAIL-M",
    ),
    "bk_sdm_v2_configs": (
        "https://github.com/Nota-NetsPresso/BK-SDM",
        "e13670c8a7e2b896f508cbece59335930703be6c",
        "CreativeML-OpenRAIL-M",
    ),
    "chroma_base_configs": (
        "lodestones/Chroma1-Base",
        "50ebe125b67794f4ed8e4985d26a4414ca62aed3",
        "Apache-2.0",
    ),
    "chroma_configs": (
        "lodestones/Chroma1-HD",
        "0e0c60ece1e82b17cb7f77342d765ba5024c40c0",
        "Apache-2.0",
    ),
    "chroma_flash_configs": (
        "lodestones/Chroma1-Flash",
        "093c4f63b60507234aa53a073f1f9f565b3f4336",
        "Apache-2.0",
    ),
    "cogview3_plus_configs": (
        "zai-org/CogView3-Plus-3B",
        "5d70e40732ac0efac98524c51a7fa9c82707f1e5",
        "Apache-2.0",
    ),
    "cogview4_configs": (
        "zai-org/CogView4-6B",
        "63a52b7f6dace7033380cd6da14d0915eab3e6b5",
        "Apache-2.0",
    ),
    "dreamshaper_8_configs": (
        "Lykon/dreamshaper-8",
        "a7e52b98680b1ba8ff7bce97c7f9f2e2e5337917",
        "CreativeML-OpenRAIL-M",
    ),
    "dreamshaper_xl_1_configs": (
        "Lykon/dreamshaper-xl-1-0",
        "41e6644752a8c9aa63930e6043c4fd83c7708420",
        "CreativeML-OpenRAIL++-M",
    ),
    "emu3_gen_configs": ("BAAI/Emu3-Gen", "40b3bb89a9cee99ab0b9deb35dead2e50d6a0d72", "Apache-2.0"),
    "ernie_image_configs": (
        "baidu/ERNIE-Image",
        "5346b31d68c9c23758ba56ef8be5e9dc174c7f99",
        "Apache-2.0",
    ),
    "flux1_configs": (
        "black-forest-labs/FLUX.1-schnell",
        "741f7c3ce8b383c54771c7003378a50191e9efe9",
        "Apache-2.0",
    ),
    "flux2_klein_4b_configs": (
        "black-forest-labs/FLUX.2-klein-4B",
        "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "Apache-2.0",
    ),
    "glm_image_configs": ("zai-org/GLM-Image", "2c433cc0cbc293bde2ac8ca9624f279b5d23fcf4", "MIT"),
    "hart_configs": (
        "mit-han-lab/hart-0.7b-1024px",
        "b1521ff1449d6817c685b08614c554e7a5b5b8eb",
        "MIT",
    ),
    "hidream_configs": (
        "HiDream-ai/HiDream-I1-Fast",
        "4856a5d8cd6fbd194780ed9f289bdf696d3afc10",
        "MIT",
    ),
    "hidream_o1_configs": (
        "HiDream-ai/HiDream-O1-Image",
        "0b0901d99f200389e138c61946af1185f5f49a13",
        "MIT",
    ),
    "infinity_configs": (
        "FoundationVision/Infinity",
        "e7108923249fb3c836ebfbc3f4b4d318d6131293",
        "MIT",
    ),
    "joyai_image_configs": (
        "jdopensource/JoyAI-Image-Edit-Diffusers and jdopensource/JoyAI-Image-Edit-Plus-Diffusers",
        "4b41fb25d961f37668750178ccbb380da326201c / c2686460c7b64d8aa11bc4d0da423fb316b33f9e",
        "Apache-2.0",
    ),
    "juggernaut_xl_v9_configs": (
        "RunDiffusion/Juggernaut-XL-v9",
        "cf419233522daa0b9ea36c3aff98fa2cab1fb0fb",
        "CreativeML-OpenRAIL-M",
    ),
    "kandinsky21_configs": (
        "kandinsky-community/kandinsky-2-1",
        "04b366659663bf60deacb96fd3e38f1cb39a3be4",
        "Apache-2.0",
    ),
    "kandinsky22_configs": (
        "kandinsky-community/kandinsky-2-2-decoder",
        "9ae140d347fed8ce6e8bb3005dcc1f48543bb8e3",
        "Apache-2.0",
    ),
    "kandinsky22_prior_configs": (
        "kandinsky-community/kandinsky-2-2-prior",
        "d9b82b7d6e78803e0553140aa47d3a279f6dc562",
        "Apache-2.0",
    ),
    "kandinsky3_configs": (
        "kandinsky-community/kandinsky-3",
        "bf79e6c219da8a94abb50235fdc4567eb8fb4632",
        "Apache-2.0",
    ),
    "kandinsky5_configs": (
        "kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers",
        "25da1e82e6adefdbc83c02a71f762287c5e471a4",
        "MIT",
    ),
    "kolors_configs": (
        "Kwai-Kolors/Kolors-diffusers",
        "7e091c75199e910a26cd1b51ed52c28de5db3711",
        "Apache-2.0",
    ),
    "ldm_configs": (
        "CompVis/ldm-text2im-large-256",
        "30de525ca11a880baea4962827fb6cb0bb268955",
        "Apache-2.0",
    ),
    "llamagen": ("peizesun/llamagen_t2i", "276f5c5a3d915b922899a03f1912605531574747", "MIT"),
    "longcat_image_configs": (
        "meituan-longcat/LongCat-Image",
        "d2ea50b79a930074c37b9b97ce45e3b2ea8cf4d8",
        "Apache-2.0",
    ),
    "lumina2_configs": (
        "Alpha-VLLM/Lumina-Image-2.0",
        "53504abd8178b30685b6c4c7a4cd181ff78b73e9",
        "Apache-2.0",
    ),
    "lumina_dimoo_configs": (
        "Alpha-VLLM/Lumina-DiMOO",
        "f014549e17679e41b119959754cdd9320af9bdc8",
        "Apache-2.0",
    ),
    "lumina_next_configs": (
        "Alpha-VLLM/Lumina-Next-SFT-diffusers",
        "0ee5ec90043acf5cb41fe96274af36eb7fad8d95",
        "Apache-2.0",
    ),
    "meissonic_configs": (
        "MeissonFlow/Meissonic",
        "7ee5068bcebaaf6165e5d77c2969aa3b38e17b38",
        "Apache-2.0",
    ),
    "nextstep_1_configs": (
        "stepfun-ai/NextStep-1-Large",
        "97dd0b9e097f60e39e3ce3bbac1a3343a09547cd",
        "Apache-2.0",
    ),
    "omnigen_configs": (
        "Shitao/OmniGen-v1-diffusers",
        "016e2f61d12a98303f6bbdf122687694d7984268",
        "MIT",
    ),
    "omnigen_tokenizer": (
        "Shitao/OmniGen-v1-diffusers",
        "016e2f61d12a98303f6bbdf122687694d7984268",
        "MIT",
    ),
    "ovis_image_configs": (
        "ATH-MaaS/Ovis-Image-7B",
        "41be1c5821a92c970d63d7eb595a2fd3fe32b22e",
        "Apache-2.0",
    ),
    "pixart_configs": (
        "PixArt-alpha model repositories",
        "e102b3591cc82e97071b8b4cb90d834d0c487207",
        "CreativeML-OpenRAIL++-M",
    ),
    "prx_configs": (
        "Photoroom/prx-1024-t2i-beta",
        "0370504b9a433fbdff814b8fcba9da03bf72a7a3",
        "Apache-2.0",
    ),
    "qwen_image_configs": (
        "Qwen/Qwen-Image",
        "75e0b4be04f60ec59a75f475837eced720f823b6",
        "Apache-2.0",
    ),
    "qwen_tokenizer": ("imagehub synthetic fixture", "project-authored", "Apache-2.0"),
    "sana_configs": (
        "Efficient-Large-Model Sana repositories",
        "ac0da2ff55fbe434795be0dce883042e4d49e2fc",
        "Apache-2.0",
    ),
    "sd1_configs": (
        "CompVis/stable-diffusion-v1-1 and stable-diffusion-v1-5/stable-diffusion-v1-5",
        "574555a657f0c8d6e970a02c7e095e9d4220dcbf / 451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "CreativeML-OpenRAIL-M",
    ),
    "sd21_configs": (
        "sd2-community/stable-diffusion-2-1",
        "bb2154823665391b4fb29b0b9cf82a198964ee05",
        "CreativeML-OpenRAIL++-M",
    ),
    "sdxl_configs": (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "462165984030d82259a11f4367a4eed129e94a7b",
        "CreativeML-OpenRAIL++-M",
    ),
    "sdxl_refiner_configs": (
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        "5d4cfe854c9a9a87939ff3653551c2b3c99a4356",
        "CreativeML-OpenRAIL++-M",
    ),
    "sdxs_configs": (
        "IDKiro/sdxs-512-0.9",
        "c332f05f60eb4b453de513be52c2a18c48d8cfe6",
        "CreativeML-OpenRAIL++-M",
    ),
    "segmind_small_sd_configs": (
        "segmind/small-sd",
        "f8b087e68071a29bbe85a4b07e6e1e8998d59aa7",
        "CreativeML-OpenRAIL-M",
    ),
    "segmind_tiny_sd_configs": (
        "segmind/tiny-sd",
        "cad0bd7495fa6c4bcca01b19a723dc91627fe84f",
        "CreativeML-OpenRAIL-M",
    ),
    "segmind_vega_configs": (
        "segmind/Segmind-Vega",
        "7714c4363e5856ff974a4f4b068e8691f26d0b40",
        "Apache-2.0",
    ),
    "show_o_configs": (
        "showlab/show-o-512x512",
        "dbe757d951099968fb102d249d35a764528df2dd",
        "Apache-2.0",
    ),
    "ssd1b_configs": ("segmind/SSD-1B", "60987f37e94cd59c36b1cba832b9f97b57395a10", "Apache-2.0"),
    "t5_tokenizer": ("imagehub synthetic fixture", "project-authored", "Apache-2.0"),
    "wuerstchen_configs": ("warp-ai/wuerstchen", "c3da41406ddd4d9c48c49aa93981a82354351b83", "MIT"),
    "z_image_configs": (
        "Tongyi-MAI/Z-Image",
        "04cc4abb7c5069926f75c9bfde9ef43d49423021",
        "Apache-2.0",
    ),
    "z_image_turbo_configs": (
        "Tongyi-MAI/Z-Image-Turbo",
        "f332072aa78be7aecdf3ee76d5c247082da564a6",
        "Apache-2.0",
    ),
}


def _source(path: Path) -> tuple[str, str, str]:
    name = path.relative_to(FIXTURES).parts[0]
    if name in SOURCES:
        return SOURCES[name]
    if name.startswith("bk_sdm_v2_"):
        return SOURCES["bk_sdm_v2_configs"]
    if name.startswith("bk_sdm_"):
        return SOURCES["bk_sdm_configs"]
    for suffix in (
        "_tensor_shapes.json",
        "_transformer_tensors.json",
        "_text_encoder_tensors.json",
    ):
        if name.endswith(suffix):
            stem = name.removesuffix(suffix)
            for candidate in (f"{stem}_configs", stem):
                if candidate in SOURCES:
                    return SOURCES[candidate]
    raise KeyError(f"missing provenance for {path.relative_to(ROOT)}")


def main() -> None:
    records = []
    for path in sorted(FIXTURES.rglob("*")):
        if not path.is_file() or path.name == "PROVENANCE.json":
            continue
        source, revision, license_name = _source(path)
        records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "source": source,
                "revision": revision,
                "license": license_name,
                "form": "synthetic" if revision == "project-authored" else "published-or-derived",
            }
        )
    (FIXTURES / "PROVENANCE.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
