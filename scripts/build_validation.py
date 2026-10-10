"""Assemble docs/validation.json from reference-test measurements and local snapshots.

The backward-compatible ``components`` and top-level ``seed_end_to_end`` fields retain SD 1.x
measurements. SDXL, Qwen-Image, SD3, FLUX.1, Krea 2, HunyuanDiT, Kandinsky 2.2, Stable Cascade,
PixArt, Sana, Lumina-Image 2.0, aMUSEd, AuraFlow, Wuerstchen v2, LongCat-Image, CogView4 and
Ideogram 4 and OmniGen
measurements are
grouped in flat
``components_sdxl``, ``components_qwen_image``, ``components_sd3``, ``components_flux1``,
``components_krea2``, ``components_hunyuan_dit``, ``components_kandinsky22``,
``components_stable_cascade``, ``components_pixart``, ``components_sana``,
``components_lumina2``, ``components_amused``, ``components_auraflow``,
``components_wuerstchen``,
``components_longcat_image``,
``components_cogview4``, ``components_ideogram4`` and ``components_omnigen`` mappings: the base
model keeps its historical unsuffixed keys, while other models use ``<key>_<model-id>``. Each
model entry may also contain its own ``seed_end_to_end`` measurement. When a model directory is
missing, its existing entry in the output is retained unchanged; a missing directory without an
existing entry is an error.

Usage:
  python scripts/build_validation.py --measurements docs/validation-measurements.json \
      --checkpoints checkpoints --machine "Apple M2 Max, 96 GB" \
      --tests-log /tmp/imagehub-reference.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import re
import sys
from pathlib import Path

import torch

import imagehub
from imagehub.catalog import MODEL_CATALOG
from imagehub.checkpoints import read_snapshot

LIMITS = [
    "Parity is claimed only on CPU in float32 against diffusers 0.40.0 and transformers 5.18.0.",
    "MPS and CUDA runs are smoke tests (finite output, shape, CPU difference recorded); "
    "no parity claim.",
    "No float16 or bfloat16 parity.",
    "No safety checker or invisible watermark; the reference ran with safety_checker=None.",
    "Only the checkpoint scheduler is implemented: PNDM (PLMS) for SD 1.x, DDIM for SD 2.1, Euler "
    "with trailing spacing for SD-Turbo, LCM for LCM Dreamshaper v7, Hyper-SDXL and DMD2, "
    "Euler for SDXL, "
    "flow-match Euler for Z-Image and SD 3.5, flow-match Euler with a static shift for FLUX.1 "
    "schnell, Lumina-Image 2.0 and AuraFlow, and dynamic shifting for FLUX.1 dev, Qwen-Image 2.1 "
    "and LongCat-Image, flow-match "
    "Euler with exponential shifting for Krea 2 and linear shifting for CogView4, DDPM for "
    "HunyuanDiT, UnCLIP and DDPM for Kandinsky 2.2, DDPM Wuerstchen for Stable Cascade and "
    "Wuerstchen v2, flow-match Euler with inverted sigmas for OmniGen, "
    "DPM-Solver++ for PixArt and Sana (flow sigmas), and the cosine masked-token scheduler for "
    "aMUSEd.",
    "No image-quality benchmark (FID, CLIP score).",
    "Text-to-image only; SD 1.x img2img is not implemented.",
]
SDXL_COMPONENT_KEYS = (
    "sdxl_tokenizers",
    "sdxl_text_encoder",
    "sdxl_text_encoder_2",
    "sdxl_unet",
    "sdxl_vae_decode",
    "sdxl_vae_encode",
    "sdxl_scheduler",
)
SD_TURBO_COMPONENT_KEYS = (
    "sd_turbo_tokenizer",
    "sd_turbo_text_encoder",
    "sd_turbo_unet",
    "sd_turbo_vae_decode",
    "sd_turbo_vae_encode",
    "sd_turbo_scheduler",
)
SD21_COMPONENT_KEYS = (
    "sd21_tokenizer",
    "sd21_text_encoder",
    "sd21_unet",
    "sd21_vae_decode",
    "sd21_vae_encode",
    "sd21_scheduler",
)
LCM_COMPONENT_KEYS = (
    "lcm_text_encoder",
    "lcm_unet",
    "lcm_vae_decode",
    "lcm_vae_encode",
    "lcm_scheduler",
)
Z_IMAGE_COMPONENT_KEYS = (
    "z_image_tokenizer",
    "z_image_text_encoder",
    "z_image_transformer",
    "z_image_vae_decode",
    "z_image_vae_encode",
    "z_image_scheduler",
)
QWEN_IMAGE_COMPONENT_KEYS = (
    "qwen_image_prompt_ids",
    "qwen_image_text_encoder",
    "qwen_image_transformer",
    "qwen_image_vae_decode",
    "qwen_image_vae_encode",
    "qwen_image_scheduler",
)
JANUS_COMPONENT_KEYS = ("janus_language_model", "janus_vq_decode")
SD3_COMPONENT_KEYS = (
    "sd3_tokenizers",
    "sd3_text_encoder",
    "sd3_text_encoder_2",
    "sd3_text_encoder_3",
    "sd3_transformer",
    "sd3_vae_decode",
    "sd3_vae_encode",
    "sd3_scheduler",
)


def header_check(header: dict | None) -> dict | None:
    """Merge per-component header records into missing/extra/mismatch lists and counts."""
    if header is None:
        return None
    if all(key in header for key in ("missing", "extra", "mismatch", "tensors")):
        return header
    if all(set(entry) == {"fp8", "scales", "tensors"} for entry in header.values()):
        return header
    merged: dict = {"missing": [], "extra": [], "mismatch": [], "tensors": {}}
    for component, entry in header.items():
        for key in ("missing", "extra", "mismatch"):
            merged[key] += [f"{component}.{item}" for item in entry[key]]
        merged["tensors"][component] = entry["tensors"]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--tests-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/validation.json"))
    args = parser.parse_args()

    import diffusers
    import transformers

    measured = json.loads(args.measurements.read_text(encoding="utf-8"))
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.is_file() else {}
    existing_models = {model["model_id"]: model for model in existing.get("models", [])}
    models = []
    for spec in MODEL_CATALOG.values():
        snapshot_dir = args.checkpoints / spec.id
        if not snapshot_dir.is_dir():
            message = f"{spec.id}: snapshot directory is missing: {snapshot_dir}"
            if spec.id in existing_models:
                print(f"warning: {message}; keeping existing entry", file=sys.stderr)
                models.append(existing_models[spec.id])
                continue
            sys.exit(message)
        snapshot = read_snapshot(snapshot_dir)
        if snapshot is None or snapshot["revision"] != spec.revision:
            sys.exit(f"{spec.id}: missing snapshot or not the validated revision")
        measurement_id = {
            "krea2-turbo": "krea2_turbo",
            "longcat-image": "longcat_image",
            "flux1-kontext-dev": "flux1_kontext_dev",
        }.get(spec.id, spec.id)
        e2e = measured.get(f"e2e_{measurement_id}")
        e2e_batch2 = measured.get(f"e2e_batch2_{spec.id}")
        e2e_multistep = measured.get(f"e2e_multistep_{spec.id}")
        e2e_guided = measured.get(f"e2e_guided_{spec.id}")
        e2e_image = measured.get(f"e2e_image_{spec.id}")
        seed_e2e = measured.get(f"seed_e2e_{measurement_id}")
        header = measured.get(f"header_{measurement_id}") or measured.get(f"header_{spec.id}")
        reference = {package: dict(details) for package, details in spec.reference.items()}
        if spec.id == "ideogram4-fp8":
            reference = {
                "ideogram4": {
                    "repository": "ideogram-oss/ideogram4",
                    "git_commit": "990fe1c4e950bb9e9dc90e01c0ad98ba434f83c2",
                },
                "transformers": {"version": "5.18.0"},
            }
        models.append(
            {
                "model_id": spec.id,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "provenance": spec.provenance,
                "catalog_id": spec.catalog_id,
                "license": spec.license,
                "reference": reference,
                "status": "full"
                if e2e
                and (
                    e2e.get("passed")
                    or (spec.id == "ideogram4-fp8" and e2e.get("tolerance") == 0.0)
                )
                else "experimental",
                "files": snapshot["files"],
                **({"base": snapshot["base"]} if "base" in snapshot else {}),
                "tensor_header_check": header_check(header),
                "end_to_end": e2e,
                **({"end_to_end_batch2": e2e_batch2} if e2e_batch2 else {}),
                **({"end_to_end_multistep": e2e_multistep} if e2e_multistep else {}),
                **({"end_to_end_guided": e2e_guided} if e2e_guided else {}),
                **({"end_to_end_image": e2e_image} if e2e_image else {}),
                "seed_end_to_end": seed_e2e,
            }
        )
    summary = re.findall(r"=+ (.+) in [\d.]+s", args.tests_log.read_text(encoding="utf-8"))
    document = {
        "version": imagehub.__version__,
        "date": datetime.date.today().isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "machine": args.machine,
            "platform": platform.platform(),
            **measured.get("environment", {}),
        },
        "models": models,
        "components": {
            key: value
            for key, value in measured.items()
            if key
            in ("tokenizer", "text_encoder", "unet", "vae_decode", "vae_encode", "seed_noise")
        },
        "components_sdxl": {
            key: value
            for key, value in measured.items()
            if key in SDXL_COMPONENT_KEYS
            or key.startswith("sdxl_lightning_unet_sdxl-lightning-")
            or key.startswith("hyper_sd_")
            or key.startswith("dmd2-")
            or any(
                key == f"{component}_{spec.id}"
                for component in SDXL_COMPONENT_KEYS
                for spec in MODEL_CATALOG.values()
                if spec.family == "stable-diffusion-xl" and spec.id != "sdxl-base-1.0"
            )
        },
        "components_sd_turbo": {
            key: value for key, value in measured.items() if key in SD_TURBO_COMPONENT_KEYS
        },
        "components_sd21": {
            key: value for key, value in measured.items() if key in SD21_COMPONENT_KEYS
        },
        "components_lcm": {
            key: value for key, value in measured.items() if key in LCM_COMPONENT_KEYS
        },
        "components_z_image": {
            key: value
            for key, value in measured.items()
            if key in Z_IMAGE_COMPONENT_KEYS
            or any(
                key == f"{component}_{spec.id}"
                for component in Z_IMAGE_COMPONENT_KEYS
                for spec in MODEL_CATALOG.values()
                if spec.family == "z-image" and spec.id != "z-image-turbo"
            )
        },
        "components_qwen_image": {
            key: value for key, value in measured.items() if key in QWEN_IMAGE_COMPONENT_KEYS
        },
        "components_janus": {
            key: value
            for key, value in measured.items()
            if key in JANUS_COMPONENT_KEYS
            or any(
                key == f"{component}_{spec.id}"
                for component in JANUS_COMPONENT_KEYS
                for spec in MODEL_CATALOG.values()
                if spec.family == "janus" and spec.id != "janus-pro-1b"
            )
        },
        "components_sd3": {
            key: value
            for key, value in measured.items()
            if key in SD3_COMPONENT_KEYS
            or any(
                key == f"{component}_{spec.id}"
                for component in SD3_COMPONENT_KEYS
                for spec in MODEL_CATALOG.values()
                if spec.family == "stable-diffusion-3" and spec.id != "sd3.5-medium"
            )
        },
        "components_flux1": {
            key: value
            for key, value in measured.items()
            if key.startswith(("flux1_", "flux_kontext_", "e2e_flux1_kontext_dev_no_image"))
        },
        "components_krea2": {
            key: value for key, value in measured.items() if key.startswith("krea2_")
        },
        "components_flux2": {
            key: value for key, value in measured.items() if key.startswith("flux2_")
        },
        "components_hunyuan_dit": {
            key: value for key, value in measured.items() if "hunyuan_dit" in key
        },
        "components_kandinsky22": {
            key: value for key, value in measured.items() if "kandinsky22" in key
        },
        "components_stable_cascade": {
            key: value
            for key, value in measured.items()
            if not key.startswith(("header_", "e2e_", "seed_e2e_"))
            and any(
                marker in key
                for marker in (
                    "stable_cascade",
                    "stable-cascade",
                    "paella",
                    "ddpm_wuerstchen",
                    "clip_text_mask",
                )
            )
        },
        "components_pixart": {key: value for key, value in measured.items() if "pixart" in key},
        "components_sana": {
            key: value for key, value in measured.items() if key.startswith("sana_")
        },
        "components_lumina2": {
            key: value for key, value in measured.items() if key.startswith("lumina2_")
        },
        "components_amused": {
            key: value for key, value in measured.items() if key.startswith(("amused_", "amused-"))
        },
        "components_auraflow": {
            key: value for key, value in measured.items() if key.startswith("auraflow_")
        },
        "components_wuerstchen": {
            key: value
            for key, value in measured.items()
            if key.startswith(("wuerstchen_", "wuerstchen-v2_"))
        },
        "components_longcat_image": {
            key: value for key, value in measured.items() if key.startswith("longcat_image_")
        },
        "components_cogview4": {
            key: value for key, value in measured.items() if key.startswith("cogview4_")
        },
        "components_ideogram4": {
            key: value for key, value in measured.items() if key.startswith("ideogram4_")
        },
        "components_omnigen": {
            key: value for key, value in measured.items() if key.startswith("omnigen_")
        },
        "seed_end_to_end": measured.get("seed_e2e_sd-v1-5"),
        "devices": {k: v for k, v in measured.items() if k.startswith("smoke_")},
        "limits": LIMITS,
        "tests": {"reference_summary": summary[-1] if summary else None},
    }
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
