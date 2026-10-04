"""Assemble docs/validation.json from reference-test measurements and local snapshots.

The backward-compatible ``components`` and top-level ``seed_end_to_end`` fields retain SD 1.x
measurements. SDXL measurements are grouped under ``components_sdxl``, and each model entry may
also contain its own ``seed_end_to_end`` measurement. Missing model directories are errors unless
``--allow-missing`` is passed, in which case they are omitted with a warning.

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
    "Only the checkpoint scheduler is implemented: PNDM (PLMS) for SD 1.x and Euler for SDXL.",
    "No image-quality benchmark (FID, CLIP score).",
    "Text-to-image only; SD 1.x img2img is not implemented.",
]


def header_check(header: dict | None) -> dict | None:
    """Merge per-component header records into missing/extra/mismatch lists and counts."""
    if header is None:
        return None
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
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    import diffusers
    import transformers

    measured = json.loads(args.measurements.read_text(encoding="utf-8"))
    models = []
    for spec in MODEL_CATALOG.values():
        snapshot_dir = args.checkpoints / spec.id
        if not snapshot_dir.is_dir():
            message = f"{spec.id}: snapshot directory is missing: {snapshot_dir}"
            if args.allow_missing:
                print(f"warning: {message}; skipping", file=sys.stderr)
                continue
            sys.exit(message)
        snapshot = read_snapshot(snapshot_dir)
        if snapshot is None or snapshot["revision"] != spec.revision:
            sys.exit(f"{spec.id}: missing snapshot or not the validated revision")
        e2e = measured.get(f"e2e_{spec.id}")
        seed_e2e = measured.get(f"seed_e2e_{spec.id}")
        header = measured.get(f"header_{spec.id}")
        models.append(
            {
                "model_id": spec.id,
                "repo_id": spec.repo_id,
                "revision": spec.revision,
                "provenance": spec.provenance,
                "catalog_id": spec.catalog_id,
                "license": spec.license,
                "status": "full" if e2e and e2e.get("passed") else "experimental",
                "files": snapshot["files"],
                "tensor_header_check": header_check(header),
                "end_to_end": e2e,
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
            if key
            in (
                "sdxl_tokenizers",
                "sdxl_text_encoder",
                "sdxl_text_encoder_2",
                "sdxl_unet",
                "sdxl_vae_decode",
                "sdxl_vae_encode",
                "sdxl_scheduler",
            )
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
