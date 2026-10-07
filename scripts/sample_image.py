"""Write the documentation sample image for a catalog model.

The sample places the reference pipeline's seed-42 output (left) next to imagehub's (right) for the
validation prompt at the model's defaults, each downscaled to 512 pixels on the long side, and
stores it as WebP in docs/images/<model-id>.webp. Both inputs are full-size PNG files; imagehub's
can also be generated from a local snapshot.

Usage:
  python scripts/sample_image.py hyper-sdxl-1step --reference ref.png --ours ours.png
  python scripts/sample_image.py hyper-sdxl-1step --reference ref.png \
      --snapshot checkpoints/hyper-sdxl-1step
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from imagehub import pipeline
from imagehub.catalog import MODEL_CATALOG

PROMPT = "a photograph of an astronaut riding a horse"
SEED = 42
SIZE = 512
GAP = 16
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "images"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model_id", choices=sorted(MODEL_CATALOG))
    parser.add_argument("--reference", type=Path, required=True, help="reference seed-42 PNG")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ours", type=Path, help="imagehub seed-42 PNG")
    source.add_argument(
        "--snapshot", type=Path, help="local snapshot to generate imagehub's output"
    )
    args = parser.parse_args()

    reference = Image.open(args.reference).convert("RGB")
    if args.ours is not None:
        ours = Image.open(args.ours).convert("RGB")
    else:
        with pipeline("t2i", backend=args.model_id, model=args.snapshot) as pipe:
            ours = pipe(PROMPT, seed=SEED).images[0].to_pil().convert("RGB")
    difference = np.abs(np.asarray(reference, dtype=np.int16) - np.asarray(ours, dtype=np.int16))
    print(f"max pixel difference: {difference.max()}, differing values: {(difference > 0).sum()}")

    for image in (reference, ours):
        image.thumbnail((SIZE, SIZE), Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGB", (reference.width + GAP + ours.width, max(reference.height, ours.height)), "white"
    )
    canvas.paste(reference, (0, 0))
    canvas.paste(ours, (reference.width + GAP, 0))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{args.model_id}.webp"
    canvas.save(target, "WEBP", quality=85, method=6)
    print(f"{target} ({target.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
