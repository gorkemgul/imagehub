"""Check that the community diffusers HunyuanImage 2.1 weights equal Tencent's official weights.

The catalog loads `hunyuanvideo-community/HunyuanImage-2.1-Diffusers`, a conversion of the official
`tencent/HunyuanImage-2.1` checkpoint. This script converts the official transformer and VAE with
diffusers' own conversion functions (`scripts/convert_hunyuan_image_to_diffusers.py` from a
diffusers checkout, passed with ``--convert-script``) and compares every converted tensor with the
local community snapshot using ``torch.equal``.

The official transformer is read tensor by tensor with HTTP range requests, so it is not stored on
disk; the 1.6 GB official VAE checkpoint is downloaded to a temporary directory.

Usage:
  python scripts/check_hunyuanimage_conversion.py --snapshot checkpoints/hunyuanimage-2-1 \
      --convert-script ../diffusers/scripts/convert_hunyuan_image_to_diffusers.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open

OFFICIAL_REPO = "tencent/HunyuanImage-2.1"
OFFICIAL_REVISION = "e435da11d9e8795a25e224c5ba27b099ed45c55b"
TRANSFORMER = "dit/hunyuanimage2.1.safetensors"
VAE = "vae/vae_2_1/pytorch_model.ckpt"
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _url(path: str) -> str:
    return f"https://huggingface.co/{OFFICIAL_REPO}/resolve/{OFFICIAL_REVISION}/{path}"


def _range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def _official_transformer() -> dict[str, torch.Tensor]:
    url = _url(TRANSFORMER)
    length = struct.unpack("<Q", _range(url, 0, 7))[0]
    header = json.loads(_range(url, 8, 7 + length))
    header.pop("__metadata__", None)
    base = 8 + length
    tensors = {}
    for index, (name, meta) in enumerate(sorted(header.items())):
        begin, end = meta["data_offsets"]
        raw = _range(url, base + begin, base + end - 1) if end > begin else b""
        tensors[name] = (
            torch.frombuffer(bytearray(raw), dtype=DTYPES[meta["dtype"]])
            .reshape(meta["shape"])
            .clone()
        )
        if index % 100 == 0:
            print(f"fetched {index + 1}/{len(header)} official transformer tensors", flush=True)
    return tensors


def _compare(converted: dict[str, torch.Tensor], directory: Path) -> dict[str, object]:
    # One published tensor at a time, so only the converted state dict stays in memory.
    # A tensor published in a narrower dtype than the official one is "rounded" when it equals the
    # official tensor cast to that dtype; anything else that is not identical is "different".
    names: set[str] = set()
    different: list[str] = []
    rounded: list[str] = []
    rounding_max_abs = 0.0
    dtypes: set[str] = set()
    official_dtypes = {str(tensor.dtype) for tensor in converted.values()}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            for name in handle.keys():
                names.add(name)
                tensor = handle.get_tensor(name)
                dtypes.add(str(tensor.dtype))
                if name not in converted:
                    continue
                official = converted[name]
                if official.dtype == tensor.dtype and torch.equal(official, tensor):
                    continue
                if torch.equal(official.to(tensor.dtype), tensor):
                    rounded.append(name)
                    error = (official.float() - tensor.float()).abs().max()
                    rounding_max_abs = max(rounding_max_abs, float(error))
                else:
                    different.append(name)
    return {
        "tensors": len(names),
        "missing": sorted(names - converted.keys()),
        "extra": sorted(converted.keys() - names),
        "different": sorted(different),
        "rounded": len(rounded),
        "rounding_max_abs": rounding_max_abs,
        "published_dtypes": sorted(dtypes),
        "official_dtypes": sorted(official_dtypes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--convert-script", type=Path, required=True)
    args = parser.parse_args()

    # The conversion script parses its own command line when it is imported.
    sys.argv = [str(args.convert_script)]
    spec = importlib.util.spec_from_file_location("convert_hunyuanimage", args.convert_script)
    assert spec is not None and spec.loader is not None
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)

    original = convert.convert_hunyuan_dict_for_tensor_parallel(_official_transformer())
    transformer, leftover = convert.convert_hunyuan_image_transformer_checkpoint_to_diffusers(
        original
    )
    result = {"transformer": _compare(transformer, args.snapshot / "transformer")}
    result["transformer"]["unconverted"] = sorted(leftover)
    del original, transformer, leftover

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vae.ckpt"
        urllib.request.urlretrieve(_url(VAE), path)

        class _Arguments:
            vae_checkpoint_path = str(path)

        vae, leftover = convert.convert_hunyuan_image_vae_checkpoint_to_diffusers(
            convert.load_original_vae_checkpoint(_Arguments())
        )
    result["vae"] = _compare(vae, args.snapshot / "vae")
    result["vae"]["unconverted"] = sorted(leftover)
    result["official"] = {"repo": OFFICIAL_REPO, "revision": OFFICIAL_REVISION}
    print(json.dumps(result, indent=2))
    failed = any(
        result[part][key]
        for part in ("transformer", "vae")
        for key in ("missing", "extra", "different", "unconverted")
    )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
