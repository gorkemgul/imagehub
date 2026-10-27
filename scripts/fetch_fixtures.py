"""Fetch optional, license-restricted test fixtures from pinned repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "fixtures" / "RESTRICTED.json"
DESTINATION = ROOT / "tests" / "fixtures" / "_restricted"


def _request(url: str, token: str | None, byte_range: str | None = None) -> bytes:
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--max-filesize",
        "16777216",
    ]
    if token:
        command.extend(["--header", f"Authorization: Bearer {token}"])
    if byte_range:
        command.extend(["--range", byte_range.removeprefix("bytes=")])
    command.append(url)
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise OSError(detail or f"curl exited with status {result.returncode}")
    return result.stdout


def _url(entry: dict[str, Any], upstream_path: str | None = None) -> str:
    path = urllib.parse.quote(upstream_path or entry["upstream_path"], safe="/")
    return f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{path}"


def _header_shapes(entry: dict[str, Any], token: str | None) -> bytes:
    merged: dict[str, list[int]] = {}
    paths = entry.get("upstream_paths", [entry["upstream_path"]])
    for upstream_path in paths:
        url = _url(entry, upstream_path)
        prefix = _request(url, token, "bytes=0-7")
        if len(prefix) < 8:
            raise ValueError(f"short safetensors prefix for {upstream_path}")
        length = struct.unpack("<Q", prefix[:8])[0]
        raw = _request(url, token, f"bytes=8-{length + 7}")
        header = json.loads(raw)
        for name, metadata in header.items():
            if name != "__metadata__":
                merged[name] = metadata["shape"]
    indent = entry.get("indent", 2)
    value: Any = sorted(merged) if entry["kind"] == "safetensors-names" else merged
    payload = json.dumps(value, indent=indent).encode()
    if entry.get("newline", False):
        payload += b"\n"
    return payload


def _safetensors_metadata(url: str, token: str | None) -> dict[str, Any]:
    prefix = _request(url, token, "bytes=0-7")
    if len(prefix) < 8:
        raise ValueError(f"short safetensors prefix for {url}")
    length = struct.unpack("<Q", prefix[:8])[0]
    return json.loads(_request(url, token, f"bytes=8-{length + 7}"))


def _derived_header(entry: dict[str, Any], token: str | None) -> bytes:
    kind = entry["kind"]
    if kind == "janus-generation-header":
        header = _safetensors_metadata(_url(entry), token)
        excluded = [
            "model.vision_model.",
            "model.aligner.",
            "lm_head.",
            "model.vqmodel.encoder.",
            "model.vqmodel.quant_conv.",
        ]
        tensors = {
            name: metadata["shape"]
            for name, metadata in header.items()
            if name != "__metadata__"
            and (
                name.startswith("model.generation_")
                or name.startswith("model.vqmodel.decoder.")
                or name.startswith("model.vqmodel.post_quant_conv.")
                or name.startswith("model.vqmodel.quantize.")
            )
        }
        value: dict[str, Any] = {"excluded_prefixes": excluded, "tensors": tensors}
    else:
        value = {"_source": {"repo": entry["repo"], "revision": entry["revision"]}}
        for path in entry["upstream_paths"]:
            component = path.split("/", 1)[0]
            header = _safetensors_metadata(_url(entry, path), token)
            value[component] = {
                name: {"dtype": metadata["dtype"], "shape": metadata["shape"]}
                for name, metadata in header.items()
                if name != "__metadata__"
            }
    return (json.dumps(value, indent=2) + "\n").encode()


def _deepseek_tiny(target: str) -> bytes:
    if target.endswith("tokenizer_config.json"):
        value = {
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": "<｜begin▁of▁sentence｜>",
            "eos_token": "<｜end▁of▁sentence｜>",
            "pad_token": "<｜▁pad▁｜>",
            "boi_token": "<begin_of_image>",
            "padding_side": "left",
        }
        return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    tokens = [
        (20, "<｜begin▁of▁sentence｜>", True),
        (21, "<｜end▁of▁sentence｜>", True),
        (22, "<｜▁pad▁｜>", False),
        (23, "<begin_of_image>", False),
        (24, "<|User|>", False),
        (25, "<|Assistant|>", False),
    ]
    value = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": normalized,
                "special": True,
            }
            for token_id, content, normalized in tokens
        ],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": "[\r\n]"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "Split",
                    "pattern": {"Regex": r"\s?[A-Za-zµÀ-ÖØ-öø-ƺ]+"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "Split",
                    "pattern": {"Regex": r"\s?[!-/:-~！-／：-～‘-‟　-。]+"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "Split",
                    "pattern": {"Regex": r"\s+$"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "Split",
                    "pattern": {"Regex": "[一-龥ࠀ-一가-퟿]+"},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {"type": "Digits", "individual_digits": True},
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": tokens[0][1], "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
            ],
            "pair": [],
            "special_tokens": {
                tokens[0][1]: {"id": tokens[0][1], "ids": [20], "tokens": [tokens[0][1]]}
            },
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {
                "h": 0,
                "i": 1,
                "hi": 2,
                "Ġ": 3,
                "c": 4,
                "a": 5,
                "t": 6,
                "Ġc": 7,
                "Ġca": 8,
                "Ġcat": 9,
                "!": 10,
                "Ċ": 11,
                "1": 12,
                "2": 13,
                "ä": 14,
                "¸": 15,
                "Ń": 16,
            },
            "merges": [["h", "i"], ["Ġ", "c"], ["Ġc", "a"], ["Ġca", "t"]],
        },
    }
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def _payload(entry: dict[str, Any], token: str | None) -> bytes:
    kind = entry.get("kind", "file")
    if kind in {"safetensors-header", "safetensors-names"}:
        return _header_shapes(entry, token)
    if kind in {"janus-generation-header", "diffusers-metadata-headers"}:
        return _derived_header(entry, token)
    if kind == "janus-text-config":
        upstream = json.loads(_request(_url(entry), token))
        value = upstream["text_config"] | {"vq_config": upstream["vq_config"]}
        return (json.dumps(value, indent=2) + "\n").encode()
    if kind == "deepseek-tiny-tokenizer":
        _request(_url(entry), token)
        return _deepseek_tiny(entry["target"])
    return _request(_url(entry), token)


def _valid(path: Path, entry: dict[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def fetch(*, check: bool = False) -> int:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    token = os.environ.get("HF_TOKEN")
    fetched = present = skipped = failed = 0
    for entry in entries:
        target = DESTINATION / entry["target"]
        if _valid(target, entry):
            present += 1
            continue
        if check:
            if entry.get("optional"):
                skipped += 1
                print(f"optional fixture unavailable: {entry['target']}")
                continue
            failed += 1
            print(f"missing or invalid: {entry['target']}")
            continue
        if entry.get("gated") and not token:
            skipped += 1
            print(f"skip gated fixture without HF_TOKEN: {entry['target']}")
            continue
        try:
            payload = _payload(entry, token)
            digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != entry["size"] or digest != entry["sha256"]:
                raise ValueError(f"verification failed (size={len(payload)}, sha256={digest})")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            fetched += 1
        except (OSError, ValueError) as error:
            if entry.get("optional"):
                skipped += 1
                print(f"skip unavailable optional fixture: {entry['target']} ({error})")
                continue
            failed += 1
            print(f"failed {entry['target']}: {error}", file=sys.stderr)
    print(
        "restricted fixtures: "
        f"{fetched} fetched, {present} present, {skipped} gated skipped, {failed} failed"
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the local fixture cache only")
    args = parser.parse_args()
    return fetch(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
