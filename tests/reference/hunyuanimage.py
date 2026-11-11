"""Isolated diffusers and transformers reference stages for HunyuanImage tests."""

import json
import sys


def run(*, body: str, args: dict[str, object], out_path: str) -> None:
    """Execute one reference-only stage and save its tensor result to ``out_path``."""
    payload = dict(args) | {"out": out_path}
    original = sys.argv
    try:
        sys.argv = [sys.argv[0], json.dumps(payload)]
        exec(body, {})  # noqa: S102
    finally:
        sys.argv = original
