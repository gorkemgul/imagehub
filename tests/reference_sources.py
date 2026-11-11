"""Locate local checkouts of the official code that some reference tests import."""

from __future__ import annotations

import os
from pathlib import Path


def reference_source(name: str) -> Path:
    """Checkout ``name`` under ``IMAGEHUB_REFERENCE_SOURCES`` (default ``checkpoints/ref-src``).

    The path may not exist; callers skip their test when the checkout is missing.
    """
    root = os.environ.get("IMAGEHUB_REFERENCE_SOURCES")
    base = Path(root) if root else Path(__file__).resolve().parents[1] / "checkpoints" / "ref-src"
    return base / name
