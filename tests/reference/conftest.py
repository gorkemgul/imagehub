import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

_MEASUREMENTS: dict[str, dict] = {}


def pytest_collection_modifyitems(config, items):
    if os.environ.get("IMAGEHUB_REFERENCE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="Set IMAGEHUB_REFERENCE_TESTS=1 to run reference comparisons")
    for item in items:
        if "reference" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def reference_threads():
    threads = int(os.environ.get("IMAGEHUB_REFERENCE_THREADS", os.cpu_count() or 1))
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    _MEASUREMENTS["environment"] = {"torch_threads": threads}
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def record():
    def add(name: str, **values):
        _MEASUREMENTS[name] = values

    return add


def pytest_sessionfinish(session, exitstatus):
    path = os.environ.get("IMAGEHUB_RECORD")
    if not path or len(_MEASUREMENTS) <= 1:
        return
    target = Path(path)
    existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    existing.update(_MEASUREMENTS)
    target.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")


REFERENCE_ROOT = Path("/Users/gorkem.gul/Personal/imagehub/checkpoints/ref-src/LlamaGen")
_LLAMAGEN_PACKAGES = ("autoregressive", "tokenizer", "utils")


@pytest.fixture(scope="module")
def llamagen_reference():
    """The official LlamaGen modules ``(gpt, generate, vq_model)`` imported from a checkout."""
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"LlamaGen reference checkout not found at {REFERENCE_ROOT}")
    sys.path.insert(0, str(REFERENCE_ROOT))
    try:
        gpt = importlib.import_module("autoregressive.models.gpt")
        generate = importlib.import_module("autoregressive.models.generate")
        vq = importlib.import_module("tokenizer.tokenizer_image.vq_model")
        yield gpt, generate, vq
    finally:
        sys.path.remove(str(REFERENCE_ROOT))
        for name in list(sys.modules):
            if name.split(".")[0] in _LLAMAGEN_PACKAGES:
                del sys.modules[name]
