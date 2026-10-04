import json
import os
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
