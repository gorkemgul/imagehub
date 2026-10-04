"""Explicitly download a catalog model as a pinned, verified local snapshot.

Usage: python -m imagehub.pretrained <model-id> <empty-directory> [--revision REF]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from imagehub import download
from imagehub.catalog import model_info
from imagehub.exceptions import DownloadError

SNAPSHOT_FILE = "imagehub_snapshot.json"


def download_pretrained(
    model_id: str,
    destination: str | os.PathLike[str],
    *,
    revision: str | None = None,
    token: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Download the catalog's file list for ``model_id`` into an empty directory.

    The default revision is the commit validated in docs/validation.json. Any other
    revision is downloaded with ``verified_revision: false`` in the manifest.
    """
    spec = model_info(model_id)
    dest = Path(destination)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"Use an empty destination for a reproducible model snapshot: {dest}")
    if spec.repo_id is None:
        raise DownloadError(f"{model_id} has no Hugging Face repository")
    headers = download.hf_headers(token)
    commit, siblings = download.resolve_revision(spec.repo_id, revision or spec.revision, headers)
    missing = [name for name in spec.files if name not in siblings]
    if missing:
        raise DownloadError(f"Published artifacts are missing at {commit}: {', '.join(missing)}")
    verified = commit == spec.revision
    if not verified:
        warnings.warn(
            f"{model_id}: commit {commit} is not the validated revision {spec.revision}; "
            "imagehub's parity evidence does not apply to it",
            UserWarning,
            stacklevel=2,
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".imagehub-download-", dir=dest.parent) as tmp:
        staging = Path(tmp) / "snapshot"
        files: dict[str, str] = {}
        for name in spec.files:
            relative = download.safe_relative(name)
            entry = siblings[name]
            url = f"https://huggingface.co/{spec.repo_id}/resolve/{commit}/" + urllib.parse.quote(
                name, safe="/"
            )
            if progress is not None:
                progress(name)
            lfs_sha256 = (entry.get("lfs") or {}).get("sha256")
            files[name] = download.fetch(
                url,
                staging / relative,
                headers,
                sha256=lfs_sha256,
                size=entry.get("size"),
                # For LFS files blobId is the pointer file's sha, not the content's.
                git_sha1=None if lfs_sha256 else entry.get("blobId"),
            )
        manifest = {
            "model_id": spec.id,
            "repo_id": spec.repo_id,
            "revision": commit,
            "verified_revision": verified,
            "files": files,
        }
        (staging / SNAPSHOT_FILE).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if dest.exists():
            dest.rmdir()  # verified empty above; never replace user files
        shutil.move(str(staging), dest)
    return dest


def read_snapshot(directory: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(directory) / SNAPSHOT_FILE
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m imagehub.pretrained",
        description="Download a catalog model as a pinned, verified local snapshot "
        "(weights: Hub sha256; other files: git blob sha1; all at one commit).",
    )
    parser.add_argument("model_id", help="catalog id, e.g. sd-v1-5")
    parser.add_argument("destination", type=Path, help="empty or missing directory")
    parser.add_argument(
        "--revision", default=None, help="branch, tag or commit (default: validated commit)"
    )
    args = parser.parse_args(argv)
    try:
        path = download_pretrained(
            args.model_id,
            args.destination,
            revision=args.revision,
            token=os.environ.get("HF_TOKEN"),
            progress=lambda name: print(f"downloading {name}", file=sys.stderr, flush=True),
        )
    except (DownloadError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
