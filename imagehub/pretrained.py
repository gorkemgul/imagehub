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

from imagehub import download
from imagehub.catalog import model_info
from imagehub.checkpoints import SNAPSHOT_FILE, read_snapshot
from imagehub.exceptions import DownloadError

__all__ = ["SNAPSHOT_FILE", "download_pretrained", "main", "read_snapshot"]


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
    base = model_info(spec.base) if spec.base is not None else None
    if base is not None and base.base is not None:
        raise DownloadError(f"{model_id}: base model {base.id} is itself composite")
    if base is not None and base.repo_id is None:
        raise DownloadError(f"{base.id} has no Hugging Face repository")
    headers = download.hf_headers(token)
    commit, siblings = download.resolve_revision(spec.repo_id, revision or spec.revision, headers)
    missing = [name for name in spec.files if name not in siblings]
    base_commit = None
    base_siblings = None
    if base is not None:
        assert base.repo_id is not None
        base_commit, base_siblings = download.resolve_revision(base.repo_id, base.revision, headers)
        missing += [
            name for name in base.files if name not in spec.overlay and name not in base_siblings
        ]
    if missing:
        raise DownloadError(f"Published artifacts are missing at {commit}: {', '.join(missing)}")
    verified = commit == spec.revision and (base is None or base_commit == base.revision)
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
        file_plan = []
        if base is not None:
            assert base.repo_id is not None
            assert base_commit is not None
            assert base_siblings is not None
            for name in base.files:
                if name not in spec.overlay:
                    file_plan.append((name, name, base.repo_id, base_commit, base_siblings[name]))
            overlaid_sources = set(spec.overlay.values())
            file_plan.extend(
                (destination, source, spec.repo_id, commit, siblings[source])
                for destination, source in spec.overlay.items()
            )
            file_plan.extend(
                (name, name, spec.repo_id, commit, siblings[name])
                for name in spec.files
                if name not in overlaid_sources
            )
        else:
            file_plan.extend(
                (name, name, spec.repo_id, commit, siblings[name]) for name in spec.files
            )
        files: dict[str, str | dict[str, str]] = {}
        for destination_name, source_name, repo_id, file_commit, entry in file_plan:
            relative = download.safe_relative(destination_name)
            download.safe_relative(source_name)
            url = f"https://huggingface.co/{repo_id}/resolve/{file_commit}/" + urllib.parse.quote(
                source_name, safe="/"
            )
            if progress is not None:
                progress(destination_name)
            lfs_sha256 = (entry.get("lfs") or {}).get("sha256")
            digest = download.fetch(
                url,
                staging / relative,
                headers,
                sha256=lfs_sha256,
                size=entry.get("size"),
                # For LFS files blobId is the pointer file's sha, not the content's.
                git_sha1=None if lfs_sha256 else entry.get("blobId"),
            )
            files[destination_name] = (
                {"repo_id": repo_id, "revision": file_commit, "sha256": digest}
                if base is not None
                else digest
            )
        manifest = {
            "model_id": spec.id,
            "repo_id": spec.repo_id,
            "revision": commit,
            "verified_revision": verified,
            "files": files,
            **(
                {"base": {"model_id": base.id, "revision": base_commit}} if base is not None else {}
            ),
        }
        (staging / SNAPSHOT_FILE).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if dest.exists():
            dest.rmdir()  # verified empty above; never replace user files
        shutil.move(str(staging), dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m imagehub.pretrained",
        description="Download a catalog model as a pinned, verified local snapshot "
        "(weights: Hub sha256; other files: git blob sha1).",
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
