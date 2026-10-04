"""Explicit, standard-library Hugging Face downloads; never run during inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from imagehub.exceptions import DownloadError

_PYTHON_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".pyw"})
USER_AGENT = "imagehub/0.1"
_SSL_HINT = (
    "TLS certificate verification failed. On python.org builds for macOS run "
    "'Install Certificates.command' from the Python application folder, or set "
    "SSL_CERT_FILE to a CA bundle."
)


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        if urllib.parse.urlparse(newurl).scheme != "https":
            raise DownloadError("Model downloads require HTTPS redirects")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and (
            urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def _open(url: str, headers: dict[str, str]) -> Any:
    opener = urllib.request.build_opener(_Redirect())
    try:
        return opener.open(urllib.request.Request(url, headers=headers), timeout=30)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            raise DownloadError(
                f"Access denied (HTTP {error.code}) for {url}. The repository may be gated: "
                "accept its terms on huggingface.co and set HF_TOKEN."
            ) from error
        if error.code == 404:
            raise DownloadError(f"File not found (HTTP 404): {url}") from error
        raise DownloadError(f"HTTP {error.code} for {url}") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, ssl.SSLCertVerificationError):
            raise DownloadError(_SSL_HINT) from error
        raise DownloadError(f"Network error for {url}: {error.reason}") from error
    except TimeoutError as error:
        raise DownloadError(f"Network error for {url}: timed out") from error


def _read_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    with _open(url, headers) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise DownloadError(f"Unexpected response from {url}")
    return value


def hf_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def resolve_revision(
    repo_id: str, revision: str, headers: dict[str, str]
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return the immutable commit for ``revision`` and its files keyed by path."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo_id):
        raise DownloadError("repo_id must have the form owner/model")
    quoted = urllib.parse.quote(revision, safe="")
    info = _read_json(
        f"https://huggingface.co/api/models/{repo_id}/revision/{quoted}?blobs=true", headers
    )
    commit = info.get("sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise DownloadError("The model API did not return an immutable commit hash")
    return commit, {entry["rfilename"]: entry for entry in info.get("siblings", [])}


def safe_relative(name: str) -> Path:
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise DownloadError(f"Unsafe path in model snapshot: {name!r}")
    if path.suffix.lower() in _PYTHON_SUFFIXES:
        raise DownloadError(f"Refusing to download Python code: {name!r}")
    return path


def fetch(
    url: str,
    target: Path,
    headers: dict[str, str],
    *,
    sha256: str | None,
    size: int | None,
    git_sha1: str | None = None,
) -> str:
    """Download ``url`` to ``target`` atomically and return its sha256.

    Verifies the size when published, the sha256 (Hub LFS files) and the git blob sha1
    (the Hub's ``blobId`` of non-LFS files). At least one checksum is required.
    """
    if not url.startswith("https://"):
        raise DownloadError("Model downloads require HTTPS")
    if sha256 is None and git_sha1 is None:
        raise DownloadError(
            f"Cannot verify {target.name}: the Hub published neither sha256 nor git blob sha1"
        )
    if git_sha1 is not None and size is None:
        raise DownloadError(
            f"Cannot verify {target.name}: its size is needed for the git blob sha1"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".imagehub-", delete=False
        ) as sink:
            temporary = sink.name
            digest = hashlib.sha256()
            blob = hashlib.sha1(b"blob %d\0" % size, usedforsecurity=False) if git_sha1 else None
            received = 0
            with _open(url, headers) as response:
                while data := response.read(1 << 20):
                    sink.write(data)
                    digest.update(data)
                    if blob is not None:
                        blob.update(data)
                    received += len(data)
        if size is not None and received != size:
            raise DownloadError(
                f"Incomplete download for {target.name}: {received} of {size} bytes"
            )
        if sha256 is not None and digest.hexdigest() != sha256:
            raise DownloadError(f"Checksum mismatch for {target.name}")
        if blob is not None and blob.hexdigest() != git_sha1:
            raise DownloadError(f"Checksum mismatch for {target.name} (git blob sha1)")
        os.replace(temporary, target)
        temporary = None
        return digest.hexdigest()
    except OSError as error:
        raise DownloadError(f"Failed to download {target.name}: {error}") from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
