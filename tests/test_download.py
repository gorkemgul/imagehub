import hashlib
import io
import json
import ssl
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from imagehub import download, pretrained
from imagehub.catalog import model_info
from imagehub.download import _Redirect, safe_relative
from imagehub.exceptions import DownloadError
from imagehub.pretrained import SNAPSHOT_FILE, download_pretrained, main, read_snapshot

SPEC = model_info("sd-v1-5")


def git_blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fake_hub(monkeypatch, *, commit=SPEC.revision, corrupt=None, short=None, drop=None):
    contents = {name: f"content of {name}".encode() for name in SPEC.files}
    siblings = []
    for name, data in contents.items():
        if name == drop:
            continue
        entry = {"rfilename": name, "size": len(data)}
        if name.endswith((".safetensors", ".bin")):
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
            entry["blobId"] = "f" * 40  # the LFS pointer's sha, which is never the content's
        else:
            entry["blobId"] = git_blob_sha1(data)
        siblings.append(entry)
    siblings.append({"rfilename": "safety_checker/model.safetensors", "size": 1})
    calls = []

    def request(url, headers):
        calls.append((url, dict(headers)))
        if "/api/models/" in url:
            return io.BytesIO(json.dumps({"sha": commit, "siblings": siblings}).encode())
        name = url.split(f"/resolve/{commit}/", 1)[1]
        data = contents[name]
        if name == corrupt:
            data = b"X" + data[1:]
        if name == short:
            data = data[:-1]
        return io.BytesIO(data)

    monkeypatch.setattr(download, "_open", request)
    return contents, calls


def fake_composite_hub(monkeypatch, *, model_id="sdxl-lightning-2step", drop=None, corrupt=None):
    spec = model_info(model_id)
    base = model_info(spec.base)
    repositories = {
        base.repo_id: {name: f"base content of {name}".encode() for name in base.files},
        spec.repo_id: {name: f"overlay content of {name}".encode() for name in spec.files},
    }
    commits = {base.repo_id: base.revision, spec.repo_id: spec.revision}
    calls = []

    def request(url, headers):
        calls.append((url, dict(headers)))
        repo_id = next(repo for repo in repositories if f"/{repo}/" in url)
        contents = repositories[repo_id]
        commit = commits[repo_id]
        if "/api/models/" in url:
            siblings = []
            for name, data in contents.items():
                if (repo_id, name) == drop:
                    continue
                siblings.append(
                    {
                        "rfilename": name,
                        "size": len(data),
                        "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                        "blobId": "f" * 40,
                    }
                )
            return io.BytesIO(json.dumps({"sha": commit, "siblings": siblings}).encode())
        name = url.split(f"/resolve/{commit}/", 1)[1]
        data = contents[name]
        if (repo_id, name) == corrupt:
            data = b"X" + data[1:]
        return io.BytesIO(data)

    monkeypatch.setattr(download, "_open", request)
    return spec, base, repositories, calls


def test_downloads_pinned_verified_snapshot(monkeypatch, tmp_path):
    contents, calls = fake_hub(monkeypatch)
    dest = download_pretrained("sd-v1-5", tmp_path / "sd15")
    manifest = read_snapshot(dest)
    assert manifest["model_id"] == "sd-v1-5"
    assert manifest["revision"] == SPEC.revision
    assert manifest["verified_revision"] is True
    assert set(manifest["files"]) == set(SPEC.files)
    for name, data in contents.items():
        assert (dest / name).read_bytes() == data
        assert manifest["files"][name] == hashlib.sha256(data).hexdigest()
    assert not (dest / "safety_checker").exists()
    assert f"/revision/{SPEC.revision}" in calls[0][0]
    assert all(f"/resolve/{SPEC.revision}/" in url for url, _ in calls[1:])
    assert all("Authorization" not in headers for _, headers in calls)
    with pytest.raises(FileExistsError):
        download_pretrained("sd-v1-5", dest)


def test_downloads_composite_snapshot_with_provenance(monkeypatch, tmp_path):
    spec = model_info("sdxl-lightning-2step")
    base = model_info(spec.base)
    overlay_source = spec.files[0]
    overlay_path = next(iter(spec.overlay))
    spec, base, repositories, calls = fake_composite_hub(
        monkeypatch, drop=(base.repo_id, overlay_path)
    )
    dest = download_pretrained(spec.id, tmp_path / "lightning")
    manifest = read_snapshot(dest)

    assert manifest["model_id"] == spec.id
    assert manifest["repo_id"] == spec.repo_id
    assert manifest["revision"] == spec.revision
    assert manifest["verified_revision"] is True
    assert manifest["base"] == {"model_id": base.id, "revision": base.revision}
    assert set(manifest["files"]) == (set(base.files) - {overlay_path}) | {overlay_path}
    assert manifest["files"]["model_index.json"] == {
        "repo_id": base.repo_id,
        "revision": base.revision,
        "sha256": hashlib.sha256(repositories[base.repo_id]["model_index.json"]).hexdigest(),
    }
    assert manifest["files"][overlay_path] == {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "sha256": hashlib.sha256(repositories[spec.repo_id][overlay_source]).hexdigest(),
    }
    assert (dest / overlay_path).read_bytes() == repositories[spec.repo_id][overlay_source]
    assert not (dest / overlay_source).exists()
    assert not any(
        f"/{base.repo_id}/resolve/{base.revision}/{overlay_path}" in url for url, _ in calls
    )


@pytest.mark.parametrize(
    "model_id", ["sana-1600m-1024", "sana-600m-1024", "lumina-image-2.0", "auraflow-v0.3"]
)
def test_downloads_single_revision_snapshot_pinned_to_its_revision(monkeypatch, tmp_path, model_id):
    spec = model_info(model_id)
    contents = {name: f"content of {name}".encode() for name in spec.files}

    def request(url, headers):
        assert f"/{spec.repo_id}/" in url
        if "/api/models/" in url:
            siblings = [
                {
                    "rfilename": name,
                    "size": len(data),
                    "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                    "blobId": "f" * 40,
                }
                for name, data in contents.items()
            ]
            return io.BytesIO(json.dumps({"sha": spec.revision, "siblings": siblings}).encode())
        return io.BytesIO(contents[url.split(f"/resolve/{spec.revision}/", 1)[1]])

    monkeypatch.setattr(download, "_open", request)
    dest = download_pretrained(model_id, tmp_path / "snapshot")
    manifest = read_snapshot(dest)
    assert manifest["model_id"] == model_id
    assert manifest["revision"] == spec.revision
    assert manifest["verified_revision"] is True
    assert set(manifest["files"]) == set(spec.files)
    for name, data in contents.items():
        assert (dest / name).read_bytes() == data


@pytest.mark.parametrize("model_id", ["kandinsky-2.2", "stable-cascade", "stable-cascade-lite"])
def test_downloads_subrepository_into_snapshot_subfolder(monkeypatch, tmp_path, model_id):
    spec = model_info(model_id)
    prior = spec.subfolders["prior"]
    repositories = {
        spec.repo_id: {name: f"decoder {name}".encode() for name in spec.files},
        prior.repo_id: {name: f"prior {name}".encode() for name in prior.files},
    }
    commits = {spec.repo_id: spec.revision, prior.repo_id: prior.revision}
    calls = []

    def request(url, headers):
        calls.append(url)
        repo_id = next(repo for repo in repositories if f"/{repo}/" in url)
        contents = repositories[repo_id]
        commit = commits[repo_id]
        if "/api/models/" in url:
            siblings = [
                {
                    "rfilename": name,
                    "size": len(data),
                    "lfs": {"sha256": hashlib.sha256(data).hexdigest()},
                    "blobId": "f" * 40,
                }
                for name, data in contents.items()
            ]
            return io.BytesIO(json.dumps({"sha": commit, "siblings": siblings}).encode())
        name = url.split(f"/resolve/{commit}/", 1)[1]
        return io.BytesIO(contents[name])

    monkeypatch.setattr(download, "_open", request)
    dest = download_pretrained(spec.id, tmp_path / "snapshot")
    manifest = read_snapshot(dest)
    assert manifest["verified_revision"] is True
    assert set(manifest["files"]) == set(spec.files) | {f"prior/{name}" for name in prior.files}
    for name, data in repositories[spec.repo_id].items():
        assert (dest / name).read_bytes() == data
        assert manifest["files"][name] == {
            "repo_id": spec.repo_id,
            "revision": spec.revision,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    for name, data in repositories[prior.repo_id].items():
        destination = f"prior/{name}"
        assert (dest / destination).read_bytes() == data
        assert manifest["files"][destination] == {
            "repo_id": prior.repo_id,
            "revision": prior.revision,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    assert any(f"/{prior.repo_id}/revision/{prior.revision}" in url for url in calls)


def test_composite_excludes_base_file_from_snapshot(monkeypatch, tmp_path):
    spec = model_info("dmd2-sdxl-1step")
    base = model_info(spec.base)
    excluded = spec.base_exclude[0]
    spec, base, repositories, calls = fake_composite_hub(
        monkeypatch,
        model_id=spec.id,
        drop=(base.repo_id, excluded),
    )
    dest = download_pretrained(spec.id, tmp_path / "dmd2")
    manifest = read_snapshot(dest)

    assert set(manifest["files"]) == (set(base.files) - set(spec.base_exclude)) | set(spec.overlay)
    assert not (dest / excluded).exists()
    assert (dest / "unet/diffusion_pytorch_model.bin").read_bytes() == repositories[spec.repo_id][
        spec.files[0]
    ]
    assert not any(f"/{base.repo_id}/resolve/{base.revision}/{excluded}" in url for url, _ in calls)


def test_composite_missing_overlay_fails_before_downloading(monkeypatch, tmp_path):
    spec = model_info("sdxl-lightning-2step")
    _, _, _, calls = fake_composite_hub(monkeypatch, drop=(spec.repo_id, spec.files[0]))
    with pytest.raises(DownloadError, match=spec.files[0]):
        download_pretrained(spec.id, tmp_path / "lightning")
    assert not any("/resolve/" in url for url, _ in calls)


def test_composite_overlay_hash_mismatch_leaves_nothing(monkeypatch, tmp_path):
    spec = model_info("sdxl-lightning-2step")
    fake_composite_hub(monkeypatch, corrupt=(spec.repo_id, spec.files[0]))
    with pytest.raises(DownloadError, match="Checksum"):
        download_pretrained(spec.id, tmp_path / "lightning")
    assert list(tmp_path.iterdir()) == []


def test_composite_base_cannot_be_composite(monkeypatch, tmp_path):
    spec = model_info("sdxl-lightning-2step")
    base = replace(model_info("sdxl-base-1.0"), base="sd-v1-5")
    monkeypatch.setattr(
        pretrained, "model_info", lambda model_id: base if model_id == base.id else spec
    )
    with pytest.raises(DownloadError, match="base.*composite"):
        download_pretrained(spec.id, tmp_path / "lightning")


def test_token_is_sent_to_hugging_face(monkeypatch, tmp_path):
    _, calls = fake_hub(monkeypatch)
    download_pretrained("sd-v1-5", tmp_path / "sd15", token="hf_test")
    assert calls[0][1]["Authorization"] == "Bearer hf_test"


def test_unvalidated_revision_is_flagged(monkeypatch, tmp_path):
    fake_hub(monkeypatch, commit="c" * 40)
    with pytest.warns(UserWarning, match="not the validated revision"):
        dest = download_pretrained("sd-v1-5", tmp_path / "sd15", revision="main")
    assert read_snapshot(dest)["verified_revision"] is False


def test_missing_artifact_fails_before_downloading(monkeypatch, tmp_path):
    _, calls = fake_hub(monkeypatch, drop="vae/config.json")
    with pytest.raises(DownloadError, match="vae/config.json"):
        download_pretrained("sd-v1-5", tmp_path / "sd15")
    assert not any("/resolve/" in url for url, _ in calls)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"corrupt": "unet/diffusion_pytorch_model.safetensors"}, "Checksum"),
        ({"corrupt": "model_index.json"}, "Checksum"),
        ({"short": "model_index.json"}, "Incomplete"),
    ],
)
def test_failed_verification_leaves_nothing_behind(monkeypatch, tmp_path, kwargs, message):
    fake_hub(monkeypatch, **kwargs)
    with pytest.raises(DownloadError, match=message):
        download_pretrained("sd-v1-5", tmp_path / "sd15")
    assert list(tmp_path.iterdir()) == []


def test_read_snapshot_missing(tmp_path):
    assert read_snapshot(tmp_path) is None


@pytest.mark.parametrize(
    "name",
    ["../escape", "/abs/path", "", "modeling.py", "a/b.PY", "x.pyc", "x.pyo", "x.pyw", "m.Py"],
)
def test_safe_relative_rejects(name):
    with pytest.raises(DownloadError):
        safe_relative(name)


def test_cross_host_redirect_does_not_forward_credentials():
    original = urllib.request.Request(
        "https://huggingface.co/owner/repo", headers={"Authorization": "Bearer example"}
    )
    redirected = _Redirect().redirect_request(
        original, None, 302, "Found", {}, "https://cdn.example/model"
    )
    assert redirected.get_header("Authorization") is None
    same_host = _Redirect().redirect_request(
        original, None, 302, "Found", {}, "https://huggingface.co/new"
    )
    assert same_host.get_header("Authorization") == "Bearer example"
    with pytest.raises(DownloadError, match="HTTPS"):
        _Redirect().redirect_request(original, None, 302, "Found", {}, "http://cdn.example/model")


def test_certificate_failure_explains_fix(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.URLError(ssl.SSLCertVerificationError("unable to get issuer"))

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(DownloadError, match="Install Certificates"):
        download._open("https://huggingface.co/api/models/x", {})


def test_main_prints_destination(monkeypatch, tmp_path, capsys):
    fake_hub(monkeypatch)
    assert main(["sd-v1-5", str(tmp_path / "sd15")]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == str((tmp_path / "sd15").resolve())
    assert "unet/diffusion_pytorch_model.safetensors" in out.err
    assert (tmp_path / "sd15" / SNAPSHOT_FILE).is_file()


def test_python_files_are_never_downloaded(monkeypatch, tmp_path):
    _, calls = fake_hub(monkeypatch)
    patched = replace(SPEC, files=(*SPEC.files, "modeling.py"))
    monkeypatch.setattr(pretrained, "model_info", lambda model_id: patched)
    with pytest.raises(DownloadError, match="modeling.py"):
        download_pretrained("sd-v1-5", tmp_path / "sd15")
    assert not any("/resolve/" in url for url, _ in calls)


def test_fetch_refuses_unverifiable_file(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "_open", lambda url, headers: io.BytesIO(b"data"))
    with pytest.raises(DownloadError, match="neither sha256 nor git blob sha1"):
        download.fetch("https://h/x", tmp_path / "x", {}, sha256=None, size=4)
    assert list(tmp_path.iterdir()) == []


def test_fetch_git_sha1_requires_size(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "_open", lambda url, headers: io.BytesIO(b"data"))
    with pytest.raises(DownloadError, match="size"):
        download.fetch(
            "https://h/x",
            tmp_path / "x",
            {},
            sha256=None,
            size=None,
            git_sha1=git_blob_sha1(b"data"),
        )


def test_fetch_verifies_git_blob_sha1(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "_open", lambda url, headers: io.BytesIO(b"data"))
    good = git_blob_sha1(b"data")
    download.fetch("https://h/x", tmp_path / "x", {}, sha256=None, size=4, git_sha1=good)
    with pytest.raises(DownloadError, match="Checksum mismatch"):
        download.fetch("https://h/x", tmp_path / "y", {}, sha256=None, size=4, git_sha1="0" * 40)
    assert [p.name for p in tmp_path.iterdir()] == ["x"]


def test_null_lfs_entry_with_size_downloads(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "_open", lambda url, headers: io.BytesIO(b"data"))
    entry = {"rfilename": "x", "size": 4, "lfs": None}
    digest = download.fetch(
        "https://h/x",
        tmp_path / "x",
        {},
        sha256=(entry.get("lfs") or {}).get("sha256"),
        size=entry.get("size"),
        git_sha1=git_blob_sha1(b"data"),
    )
    assert digest == hashlib.sha256(b"data").hexdigest()


def test_null_lfs_in_download_pretrained(monkeypatch, tmp_path):
    fake_hub(monkeypatch)
    real = download.resolve_revision

    def patched(repo_id, revision, headers):
        commit, siblings = real(repo_id, revision, headers)
        for entry in siblings.values():
            if entry.get("lfs") is None:
                entry["lfs"] = None
        return commit, siblings

    monkeypatch.setattr(download, "resolve_revision", patched)
    download_pretrained("sd-v1-5", tmp_path / "sd15")


def _failing_opener(monkeypatch, error):
    class Opener:
        def open(self, request, timeout):
            raise error

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())


def _http_error(code):
    return urllib.error.HTTPError("https://huggingface.co/x", code, "msg", {}, None)


@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors_mention_gated(monkeypatch, code):
    _failing_opener(monkeypatch, _http_error(code))
    with pytest.raises(DownloadError, match="gated") as info:
        download._open("https://huggingface.co/x", {})
    assert "HF_TOKEN" in str(info.value)


def test_not_found_error(monkeypatch):
    _failing_opener(monkeypatch, _http_error(404))
    with pytest.raises(DownloadError, match="not found") as info:
        download._open("https://huggingface.co/x", {})
    assert "https://huggingface.co/x" in str(info.value)


def test_other_http_error_has_code(monkeypatch):
    _failing_opener(monkeypatch, _http_error(500))
    with pytest.raises(DownloadError, match="500"):
        download._open("https://huggingface.co/x", {})


@pytest.mark.parametrize("error", [urllib.error.URLError("dns failure"), TimeoutError("timed out")])
def test_network_errors_are_download_errors(monkeypatch, error):
    _failing_opener(monkeypatch, error)
    with pytest.raises(DownloadError, match="Network error"):
        download._open("https://huggingface.co/x", {})


def test_stream_oserror_becomes_download_error(monkeypatch, tmp_path):
    class Broken:
        def read(self, n):
            raise ConnectionResetError("reset")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(download, "_open", lambda url, headers: Broken())
    with pytest.raises(DownloadError, match="x.bin"):
        download.fetch("https://h/x.bin", tmp_path / "x.bin", {}, sha256="0" * 64, size=4)
    assert list(tmp_path.iterdir()) == []


def test_main_reports_errors_without_traceback(tmp_path, capsys):
    (tmp_path / "occupied").mkdir()
    (tmp_path / "occupied" / "f").write_text("x")
    assert main(["sd-v1-5", str(tmp_path / "occupied")]) == 1
    assert capsys.readouterr().err.startswith("error:")
