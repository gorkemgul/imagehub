import hashlib
import io
import json
import ssl
import urllib.error
import urllib.request

import pytest

from imagehub import download
from imagehub.catalog import model_info
from imagehub.download import _Redirect, safe_relative
from imagehub.exceptions import DownloadError
from imagehub.pretrained import SNAPSHOT_FILE, download_pretrained, main, read_snapshot

SPEC = model_info("sd-v1-5")


def fake_hub(monkeypatch, *, commit=SPEC.revision, corrupt=None, short=None, drop=None):
    contents = {name: f"content of {name}".encode() for name in SPEC.files}
    siblings = []
    for name, data in contents.items():
        if name == drop:
            continue
        entry = {"rfilename": name, "size": len(data)}
        if name.endswith((".safetensors", ".bin")):
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
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


@pytest.mark.parametrize("name", ["../escape", "/abs/path", ""])
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
