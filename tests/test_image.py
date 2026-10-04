import io
import sys

import pytest
import torch

from imagehub.exceptions import MissingDependencyError
from imagehub.image import Image


def random_image(h=5, w=7, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (h, w, c), generator=g, dtype=torch.uint8)


@pytest.mark.parametrize("channels", [3, 4])
def test_png_roundtrip(tmp_path, channels):
    pixels = random_image(c=channels)
    path = Image(pixels).save(tmp_path / "x.png")
    restored = Image.open(path)
    assert torch.equal(restored.tensor, pixels)
    assert (restored.width, restored.height, restored.channels) == (7, 5, channels)


def test_rejects_invalid_tensors():
    with pytest.raises(TypeError):
        Image(torch.zeros(4, 4, 3))
    with pytest.raises(ValueError):
        Image(torch.zeros(4, 4, 2, dtype=torch.uint8))
    with pytest.raises(ValueError):
        Image(torch.zeros(4, 4, dtype=torch.uint8))


def test_save_requires_png(tmp_path):
    with pytest.raises(ValueError, match="png"):
        Image(random_image()).save(tmp_path / "x.jpg")


def test_from_png_rejects_corruption():
    data = bytearray(Image(random_image()).to_png())
    with pytest.raises(ValueError, match="PNG"):
        Image.from_png(b"not a png")
    data[20] ^= 0xFF  # inside IHDR body -> CRC mismatch
    with pytest.raises(ValueError, match="CRC"):
        Image.from_png(bytes(data))


def test_reads_pillow_filtered_png_and_pillow_reads_ours():
    PIL = pytest.importorskip("PIL.Image")
    pixels = random_image(h=33, w=29, c=3, seed=1)
    gradient = torch.arange(33 * 29 * 3, dtype=torch.int64).reshape(33, 29, 3) % 256
    for source in (pixels, gradient.to(torch.uint8)):
        buffer = io.BytesIO()
        PIL.frombytes("RGB", (29, 33), bytes(source.flatten().tolist())).save(buffer, "PNG")
        assert torch.equal(Image.from_png(buffer.getvalue()).tensor, source)
        decoded = PIL.open(io.BytesIO(Image(source).to_png())).convert("RGB")
        assert bytes(decoded.tobytes()) == bytes(source.flatten().tolist())


def test_to_pil_without_pillow(monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(MissingDependencyError, match=r"imagehub\[image\]"):
        Image(random_image()).to_pil()


def test_image_owns_a_detached_copy():
    pixels = random_image()
    image = Image(pixels)
    pixels.zero_()
    assert image.tensor.sum() > 0
    with torch.inference_mode():
        produced = random_image(seed=2)
    owned = Image(produced)
    assert not owned.tensor.is_inference()
    assert not owned.tensor.requires_grad
