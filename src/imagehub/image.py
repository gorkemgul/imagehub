"""Image container with dependency-free PNG encoding and decoding."""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from imagehub.exceptions import MissingDependencyError

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS = {2: 3, 6: 4}  # PNG color type -> channels (8-bit RGB, 8-bit RGBA)


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def _unfilter(raw: bytes, width: int, height: int, channels: int) -> bytearray:
    stride = width * channels
    out = bytearray(height * stride)
    previous = bytearray(stride)
    pos = 0
    for y in range(height):
        kind = raw[pos]
        line = bytearray(raw[pos + 1 : pos + 1 + stride])
        pos += stride + 1
        if kind == 1:
            for x in range(channels, stride):
                line[x] = (line[x] + line[x - channels]) & 0xFF
        elif kind == 2:
            for x in range(stride):
                line[x] = (line[x] + previous[x]) & 0xFF
        elif kind == 3:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                line[x] = (line[x] + ((left + previous[x]) >> 1)) & 0xFF
        elif kind == 4:
            for x in range(stride):
                left = line[x - channels] if x >= channels else 0
                upper_left = previous[x - channels] if x >= channels else 0
                line[x] = (line[x] + _paeth(left, previous[x], upper_left)) & 0xFF
        elif kind != 0:
            raise ValueError(f"Unsupported PNG filter type {kind}")
        out[y * stride : (y + 1) * stride] = line
        previous = line
    return out


class Image:
    """An 8-bit RGB or RGBA image stored as a (height, width, channels) uint8 tensor."""

    __slots__ = ("tensor",)

    def __init__(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor) or tensor.dtype != torch.uint8:
            raise TypeError("Image expects a uint8 tensor")
        if tensor.ndim != 3 or tensor.shape[2] not in (3, 4) or min(tensor.shape[:2]) < 1:
            raise ValueError("Image tensor must have shape (height, width, 3 or 4)")
        self.tensor = tensor.detach().cpu().contiguous()

    @property
    def height(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def channels(self) -> int:
        return int(self.tensor.shape[2])

    def _bytes(self) -> bytes:
        return bytes(self.tensor.flatten().tolist())

    def to_png(self) -> bytes:
        h, w, c = self.height, self.width, self.channels
        rows = self.tensor.reshape(h, w * c)
        raw = torch.cat([torch.zeros(h, 1, dtype=torch.uint8), rows], dim=1)
        header = struct.pack(">IIBBBBB", w, h, 8, 2 if c == 3 else 6, 0, 0, 0)
        return (
            _SIGNATURE
            + _chunk(b"IHDR", header)
            + _chunk(b"IDAT", zlib.compress(bytes(raw.flatten().tolist()), 6))
            + _chunk(b"IEND", b"")
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".png":
            raise ValueError("Only .png output is supported; use to_pil() for other formats")
        target.write_bytes(self.to_png())
        return target

    @classmethod
    def from_png(cls, data: bytes) -> Image:
        if not data.startswith(_SIGNATURE):
            raise ValueError("Not a PNG file")
        pos, header, idat = len(_SIGNATURE), None, bytearray()
        while pos + 8 <= len(data):
            length, kind = struct.unpack(">I4s", data[pos : pos + 8])
            body = data[pos + 8 : pos + 8 + length]
            crc = data[pos + 8 + length : pos + 12 + length]
            if len(body) != length or len(crc) != 4:
                raise ValueError("Truncated PNG chunk")
            if zlib.crc32(kind + body) & 0xFFFFFFFF != struct.unpack(">I", crc)[0]:
                raise ValueError("PNG chunk CRC mismatch")
            pos += 12 + length
            if kind == b"IHDR":
                header = struct.unpack(">IIBBBBB", body)
            elif kind == b"IDAT":
                idat += body
            elif kind == b"IEND":
                break
        if header is None:
            raise ValueError("PNG is missing its IHDR chunk")
        width, height, depth, color, compression, filtering, interlace = header
        if depth != 8 or color not in _CHANNELS or compression or filtering or interlace:
            raise ValueError("Only 8-bit non-interlaced RGB/RGBA PNG files are supported")
        channels = _CHANNELS[color]
        raw = zlib.decompress(bytes(idat))
        if len(raw) != height * (width * channels + 1):
            raise ValueError("PNG image data has the wrong size")
        pixels = _unfilter(raw, width, height, channels)
        tensor = torch.frombuffer(pixels, dtype=torch.uint8).reshape(height, width, channels)
        return cls(tensor.clone())

    @classmethod
    def open(cls, path: str | os.PathLike[str]) -> Image:
        return cls.from_png(Path(path).read_bytes())

    def to_pil(self) -> Any:
        try:
            from PIL import Image as PILImage
        except ImportError as error:
            raise MissingDependencyError(
                "Image.to_pil() requires Pillow; install it with `pip install imagehub[image]`"
            ) from error
        mode = "RGB" if self.channels == 3 else "RGBA"
        return PILImage.frombytes(mode, (self.width, self.height), self._bytes())

    def __repr__(self) -> str:
        return f"Image(width={self.width}, height={self.height}, channels={self.channels})"
